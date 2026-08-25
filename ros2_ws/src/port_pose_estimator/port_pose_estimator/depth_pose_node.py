"""Publish object pose from the RealSense depth stream, without FoundationPose.

Drop-in alternative to fp_pose_bridge: same topics, same frames, same
world-frame composition, so arm_cmd needs no change and either source can be
run (never both at once -- they would fight over the topic).

What it removes, compared with the FoundationPose route: the GPU, the separate
micromamba environment, the TCP bridge between them, LangSAM, and the 180 deg
silhouette ambiguity. What it needs in exchange: the part must lie flat with its
opening facing the camera, which is how this rig works anyway.

Unlike that route, this one subscribes to realsense2_camera instead of opening
the device itself, so do NOT kill the camera node before starting it -- that
step in the launch notes exists only because FoundationPose wants the device
directly.

    ros2 run port_pose_estimator depth_pose_node
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import TransformBroadcaster
from tm_msgs.msg import FeedbackState

from port_pose_estimator import depth_pose_lib as dpl
from port_pose_estimator import mono_pose_lib as mpl
from port_pose_estimator.fp_pose_bridge import (
    load_T_G_C, tool_pose_is_sane, tool_pose_to_matrix, rotmat_to_quat,
    FEEDBACK_MAX_AGE_S)

REFERENCE_PATH = os.path.join(get_package_share_directory('port_pose_estimator'),
                              'config', 'opening_reference.json')


class MjpegServer:
    """Serve the latest frames as MJPEG, so bring-up needs nothing but a browser.

    Two views, because they answer different questions. The annotated one says
    what the detector decided; the raw one says what it had to work with, which
    is what you want when it decided nothing -- a panel half out of frame, a
    glare patch, the gripper's own shadow across the ports. Reading those off
    the annotated view is hard precisely when it matters, since a frame that
    failed carries almost no annotation.

    Port 8091 rather than 8090 so it never collides with the FoundationPose
    script if that happens to still be running. Built on http.server to avoid
    pulling Flask into the ROS environment.
    """

    PAGE = b"""<!doctype html><meta charset=utf-8><title>depth_pose_node</title>
<style>
 body{margin:0;background:#111;color:#ccc;font:13px system-ui,sans-serif}
 .wrap{display:flex;flex-wrap:wrap;gap:12px;padding:12px}
 figure{margin:0;flex:1 1 420px}
 figcaption{padding:4px 2px;color:#8a8a8a}
 img{width:100%;height:auto;background:#000;border-radius:4px}
</style>
<div class=wrap>
 <figure><img src="/stream"><figcaption>detected &mdash; blue outline is the
  panel, green circles are the ports it matched, red crosses are where the
  solved pose puts them (the gap between the two is the reprojection error)
  </figcaption></figure>
 <figure><img src="/raw"><figcaption>raw camera</figcaption></figure>
</div>"""

    def __init__(self, port=8091, logger=None):
        self.frame = None
        self.raw = None
        self.lock = threading.Lock()
        self.port = port
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass                                  # keep it out of the ROS log

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(outer.PAGE)))
                    self.end_headers()
                    self.wfile.write(outer.PAGE)
                    return
                if self.path == '/stream':
                    attr = 'frame'
                elif self.path == '/raw':
                    attr = 'raw'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type',
                                 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    while True:
                        with outer.lock:
                            buf = getattr(outer, attr)
                        if buf is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(buf)
                        self.wfile.write(b'\r\n')
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass                              # viewer closed the tab

        self.server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        if logger:
            logger.info(f'debug stream on http://localhost:{port}/  '
                        f'(annotated /stream, raw /raw)')

    def _encode(self, bgr, max_side):
        scale = max_side / max(bgr.shape[:2])
        if scale < 1.0:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def update(self, bgr):
        buf = self._encode(bgr, 900)
        if buf:
            with self.lock:
                self.frame = buf

    def update_raw(self, bgr, max_side=640):
        """The camera frame as it arrived. Scaled down -- this is for looking at,
        and shipping 1280x720 at frame rate to a browser buys nothing."""
        buf = self._encode(bgr, max_side)
        if buf:
            with self.lock:
                self.raw = buf


class HeadingLock:
    """Keep the published heading consistent from frame to frame.

    An earlier version of this voted on the raw flip sign, which was a mistake:
    the sign is only meaningful against the angle the moments happened to report
    that frame, and moments return the axis modulo 180 deg, so the sign flips
    harmlessly whenever the reported angle does. Accumulating those votes mixed
    two different conventions together and produced exactly the instability it
    was meant to remove -- headings jumping, and frames refusing to decide.

    Measured over 39 consecutive frames, the per-frame heading was already 100%
    consistent once read as a direction rather than as a sign. So this keeps a
    reference direction and only flips a new heading into agreement with it,
    which costs nothing when the detector is right and contains the damage when
    it is not.
    """

    def __init__(self, move_tol=0.010):
        self.move_tol = move_tol
        self.ref = None
        self.anchor = None

    def apply(self, heading, position):
        """-> heading, turned to agree with the running reference."""
        if self.anchor is None or np.linalg.norm(position - self.anchor) > self.move_tol:
            self.ref = None                 # the part moved; start again
        self.anchor = position
        if self.ref is None:
            self.ref = heading
            return heading, True
        if float(np.dot(heading, self.ref)) < 0:
            heading = -heading              # same axis, opposite end
        self.ref = 0.9 * self.ref + 0.1 * heading
        self.ref /= max(np.linalg.norm(self.ref), 1e-9)
        return heading, False


class DepthPoseNode(Node):
    def __init__(self):
        super().__init__('depth_pose_node')
        # Depth aligned to *colour*, not the raw depth stream: the hand-eye
        # calibration T_G_C was measured against camera_color_optical_frame, so
        # feeding it poses expressed in the depth frame would bake the
        # depth-to-colour extrinsic in as a constant error.
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('object_frame', 'object')
        self.declare_parameter('part', 'rj45_test')
        # 'mono' solves the pose from the colour image alone, against the CAD
        # port table; 'depth' is the original route, which measures the part
        # first. Mono is the default for anything with a port table because the
        # platform is black and its depth returns cannot carry a scale -- see
        # mono_pose_lib. Depth stays reachable for the pale RJ45 jig, which has
        # no port table and so cannot use the pattern match at all.
        self.declare_parameter('method', 'mono')
        # highlight one port in the debug stream; purely for bring-up
        self.declare_parameter('target_port', '')
        self.declare_parameter('min_opening_area', 2e-5)
        self.declare_parameter('max_opening_area', 5e-4)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('stream_port', 8091)
        # height band above the fitted table that counts as "the work"
        self.declare_parameter('min_height', 0.008)
        self.declare_parameter('max_height', 0.060)
        # how far from the work's centre an opening may sit (CAD: ~1mm)
        self.declare_parameter('max_centre_offset_m', 0.008)
        self.declare_parameter('segment_by_grey', False)
        # -1 or +1 pins the 180deg heading by hand; 0 leaves it to the vote
        self.declare_parameter('force_flip', 0)

        self.camera_frame = self.get_parameter('camera_frame').value
        self.object_frame = self.get_parameter('object_frame').value
        part = self.get_parameter('part').value

        with open(REFERENCE_PATH) as f:
            refs = json.load(f)
        if part not in refs:
            raise RuntimeError(f'{part!r} not in {REFERENCE_PATH}; '
                               f'have {sorted(refs)} -- run tools/build_reference.py')
        self.ref = refs[part]
        self.method = self.get_parameter('method').value
        if self.method == 'mono' and not self.ref.get('ports'):
            self.get_logger().warn(
                f'{part!r} has no port table, so the monocular pattern match has '
                'nothing to match against; falling back to the depth route')
            self.method = 'depth'
        self.get_logger().info(
            f'loaded opening reference for {part!r}, method {self.method!r}')

        self.T_G_C = load_T_G_C()
        self.T_world_arm = None
        self.T_world_arm_stamp = None
        self.K = None
        self.bgr = None             # latest colour frame
        self.grey = None            # and its greyscale, which is what gets read
        self.depth = None           # kept even in mono mode, for the tilt check
        self.heading = HeadingLock()

        self.pose_pub = self.create_publisher(PoseStamped, 'camera_frame/object_pose', 10)
        self.world_pose_pub = self.create_publisher(PoseStamped, 'world_frame/object_pose', 10)
        # what the detector saw, for bring-up: a pose alone cannot tell you
        # whether the plane fit or the blob shape was the thing that went wrong
        self.debug_pub = self.create_publisher(Image, 'depth_pose/debug_image', 1)
        self.stream = MjpegServer(self.get_parameter('stream_port').value, self.get_logger())
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(CameraInfo, self.get_parameter('info_topic').value,
                                 self._info_cb, 10)
        self.create_subscription(Image, self.get_parameter('depth_topic').value,
                                 self._depth_cb, 10)
        self.create_subscription(Image, self.get_parameter('color_topic').value,
                                 self._color_cb, 1)
        self.create_subscription(FeedbackState, 'feedback_states', self._feedback_cb, 10)

    # ------------------------------------------------------------ callbacks

    def _info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f'camera intrinsics received: fx={self.K[0,0]:.1f}')

    def _color_cb(self, msg):
        a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        bgr = np.ascontiguousarray(a[:, :, ::-1] if msg.encoding == 'rgb8' else a)
        self.bgr = bgr
        self.grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # published whatever happens downstream: the frame that produced no pose
        # is exactly the one worth being able to look at
        self.stream.update_raw(bgr)
        if self.method != 'mono' or self.K is None:
            return
        # In mono mode this callback is the pipeline -- the depth stream is only
        # consulted for the tilt cross-check, so nothing waits on it.
        T_cam_obj, info, ctx = self._estimate_mono()
        if ctx is not None:
            self._publish_mono_debug(ctx, msg.header.stamp)
        if T_cam_obj is None:
            self.get_logger().warn(f'no pose this frame: {info}',
                                   throttle_duration_sec=2)
            return
        self.get_logger().info(info, throttle_duration_sec=5)
        self._publish(T_cam_obj, msg.header.stamp)

    def _feedback_cb(self, msg):
        # same guards as fp_pose_bridge: a plausible-but-stale tool_pose is more
        # dangerous than an obviously broken one, since it passes every value check
        if not tool_pose_is_sane(msg.tool_pose):
            self.get_logger().error(
                f'tool_pose from tm_driver is garbage ({list(msg.tool_pose)})',
                throttle_duration_sec=5)
            self.T_world_arm = None
            self.T_world_arm_stamp = None
            return
        self.T_world_arm = tool_pose_to_matrix(msg.tool_pose)
        self.T_world_arm_stamp = time.monotonic()

    def _depth_cb(self, msg):
        if self.K is None:
            return
        depth = self._decode(msg)
        if depth is None:
            return
        self.depth = depth
        if self.method == 'mono':
            return                      # the colour callback runs that pipeline
        T_cam_obj, info, ctx = self._estimate(depth)
        # publish the debug view even on failure -- a frame that produced no pose
        # is exactly the one worth looking at
        if ctx is not None:
            self._publish_debug(*ctx, msg.header.stamp)
        if T_cam_obj is None:
            self.get_logger().warn(f'no pose this frame: {info}', throttle_duration_sec=2)
            return
        self._publish(T_cam_obj, msg.header.stamp)

    # -------------------------------------------------------------- helpers

    def _decode(self, msg):
        if msg.encoding == '16UC1':
            raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            d = raw.astype(np.float64) * 0.001          # RealSense 16UC1 is mm
        elif msg.encoding == '32FC1':
            d = np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.width).astype(np.float64)
        else:
            self.get_logger().error(f'unsupported depth encoding {msg.encoding}',
                                    throttle_duration_sec=10)
            return None
        d[d <= 0] = np.nan                              # 0 means "no return"
        return d

    def _publish_debug(self, rect, openings, chosen, axis, axis_y, stamp):
        if not self.get_parameter('publish_debug_image').value:
            return
        self._send_debug(dpl.debug_image(rect, openings, chosen, axis, axis_y), stamp)

    def _send_debug(self, vis, stamp):
        self.stream.update(vis)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame
        msg.height, msg.width = vis.shape[:2]
        msg.encoding = 'bgr8'
        msg.step = vis.shape[1] * 3
        msg.data = vis.tobytes()
        self.debug_pub.publish(msg)

    def _publish_mono_debug(self, ctx, stamp):
        if not self.get_parameter('publish_debug_image').value:
            return
        target = self.get_parameter('target_port').value or None
        vis = mpl.debug_image(*ctx, self.K, target=target)
        self._send_debug(vis, stamp)

    def _estimate_mono(self):
        """-> (T_camera_object | None, info, debug context | None).

        Every stage here reads the colour image. The order matters: find the
        part, then look for ports only inside it, then decide which port is
        which, and only then solve a pose. Detecting ports over the whole frame
        instead works on the bench and fails the moment anything else bright is
        in shot, and there always is.
        """
        grey, bgr = self.grey, self.bgr
        ports = self.ref['ports']

        # Which dark blob is the part is settled by whether the CAD pattern
        # registers inside it, not by how it looks. The gripper's own black body,
        # a cable and a monitor bezel have between them outscored the real panel
        # on brightness and shape -- they are all just dark rectangles -- so the
        # blobs are tried in order and the first one that yields ports wins.
        blobs = mpl.platform_candidates(grey)
        if not blobs:
            return None, 'no dark part found in the frame', None
        # Start from the blob that worked last time, for the same reason
        # find_ports starts from the scale that worked last time.
        prev_blob = getattr(self, '_last_blob', 0)
        order = sorted(range(len(blobs)), key=lambda i: (i != prev_blob, i))
        mask = outline = px_hint = None
        cands, match = [], None
        for k in order:
            m_, o_ = blobs[k]
            hint = mpl.silhouette_scale(o_, self.ref['work_size_m'])
            if not hint:
                continue
            c_, mt_ = mpl.find_ports(grey, m_, hint, ports, K=self.K,
                                     prefer=getattr(self, '_last_tried', None))
            if mask is None:                    # keep the best-scoring for debug
                mask, outline, px_hint, cands = m_, o_, hint, c_
            if mt_ is not None and mt_.get('verified'):
                mask, outline, px_hint, cands, match = m_, o_, hint, c_, mt_
                self._last_blob = k
                self._last_tried = mt_.get('tried')
                if k:
                    self.get_logger().info(
                        f'the top dark blob held no ports; used candidate {k + 1} '
                        f'of {len(blobs)}', throttle_duration_sec=10)
                break
        ctx = (bgr, outline, cands, None, ports, None, None, None)
        if match is None:
            return None, (f'no dark region registered against the {len(ports)}-port '
                          f'CAD table ({len(blobs)} tried, best had {len(cands)} '
                          f'port candidates)'), ctx

        img = np.array([cands[di]['centre'] for _, di in match['pairs']])
        sol = mpl.solve_pose(match['pairs'], ports, img, self.K)
        if sol is None:
            return None, ('no pose with the port face toward the camera -- the '
                          'match is probably mirrored'), ctx
        rvec, tvec, _ = sol

        # second pass: re-measure each port inside its own projected outline
        fixed = mpl.refine_centroids(grey, match['pairs'], ports, rvec, tvec, self.K)
        img = np.array([f if f is not None else img[i]
                        for i, f in enumerate(fixed)])
        sol = mpl.solve_pose(match['pairs'], ports, img, self.K) or sol
        rvec, tvec, err = sol

        # A coplanar target always admits two PnP solutions, and solve_pose's
        # only way to choose between them is reprojection error. When the
        # camera looks nearly square onto the panel that is not enough: both
        # solutions can pass the facing check, and picking by error alone lets
        # image noise flip the choice frame to frame. Seen live, this is what
        # "tilt disagrees with depth by 13-21 deg" actually was -- the wrong
        # twin, not a measurement problem -- and it is also what reads on
        # screen as the X/Y axes suddenly swapping: the two solutions are
        # related by a reflection, so getting the wrong one does not just add
        # noise, it hands back a different, self-consistent-looking heading.
        #
        # Depth breaks the tie outright, because it does not have this
        # ambiguity: a RANSAC plane fit over tens of thousands of points has
        # exactly one normal, not two. Re-solving with that normal held fixed
        # removes the coplanar degeneracy structurally rather than picking
        # between its two symptoms, so it converges to the right answer
        # regardless of which twin solve_pose happened to return first.
        if self.depth is not None and self.depth.shape == grey.shape:
            n = mpl.plane_normal_from_depth(
                self.depth, self.K, cv2.erode(mask, np.ones((21, 21), np.uint8)))
            if n is not None:
                tilt = mpl.tilt_degrees(rvec, n)
                fixed_sol = mpl.refit_with_normal(match['pairs'], ports, img,
                                                  self.K, n, rvec, tvec)
                if fixed_sol is not None and fixed_sol[2] <= max(err * 2, 4.0):
                    rvec, tvec, err = fixed_sol
                    tilt = 0.0          # z is now literally the depth normal
                elif tilt > 10.0:
                    self.get_logger().warn(
                        f'pose tilt disagrees with the depth plane by {tilt:.0f} '
                        'deg and the fixed-normal refit did not reproject well '
                        '-- keeping the image-only pose', throttle_duration_sec=10)
                self._tilt = tilt

        ctx = (bgr, outline, cands, match['pairs'], ports, img, rvec, tvec)
        mm_per_px = float(tvec.ravel()[2]) / self.K[0, 0] * 1000
        info = (f"{match['n_matched']}/{match['n_cad']} ports matched, "
                f"reprojection {err:.2f} px ({err * mm_per_px:.2f} mm), "
                f"range {tvec.ravel()[2] * 1000:.0f} mm")
        if getattr(self, '_tilt', None) is not None:
            info += f', tilt agrees with depth to {self._tilt:.1f} deg'
        return mpl.pose_matrix(rvec, tvec), info, ctx

    def _estimate(self, depth):
        """-> (T_camera_object | None, info, debug context | None)."""
        pts = dpl.deproject(depth, self.K)
        if len(pts) < 500:
            return None, f'only {len(pts)} valid depth points', None

        # Stage 1: the dominant plane in the frame is the table. Fitting it
        # first is what makes the work separable -- the tilt of the camera puts
        # the table across a wide range of depths, so nothing simpler works.
        table = dpl.fit_plane_ransac(pts)
        if table is None:
            return None, 'table plane fit failed', None
        t_normal, t_origin, t_inliers = table
        if t_inliers < 2000:
            return None, f'table plane has only {t_inliers} inliers', None

        # Stage 2: the largest contiguous thing standing on that table is the
        # work. Taking every point above the plane instead lets the table's own
        # depth noise in, and on a white matte surface there is a lot of it.
        expected = self.ref.get('work_size_m')
        work = work_mask = None
        # Segmenting the work by brightness is tempting on a black part -- the
        # depth silhouette comes back ragged and ~35% oversized -- but measured
        # side by side it found fewer ports downstream (4 against 7) because the
        # tighter mask clips ports near the edges. Off by default until that is
        # sorted out; the port detection itself does use greyscale, where it
        # clearly wins.
        if (self.get_parameter('segment_by_grey').value
                and self.grey is not None and self.grey.shape == depth.shape
                and expected):
            work, work_mask = dpl.object_from_grey(
                self.grey, depth, self.K, t_normal, t_origin, expected,
                min_h=self.get_parameter('min_height').value,
                max_h=self.get_parameter('max_height').value)
        if work is None:
            work, work_mask = dpl.object_above_plane(
                depth, self.K, t_normal, t_origin,
                min_h=self.get_parameter('min_height').value,
                max_h=self.get_parameter('max_height').value,
                expected_size=expected)
        if work is None or len(work) < 300:
            n = 0 if work is None else len(work)
            return None, (f'no object found above the table '
                          f'({n} points, table had {t_inliers})'), None

        # Stage 3: the work's top face. It lies flat, so its face is parallel to
        # the table -- reuse that normal, which came from far more points than
        # the small top face could ever provide, and take only the height from
        # the work itself.
        # Where to put the reference plane. Measuring it off the work's own top
        # points is the obvious choice and the wrong one here: this platform is
        # black, and black absorbs the projector's IR, so its depth reads
        # systematically far. That pushed the plane ~7% beyond the real surface,
        # and since the greyscale is sampled by projecting grid cells onto that
        # plane, every feature came back 7% oversized -- enough to stop the port
        # pattern matching the CAD at all.
        #
        # The table is white and gives 50k+ clean points, and the CAD says how
        # thick the part is, so the top face is simply the table lifted by that.
        top_h = np.percentile(dpl.height_above_plane(work, t_normal, t_origin), 85)
        top = work[dpl.height_above_plane(work, t_normal, t_origin) > top_h - 0.004]
        if len(top) < 200:
            return None, f'only {len(top)} points on the top face', None
        normal, origin = t_normal, top.mean(axis=0)

        # cell size follows the sensor's own resolution at this range; a finer
        # grid than the data supports breaks the surface into speckle
        step = dpl.grid_step_for(float(np.median(work[:, 2])), self.K[0, 0])
        rect = dpl.rectify(work, normal, origin, step=step)
        if self.grey is not None and self.grey.shape == depth.shape:
            rect['value_grid'] = dpl.rectify_image(rect, self.grey, self.K)
            # then correct the plane's distance against the CAD's own dimensions
            # and rebuild -- see correct_plane_scale for why depth alone is not
            # good enough here
            fixed, ratio = dpl.correct_plane_scale(
                rect, rect['value_grid'], self.ref['work_size_m'], normal, origin)
            if fixed is not None and abs(ratio - 1.0) > 0.02:
                self.get_logger().info(
                    f'measured {(ratio-1)*100:+.0f}% oversize against the CAD; '
                    'rescaling the reference plane', throttle_duration_sec=10)
                origin = fixed
                step = dpl.grid_step_for(float(origin[2]), self.K[0, 0])
                rect = dpl.rectify(work, normal, origin, step=step)
                rect['value_grid'] = dpl.rectify_image(rect, self.grey, self.K)
        openings = dpl.find_openings(
            rect,
            min_area=self.get_parameter('min_opening_area').value,
            max_area=self.get_parameter('max_opening_area').value)
        if not openings:
            return None, 'no openings found', (rect, [], None, None, None)

        # settle the flip by vote rather than per frame -- see FlipVoter
        # A part with a pattern of ports settles its own heading: matching the
        # detected centres against the CAD table leaves only one way to lie, so
        # none of the single-opening machinery below (long axis from moments, the
        # 180deg probe, the heading lock) is needed or used.
        ports = self.ref.get('ports')
        if ports:
            # ports come from greyscale when it is available -- depth does not
            # resolve them on a black body (see find_openings_grey)
            grey_ops = dpl.find_openings_grey(rect)
            if len(grey_ops) >= 3:
                openings = grey_ops
            T, info = dpl.solve_opening_pattern(rect, openings, ports)
            if T is None:
                return None, info, (rect, openings, None, None, None)
            x_cam, y_cam = T[:3, 0], T[:3, 1]
            return T, info, (rect, openings, None,
                             (float(x_cam @ rect['a1']), float(x_cam @ rect['a2'])),
                             (float(y_cam @ rect['a1']), float(y_cam @ rect['a2'])))

        # Anchor the choice to the middle of the work rather than to wherever it
        # was last frame: the CAD puts the socket essentially at the jig's centre
        # (0.06, -0.65mm on a 43x46mm part), while the no-return patches that
        # keep competing with it sit around the edges. A temporal anchor would
        # instead latch onto the first mistake and stay there.
        jig_centre = np.array(dpl.camera_to_grid(rect, work.mean(axis=0)))
        op, pick_note = dpl.pick_opening(
            openings, self.ref['opening_size_m'], prefer_near=jig_centre,
            max_centre_offset=self.get_parameter('max_centre_offset_m').value / step)
        if op is None:
            return None, pick_note, (rect, openings, None, None, None)
        centroid = dpl.grid_to_camera(rect, op['ci'], op['cj'])
        # The depth width probe stays primary: measured side by side over 40
        # frames it decided 39 of them with a fully consistent heading, against
        # 27 of 40 for the greyscale shell. The shell only comes in when the
        # probe cannot call it, which is where its independent evidence helps.
        frame_sign, margin = dpl.resolve_flip(op, rect, centroid,
                                              self.ref['flip_probe'])
        if frame_sign == 0:
            shell = dpl.find_shell(rect)
            if shell is not None:
                frame_sign, _ratio = dpl.resolve_flip_by_shell(shell, rect, op)

        forced = self.get_parameter('force_flip').value
        sign = forced if forced in (-1, 1) else frame_sign
        if sign == 0:
            return None, 'flip undecidable this frame', (rect, openings, op, None, None)

        T, info = dpl.solve_single_opening(
            rect, openings,
            np.array(self.ref['opening_centroid_in_mesh']),
            self.ref['flip_probe'],
            force_sign=sign, opening=op)
        if T is None:
            return None, info, (rect, openings, op, None, None)

        # lock the heading to the running reference before anything downstream
        # sees it, then rebuild the frame around whichever end that settled on
        x_cam = T[:3, 0]
        x_cam, fresh = self.heading.apply(x_cam, T[:3, 3])
        z_cam = T[:3, 2]
        y_cam = np.cross(z_cam, x_cam)
        T[:3, 0], T[:3, 1] = x_cam, y_cam
        if fresh:
            self.get_logger().info('heading reference set')

        axis = (float(x_cam @ rect['a1']), float(x_cam @ rect['a2']))
        axis_y = (float(y_cam @ rect['a1']), float(y_cam @ rect['a2']))
        return T, info, (rect, openings, info['chosen'], axis, axis_y)

    def _publish(self, T_cam_obj, stamp):
        self._send(self.pose_pub, T_cam_obj, self.camera_frame, stamp)

        if self.T_world_arm is None:
            self.get_logger().warn('no feedback_states yet, skipping world-frame pose',
                                   throttle_duration_sec=5)
            return
        age = time.monotonic() - self.T_world_arm_stamp
        if age > FEEDBACK_MAX_AGE_S:
            self.get_logger().error(
                f'feedback_states is {age:.1f}s stale -- refusing to publish a '
                'world-frame pose built on it', throttle_duration_sec=5)
            return

        self._send(self.world_pose_pub, self.T_world_arm @ self.T_G_C @ T_cam_obj,
                   'world', stamp)

    def _send(self, pub, T, frame_id, stamp):
        t = T[:3, 3]
        qx, qy, qz, qw = rotmat_to_quat(T[:3, :3])

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = frame_id
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = t.tolist()
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        pub.publish(ps)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = frame_id
        tf.child_frame_id = self.object_frame
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = t.tolist()
        tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = DepthPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
