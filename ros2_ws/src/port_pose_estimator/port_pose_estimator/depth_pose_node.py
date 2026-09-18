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
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
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
 figure{margin:0;flex:0 1 auto}
 figcaption{padding:4px 2px;color:#8a8a8a}
 img{display:block;max-width:100%;max-height:80vh;width:auto;height:auto;
     background:#000;border-radius:4px}
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
        # When true, the color/depth callbacks stop overwriting self.bgr/
        # self.grey/self.depth with whatever the camera just sent, pinning
        # the pipeline to whichever single frame was live at the moment this
        # flipped on -- every subsequent "port <name>"/hover/where/truth
        # after that is then computed from that one exact image, over and
        # over, instead of a fresh capture each time. Lets arm_cmd's "save0"/
        # "end0" isolate frame-to-frame vision noise (and identification
        # flipping between candidates) from everything else in the chain: if
        # per-hole error still varies while frozen, the camera/detector is
        # not the cause.
        self.declare_parameter('freeze_frame', False)
        self.declare_parameter('part', 'rj45_test')
        # Same upstream task file arm_cmd reads. Here only the panel name
        # matters -- it selects the CAD table -- but the target is taken too,
        # so the debug stream highlights the socket the job is about.
        self.declare_parameter('task_json', '')
        # 'mono' solves the pose from the colour image alone, against the CAD
        # port table; 'depth' is the original route, which measures the part
        # first. Mono is the default for anything with a port table because the
        # platform is black and its depth returns cannot carry a scale -- see
        # mono_pose_lib. Depth stays reachable for the pale RJ45 jig, which has
        # no port table and so cannot use the pattern match at all.
        self.declare_parameter('method', 'mono')
        # highlight one port in the debug stream; purely for bring-up
        self.declare_parameter('target_port', '')
        # A frame this poorly self-consistent is not worth feeding to the
        # smoother at all -- see _estimate_mono's gate just before the
        # PoseSmoother push. 4.0 px is the tilt+edge-centroids pipeline's
        # own measured range on a good real capture (1.65-4.17 px); this
        # does not reject the panel's own hard cases outright; it stops one
        # from being smoothed into looking like a settled answer.
        self.declare_parameter('max_reproj_px', 4.0)
        # How many recent accepted frames the published pose is a median of.
        # This is the knob that actually controls how often a visibly crooked
        # frame reaches the arm -- see PoseSmoother's docstring for the
        # measurement behind the default.
        self.declare_parameter('pose_window', 9)
        # The per-hole route (frame_rect/ring_rect + ray_plane -- see
        # mono_pose_lib.HoleSmoother) that aims at the target port's own
        # measured pixel instead of pose (x) CAD's projection of it. Same
        # window-size rationale as pose_window; max_gap is frames, not
        # seconds, since it is compared against a plain miss counter rather
        # than a timestamp.
        self.declare_parameter('hole_window', 9)
        self.declare_parameter('hole_max_gap', 30)
        # Off switches this back to pose (x) CAD for every port, with no code
        # change -- for if the live stream shows this method doing worse
        # than the old one on some port the bench frames didn't cover.
        self.declare_parameter('hole_measure_enabled', True)
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

        task_path = self.get_parameter('task_json').value
        self.task_port = None
        if task_path:
            self.get_logger().error(
                f'task_json={task_path!r} given, but task_file support is not '
                f'wired up in this build -- keeping part={part!r}')

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
        # _color_cb runs on its own callback group/thread (see its
        # create_subscription call and main()'s MultiThreadedExecutor), so it
        # can no longer starve feedback_states off the default group -- before
        # that split, a multi-second vision frame delayed this callback too,
        # and by the time it ran, "just received" would already read stale.
        # This lock guards T_world_arm/T_world_arm_stamp as a pair so the
        # publish path (running on the other thread) never sees a stamp from
        # one update paired with the matrix from another.
        self._feedback_lock = threading.Lock()
        self.T_world_arm = None
        self.T_world_arm_stamp = None
        self.K = None
        self.bgr = None             # latest colour frame
        self.grey = None            # and its greyscale, which is what gets read
        self.depth = None           # kept even in mono mode, for the tilt check
        self.heading = HeadingLock()
        # The part sits still between attempts, so consecutive frames should
        # barely differ -- median-filtering a short window over trusting
        # whichever frame just happened to solve is free variance reduction,
        # and rejecting a single-frame jump against recent history catches
        # the flicker a weak candidate on that one frame's ladder attempt can
        # cause. See mono_pose_lib.PoseSmoother.
        self.smoother = mpl.PoseSmoother(
            size=int(self.get_parameter('pose_window').value))
        # Which target_port the smoother's history was built against -- see
        # its own reset() call in _estimate_mono, right where target_port is
        # re-read each frame.
        self._last_target = None
        # Aims the target port's published position at the pixel the camera
        # actually shows for it, not at pose (x) CAD's projection -- see
        # _estimate_mono's per-hole correction block, and
        # tools/test_direct_target.py for the experiment this is lifted from.
        self.hole_smoother = mpl.HoleSmoother(
            size=int(self.get_parameter('hole_window').value),
            max_gap=int(self.get_parameter('hole_max_gap').value))
        self._measured_uvs = None
        self._panel_centre = None
        self._layout_consistent = True
        self._recovered_uvs = {}
        self._recovered_3d = {}
        self._all_ports_found = False
        # find_outer_quad is a single Canny detection with no temporal memory
        # of its own -- unlike each hole's own frame_rect/ring_rect reading,
        # which HoleSmoother already holds across a short gap. Without the
        # same holding here the axes would blink out on any frame Canny
        # happens to miss the panel's outer edge, even though nothing about
        # the panel actually changed. size=1 is deliberate: this is a
        # position/orientation pair, not a per-pixel measurement worth
        # median-filtering, so only the gap-holding half of HoleSmoother is
        # wanted here.
        # size=1 (pure hold, no averaging) made the displayed arrow jump by
        # the full amount of any two consecutive readings' own small Canny/
        # sub-pixel noise -- confirmed live, ~12px between successful reads
        # 2026-09-08. size=5 medians that noise down the same way
        # HoleSmoother already does for each hole's own pixel, at the cost
        # of reacting to a genuine panel move a few frames slower.
        self._panel_axes_smoother = mpl.HoleSmoother(size=5, max_gap=30)

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
        # _color_cb is the one that runs case1's multi-second, 19-port,
        # port_ladder-retrying match -- it gets its own callback group so that
        # run can never delay anything else: not feedback_states (see the
        # _feedback_lock comment above), and not the node's own built-in
        # ~/set_parameters service either (arm_cmd's set_target_port/freeze
        # call that, and it shares the default group with everything below).
        vision_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, self.get_parameter('color_topic').value,
                                 self._color_cb, 1, callback_group=vision_group)
        self.create_subscription(FeedbackState, 'feedback_states', self._feedback_cb, 10)

    # ------------------------------------------------------------ callbacks

    def _info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f'camera intrinsics received: fx={self.K[0,0]:.1f}')

    def _color_cb(self, msg):
        frozen = bool(self.get_parameter('freeze_frame').value)
        if not (frozen and self.bgr is not None):
            a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            bgr = np.ascontiguousarray(a[:, :, ::-1] if msg.encoding == 'rgb8' else a)
            self.bgr = bgr
            self.grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # published whatever happens downstream: the frame that produced no pose
        # is exactly the one worth being able to look at
        self.stream.update_raw(self.bgr)
        if self.method != 'mono' or self.K is None:
            return
        # In mono mode this callback is the pipeline -- the depth stream is only
        # consulted for the tilt cross-check, so nothing waits on it.
        T_cam_obj, info, ctx, confirmed = self._estimate_mono()
        if ctx is not None:
            self._publish_mono_debug(ctx, msg.header.stamp, confirmed)
        if T_cam_obj is None:
            self.get_logger().warn(f'no pose this frame: {info}',
                                   throttle_duration_sec=2)
            return
        self.get_logger().info(info, throttle_duration_sec=5)
        self._publish(T_cam_obj, msg.header.stamp, confirmed)

    def _feedback_cb(self, msg):
        # same guards as fp_pose_bridge: a plausible-but-stale tool_pose is more
        # dangerous than an obviously broken one, since it passes every value check
        if not tool_pose_is_sane(msg.tool_pose):
            self.get_logger().error(
                f'tool_pose from tm_driver is garbage ({list(msg.tool_pose)})',
                throttle_duration_sec=5)
            with self._feedback_lock:
                self.T_world_arm = None
                self.T_world_arm_stamp = None
            return
        T_world_arm = tool_pose_to_matrix(msg.tool_pose)
        with self._feedback_lock:
            self.T_world_arm = T_world_arm
            self.T_world_arm_stamp = time.monotonic()

    def _depth_cb(self, msg):
        if self.K is None:
            return
        if self.get_parameter('freeze_frame').value and self.depth is not None:
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
        # This route has no PoseSmoother (see _estimate_mono) -- it always
        # published every frame it solved before that existed, so it keeps
        # doing exactly that; confirmed=True here is "unconditionally", not
        # "settled".
        self._publish(T_cam_obj, msg.header.stamp, confirmed=True)

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

    def _publish_mono_debug(self, ctx, stamp, confirmed):
        if not self.get_parameter('publish_debug_image').value:
            return
        target = (self.get_parameter('target_port').value
                  or getattr(self, 'task_port', None) or None)
        vis = mpl.debug_image(*ctx, self.K, target=target,
                              measured_uv=self._measured_uvs,
                              show_other=False, show_outline=False,
                              show_candidates=False, panel_centre=self._panel_centre,
                              recovered_uv=self._recovered_uvs)
        # The rejection text already lived in the log line (see
        # _estimate_mono's info string), which nobody staring at the MJPEG
        # stream in a browser ever sees. This is the same status, on the
        # thing actually being watched: world_frame/object_pose -- what
        # arm_cmd reads -- only publishes once this reads CONFIRMED, so a
        # viewer can tell "not moving yet" from "moved, just looks off"
        # without cross-referencing the log. Split the "not yet" case in
        # two: a pose that just has not settled looks different on screen
        # (and means something different) from one where the holes
        # currently on screen actively contradict CAD's own layout -- see
        # mpl.inconsistent_ports and _estimate_mono's confirmed gate.
        if confirmed:
            label, col = 'CONFIRMED - published to arm', (0, 200, 0)
        elif not self._layout_consistent:
            label, col = 'holes disagree with CAD layout - not published', (0, 0, 220)
        else:
            label, col = 'settling... not published', (0, 140, 255)
        y0 = vis.shape[0] - 22
        cv2.rectangle(vis, (0, y0), (vis.shape[1], vis.shape[0]), col, -1)
        cv2.putText(vis, label, (4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 0, 0), 1, cv2.LINE_AA)
        # Current range, top-right, small -- self._panel_centre, not ctx's
        # own tvec. tvec is target-dependent (the target's own ray_plane
        # point when one is set and measured/recovered this frame, but the
        # coarse *identification* pose -- PnP, not ray_plane at all --
        # otherwise), so the same number meant two different things
        # depending on target_port state. panel_centre is always the
        # panel's own outer-quad centre via ray_plane (see find_outer_quad/
        # panel_axes_pose), regardless of target_port, so this reads the
        # same way every time.
        pc = self._panel_centre
        if pc is not None:
            depth_text = f'{float(pc[2]) * 1000:.0f}mm'
            (tw, th), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            pos = (vis.shape[1] - tw - 6, th + 6)
            # Dark outline first -- a pale colour picked to stay visible on a
            # bright panel disappears the moment it crosses a dark socket,
            # and vice versa; the outline keeps it legible on either.
            cv2.putText(vis, depth_text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                       (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, depth_text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                       (255, 200, 120), 1, cv2.LINE_AA)
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
        # Reset up front, not just where they get actually computed further
        # down: several stages between here and there can return early
        # (no dark part found, no identification, no pose), and without
        # this a frame that fails one of those would show whatever these
        # held from a previous, unrelated frame instead of the neutral
        # "did not get that far".
        self._layout_consistent = True
        self._all_ports_found = True
        self._recovered_uvs = {}
        self._recovered_3d = {}

        # Which dark blob is the part is settled by whether the CAD pattern
        # registers inside it, not by how it looks. The gripper's own black body,
        # a cable and a monitor bezel have between them outscored the real panel
        # on brightness and shape -- they are all just dark rectangles -- so the
        # blobs are tried in order and the first one that yields ports wins.
        #
        # Otsu over the whole frame (platform_candidates) cannot separate the
        # part from equally dark clutter touching it in shot -- a chair, boxes,
        # shelving. A panel whose reference entry sets platform_from: "depth"
        # uses the depth stream to isolate it instead: the part sits close to
        # the camera and clutter sits further back, so gating the near cluster
        # of the depth histogram and keeping only the pixels recessed below its
        # fitted plane finds the part's own face regardless of what else is in
        # frame. See mono_pose_lib.platform_from_depth.
        depth_platform = self.ref.get('platform_from') == 'depth'
        # Identify with Canny outright, for a panel using the image-only
        # hole-centre method: confirmed on real captures this session that
        # canny_candidates reliably finds usb1/usb2 where the brightness
        # ladder below routinely does not (it never finds usb1 at all), so
        # running the ladder to identify and then canny_candidates again a
        # few lines later just to place the measurement window was two
        # passes doing the job one already did better. See canny_candidates.
        use_canny_id = depth_platform and self.get_parameter('hole_measure_enabled').value
        if depth_platform:
            if self.depth is None or self.depth.shape != grey.shape:
                return None, 'depth frame not ready for a depth-isolated platform', None, False
            blobs = mpl.platform_from_depth(self.depth, self.K, max_n=3)
        else:
            blobs = mpl.platform_candidates(grey)
        if not blobs:
            return None, 'no dark part found in the frame', None, False
        # Start from the blob that worked last time, for the same reason
        # find_ports starts from the scale that worked last time.
        prev_blob = getattr(self, '_last_blob', 0)
        order = sorted(range(len(blobs)), key=lambda i: (i != prev_blob, i))
        mask = outline = px_hint = None
        cands, match = [], None
        ladder = self.ref.get('port_ladder')
        for k in order:
            m_, o_ = blobs[k]
            # silhouette_scale assumes the outline IS the CAD-sized footprint,
            # which a depth-isolated mask is not (it can exclude the raised
            # frame, or include background that bled in) -- scale_from_depth
            # reads the real range off the depth stream instead, sidestepping
            # the mask's own shape. See mono_pose_lib.scale_from_depth.
            if depth_platform:
                hint = mpl.scale_from_depth(self.depth, m_, self.K)
                # A depth-isolated mask that is not actually the panel (too
                # small, too sparse, a spurious recessed sliver) still gives
                # scale_from_depth *a* number -- checking it against the CAD
                # panel's real size here stops a contaminated scale before
                # it can poison the port search. See mask_size_consistent.
                if hint and not mpl.mask_size_consistent(m_, hint, self.ref['work_size_m']):
                    # The commonest way this fails live: the near-depth
                    # cluster fuses the panel with something else at a
                    # similar range (this rig's own vent grille, sitting
                    # right next to the I/O shield) into one connected
                    # blob roughly 2x too wide -- confirmed on real
                    # captures, not assumed. scale_from_depth's own number
                    # is usually still fine even then, since it only reads
                    # the *median depth*, not the blob's shape -- so
                    # find_outer_quad gets a real chance to recover the
                    # true rectangle by shape+size inside this blob's own
                    # (generously padded) region, the same Canny search
                    # panel_axes_pose uses for orientation, repurposed here
                    # to rescue the mask itself. Recovered 2 of 3 real
                    # fused-blob failures in a live burst test.
                    quad = mpl.find_outer_quad(grey, m_, hint, self.ref['work_size_m'])
                    if quad is not None:
                        m2 = np.zeros_like(m_)
                        cv2.fillPoly(m2, [np.round(quad).astype(np.int32)], 255)
                        hint2 = mpl.scale_from_depth(self.depth, m2, self.K)
                        if hint2 and mpl.mask_size_consistent(m2, hint2, self.ref['work_size_m']):
                            m_, o_, hint = m2, np.round(quad).astype(np.int32).reshape(-1, 1, 2), hint2
                        else:
                            hint = None
                    else:
                        hint = None
            else:
                hint = mpl.silhouette_scale(o_, self.ref['work_size_m'])
            if not hint:
                continue
            if use_canny_id:
                inner = cv2.erode(m_, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
                c_ = mpl.canny_candidates(grey, inner, hint, ports)
                mt_ = mpl.match_ports(c_, ports, hint, K=self.K,
                                     min_pairs=self.ref.get('min_pairs'))
            elif ladder:
                c_, mt_ = mpl.find_ports_ladder(
                    grey, m_, hint, ports, ladder, K=self.K,
                    prefer=getattr(self, '_last_tried', None),
                    min_pairs=self.ref.get('min_pairs'),
                    max_reproj_px=self.ref.get('max_reproj_px', 6.0))
            else:
                c_, mt_ = mpl.find_ports(grey, m_, hint, ports, K=self.K,
                                         prefer=getattr(self, '_last_tried', None))
            if mask is None:                    # keep the best-scoring for debug
                mask, outline, px_hint, cands = m_, o_, hint, c_
            if mt_ is not None and mt_.get('verified'):
                mask, outline, px_hint, cands, match = m_, o_, hint, c_, mt_
                self._last_blob = k
                if not use_canny_id:
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
                          f'port candidates)'), ctx, False

        img = np.array([cands[di]['centre'] for _, di in match['pairs']])
        if use_canny_id:
            # match['pairs'] are Canny's own candidates here (the loop above
            # ran canny_candidates+match_ports directly), so this pose IS
            # already the accurate coarse pose frame_rect/ring_rect need --
            # no second Canny pass to get one, unlike the brightness route
            # below. solve_and_refine's extra refinement stages
            # (edge_centroids, refine_centroids) were tuned against
            # brightness-sourced candidates specifically; plain solve_pose
            # here matches what tools/test_direct_target.py already
            # validated this same candidate source against.
            sol = mpl.solve_pose(match['pairs'], ports, img, self.K, None)
            if sol is None:
                return None, ('no pose with the port face toward the camera -- '
                              'the match is probably mirrored'), ctx, False
            rvec, tvec, err = sol
            pairs = match['pairs']
        else:
            # solve, then re-measure each port inside its own projected outline,
            # then check for any inlier whose blob measured far off its assigned
            # port's own size -- keeping whichever of the resulting poses
            # reprojects best. See mono_pose_lib.solve_and_refine: the refined
            # pass is not always the one to trust, and past a point candidates
            # is worth passing precisely because match_ports never rejects a
            # correspondence for its measured size, only for its position.
            sol = mpl.solve_and_refine(match['pairs'], ports, img, grey, self.K,
                                       candidates=cands)
            if sol is None:
                return None, ('no pose with the port face toward the camera -- the '
                              'match is probably mirrored'), ctx, False
            rvec, tvec, err, img, pairs = sol
        # frame_rect/ring_rect's window placement below needs a coarse
        # pose, not whatever the tilt refit two paragraphs down replaces
        # rvec/tvec with -- the refit's own few-px shift is enough to flip
        # which metal fragment counts as "closest to the quad centre" on an
        # asymmetric L-bracket. See mono_pose_lib.frame_rect.
        rvec0, tvec0 = rvec, tvec
        if not use_canny_id:
            # It also should not be *this* coarse pose when it can be helped:
            # production identification (find_ports_ladder, just above) is
            # brightness-based and routinely never matches usb1 and misses
            # usb2 too -- confirmed on a real capture, not assumed -- so
            # their own window would be placed from a pose fit entirely
            # without their own evidence, off by just enough to break
            # frame_rect's fragment pick for exactly those two ports.
            # canny_candidates finds the boundary every socket actually has
            # (silver frame against dark interior) rather than thresholding
            # for brightness, and reliably includes both. Only reachable
            # for a panel not using the Canny-identification route above,
            # where this second pass is genuinely new information rather
            # than a repeat of it.
            inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            ccands = mpl.canny_candidates(grey, inner, px_hint, ports)
            cmatch = mpl.match_ports(ccands, ports, px_hint, K=self.K,
                                     min_pairs=self.ref.get('min_pairs'))
            if cmatch is not None and cmatch.get('verified'):
                cimg = np.array([ccands[di]['centre'] for _, di in cmatch['pairs']])
                csol = mpl.solve_pose(cmatch['pairs'], ports, cimg, self.K, None)
                if csol is not None:
                    rvec0, tvec0, _ = csol

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
        eroded = None
        if self.depth is not None and self.depth.shape == grey.shape:
            eroded = cv2.erode(mask, np.ones((21, 21), np.uint8))
            n = mpl.plane_normal_from_depth(self.depth, self.K, eroded)
            if n is not None:
                tilt = mpl.tilt_degrees(rvec, n)
                fixed_sol = mpl.refit_with_normal(pairs, ports, img,
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

        # Every real socket's own pixel, not pose (x) CAD's projection of
        # it -- see mono_pose_lib's "image-only hole centre"
        # (frame_rect/ring_rect/plane_from_depth/ray_plane), and
        # tools/test_direct_target.py for the repeatability experiment this
        # is lifted from. The identification pipeline above (find_ports*,
        # match_ports, solve_and_refine, edge_centroids) still only says
        # which candidate is which named port; nothing about the *position*
        # or *orientation* actually published comes from it at all any more.
        #
        # Every port gets measured and turned into its own 3D point
        # (self._measured_uvs / self._measured_3d), with hand-eye alone
        # (K + the depth-measured plane -- see ray_plane): no CAD position
        # anywhere in that chain. The published tvec becomes the current
        # job's own target port's own point directly -- not a shift applied
        # to the identification pose's tvec -- so arm_cmd must not add its
        # usual CAD centre_in_mesh on top; see arm_cmd's
        # direct_hole_position parameter, which this pairs with. If the
        # target port has no measurement this frame (HoleSmoother gave up
        # holding one), this is "no pose this frame" outright -- there is no
        # CAD fallback for it any more; publishing pose (x) CAD's guess here
        # is exactly the contamination this whole method exists to remove,
        # and a stale-but-real position from PoseSmoother's own history is a
        # better wrong answer than a fresh CAD one anyway.
        #
        # Orientation the same way: mpl.panel_axes_pose reads the panel's own
        # outer rectangle (mpl.find_outer_quad, Canny) for which way is the
        # long axis (screen up = +Y) and which is the short one (screen
        # right = +X), each actually measured in 3D via ray_plane against
        # the depth plane -- not assumed from on-screen angles, which
        # perspective distorts, and not a PnP result. Z is the depth plane's
        # own normal. No per-hole baseline voting: one rectangle, always the
        # same physical edges, is simpler and does not change answer
        # depending on which two holes happened to be measured this frame.
        #
        # depth_platform gates this because plane_from_depth's recess-
        # trimming assumes the depth-isolated panel mask, which is what that
        # flag selects -- a greyscale silhouette mask on some other jig is
        # not what this was tuned or verified against.
        target = (self.get_parameter('target_port').value
                  or getattr(self, 'task_port', None) or None)
        if target != self._last_target:
            # A target switch is a deliberate jump to a different port's own
            # position, not per-frame sensor noise -- without this,
            # PoseSmoother's own jump-rejection (max_trans_jump_m, see its
            # docstring) treats the new target's very different tvec as a
            # bad frame and keeps rejecting it until the old target's
            # history individually ages past max_age_s (2s), costing a
            # ~2s stall on every switch for no reason. reset() clears that
            # history outright so the very next frame is judged on its own.
            self.smoother.reset()
            self._last_target = target
        self._measured_uvs = None
        self._measured_3d = {}
        self._panel_centre = None
        self._recovered_uvs = {}
        self._recovered_3d = {}
        self._all_ports_found = False
        if (depth_platform and eroded is not None
                and self.get_parameter('hole_measure_enabled').value):
            plane = mpl.plane_from_depth(self.depth, self.K, eroded)
            if plane is not None:
                origin, plane_normal = plane
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                # Every port's own raw pixel this frame, before any of it
                # touches HoleSmoother's history -- inconsistent_ports needs
                # this frame's own readings together to cross-check them
                # against each other, which a value already blended with
                # past frames could no longer do honestly.
                raw_3d = {}
                raw_uv = {}
                # match_ports already knows exactly where each matched port's
                # own candidate sits in the image -- that correspondence came
                # from a robust whole-panel search, not from this coarse
                # pose. Anchoring frame_rect/ring_rect's window there instead
                # of at the pose's own projection keeps the CAD-accurate
                # size/orientation _port_quad still provides, but removes the
                # failure mode where two closely-spaced ports' windows get
                # pulled toward each other by a few px of pose-fit noise that
                # has nothing to do with either port's own detection.
                #
                # Live testing (2026-09-08): previously gated to nearest-CAD-
                # neighbour < 10mm only, because unconditional anchoring once
                # measured worse for hdmi1 offline (0.72mm -> 1.24mm mean
                # CAD-distance error on one frame, isolated ports gaining
                # nothing from rescue while trading away the coarse pose's
                # multi-point averaging). Live match rates this session ran
                # far lower than the offline test frames (7-10/19 rather
                # than typically higher), so usb1/usb2 kept falling back to
                # the ungated, drift-prone path anyway. Removed the gate to
                # test unconditionally -- every matched port anchors on its
                # own candidate now, isolated or not. Revert by restoring
                # the `near[name] < 0.010` condition below if this regresses
                # ports like hdmi1 on hardware.
                own_quads = {p['name']: mpl._port_quad(p, rvec0, tvec0, self.K, None)
                            for p in ports if p.get('kind') != 'other'}
                own_centres = {n: q.mean(axis=0) for n, q in own_quads.items()}
                matched_uv = {ports[ci]['name']: cands[di]['centre'] for ci, di in pairs}
                for port in ports:
                    if port.get('kind') == 'other':
                        continue
                    name = port['name']
                    quad = own_quads[name]
                    if name in matched_uv:
                        quad = quad - own_centres[name] + matched_uv[name]
                    rect = (mpl.ring_rect(grey, hsv, quad) if name == 'hdmi1'
                           else mpl.frame_rect(grey, hsv, quad))
                    if rect is None:
                        continue
                    l, t, r, b = rect
                    # A wrong fragment doesn't only drift toward a neighbour
                    # -- it can just be the wrong *shape*, e.g. a sliver of
                    # reflection or a partial edge, still sitting roughly
                    # where the real opening is. CAD already gives this
                    # port's own expected size (own_quads' own bbox, at
                    # today's scale); a result under half of it in either
                    # dimension is not this opening at all. Caught two real
                    # bad reads this way offline that nothing else here did
                    # (case102's rj451 at 0.20x/0.28x expected, case110's
                    # usb4 at 0.36x/0.00x -- a zero-height box).
                    ow = own_quads[name][:, 0].max() - own_quads[name][:, 0].min()
                    oh = own_quads[name][:, 1].max() - own_quads[name][:, 1].min()
                    # 0.55, not 0.5: offline against the captured test frames,
                    # raising it this far catches one more real borderline
                    # case (case110's usb5, sitting right against the two
                    # confirmed-bad readings at 0.52x) with no cost -- every
                    # legitimate reading in that set stayed at 0.59x or
                    # above. Higher starts cutting those (case106's usb4 at
                    # 0.59x is the next one lost, at 0.6).
                    if (r - l) < 0.55 * ow or (b - t) < 0.55 * oh:
                        continue
                    uv_c = np.array([(l + r) / 2, (t + b) / 2])
                    # Rejects rather than guesses when the measured centre
                    # ended up closer to a NEIGHBOUR's own expected slot than
                    # to this port's own -- catches the drift-into-neighbour
                    # failure regardless of whether the anchoring above fired
                    # for this port this frame.
                    own_d = np.linalg.norm(uv_c - own_centres[name])
                    nearest_other = min(
                        (np.linalg.norm(uv_c - c) for n2, c in own_centres.items() if n2 != name),
                        default=np.inf)
                    if nearest_other < own_d:
                        continue
                    # ray_plane, not the port's own depth pixel: a hole is a
                    # recess, so its own depth reads "behind the panel face",
                    # not "where the opening is" -- see plane_from_depth.
                    P_c = mpl.ray_plane(uv_c, self.K, origin, plane_normal)
                    if P_c is not None:
                        raw_uv[name] = uv_c
                        raw_3d[name] = P_c

                # A wrong metal fragment still returns a confident, in-range
                # pixel -- nothing about it alone looks wrong. But it is not
                # this hole's own frame, so its distance to every other
                # measured hole will not match what CAD says that distance
                # is, even though no single frame_rect/ring_rect call could
                # have caught that on its own. See mpl.inconsistent_ports.
                # Flagged ports are treated as unmeasured this frame -- a
                # miss HoleSmoother already knows how to hold through --
                # rather than let a wrong-but-confident reading into its
                # history at all.
                bad = mpl.inconsistent_ports(raw_3d, ports)
                if bad:
                    self.get_logger().warn(
                        f'{sorted(bad)} disagree with the rest of the panel\'s '
                        'own CAD layout this frame -- treating as unmeasured',
                        throttle_duration_sec=5)
                self._layout_consistent = len(raw_3d) - len(bad) >= 2 and not bad

                self._measured_uvs = {}
                for port in ports:
                    if port.get('kind') == 'other':
                        continue
                    name = port['name']
                    uv_raw = raw_uv.get(name) if name not in bad else None
                    uv = self.hole_smoother.push(name, uv_raw)
                    self._measured_uvs[name] = uv
                    if uv is None:
                        continue
                    P_meas = mpl.ray_plane(uv, self.K, origin, plane_normal)
                    if P_meas is not None:
                        self._measured_3d[name] = P_meas

                # Held across a short gap (self._panel_axes_smoother, size=1
                # so this is pure hold, no averaging) the same way each
                # hole's own reading is -- find_outer_quad is one Canny
                # detection with no memory of its own, and without holding,
                # the axes would blink out on any single frame it happens to
                # miss the panel's outer edge, even though the panel itself
                # has not moved.
                quad_outer = mpl.find_outer_quad(grey, mask, px_hint,
                                                 self.ref['work_size_m'])
                rvec_a = centre_a = None
                if quad_outer is not None:
                    rvec_a = mpl.panel_axes_pose(quad_outer, self.K, origin, plane_normal)
                    centre_a = mpl.ray_plane(
                        quad_outer.mean(axis=0), self.K, origin, plane_normal)
                held_rvec = self._panel_axes_smoother.push(
                    'rvec', rvec_a.ravel() if rvec_a is not None else None)
                # For the debug view only -- nothing published depends on
                # this point.
                self._panel_centre = self._panel_axes_smoother.push('centre', centre_a)
                if held_rvec is not None:
                    rvec = held_rvec.reshape(3, 1)

                # Every named socket needs *a* position for CONFIRMED now,
                # not just whichever ones this frame's own measurements
                # happened to cover -- see mpl.recover_missing. This is
                # pose (x) CAD again, deliberately, for exactly the sockets
                # that were not real measurements anyway.
                #
                # 2026-09-08: at the user's explicit request, the target
                # port's own check below now falls back to this too when it
                # has no direct measurement -- previously refused outright.
                # This is a real relaxation of the rule stated just above:
                # the arm can now move to a pose (x) CAD guess, not only a
                # directly measured one. self._recovered_3d keeps the raw
                # point (self._recovered_uvs is only its 2D projection, for
                # the yellow-cross display) so the target check below can
                # use it.
                num_real = sum(1 for p in ports if p.get('kind') != 'other')
                self._recovered_3d = {}
                if 0 < len(self._measured_3d) < num_real:
                    Rm_panel = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
                    self._recovered_3d = mpl.recover_missing(self._measured_3d, ports, Rm_panel)
                    for name, P in self._recovered_3d.items():
                        rp, _ = cv2.projectPoints(P.reshape(1, 3), np.zeros(3),
                                                  np.zeros(3), self.K, None)
                        self._recovered_uvs[name] = rp.reshape(2)
                self._all_ports_found = (
                    len(self._measured_3d) + len(self._recovered_uvs) >= num_real)

                if target:
                    P_final = self._measured_3d.get(target)
                    from_recovery = P_final is None
                    if P_final is None:
                        P_final = self._recovered_3d.get(target)
                    if P_final is None:
                        ctx = (bgr, outline, cands, pairs, ports, img, rvec, tvec)
                        return None, (f'{target} has no image measurement this '
                                      'frame (and none held, and no neighbour to '
                                      'recover it from)'), ctx, False
                    if from_recovery:
                        self.get_logger().warn(
                            f"target '{target}' has no direct measurement this "
                            'frame -- moving to a pose (x) CAD guess from its '
                            'nearest measured neighbour instead',
                            throttle_duration_sec=2)
                    tvec = P_final.reshape(3, 1)

        # ctx keeps this frame's own rvec/tvec -- the debug view is honest
        # about what this one frame's detector and fit produced. What gets
        # published is the median-filtered, jump-rejecting pose instead; see
        # the smoother's own docstring for why they are allowed to differ.
        ctx = (bgr, outline, cands, pairs, ports, img, rvec, tvec)
        mm_per_px = float(tvec.ravel()[2]) / self.K[0, 0] * 1000
        info = (f"{match['n_matched']}/{match['n_cad']} ports matched, "
                f"reprojection {err:.2f} px ({err * mm_per_px:.2f} mm), "
                f"range {tvec.ravel()[2] * 1000:.0f} mm")
        if getattr(self, '_tilt', None) is not None:
            info += f', tilt agrees with depth to {self._tilt:.1f} deg'
        if target:
            info += (f', {target} aimed at its own measured pixel'
                     if self._measured_3d.get(target) is not None
                     else f', {target} unmeasured this frame -- aimed at a pose (x) '
                          'CAD guess from its nearest measured neighbour instead')
        if self._measured_uvs:
            n_measured = sum(v is not None for v in self._measured_uvs.values())
            info += f', {n_measured}/{len(self._measured_uvs)} sockets image-measured'

        # A frame whose own fit does not agree with itself this well is not
        # worth handing to the smoother at all -- PoseSmoother only checks
        # agreement with recent history, so a frame that is consistently bad
        # in the same way every time (the panel hasn't moved, the same weak
        # candidates keep winning) would sail through that check and get
        # smoothed into looking settled. Filtering on this frame's own
        # reprojection error first, before smoothing ever sees it, is what
        # "only the frames that are actually good" means concretely.
        max_reproj = self.get_parameter('max_reproj_px').value
        if err > max_reproj:
            return None, (f'reprojection {err:.2f} px exceeds max_reproj_px '
                          f'({max_reproj:.1f}) -- not accurate enough to use'), ctx, False

        s_rvec, s_tvec, accepted, confirmed = self.smoother.push(
            rvec, tvec, time.monotonic())
        if not accepted:
            info += ' -- REJECTED as a frame-to-frame jump, held last pose'
            self.get_logger().warn(
                f'pose jumped too far from recent history; publishing the '
                f'median instead', throttle_duration_sec=5)
        # A settled pose stream is not enough on its own: PoseSmoother only
        # checks agreement with its *own* recent history, which a hole that
        # has been wrong in the same way for several frames in a row would
        # sail through. self._layout_consistent is this frame's own
        # measured holes agreeing with each other's CAD distances (see
        # mpl.inconsistent_ports); self._all_ports_found is every named
        # socket having *a* position at all, measured or recovered (see
        # mpl.recover_missing) -- all three have to hold before this is
        # confirmed to arm_cmd.
        confirmed = confirmed and self._layout_consistent and self._all_ports_found
        info += ' [CONFIRMED]' if confirmed else ' [settling...]'
        return mpl.pose_matrix(s_rvec, s_tvec), info, ctx, confirmed

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

    def _publish(self, T_cam_obj, stamp, confirmed):
        # camera_frame publishes every frame regardless -- it is what the
        # debug stream and any monitor watches, and the frame that has not
        # settled yet is exactly the one worth being able to see. world_frame
        # is what arm_cmd actually reads to decide where to move (see
        # arm_cmd.py's _hole_frame), so that is the one gated: publishing it
        # only once PoseSmoother has confirmed a run of consecutive agreeing
        # frames means arm_cmd's own existing "latest_object_pose is None"
        # check already refuses to act on a pose that has not settled,
        # with no change needed on that side.
        self._send(self.pose_pub, T_cam_obj, self.camera_frame, stamp)
        if not confirmed:
            return

        # snapshot together -- _feedback_cb runs on its own thread and updates
        # both under the same lock, so this must read both under it too, or a
        # stamp from one update could get paired with the matrix from another
        with self._feedback_lock:
            T_world_arm, T_world_arm_stamp = self.T_world_arm, self.T_world_arm_stamp
        if T_world_arm is None:
            self.get_logger().warn('no feedback_states yet, skipping world-frame pose',
                                   throttle_duration_sec=5)
            return
        age = time.monotonic() - T_world_arm_stamp
        if age > FEEDBACK_MAX_AGE_S:
            self.get_logger().error(
                f'feedback_states is {age:.1f}s stale -- refusing to publish a '
                'world-frame pose built on it', throttle_duration_sec=5)
            return

        self._send(self.world_pose_pub, T_world_arm @ self.T_G_C @ T_cam_obj,
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
    # _color_cb is on its own callback group (see its create_subscription
    # call) specifically so it needs a real thread to run on concurrently with
    # everything else -- a single-threaded spin() would still starve the
    # default group behind it, group split or no group split. 2 threads
    # covers it: one for _color_cb's group, one for the default group
    # (info/depth/feedback_states callbacks, and the node's own built-in
    # ~/set_parameters service that arm_cmd's set_target_port/freeze call).
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()