
import rclpy.time_source
from tm_msgs.srv import SetPositions,SetEvent,SetIO
from tm_msgs.msg import FeedbackState
from geometry_msgs.msg import PoseStamped

import json
import os

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
import numpy as np
from scipy.spatial.transform import Rotation
import threading
import time

# Geometry of the RJ45 jig, measured from objects/rj45_test/rj45_test.obj after
# recentring (see fp_pose_bridge / track_publish_and_stream): the socket opening
# sits on the +Z face, 15.2mm along X by 12.2mm along Y, its centre offset +2mm
# in Y from the jig centre. Mesh +Z points *out* of the hole, i.e. the direction
# a plug approaches from -- and the flange's +Z points the opposite way (down,
# toward the work), so aligning means flange Z anti-parallel to object Z.
HOLE_CENTRE_IN_MESH = np.array([0.0, 0.002, 0.010])
HOLE_LONG_AXIS_IN_MESH = np.array([1.0, 0.0, 0.0])   # 15.2mm side runs along X
HOLE_NORMAL_IN_MESH = np.array([0.0, 0.0, 1.0])      # +Z, pointing out of the hole
# 0.34 -0.47 0.19 target
# move 0.34 -0.47 0.3
# move 0.34 -0.47 0.19
# pick
# move 0.34 -0.47 0.3
# move 0.2 -0.3 0.3
# move 0.2 -0.3 0.19
# place

class ArmCmd(Node):
    def __init__(self):
        super().__init__('arm_cmd')
        self.pos_cli = self.create_client(SetPositions, 'set_positions')
        self.event_cli = self.create_client(SetEvent, 'set_event')
        self.io_cli = self.create_client(SetIO, 'set_io')
        self.pos_sub = self.create_subscription(FeedbackState, 'feedback_states', self.pos_callback, 10)
        self.latest_object_pose = None
        self.object_pose_sub = self.create_subscription(PoseStamped, 'world_frame/object_pose', self.object_pose_callback, 10)
        while not self.pos_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        while not self.event_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        while not self.io_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.set_positions_req = SetPositions.Request()
        self.set_event_req = SetEvent.Request()
        self.declare_parameter('part', 'rj45_test')
        # Height of the bench top in world/base coordinates. The arm reports
        # tool_pose against its own base frame, and nothing in the driver knows
        # where the table is, so the one command that is specified relative to
        # the table has to be told. Measured on this rig: the bench sits 5cm
        # above the robot base. Override with -p table_z:=... if the arm is
        # remounted or moved to another bench.
        self.declare_parameter('table_z', 0.05)
        # Roll added automatically for each kind of plug. The alignment code
        # squares the flange's X axis to the socket's long axis, which puts the
        # jaws in the right plane but not necessarily the right way up: how far
        # they then have to turn depends on how that particular plug sits
        # between them, and an RJ45 body does not sit the way a flat USB or
        # HDMI one does. Measured on this rig, USB and HDMI need a quarter turn
        # and RJ45 needs none, which is why this is per kind rather than one
        # constant -- an earlier version used a single global offset and was
        # wrong for RJ45 by 90 deg.
        #
        # Sign: positive roll increases the target's rz, a counter-clockwise
        # turn seen from above. Note +90 and -90 both leave the jaws parallel to
        # the socket -- a long axis is a line, not a direction -- and differ
        # only in which way round the plug ends up, so if one turns out to be
        # facing backwards on first insertion, negate that kind's value.
        # Height of the part standing on the bench, when it differs from the CAD.
        # Only the clearance readout in descend() uses this -- nothing about
        # targeting a port depends on it, because the port's z offset appears
        # identically in the PnP object points and in the arm's hole_pos sum and
        # cancels out. But a clearance number that is optimistic is the wrong
        # kind of wrong, so it is worth being able to correct when the hardware
        # is shimmed or rebuilt taller than the model. 0.0 means "use the CAD".
        self.declare_parameter('part_height_m', 0.0)
        self.declare_parameter('roll_usb', 90.0)
        self.declare_parameter('roll_hdmi', 90.0)
        self.declare_parameter('roll_rj45', 0.0)
        # for the single-opening jig, which has no port table and so no kind
        self.declare_parameter('roll_default', 0.0)
        self.part = self.get_parameter('part').value
        self._ports = None
        self.target_positions = [0.2, -0.4, 0.35, 3.14159, 0.0, -1.57]
        self.current_positions = [0.2, -0.4, 0.35, 3.14159, 0.0, -1.57]
        
    def pos_callback(self,msg):
        self.current_positions = msg.tool_pose
        # self.get_logger().info("Current Position: %s" % self.current_positions)

    def object_pose_callback(self,msg):
        self.latest_object_pose = msg

    def target_is_sane(self, target):
        # Last line of defence before a motion command. A corrupted
        # feedback_states layout in tm_driver once produced a target of
        # [0.042, 289590.5, 4.6e-44, ...] here; only the TM controller's own
        # range check stopped the arm. Refuse anything outside the TM5-900's
        # ~0.9m reach rather than relying on that.
        import math
        for v in target[:3]:
            if not math.isfinite(v) or abs(v) > 2.0:
                self.get_logger().error(f'refusing implausible target {target} -- check tm_driver/feedback_states')
                return False
        return True

    def descend(self, drop=0.05, move=True):
        """Straight vertical move: keep X/Y and orientation, only lower Z.

        Does not touch the tracked object pose at all, so it is the one motion
        that behaves identically whether or not vision has a lock.

        The log line reports what will be left underneath rather than refusing
        anything. Descending is the direction that can hit something, and the
        useful thing to know before committing is the remaining clearance --
        both to the bench and to the top of the part standing on it. Note that
        clearance is measured to the tool_pose the driver reports, so whatever
        is held in the jaws hangs below the number shown.
        """
        target = list(self.current_positions[:6])
        target[2] -= drop
        if not self.target_is_sane(target):
            return None
        table_z = float(self.get_parameter('table_z').value)
        part_h = float(self.get_parameter('part_height_m').value)
        source = 'measured'
        if part_h <= 0.0:
            source = 'CAD'
            try:
                part_h = float(json.load(open(os.path.join(
                    get_package_share_directory('py_gripper'), 'config',
                    'opening_reference.json')))[self.part].get('work_height_m', 0.0))
            except (OSError, KeyError, ValueError, TypeError):
                part_h = 0.0
        above_table = target[2] - table_z
        note = f'{above_table*100:.1f}cm above the bench'
        if part_h > 0:
            note += (f', {(above_table - part_h)*100:.1f}cm above the part '
                     f'({part_h*100:.1f}cm tall, {source})')
        self.get_logger().info(
            f'descend {drop*100:.0f}cm straight down -> '
            f'{[round(v, 4) for v in target]}  ({note})')
        if not move:
            self.get_logger().info('(dry run, not moving)')
            return None
        return self.send_request(target)

    def ready_pose(self, height=0.45, move=True):
        """Park the flange `height` above the table, its face parallel to it.

        The starting pose for a run. Two things are being set: the height, and
        the flange plane -- "parallel to the table" means the flange's own Z
        axis points straight down, which is the same convention every other
        command in this file uses (euler xyz of pi, 0, heading).

        The heading is deliberately left alone. Levelling the flange does not
        require picking a yaw, and carrying the current one over keeps this to
        the smallest motion that satisfies the request; the alignment commands
        set the heading themselves when they run.

        Height is measured from the `table_z` parameter, not from z=0, because
        the arm's base frame is not the table. It is also measured to the
        *tool_pose* the driver reports -- if a TCP offset is configured in
        TMflow, that is the point being placed 40cm up, not the flange face,
        and anything held in the jaws hangs below it.
        """
        table_z = float(self.get_parameter('table_z').value)

        # level the flange while keeping whatever heading it already has
        R_current = Rotation.from_euler('xyz', self.current_positions[3:6]).as_matrix()
        flange_z = np.array([0.0, 0.0, -1.0])
        flange_x = R_current[:, 0].copy()
        flange_x[2] = 0.0
        n = np.linalg.norm(flange_x)
        if n < 1e-6:
            # the flange's X currently points straight up or down, so there is
            # no heading to preserve -- any horizontal one will do
            flange_x = np.array([1.0, 0.0, 0.0])
        else:
            flange_x /= n
        flange_y = np.cross(flange_z, flange_x)
        R_target = np.column_stack([flange_x, flange_y, flange_z])
        rx, ry, rz = Rotation.from_matrix(R_target).as_euler('xyz')

        target = [float(self.current_positions[0]),
                  float(self.current_positions[1]),
                  float(table_z + height),
                  float(rx), float(ry), float(rz)]
        if not self.target_is_sane(target):
            return None
        reorient_deg = np.degrees(
            Rotation.from_matrix(R_current.T @ R_target).magnitude())
        self.get_logger().info(
            f'ready pose: {height*100:.0f}cm above table (table_z={table_z:.3f}), '
            f'flange levelled -> {[round(v, 4) for v in target]} '
            f'(moving {abs(target[2]-self.current_positions[2])*100:.1f}cm in Z, '
            f'reorienting {reorient_deg:.0f}deg)')
        if not move:
            self.get_logger().info('(dry run, not moving)')
            return None
        return self.send_request(target)

    def _port_geometry(self, port=None):
        """Where the target opening sits in the object's own frame.

        With one opening there is nothing to choose and the constants at the top
        of this file apply. A multi-port panel keeps its table in
        config/opening_reference.json, built from the CAD, and the port is picked
        by name -- vision never has to tell a USB from an HDMI, which is the one
        thing it would be unreliable at.
        """
        if port is None:
            return (HOLE_CENTRE_IN_MESH, HOLE_LONG_AXIS_IN_MESH,
                    'single opening', None)
        table = self.port_table()
        if not table:
            self.get_logger().error('no port table loaded; run tools/build_reference.py')
            return None
        if port not in table:
            self.get_logger().error(f'unknown port {port!r}; have {sorted(table)}')
            return None
        p = table[port]
        # A port's polarity -- which end of its long axis the plug's latch has to
        # meet. Note this is NOT detected from the camera: the value comes from
        # config and is the same every frame. That is deliberate and sufficient,
        # because the platform's own orientation is already settled outright by
        # the port-pattern match, so a port's polarity relative to the platform
        # is a fixed fact about the hardware rather than something to re-measure.
        #
        # An earlier version of this comment claimed the CAD models these
        # cavities as plain symmetric slots. That was wrong -- measured by
        # ray-casting server1_all.STL at 0.1mm, every cavity is strongly
        # asymmetric under a half turn (USB 118%, RJ45 37%, HDMI 10% of cells
        # unmatched), and the asymmetry runs across the *short* axis: USB is
        # 0.5mm deeper on -x (the tongue sits on +x), RJ45 0.7mm deeper on +x
        # (the latch slot). All eight ports agree in direction, none is mirrored,
        # which is why flip=1 throughout is self-consistent. It does mean the
        # polarity could be derived from the CAD instead of hardcoded, which
        # would matter on a platform that mounts some ports turned around.
        flip = int(p.get('flip', 1))
        return (np.array(p['centre']), np.array(p['long_axis']) * flip,
                f"{port} ({p['kind']}{', flipped' if flip < 0 else ''})",
                p['kind'])

    def roll_for_kind(self, kind):
        """Automatic roll for this kind of plug, in degrees."""
        name = f'roll_{kind}' if kind else 'roll_default'
        try:
            return float(self.get_parameter(name).value)
        except Exception:
            self.get_logger().warn(
                f'no {name} parameter for a {kind!r} plug; using 0deg',
                throttle_duration_sec=30)
            return 0.0

    def port_table(self):
        """Named ports for the currently selected part, loaded once."""
        if self._ports is not None:
            return self._ports
        path = os.path.join(get_package_share_directory('py_gripper'),
                            'config', 'opening_reference.json')
        try:
            refs = json.load(open(path))
        except OSError as e:
            self.get_logger().error(f'cannot read {path}: {e}')
            return {}
        entry = refs.get(self.part, {})
        self._ports = {p['name']: p for p in entry.get('ports', [])}
        if self._ports:
            self.get_logger().info(
                f"loaded {len(self._ports)} ports for {self.part!r}: "
                f"{', '.join(sorted(self._ports))}")
        return self._ports

    def _hole_frame(self, roll_deg=0.0, port=None):
        # Shared geometry for the hole-aligned commands: where the socket
        # opening is in world coords, which way it faces, and the flange
        # rotation that squares the gripper up to it.
        #
        # roll_deg is the one thing that can't be derived from the mesh or the
        # calibration: it depends on how the plug happens to sit in the gripper
        # jaws. Try 0/90/180/270 and keep whichever makes the plug's rectangle
        # line up with the socket's.
        if self.latest_object_pose is None:
            self.get_logger().warn('no world_frame/object_pose received yet, cannot align to hole')
            return None

        p = self.latest_object_pose.pose.position
        q = self.latest_object_pose.pose.orientation
        R_world_obj = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        obj_pos = np.array([p.x, p.y, p.z])

        geom = self._port_geometry(port)
        if geom is None:
            return None
        centre_in_mesh, long_in_mesh, label, kind = geom
        hole_pos = obj_pos + R_world_obj @ centre_in_mesh
        long_axis = R_world_obj @ long_in_mesh

        # The jig always lies flat on the table with the socket facing
        # straight up, so the true hole normal is exactly world +Z regardless
        # of what FoundationPose's rotation estimate says -- trust that
        # physical constraint instead of the tracked normal, which is both
        # noisy and prone to a ~180deg ambiguity on this box shape (see
        # track_publish_and_stream.py). The only thing that actually needs to
        # come from tracking is the in-plane heading that squares the gripper
        # to the hole's long edge.
        normal = np.array([0.0, 0.0, 1.0])
        flange_z = -normal                                   # face the hole, straight down

        flange_x = long_axis.copy()
        flange_x[2] = 0.0                                    # heading only -- ignore tilt/noise
        n = np.linalg.norm(flange_x)
        if n < 1e-6:
            self.get_logger().error('degenerate hole heading (long axis reads as vertical), refusing to move')
            return None
        flange_x /= n
        flange_y = np.cross(flange_z, flange_x)
        R_world_flange = np.column_stack([flange_x, flange_y, flange_z])
        auto_roll = self.roll_for_kind(kind)
        roll_total = roll_deg + auto_roll
        if auto_roll:
            self.get_logger().info(
                f'{kind} plug: adding {auto_roll:+.0f}deg automatic roll '
                f'(total {roll_total:+.0f}deg)')
        R_world_flange = R_world_flange @ Rotation.from_euler(
            'z', roll_total, degrees=True).as_matrix()
        return hole_pos, normal, R_world_flange

    def _send_hole_target(self, target_pos, R_world_flange, label, move):
        rx, ry, rz = Rotation.from_matrix(R_world_flange).as_euler('xyz')
        target = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
                  float(rx), float(ry), float(rz)]
        if not self.target_is_sane(target):
            return None
        R_current = Rotation.from_euler('xyz', self.current_positions[3:6]).as_matrix()
        reorient_deg = np.degrees(Rotation.from_matrix(R_current.T @ R_world_flange).magnitude())
        self.get_logger().info(f'{label} -> {[round(v,4) for v in target]} '
                               f'(reorienting {reorient_deg:.0f}deg)')
        if not move:
            self.get_logger().info('(dry run, not moving)')
            return None
        return self.send_request(target)

    def align_to_hole(self, standoff=0.15, roll_deg=0.0, move=True, port=None):
        # Park standoff metres straight out along the hole's normal, squared up
        # to the socket. Unlike hover/align_xy this uses the object's
        # orientation, so the wrist rotates instead of staying fixed downward.
        f = self._hole_frame(roll_deg, port)
        if f is None:
            return None
        hole_pos, normal, R_world_flange = f
        return self._send_hole_target(
            hole_pos + normal * standoff, R_world_flange,
            f'align_to_hole [{port or "hole"}] standoff={standoff} roll={roll_deg}deg', move)

    def align_hole_xy(self, roll_deg=0.0, move=True, port=None):
        # Same squaring-up rotation as align_to_hole, but XY only: Z stays at
        # whatever height the arm is already at, so orientation and XY can be
        # dialled in without the tool ever creeping closer to the work.
        f = self._hole_frame(roll_deg, port)
        if f is None:
            return None
        hole_pos, _normal, R_world_flange = f
        target_pos = np.array([hole_pos[0], hole_pos[1], self.current_positions[2]])
        return self._send_hole_target(
            target_pos, R_world_flange,
            f'align_hole_xy [{port or "hole"}] (keeping current Z) roll={roll_deg}deg', move)

    def align_xy(self):
        # keep Z (and current height) exactly where the arm already is;
        # only update X/Y to the latest detected object position. For
        # iterating on XY alignment without the risk of Z creeping down
        # on every retry.
        if self.latest_object_pose is None:
            self.get_logger().warn('no world_frame/object_pose received yet, cannot align')
            return None
        p = self.latest_object_pose.pose.position
        target = [p.x, p.y, self.current_positions[2], 3.14159, 0.0, -1.57]
        if not self.target_is_sane(target):
            return None
        self.get_logger().info(f'aligning XY only, keeping current Z: {target}')
        return self.send_request(target)

    def is_arrived(self,error=0.01):
        if sum((self.target_positions[i]-self.current_positions[i])**2 for i in range(3)) > error**2:
            return False
        return True

    def wait_until_arrived(self, timeout=15.0, error=0.005):
        # send_request() only dispatches the service call -- rclpy's
        # Future.result() returns None rather than blocking when the call
        # hasn't completed, so it returns as soon as the request is queued.
        # That's fine when a human is pacing the commands, but chaining moves
        # in code needs an explicit wait or every step fires at once.
        #
        # Polls current_positions, which the background spin thread keeps
        # updated. Deliberately does NOT call rclpy.spin_once(): that thread is
        # already spinning this node and spinning it from two threads is unsafe.
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.is_arrived(error):
                return True
            time.sleep(0.05)
        self.get_logger().warn(
            f'move did not reach target within {timeout}s '
            f'(target={[round(v,4) for v in self.target_positions[:3]]}, '
            f'current={[round(v,4) for v in self.current_positions[:3]]})')
        return False

    def send_request(self,positions=[0.2, -0.4, 0.35, 3.14159, 0.0, -1.57],
                     velocity=0.1, acc_time=0.5, blend_percentage=100, fine_goal=False):
        
        self.target_positions = positions
        print(self.target_positions)
        set_positions_req = SetPositions.Request()
        set_positions_req.motion_type = SetPositions.Request.LINE_T
        set_positions_req.positions = positions
        set_positions_req.velocity = velocity
        set_positions_req.acc_time = acc_time
        set_positions_req.blend_percentage = blend_percentage
        set_positions_req.fine_goal = fine_goal
        future = self.pos_cli.call_async(set_positions_req)
        # rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def send_gripper(self,gap=0.085):
        # Toyo CHG2: binary open/close via End Effector DO_0 (H=close, L=open).
        # Keeps the old continuous "gap" arg so existing call sites don't change;
        # gap >= 0.0425 (half-open) opens, below that closes.
        gap = gap if gap < 0.085 else 0.085
        gap = gap if gap > 0.0 else 0.0
        close = gap < 0.0425
        print(gap, "-> close" if close else "-> open")
        req = SetIO.Request()
        req.module = SetIO.Request.MODULE_ENDEFFECTOR
        req.type = SetIO.Request.TYPE_DIGITAL_OUT
        req.pin = 0
        req.state = 1.0 if close else 0.0
        future = self.io_cli.call_async(req)
        return future

    def send_event(self):
        self.set_event_req = SetEvent.Request()
        self.set_event_req.func = SetEvent.Request.STOP
        self.set_event_req.arg0 = 0
        self.set_event_req.arg1 = 0
        future = self.event_cli.call_async(self.set_event_req)
        # rclpy.spin_until_future_complete(self, future)
        return future.result()


def parse_hole_args(args):
    """Arguments after hole/holexy -> (port name, roll degrees, dry run, standoff m).

    A port name and a roll angle are both optional and tell themselves apart:
    anything that parses as a number is the angle, anything else is a name. So
    "holexy", "holexy 90", "holexy usb2" and "holexy usb2 90 dry" all read the
    way they look.

    Standoff is written with an explicit "cm" suffix ("hole usb2 3cm") rather
    than as a second bare number, because a bare number is already spoken for
    -- it is the roll angle -- and there is no position in the argument list
    that would tell the two apart otherwise. Only "hole" uses it; "holexy"
    keeps Z wherever the arm already is; standoff=None there.
    """
    port, roll, dry, standoff = None, 0.0, False, None
    for a in args:
        if a in ('dry', 'd'):
            dry = True
            continue
        for suffix, scale in (('mm', 0.001), ('cm', 0.01)):
            if a.endswith(suffix):
                try:
                    standoff = float(a[:-len(suffix)]) * scale
                    break
                except ValueError:
                    pass
        else:
            suffix = None
        if suffix is not None and a.endswith(suffix):
            try:
                float(a[:-len(suffix)])
                continue
            except ValueError:
                pass
        try:
            roll = float(a)
        except ValueError:
            port = a
    return port, roll, dry, standoff


def main(args=None):
    rclpy.init(args=args)
    armCmd = ArmCmd()
    rclpy.spin_once(armCmd)
    armCmd.send_gripper(0.085)

    # response = armCmd.send_request()
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)

    # response = armCmd.send_request([0.33, -0.47, 0.35, 3.14159, 0.0, -1.57])    
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)

    # response = armCmd.send_request([0.33, -0.47, 0.19, 3.14159, 0.0, -1.57])
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)
    
    # armCmd.send_gripper(0.03)
    # time.sleep(1.5)
    # print("pick")
    
    # response = armCmd.send_request([0.33, -0.46, 0.35, 3.14159, 0.0, -1.57])
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)

    # response = armCmd.send_request([0.2, -0.3, 0.35, 3.14159, 0.0, -1.57])
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)

    # response = armCmd.send_request([0.2, -0.3, 0.191, 3.14159, 0.0, -1.57])
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)

    # armCmd.send_gripper(0.085)
    # time.sleep(1.5)
    # print("place")

    # response = armCmd.send_request([0.2, -0.3, 0.35, 3.14159, 0.0, -1.57])
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)

    # response = armCmd.send_request()
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)
    # ######################################################

    # response = armCmd.send_request([0.2, -0.3, 0.35, 3.14159, 0.0, -1.57])
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)

    # response = armCmd.send_request([0.2, -0.3, 0.191, 3.14159, 0.0, -1.57])
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)

    # armCmd.send_gripper(0.03)
    # time.sleep(1.5)
    # print("pick")

    # response = armCmd.send_request([0.2, -0.3, 0.35, 3.14159, 0.0, -1.57])
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)

    # response = armCmd.send_request([0.33, -0.47, 0.35, 3.14159, 0.0, -1.57])
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)

    # response = armCmd.send_request([0.33, -0.47, 0.19, 3.14159, 0.0, -1.57])
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)
    
    # armCmd.send_gripper(0.085)
    # time.sleep(1.5)
    # print("place")

    # response = armCmd.send_request([0.33, -0.47, 0.35, 3.14159, 0.0, -1.57])
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)

    # response = armCmd.send_request()
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)

    #############################
    # response = armCmd.send_request([0.0, -0.3, 0.35, 3.14159, 0.0, -1.57])
    # while not armCmd.is_arrived():
    #     rclpy.spin_once(armCmd)
    # print("move",armCmd.target_positions)

    # HZ = 100
    # distance = 0.400 #(m)
    # speed = 0.1 #(m/s)
    # total_time = distance/speed

    # duration = 1/(HZ/3)
    # fragment_size = speed*duration
    # p = 0
    # print("total_time %.3f" % total_time,"fragment_size %.3f" % fragment_size ,"duration %.3f" % duration)
    # last = time.time()
    # while True:
        
    #     if p > distance:
    #         break
    #     rclpy.spin_once(armCmd)
    #     if armCmd.is_arrived():
    #         # speed += 0.02
    #         p += fragment_size
    #         armCmd.send_request([p, -0.3, 0.35, 3.14159, 0.0, -1.57],speed,0.001)
    #         print("move",end=" ")
    #         for pos in armCmd.current_positions:
    #             print("%.3f"%pos,end=" ")
    #         print()

    #     while (time.time() - last) < (1/HZ):
    #         rclpy.spin_once(armCmd)
            
    #     print(1/(time.time() - last))
    #     last = time.time()
    # subscriptions/service futures need the node spinning to actually receive
    # anything; the interactive input() loop below blocks the main thread, so
    # spin in the background instead.
    spin_thread = threading.Thread(target=rclpy.spin, args=(armCmd,), daemon=True)
    spin_thread.start()

    while True:
        raw = input("Positions: ").strip()
        # "h" / "h 10cm" / "h 10cm dry" -- straight down, pose untouched.
        if raw.split() and raw.split()[0] in ('h', 'hover'):
            _p, _roll, dry, drop = parse_hole_args(raw.split()[1:])
            kwargs = {} if drop is None else {'drop': drop}
            armCmd.descend(move=not dry, **kwargs)
            continue
        if raw in ('xy', 'align'):
            armCmd.align_xy()
            continue
        # "ready" / "ready 30cm" / "ready dry" -- starting pose for a run:
        # a set height above the table with the flange face levelled.
        if raw.split() and raw.split()[0] in ('ready', 'rdy'):
            _p, _roll, dry, height = parse_hole_args(raw.split()[1:])
            kwargs = {} if height is None else {'height': height}
            armCmd.ready_pose(move=not dry, **kwargs)
            continue
        # "hole" / "hole 90" / "hole 90 dry"  -- roll angle in degrees, plus an
        # optional dry run that prints the computed target without moving.
        if raw.split() and raw.split()[0] in ('hole', 'a'):
            port, roll, dry, standoff = parse_hole_args(raw.split()[1:])
            kwargs = {} if standoff is None else {'standoff': standoff}
            armCmd.align_to_hole(roll_deg=roll, move=not dry, port=port, **kwargs)
            continue
        # "holexy" / "holexy 90" / "holexy 90 dry" -- same alignment, Z untouched
        if raw.split() and raw.split()[0] in ('holexy', 'hxy'):
            port, roll, dry, _standoff = parse_hole_args(raw.split()[1:])
            armCmd.align_hole_xy(roll_deg=roll, move=not dry, port=port)
            continue
        positions = list(map(float, raw.split()))
        if len(positions) == 3:
            positions = positions + [3.14159, 0.0, 3.14]
        if len(positions) == 1:
            # The CHG2 is driven by a single digital output, so it is purely
            # open/close -- no intermediate width. 1 opens, 0 closes (any other
            # non-zero also opens, so older habits like "85" still work).
            armCmd.send_gripper(0.085 if positions[0] != 0 else 0.0)
            continue
        if len(positions) == 0:
            armCmd.send_event()
        response = armCmd.send_request(positions)
        armCmd.get_logger().info("Response: %s" % response)

    rclpy.spin(armCmd)
    rclpy.shutdown()


if __name__ == '__main__':
    main()