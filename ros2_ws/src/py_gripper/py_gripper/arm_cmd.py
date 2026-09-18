
import rclpy.time_source
from tm_msgs.srv import SetPositions,SetEvent,SetIO
from tm_msgs.msg import FeedbackState
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
import numpy as np
import threading
import time
import json
import os

# Where "save"/"back" park a single remembered pose (e.g. a standard
# observation stance to measure holes from consistently -- see the
# discussion on hand-eye residual error depending on viewing distance/angle).
# Outside install/ and src/ on purpose: install gets overwritten by every
# colcon build, and this is runtime scratch, not something to version-control.
SAVED_POSE_PATH = os.path.expanduser('~/.py_gripper_saved_pose.json')

# hover_at()'s default tool-tip distance, ruler-measured from G (the point
# tool_pose/current_positions actually reports, NOT the raw mechanical
# flange -- on this rig G sits 4.425cm further out along Z than the flange
# itself, confirmed against the teach pendant's own TCP setting) -- see
# that method's own docstring for why it must be measured from G, not the
# camera. Update this if the held connector changes length; "above" still
# takes an explicit second number to override it for one call without
# editing code.
DEFAULT_TIP_OFFSET_CM = 16.99  # measured directly from G, 2026-09-10

# Empirically observed systematic offset: the arm consistently lands
# too high relative to where each hole actually is, checked across
# several different ports -- confirmed live 2026-09-10. A fixed,
# same-direction offset across unrelated ports is what a correctable
# systematic bias looks like (independent per-hole measurement noise
# would not agree in direction and size like that).
# Confirmed to be the WORLD FRAME's own Z axis specifically, by directly
# nudging one absolute-position coordinate at a time and watching which
# one moved the debug view up/down on screen -- not assumed from the
# panel's own measured orientation or any other derived axis. Re-measure
# and update this if it stops holding once other accuracy work
# (K/distortion, T_G_E, etc.) changes the picture -- this is a patch
# over today's residual, not a derived constant.
#
# usb1/usb2 measured 2.5mm; every other port measured 1mm more (3.5mm) --
# confirmed live 2026-09-15. Not assumed to generalise to ports not yet
# tested; a newly added port defaults to Z_BIAS_M_OTHER until it gets its
# own measurement.
Z_BIAS_M_USB12 = 0.0025       # usb1, usb2
Z_BIAS_M_OTHER = 0.0025       # every other port
_Z_BIAS_USB12_PORTS = ('usb1', 'usb2')


def _z_bias_for(port_name):
    return Z_BIAS_M_USB12 if port_name in _Z_BIAS_USB12_PORTS else Z_BIAS_M_OTHER


# World-frame Y axis, same convention as the Z bias above (confirmed axis/
# direction, not derived). Same correction for every port for now -- unlike
# Z, not yet split per-port. Starting value, not yet tuned per port.
Y_BIAS_M = 0.001 # -Y direction, all ports

# 0.34 -0.47 0.19 target
# move 0.34 -0.47 0.3
# move 0.34 -0.47 0.19
# pick
# move 0.34 -0.47 0.3
# move 0.2 -0.3 0.3
# move 0.2 -0.3 0.19
# place

class ArmCmd(Node):
    def __init__(self, node_name='arm_cmd'):
        # node_name overridable so a second ArmCmd-based process (e.g. an
        # action server) can run alongside the interactive `arm_cmd` CLI
        # without a node-name collision on the same ROS domain.
        super().__init__(node_name)
        self.pos_cli = self.create_client(SetPositions, 'set_positions')
        self.event_cli = self.create_client(SetEvent, 'set_event')
        self.io_cli = self.create_client(SetIO, 'set_io')
        self.pos_sub = self.create_subscription(FeedbackState, 'feedback_states', self.pos_callback, 10)
        self.latest_object_pose = None
        self._current_target_port = None  # for _target_xyz's per-port Z bias
        self.object_pose_sub = self.create_subscription(PoseStamped, 'world_frame/object_pose', self.object_pose_callback, 10)
        # not waited-for at startup -- depth_pose_node may not be up yet (or
        # this session may not need it at all), so only block on it lazily,
        # inside set_target_port itself, when the user actually asks for it.
        self.set_param_cli = self.create_client(SetParameters, '/depth_pose_node/set_parameters')
        while not self.pos_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        while not self.event_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        while not self.io_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.set_positions_req = SetPositions.Request()
        self.set_event_req = SetEvent.Request()
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

    def _set_remote_param(self, name, value):
        # Shared by anything that drives depth_pose_node's own parameters
        # through its standard set_parameters service, so they can be changed
        # from here without a separate `ros2 param set` terminal or
        # restarting depth_pose_node. bool/str inferred from value's type.
        if not self.set_param_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('depth_pose_node not reachable (set_parameters service unavailable) -- is it running?')
            return None
        if isinstance(value, bool):
            pval = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value)
        else:
            pval = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value)
        req = SetParameters.Request()
        req.parameters = [Parameter(name=name, value=pval)]
        future = self.set_param_cli.call_async(req)
        for _ in range(100):
            if future.done():
                break
            time.sleep(0.05)
        if not future.done():
            self.get_logger().warn('set_parameters call timed out')
            return None
        return future.result().results[0]

    def set_target_port(self, name):
        result = self._set_remote_param('target_port', name)
        if result is None:
            return None
        if not result.successful:
            self.get_logger().warn(f"depth_pose_node rejected target_port='{name}': {result.reason}")
            return False
        # world_frame/object_pose is only published once depth_pose_node has
        # a CONFIRMED reading for this port (see its own _publish) -- clear
        # whatever pose is left from the previous port so a chained move
        # right after this can't act on a stale, wrong-port position while
        # this one is still settling.
        self.latest_object_pose = None
        self._current_target_port = name
        self.get_logger().info(f"target_port -> '{name}', waiting for a confirmed pose...")
        t0 = time.time()
        while time.time() - t0 < 5.0:
            if self.latest_object_pose is not None:
                self.get_logger().info(f"'{name}' confirmed ({time.time() - t0:.1f}s)")
                return True
            time.sleep(0.05)
        self.get_logger().warn(f"'{name}' not confirmed within 5s -- not moving")
        return False

    def freeze(self, on):
        # Pins depth_pose_node to a single camera frame (see its own
        # 'freeze_frame' parameter) so every "port <name>"/xy/where/truth
        # after this reads the SAME image over and over, instead of a fresh
        # capture each time -- isolates frame-to-frame vision noise (and
        # identification flipping between candidates) from every other error
        # source: if per-hole error still varies while frozen, the camera/
        # detector is not the cause.
        result = self._set_remote_param('freeze_frame', on)
        if result is None:
            return None
        if not result.successful:
            self.get_logger().warn(f'depth_pose_node rejected freeze_frame={on}: {result.reason}')
            return False
        self.get_logger().info(
            'frame FROZEN -- every port lookup from here reuses this one image'
            if on else 'frame live again')
        return True

    def _target_xyz(self):
        """The current target's own world-frame position, with its
        per-port Z bias subtracted (_z_bias_for -- see that and
        Z_BIAS_M_USB12/_OTHER's own comment for how the axis and
        direction were confirmed) and the flat Y bias subtracted (see
        Y_BIAS_M's own comment). Shared by hover() and hover_at() so
        the correction only lives in one place. -> (3,) or None.
        """
        if self.latest_object_pose is None:
            return None
        p = self.latest_object_pose.pose.position
        z_bias = _z_bias_for(self._current_target_port)
        return np.array([p.x, p.y - Y_BIAS_M, p.z - z_bias])

    def hover(self, move=True):
        """Line up X/Y over the target, without following it along the
        flange's own Z -- the panel's approach/normal axis on this
        side-mounted setup. Keeps whatever standoff distance from the panel
        the arm already has; only slides sideways. Orientation untouched.

        A depth-safe first step: no risk from the depth measurement being
        off, confirm visually, then close the remaining distance
        deliberately with approach() or hover_at().
        """
        target_xyz = self._target_xyz()
        if target_xyz is None:
            self.get_logger().warn('no world_frame/object_pose received yet, cannot align')
            return None
        cur = np.array(self.current_positions[:3])
        flange_z = Rotation.from_euler('xyz', self.current_positions[3:6]).as_matrix()[:, 2]
        delta = target_xyz - cur
        depth = float(np.dot(delta, flange_z))
        delta_perp = delta - depth * flange_z
        target = list(cur + delta_perp) + list(self.current_positions[3:6])
        if not self.target_is_sane(target):
            return None
        dist = float(np.linalg.norm(delta_perp))
        self.get_logger().info(
            f'hovering over target, flange-Z depth held fixed (target is '
            f'{depth*100:+.1f}cm away along that axis -- not followed) -> '
            f'{[round(v, 4) for v in target]} ({dist*100:.1f}cm sideways move)')
        if not move:
            self.get_logger().info('(dry run, not moving)')
            return None
        return self.send_request(target)

    def hover_at(self, standoff_cm, tip_offset_cm=DEFAULT_TIP_OFFSET_CM, move=True):
        """Go directly to standoff_cm before the target, along the flange's
        own approach axis -- both X/Y and depth in one move, unlike hover()
        (X/Y only, keeps whatever depth the arm already happens to be at).

        Both X/Y and depth come from the target's own WORLD-frame position
        (world_frame/object_pose -- hand-eye + tool_pose already applied),
        the same one hover()/where() read, confirmed accurate live via the
        roll()+where() hand-eye check (1.2mm drift over a 20deg turn).
        tip_offset_cm is added directly onto standoff_cm in the retreat
        distance, so the total distance held back from the target is
        (standoff_cm + tip_offset_cm)/100 metres along flange Z.

        tip_offset_cm MUST be measured from G -- the point tool_pose /
        current_positions actually reports, i.e. wherever send_request()'s
        target actually lands -- not the camera, and not assumed to be the
        raw mechanical flange either. On this rig G sits 4.425cm further
        out along Z than the flange itself (confirmed against the teach
        pendant's own TCP setting); the two are not the same point, so a
        distance measured from the flange face still needs that 4.425cm
        subtracted before it is usable here. DEFAULT_TIP_OFFSET_CM was
        measured directly from G with the connector already mounted,
        sidestepping that arithmetic.

        An earlier version of this method used a camera-frame depth
        reading (camera_frame/object_pose) combined with a camera-to-
        tool-tip ruler measurement, reasoning that camera_frame/object_pose
        sidesteps the hand-eye chain entirely. That reasoning had a real
        hole: the camera's own optical axis is NOT parallel to the
        flange's Z axis -- computed from this rig's own T_G_C, they are
        about 23 degrees apart -- so a distance measured along the
        camera's axis cannot be applied as a move along the flange's axis
        without introducing exactly that angular error. Confirmed live
        2026-09-10: usb1 and usb3, dry-run from the same stationary pose
        with tilt already within 2deg, reported depths 3.2cm apart -- far
        more than 2deg of panel tilt could produce (well under 1mm over
        this panel's own size), consistent with the computed 23deg
        camera/flange offset instead. Measuring tip_offset_cm from G and
        staying entirely in world frame avoids mixing the two axes at all.
        """
        target_xyz = self._target_xyz()
        if target_xyz is None:
            self.get_logger().warn('no world_frame/object_pose received yet, cannot align')
            return None
        flange_z = Rotation.from_euler('xyz', self.current_positions[3:6]).as_matrix()[:, 2]
        total_standoff_m = (standoff_cm + tip_offset_cm) / 100.0
        target = list(target_xyz - flange_z * total_standoff_m) + list(self.current_positions[3:6])
        if not self.target_is_sane(target):
            return None
        dist = float(np.linalg.norm(np.array(target[:3]) - np.array(self.current_positions[:3])))
        self.get_logger().info(
            f'moving to {standoff_cm:.1f}cm + {tip_offset_cm:.1f}cm tip offset before '
            f'target along the approach axis -> {[round(v, 4) for v in target]} '
            f'({dist*100:.1f}cm from current)')
        if not move:
            self.get_logger().info('(dry run, not moving)')
            return None
        return self.send_request(target)

    def approach(self, distance_cm, move=True):
        """Slide the flange a given distance along its own +Z axis -- the
        panel's approach/normal direction here, since the panel is
        side-mounted and the flange faces it head-on. Positive moves further
        the way the flange is already pointing; negative backs away.
        Orientation untouched.
        """
        cur = np.array(self.current_positions[:3])
        flange_z = Rotation.from_euler('xyz', self.current_positions[3:6]).as_matrix()[:, 2]
        target = list(cur + flange_z * (distance_cm / 100.0)) + list(self.current_positions[3:6])
        if not self.target_is_sane(target):
            return None
        self.get_logger().info(
            f'moving {distance_cm:+.1f}cm along flange Z -> {[round(v, 4) for v in target]}')
        if not move:
            self.get_logger().info('(dry run, not moving)')
            return None
        return self.send_request(target)

    def _wait_fresh_pose(self, timeout=5.0):
        # depth_pose_node's panel-orientation smoother holds a short window
        # of recent frames (median-filtered, see its own HoleSmoother) --
        # right after ANY real motion (this file's own square/approach/
        # hover/pos, or a manual jog), that window is still part pre-move,
        # part post-move for a few frames, so a value read immediately after
        # moving can describe where the camera *was* rather than where it
        # is now. Clearing and waiting for a new publish (same trick
        # set_target_port already uses when switching ports) means whatever
        # gets read next reflects the camera's current pose.
        self.latest_object_pose = None
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.latest_object_pose is not None:
                return True
            time.sleep(0.05)
        return False

    def check_tilt(self):
        """How far the flange's own approach axis is from square to the
        panel. world_frame/object_pose's orientation IS the panel's own
        measured frame (mpl.panel_axes_pose: Y=long axis, X=short axis,
        Z=depth-plane normal -- the same measurement drawn as the two arrows
        in the debug view, not a separate/decorative thing), so this is a
        real comparison against vision, not a guess.

        0 deg means flange Z is exactly parallel (or exactly anti-parallel --
        either is "square", direction is a sign convention this doesn't need
        to resolve) to the panel's own measured normal.
        """
        if not self._wait_fresh_pose():
            self.get_logger().warn('no fresh world_frame/object_pose within 5s, cannot check tilt')
            return None
        q = self.latest_object_pose.pose.orientation
        panel_normal = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()[:, 2]
        flange_z = Rotation.from_euler('xyz', self.current_positions[3:6]).as_matrix()[:, 2]
        angle = float(np.degrees(np.arccos(np.clip(np.dot(flange_z, panel_normal), -1.0, 1.0))))
        tilt = min(angle, 180.0 - angle)
        self.get_logger().info(
            f'flange approach axis is {tilt:.1f} deg off square to the panel '
            f'(0 = perfectly parallel)')
        return tilt

    def where(self):
        """The current target hole's world-frame position, plus how far it
        has moved since the last call.

        The hand-eye check: a hole that has not physically moved must report
        the same world coordinates no matter where the arm looks at it from,
        because T_world_arm is supposed to cancel the arm's own motion out
        (world = T_world_arm (x) T_G_C (x) camera). Read it here, move the arm,
        read it again -- whatever the number drifts by is hand-eye error (or a
        tool_pose/TCP mismatch), not hole-measurement noise, since both
        readings are of the same hole.

        The arm's own reorientation between the two reads is reported
        alongside, because the two numbers only mean something together: a
        hand-eye error affects both readings almost identically when the
        wrist has NOT turned, so it cancels in the difference and a pure
        translation between reads can show ~0 drift however bad the
        calibration is. Turning the wrist is what separates them.
        """
        if self.latest_object_pose is None:
            self.get_logger().warn('no world_frame/object_pose received yet')
            return None
        p = self.latest_object_pose.pose.position
        pos = np.array([p.x, p.y, p.z])
        R_now = Rotation.from_euler('xyz', self.current_positions[3:6])
        msg = f'target in world frame: {[round(v, 4) for v in pos]}'
        prev = getattr(self, '_last_where', None)
        if prev is not None:
            prev_pos, prev_R = prev
            d = pos - prev_pos
            turned = float(np.degrees((prev_R.inv() * R_now).magnitude()))
            msg += (f' | drifted {np.linalg.norm(d)*1000:.1f}mm '
                    f'(dx {d[0]*1000:+.1f}, dy {d[1]*1000:+.1f}, dz {d[2]*1000:+.1f} mm) '
                    f'while the wrist turned {turned:.1f} deg')
            if turned < 10.0:
                msg += ('  [wrist barely turned -- a hand-eye error largely '
                        'cancels between two reads like this, so a small '
                        'drift here does NOT clear the calibration]')
        self._last_where = (pos, R_now)
        self.get_logger().info(msg)
        return pos

    def square_up(self, move=True):
        """Reorient the flange so its approach axis is exactly parallel to
        the panel's own measured normal (see check_tilt) -- position
        untouched, this only rotates in place.

        Rather than jogging by hand toward 0 deg (imprecise, and 2 deg was
        apparently the practical floor for that), this reads the same
        vision-measured normal check_tilt compares against and solves for
        the exact orientation directly -- limited by measurement noise, not
        by how finely a human can jog.

        Whatever heading (roll about the new approach axis) the flange
        currently has is kept as close as possible rather than picked
        arbitrarily, the same spirit as the old ready_pose's flange-levelling
        -- this should not spin the tool around unnecessarily to get there.
        """
        if not self._wait_fresh_pose():
            self.get_logger().warn('no fresh world_frame/object_pose within 5s, cannot square up')
            return None
        q = self.latest_object_pose.pose.orientation
        panel_normal = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()[:, 2]
        R_cur = Rotation.from_euler('xyz', self.current_positions[3:6]).as_matrix()
        flange_z_cur, flange_x_cur = R_cur[:, 2], R_cur[:, 0]
        # match whichever sign of the normal the flange is already closer to,
        # so this does not flip the tool through 180 deg to "align" the other way
        target_z = panel_normal if np.dot(panel_normal, flange_z_cur) > 0 else -panel_normal
        x_proj = flange_x_cur - np.dot(flange_x_cur, target_z) * target_z
        n = np.linalg.norm(x_proj)
        if n < 1e-6:
            # current X is now parallel to the new Z -- no heading to keep,
            # any perpendicular direction will do
            x_proj = R_cur[:, 1]
            x_proj = x_proj - np.dot(x_proj, target_z) * target_z
            n = np.linalg.norm(x_proj)
        x_new = x_proj / n
        y_new = np.cross(target_z, x_new)
        rx, ry, rz = Rotation.from_matrix(
            np.column_stack([x_new, y_new, target_z])).as_euler('xyz')
        target = list(self.current_positions[:3]) + [float(rx), float(ry), float(rz)]
        if not self.target_is_sane(target):
            return None
        reorient_deg = float(np.degrees(
            Rotation.from_matrix(R_cur.T @ np.column_stack([x_new, y_new, target_z])).magnitude()))
        self.get_logger().info(
            f'squaring up to the panel -> {[round(v, 4) for v in target]} '
            f'(reorienting {reorient_deg:.1f} deg, position unchanged)')
        if not move:
            self.get_logger().info('(dry run, not moving)')
            return None
        return self.send_request(target)

    def roll(self, degrees, move=True):
        """Turn the flange about its own approach axis, position unchanged.

        The motion the hand-eye check in where() needs: the camera sits ~6cm
        off this axis (T_G_C's own translation), so rolling about it swings
        the camera around the target -- a genuinely different viewpoint AND a
        genuinely different wrist orientation, while the target stays in
        frame. Turning about a world axis instead would just aim the camera
        off the panel.
        """
        R_cur = Rotation.from_euler('xyz', self.current_positions[3:6])
        R_new = R_cur * Rotation.from_euler('z', degrees, degrees=True)
        rx, ry, rz = R_new.as_euler('xyz')
        target = list(self.current_positions[:3]) + [float(rx), float(ry), float(rz)]
        if not self.target_is_sane(target):
            return None
        self.get_logger().info(
            f'rolling {degrees:+.1f} deg about the approach axis, position held '
            f'-> {[round(v, 4) for v in target]}')
        if not move:
            self.get_logger().info('(dry run, not moving)')
            return None
        return self.send_request(target)

    def save_pose(self):
        # A single remembered slot, not a named set -- the use case is one
        # standard stance to return to (e.g. to measure every hole from a
        # consistent distance/angle), not a library of poses.
        self._saved_pose = list(self.current_positions)
        try:
            with open(SAVED_POSE_PATH, 'w') as f:
                json.dump(self._saved_pose, f)
        except OSError as e:
            self.get_logger().warn(f'could not persist saved pose to {SAVED_POSE_PATH}: {e}')
        self.get_logger().info(f'saved current pose -> {[round(v, 4) for v in self._saved_pose]}')

    def restore_pose(self, move=True):
        pose = getattr(self, '_saved_pose', None)
        if pose is None and os.path.exists(SAVED_POSE_PATH):
            with open(SAVED_POSE_PATH) as f:
                pose = json.load(f)
            self._saved_pose = pose
        if pose is None:
            self.get_logger().warn("no saved pose yet -- use 'save' first")
            return None
        if not self.target_is_sane(pose):
            return None
        dist = float(np.linalg.norm(np.array(pose[:3]) - np.array(self.current_positions[:3])))
        self.get_logger().info(
            f'restoring saved pose -> {[round(v, 4) for v in pose]} ({dist*100:.1f}cm from current)')
        if not move:
            self.get_logger().info('(dry run, not moving)')
            return None
        return self.send_request(pose)

    def is_arrived(self,error=0.01):
        if sum((self.target_positions[i]-self.current_positions[i])**2 for i in range(3)) > error**2:
            return False
        return True

    def wait_until_arrived(self, timeout=15.0, error=0.005):
        # send_request() only dispatches the service call and returns as
        # soon as the request is queued -- it does not wait for the arm to
        # actually get there. That's fine when a human is pacing the
        # commands, but chaining moves in code needs an explicit wait or
        # every step fires at once.
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
        """-> True once the move is dispatched -- NOT once the arm has
        arrived, and not the service's own response either: the response
        is handled by the background spin thread, arriving well after this
        returns, so returning future.result() here gave back None on
        essentially every call regardless of success (confirmed live:
        code that branched on it, see arm_cmd_cycle.py, mistook every
        single dispatched move for a failure). Callers that need to know
        the move actually finished must follow this with
        wait_until_arrived(), which is what that is for.
        """
        self.target_positions = positions
        print(self.target_positions)
        set_positions_req = SetPositions.Request()
        set_positions_req.motion_type = SetPositions.Request.LINE_T
        set_positions_req.positions = positions
        set_positions_req.velocity = velocity
        set_positions_req.acc_time = acc_time
        set_positions_req.blend_percentage = blend_percentage
        set_positions_req.fine_goal = fine_goal
        self.pos_cli.call_async(set_positions_req)
        return True

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


def main(args=None):
    rclpy.init(args=args)
    armCmd = ArmCmd()
    rclpy.spin_once(armCmd)

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
        # "xy [name] [dry]" -- optionally switch target_port first, then
        # line up X/Y over it without following it along the flange's own Z
        # (the panel's approach axis). Was "hover".
        if raw.split() and raw.split()[0] == 'xy':
            parts = raw.split()[1:]
            dry = 'dry' in parts
            names = [x for x in parts if x != 'dry']
            if len(names) > 1:
                armCmd.get_logger().warn('usage: xy [name] [dry], e.g. xy usb6')
            elif not names or armCmd.set_target_port(names[0]):
                armCmd.hover(move=not dry)
            continue
        # "z <cm>" / "z <cm> dry" -- slide that many cm along the flange's
        # own +Z (toward the panel here); negative backs away. Was "approach".
        if raw.split() and raw.split()[0] == 'z':
            parts = raw.split()
            dry = 'dry' in parts[1:]
            nums = [x for x in parts[1:] if x != 'dry']
            if len(nums) != 1:
                armCmd.get_logger().warn('usage: z <cm> [dry], e.g. z 3')
            else:
                armCmd.approach(float(nums[0]), move=not dry)
            continue
        # "above <cm> [tip_cm] [name] [dry]" -- go directly to <cm> before
        # the target (optionally switching target_port first), X/Y and
        # depth together. tip_cm defaults to DEFAULT_TIP_OFFSET_CM (the
        # G-to-tool-tip distance, ruler-measured -- see hover_at's own
        # docstring for why it must be measured from G, not the flange
        # face and not the camera) and only needs typing to override it
        # for one call.
        if raw.split() and raw.split()[0] == 'above':
            parts = raw.split()[1:]
            dry = 'dry' in parts
            rest = [x for x in parts if x != 'dry']
            nums = [x for x in rest if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
            names = [x for x in rest if x not in nums]
            if len(nums) not in (1, 2) or len(names) > 1:
                armCmd.get_logger().warn(
                    f'usage: above <cm> [tip_cm] [name] [dry], e.g. above 0.5 usb6 '
                    f'(tip_cm defaults to {DEFAULT_TIP_OFFSET_CM})')
            elif not names or armCmd.set_target_port(names[0]):
                tip_cm = float(nums[1]) if len(nums) == 2 else DEFAULT_TIP_OFFSET_CM
                armCmd.hover_at(float(nums[0]), tip_cm, move=not dry)
            continue
        # "save" -- remember the current pose. "back" / "back dry" -- return
        # to it. One slot, persisted across restarts.
        if raw.split() and raw.split()[0] in ('save', 'mark'):
            armCmd.save_pose()
            continue
        if raw.split() and raw.split()[0] in ('back', 'restore'):
            dry = 'dry' in raw.split()[1:]
            armCmd.restore_pose(move=not dry)
            continue
        # "save0" -- freeze depth_pose_node on the current camera frame, so
        # every port lookup until "end0" reuses that one image.
        if raw.strip() == 'save0':
            armCmd.freeze(True)
            continue
        if raw.strip() == 'end0':
            armCmd.freeze(False)
            continue
        # "t" -- how far off square the flange currently is to the panel.
        # Was "tilt".
        if raw.strip() == 't':
            armCmd.check_tilt()
            continue
        # "roll <deg>" -- turn about the approach axis in place (the motion
        # the "where" hand-eye check needs).
        if raw.split() and raw.split()[0] == 'roll':
            parts = raw.split()
            dry = 'dry' in parts[1:]
            nums = [x for x in parts[1:] if x != 'dry']
            if len(nums) != 1:
                armCmd.get_logger().warn('usage: roll <deg> [dry], e.g. roll 30')
            else:
                armCmd.roll(float(nums[0]), move=not dry)
            continue
        # "where" -- current target's world position + drift since last read.
        if raw.strip() == 'where':
            armCmd.where()
            continue
        # "s" / "s dry" -- rotate in place to exactly 0 deg tilt. Was "square".
        if raw.split() and raw.split()[0] == 's':
            dry = 'dry' in raw.split()[1:]
            armCmd.square_up(move=not dry)
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