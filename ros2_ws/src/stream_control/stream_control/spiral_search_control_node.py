#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flange-frame force control node for TM PVT streaming control.

PVT commands the FLANGE (G) directly, and every control point sent out --
position AND orientation -- is derived by composing a small LOCAL delta pose
onto the flange's LIVE current pose:

    T_B_G : base <- flange (G). tool_pose from feedback_states is the ATC
            flange pose, so this is LIVE -- rebuilt every time feedback
            arrives (see _fb_cb / _make_transform).

Every tick, the controller (force servo + spiral) produces a pure
translation in the flange's own local axes -- [dx_g, dy_g, dz_g] -- with NO
rotation component, since the F/T sensor and the spiral search are both
already flange-frame quantities (no separate hand-eye calibration is needed
here: there is no end-effector frame in play in this node at all). That
local delta is composed onto the live flange pose:

    target_pos_base    = T_B_G[:3,:3] @ [dx_g,dy_g,dz_g] + T_B_G[:3,3]
    target_orient_base = T_B_G[:3,:3]     # local rotation delta is identity

This is why the node never sets a target orientation itself: since the
local rotation delta is always identity, the target orientation is simply
"whatever the flange's orientation is measured to be right now," which
holds it wherever it happened to be when the node started, with no
hardcoded rx/ry/rz to configure. (This is a PASSIVE hold -- it never
resists an external disturbance nudging the orientation, it just reports
back whatever is currently measured. An ACTIVE hold, closed-loop toward a
value captured once at startup, would be a different and slightly bigger
change -- see _ctrl_tick if that's ever needed.)

The spiral trajectory is pre-generated as PER-TICK DELTAS (this point minus
the last), not absolute offsets from a center, and combined with always
composing onto the live pose, this removes the need for any frozen "anchor"
transform: "hold position" is just "emit a zero delta," and
"resume/re-center the spiral" is just "start emitting deltas again from
wherever the flange currently is." The one place a fixed reference point is
still needed is HOLE_TESTING/INSERTING's steady-state and surface-height
checks, which have to compare against real feedback (not the commanded
deltas -- if the tool is resting against a hard stop, the commanded delta
and the actual movement disagree, and that disagreement is exactly what
those checks are trying to detect). That reference is _monitor_ref_pos_base,
a plain base-frame position snapshot.

PVTCommand is a service rather than a topic specifically so this node
cannot start dispatching points before the control node is actually up:
a service client only becomes "ready" once it has discovered a matching
server, so streaming is gated on `cmd_client.service_is_ready()` in
addition to having a start pose. Topics have no such guarantee -- points
published before the subscriber matches are silently dropped.

This node has NO knowledge of the send_script service, SendScript I/O,
ack latency, or CSV/PNG logging -- that all lives in robot_control_node.py.
"""

import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node

from stream_control.PI_controller import PI_controller

from geometry_msgs.msg import WrenchStamped
from tm_msgs.msg import FeedbackState
from custom_interface.srv import PVTCommand


class SearchState(Enum):
    """
    WAITING_TOUCH -> SEARCHING -> HOLE_TESTING -> (SEARCHING | INSERTING)

    Every tick's control point is a pure-translation delta in the flange's
    own local axes, composed onto the flange's LIVE pose (position AND
    orientation -- see class docstring):

        * Fz is read directly off the F/T sensor, which is already a
          flange-frame reading.
        * The PI controller's output is a step along the flange's own
          z-axis (the insertion direction) taken *from wherever the flange
          is right now* -- self-correcting every tick.
        * The spiral search supplies a per-tick step in the flange's x-y
          plane. Outside SEARCHING this step is simply zero, which is what
          makes "hold" and "freeze" trivial in a delta scheme -- no offset
          to keep re-applying, just nothing further to add.

    WAITING_TOUCH : z-force target = touch_force, no xy step.
                    -> SEARCHING when |touch_force-Fz|<=tol
    SEARCHING     : z-force target = touch_force, xy steps along the spiral.
                    -> HOLE_TESTING when Fz < align_force
    HOLE_TESTING  : z-force target = insert_force, no xy step. If the
                    flange's actual travel along its z-axis (measured from
                    live feedback against _monitor_ref_pos_base) settles
                    within position_steady_state_tolerance_mm of the recorded
                    surface_height, it wasn't a real hole -> back to
                    SEARCHING, the monitor point re-recorded here. If it
                    sinks past that tolerance, it's a real hole -> INSERTING.
    INSERTING     : z-force target = insert_force, no xy step. Terminal:
                    stops once the measured insertion-axis position is
                    steady within tolerance.
    """
    WAITING_TOUCH = auto()
    SEARCHING = auto()
    HOLE_TESTING = auto()
    INSERTING = auto()


class SpiralSearchControllerNode(Node):
    def __init__(self):
        super().__init__("spiral_search_controller")
        self.PVT_SERVER_CLIENT_ID = "spiral_search_node"

        self.touch_force = 1  # 1N

        # ----- parameters -----
        self.declare_parameter("PI_controller_Kp", 2e-4)
        self.declare_parameter("PI_controller_Ki", 1e-4)
        self.declare_parameter("PI_controller_integral_limit", 0.05)

        self.declare_parameter("ctrl_hz", 100.0)
        self.declare_parameter("duration_s", 10.0)
        self.declare_parameter("pvt_point_time_ratio", 0.9)  # Don't touch this
        self.declare_parameter("command_service", "pvt_command")
        self.declare_parameter("service_wait_log_period_s", 2.0)

        # ----- spiral search parameters (in the flange's x-y plane) -----
        # Archimedean spiral: r(theta) = (spiral_pitch_mm / 2*pi) * theta
        self.declare_parameter("spiral_pitch_mm", 2.0)          # radial growth per revolution
        self.declare_parameter("spiral_max_radius_mm", 15.0)    # stop generating past this radius
        self.declare_parameter("spiral_search_speed_mm_s", 3.0) # constant speed along the spiral path

        # ----- touch detection -----
        self.declare_parameter("touch_force_tol_n", 0.5)  # |Fz - touch_force| <= tol => "touched"

        # ----- alignment / insertion -----
        self.declare_parameter('position_steady_state_tolerance_mm', 0.05)
        self.declare_parameter('hole_testing_threshold_mm', 5)
        self.declare_parameter('steady_state_window_s', 0.3)
        self.declare_parameter("align_force", 0.4)   # Fz drops below this => hole/peg aligned
        self.declare_parameter("insert_force", 5.0)  # z-force target once aligned (the "peg" phase)
        self.declare_parameter("max_force_n", 8.0)   # hard safety cutoff, must stay above insert_force

        self.force_controller = PI_controller(
            Kp = self.get_parameter("PI_controller_Kp").value ,
            Ki = self.get_parameter("PI_controller_Ki").value ,
            integral_limit = self.get_parameter("PI_controller_integral_limit").value ,
        )

        self.ctrl_hz = float(self.get_parameter("ctrl_hz").value)
        self.duration_s = float(self.get_parameter("duration_s").value)
        self.pvt_point_time_ratio = float(self.get_parameter("pvt_point_time_ratio").value)

        if self.ctrl_hz <= 0.0:
            raise ValueError("ctrl_hz must be > 0")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be > 0")
        if self.pvt_point_time_ratio <= 0.0:
            raise ValueError("pvt_point_time_ratio must be > 0")

        self.ctrl_dt = 1.0 / self.ctrl_hz
        self.total_points = int(round(self.duration_s * self.ctrl_hz))

        self.spiral_pitch_mm = float(self.get_parameter("spiral_pitch_mm").value)
        self.spiral_max_radius_mm = float(self.get_parameter("spiral_max_radius_mm").value)
        self.spiral_search_speed_mm_s = float(self.get_parameter("spiral_search_speed_mm_s").value)
        self.touch_force_tol_n = float(self.get_parameter("touch_force_tol_n").value)

        if self.spiral_pitch_mm <= 0.0:
            raise ValueError("spiral_pitch_mm must be > 0")
        if self.spiral_max_radius_mm <= 0.0:
            raise ValueError("spiral_max_radius_mm must be > 0")
        if self.spiral_search_speed_mm_s <= 0.0:
            raise ValueError("spiral_search_speed_mm_s must be > 0")
        if self.touch_force_tol_n <= 0.0:
            raise ValueError("touch_force_tol_n must be > 0")

        self.position_steady_state_tolerance_m = (
            self.get_parameter('position_steady_state_tolerance_mm').value / 1000.0
        )

        self.hole_testing_threshold_m = (
            self.get_parameter('hole_testing_threshold_mm').value / 1000.0
        )

        steady_state_window_s = self.get_parameter('steady_state_window_s').value
        # ctrl_hz is fixed by the timer period, so a time window converts directly
        # to a sample count for the rolling buffers below.
        self._steady_state_len = max(2, round(steady_state_window_s * self.ctrl_hz))

        self._hole_test_z_history = deque(maxlen=self._steady_state_len)
        self._insert_z_history = deque(maxlen=self._steady_state_len)
        self.surface_height_g = None
        self._shutdown_requested = False

        self.align_force = float(self.get_parameter("align_force").value)
        self.insert_force = float(self.get_parameter("insert_force").value)
        self.max_force_n = float(self.get_parameter("max_force_n").value)

        if self.align_force <= 0.0:
            raise ValueError("align_force must be > 0")
        if self.insert_force <= 0.0:
            raise ValueError("insert_force must be > 0")
        if self.max_force_n <= self.insert_force:
            raise ValueError(
                "max_force_n must be greater than insert_force, otherwise the "
                "safety cutoff would trip as soon as the peg reaches its "
                "insertion force target"
            )

        # explicit finite state machine -- see SearchState docstring
        self.state = SearchState.WAITING_TOUCH

        # T_B_G: base <- flange, updated every time feedback_states arrives
        self.T_B_G = np.eye(4)

        # A plain base-frame position snapshot -- just a reference point,
        # not a transform -- recorded at startup and re-recorded whenever
        # the spiral re-centers. Used only to measure how far the flange
        # has actually travelled along its own z-axis since that moment
        # (see _ctrl_tick); it plays no part in generating the per-tick
        # command itself.
        self._monitor_ref_pos_base: Optional[np.ndarray] = None

        self.spiral_traj: List[tuple] = []
        self.spiral_idx = 0

        command_service = self.get_parameter("command_service").value
        self.service_wait_log_period_s = float(self.get_parameter("service_wait_log_period_s").value)
        self.cmd_client = self.create_client(PVTCommand, command_service)

        self.create_subscription(FeedbackState, "feedback_states", self._fb_cb, 10)
        self.create_subscription(WrenchStamped, 'optoforce/wrench', self._FTsensor_cb, 10)

        self.feedback_is_avaliable = False

        self.FT_is_avaliable = False
        self.force_observe = None
        self.torque_observe = None

        self.tick = 0
        self.done = False

        self._inflight = 0
        self._error_count = 0
        self._last_wait_log_wall = 0.0

        self.startup_timer = self.create_timer(0.05, self._startup_tick)
        self.ctrl_timer = None

        self.prev_time = None  # For control loop

        self.get_logger().info(
            f"{self.ctrl_hz:.1f}Hz, {self.total_points} points, "
            f"calling PVTCommand service '{command_service}', "
            f"orientation held passively at whatever it is on startup"
        )

    # ---------- ROS callbacks ----------
    def _fb_cb(self, msg: FeedbackState):
        if not msg.tool_pose or len(msg.tool_pose) < 6:
            return
        # tool_pose = ATC flange_pose (base <- flange), metres + radians
        x, y, z = msg.tool_pose[:3]
        quat = R.from_euler('xyz', msg.tool_pose[3:], degrees=False).as_quat()
        self.T_B_G = self._make_transform(x, y, z, quat)
        self.feedback_is_avaliable = True

    def _FTsensor_cb(self, msg: WrenchStamped):
        self.force_observe = msg.wrench.force
        self.torque_observe = msg.wrench.torque

        self.FT_is_avaliable = True
        #self.get_logger().info(f"Force-Torque capture")

    # ---------- helpers ----------
    @staticmethod
    def _make_transform(x, y, z, quat_xyzw):
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        T[:3, 3] = [x, y, z]
        return T

    def _generate_spiral_trajectory(self) -> List[tuple]:
        """Pre-generate an Archimedean spiral in the flange's x-y plane as
        PER-TICK DELTAS (this point minus the last), not absolute offsets
        from a center. That is what lets the control loop just keep adding
        each tick's delta onto wherever the flange currently is, with no
        separate spiral-center bookkeeping.

        r(theta) = b * theta, with b = spiral_pitch_mm / (2*pi), so the
        radius grows by spiral_pitch_mm every full revolution.

        Points are sampled at ctrl_dt such that the flange travels along
        the spiral at a constant spiral_search_speed_mm_s.

        Returns a list of (ddx_g_m, ddy_g_m, vx_g_mps, vy_g_mps) tuples,
        where ddx_g/ddy_g are this sample's step relative to the previous
        sample (the first sample's delta is relative to the spiral's own
        start point, i.e. effectively zero).
        """
        b = self.spiral_pitch_mm / (2.0 * math.pi)
        v = self.spiral_search_speed_mm_s
        dt = self.ctrl_dt
        max_r = self.spiral_max_radius_mm

        traj: List[tuple] = []
        theta = 1e-6  # avoid the singularity at theta = 0
        prev_x_m, prev_y_m = 0.0, 0.0

        while True:
            r = b * theta
            if r >= max_r:
                break

            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            x_mm = r * cos_t
            y_mm = r * sin_t

            # tangential velocity components (analytic derivative wrt time)
            dx_dtheta = b * (cos_t - theta * sin_t)
            dy_dtheta = b * (sin_t + theta * cos_t)
            theta_dot = v / (b * math.sqrt(theta * theta + 1.0))  # rad/s

            vx_mm_s = dx_dtheta * theta_dot
            vy_mm_s = dy_dtheta * theta_dot

            x_m, y_m = x_mm / 1000.0, y_mm / 1000.0
            traj.append((
                x_m - prev_x_m,
                y_m - prev_y_m,
                vx_mm_s / 1000.0,
                vy_mm_s / 1000.0,
            ))
            prev_x_m, prev_y_m = x_m, y_m

            theta += theta_dot * dt

        if not traj:
            traj.append((0.0, 0.0, 0.0, 0.0))

        return traj

    def _check_steady_state(self, history: deque, z: float) -> bool:
        """Push z into a rolling window; True once the window is full and its
        spread (max-min) is within position_steady_state_tolerance_m.

        Same discard-until-full behaviour as the OptoForce moving-average filter:
        a window that isn't full yet says nothing about steadiness, so it must
        not report True just because the spread-so-far happens to be small.
        """
        history.append(z)
        if len(history) < history.maxlen:
            return False
        return (max(history) - min(history)) <= self.position_steady_state_tolerance_m

    def _startup_tick(self):
        if (not self.feedback_is_avaliable) or (not self.FT_is_avaliable):
            return

        if not self.cmd_client.service_is_ready():
            now = time.time()
            if (now - self._last_wait_log_wall) >= self.service_wait_log_period_s:
                self._last_wait_log_wall = now
                self.get_logger().info(
                    "Waiting for robot control node's PVTCommand service..."
                )
            return  # do NOT generate/start streaming until the server exists

        self.startup_timer.cancel()

        self.get_logger().info("✓ PVTCommand service is up")

        # record the starting flange position, purely as the reference point
        # for insertion-axis travel monitoring (see class docstring) --
        # the command loop itself needs no such reference
        self._monitor_ref_pos_base = self.T_B_G[:3, 3].copy()
        self.spiral_traj = self._generate_spiral_trajectory()

        self.get_logger().info(
            f"Generated spiral trajectory: {len(self.spiral_traj)} points "
            f"(pitch={self.spiral_pitch_mm:.2f}mm, max_r={self.spiral_max_radius_mm:.2f}mm, "
            f"speed={self.spiral_search_speed_mm_s:.2f}mm/s), "
            f"monitor ref(base)={np.round(self._monitor_ref_pos_base, 4).tolist()}"
        )

        self.ctrl_timer = self.create_timer(self.ctrl_dt, self._ctrl_tick)

    def _ctrl_tick(self):
        if (not self.feedback_is_avaliable) or (not self.FT_is_avaliable):
            return
        if self._shutdown_requested:
            return

        now = self.get_clock().now()

        if self.prev_time is None:
            self.prev_time = now
            return

        dt = (now - self.prev_time).nanoseconds * 1e-9
        self.prev_time = now

        if dt <= 0.0:
            return

        # live flange pose (base <- flange). Used as-is: position and
        # orientation are both taken straight from current feedback, since
        # every control point this tick is a pure-translation LOCAL delta
        # composed onto it (see class docstring).
        current_flange_pos_base = self.T_B_G[:3, 3]
        R_B_G_live = self.T_B_G[:3, :3]

        fz = self.force_observe.z  # F/T sensor is mounted at the flange -- already flange-frame

        # How far the flange has actually travelled along its own z-axis
        # since _monitor_ref_pos_base was last (re)recorded. This is real
        # feedback, not the commanded deltas below, which matters exactly
        # when they'd disagree -- e.g. resting against a hard stop, where
        # the commanded step keeps arriving but the flange stops moving.
        # That disagreement is what HOLE_TESTING/INSERTING need to detect,
        # so it has to be measured, not assumed.
        gz_base_now = R_B_G_live @ np.array([0.0, 0.0, 1.0])  # flange's own z-axis, expressed in base, right now
        current_z_g = float((current_flange_pos_base - self._monitor_ref_pos_base) @ gz_base_now)

        # ================= 1) STATE TRANSITIONS =================
        if self.state == SearchState.WAITING_TOUCH:
            if abs(self.touch_force - fz) <= self.touch_force_tol_n:
                self.state = SearchState.SEARCHING
                self.spiral_idx = 0
                self.get_logger().info(
                    f"[TOUCH] contact detected (Fz={fz:.3f}N) -- starting spiral search"
                )

        elif self.state == SearchState.SEARCHING:
            if fz < self.align_force:
                self.state = SearchState.HOLE_TESTING
                self.surface_height_g = current_z_g
                self._hole_test_z_history.clear()
                self.get_logger().info(
                    f"[CANDIDATE] possible alignment (Fz={fz:.3f}N < "
                    f"align_force={self.align_force:.3f}N) -- testing for a real "
                    f"hole from surface_height(g)={self.surface_height_g:.5f}m"
                )

        elif self.state == SearchState.HOLE_TESTING:
            if self._check_steady_state(self._hole_test_z_history, current_z_g):
                moved_m = abs(current_z_g - self.surface_height_g)
                if moved_m <= self.hole_testing_threshold_m:
                    # settled without sinking -- resting on the surface, not a hole
                    self.get_logger().info(
                        f"[NOT A HOLE] settled {moved_m*1000:.3f}mm from surface "
                        f"(tol={self.hole_testing_threshold_m*1000:.3f}mm) "
                        f"-- resuming spiral from the current position"
                    )
                    self._monitor_ref_pos_base = current_flange_pos_base.copy()
                    self.spiral_idx = 0
                    self._hole_test_z_history.clear()
                    self.state = SearchState.SEARCHING
                else:
                    # sank in past tolerance -- this is the hole
                    self.get_logger().info(
                        f"[HOLE CONFIRMED] settled {moved_m*1000:.3f}mm from "
                        f"surface -- inserting to {self.insert_force:.2f}N"
                    )
                    self._insert_z_history.clear()
                    self.state = SearchState.INSERTING

        elif self.state == SearchState.INSERTING:
            if self._check_steady_state(self._insert_z_history, current_z_g):
                self.get_logger().info(
                    f"[INSERT COMPLETE] z steady within "
                    f"{self.position_steady_state_tolerance_m*1000:.3f}mm over "
                    f"{self._steady_state_len} ticks -- holding and stopping"
                )
                self._shutdown_requested = True

        # ============ 2) Z FORCE OUTPUT (insertion axis) ============
        # A step from wherever the flange is *right now* -- no reference
        # point needed, this is a pure delta.
        z_target = (self.insert_force
                    if self.state in (SearchState.HOLE_TESTING, SearchState.INSERTING)
                    else self.touch_force)

        error = z_target - fz
        delta_z = self.force_controller.update(error, dt)

        dz_g = delta_z
        vz_g = delta_z * self.ctrl_hz

        # ============ 3) XY OUTPUT (spiral search plane) ============
        # Also a pure delta: outside SEARCHING it is simply zero, which is
        # exactly "add nothing further" -- there is no offset to keep
        # re-applying, so freezing the spiral is free.
        if self.state == SearchState.SEARCHING and self.spiral_traj:
            if self.spiral_idx < len(self.spiral_traj):
                ddx_g, ddy_g, vx_g, vy_g = self.spiral_traj[self.spiral_idx]
                self.spiral_idx += 1
            else:
                # spiral exhausted (max radius reached) with no contact yet --
                # hold here rather than re-emitting the last nonzero delta,
                # which would otherwise keep pushing outward forever
                ddx_g, ddy_g, vx_g, vy_g = 0.0, 0.0, 0.0, 0.0
        else:
            ddx_g, ddy_g, vx_g, vy_g = 0.0, 0.0, 0.0, 0.0

        # Insertion just concluded: emit no further z step. Position has
        # stopped moving but the force error may not have (e.g. mechanically
        # bottomed out), so applying the PI output here would keep
        # commanding further push against a hard stop.
        if self._shutdown_requested:
            dz_g = 0.0
            vz_g = 0.0

        # The control point: a pure-translation local delta in the flange's
        # own axes, no rotation component.
        delta_g = np.array([ddx_g, ddy_g, dz_g])
        velocity_g = np.array([vx_g, vy_g, vz_g])

        # Compose onto the live flange pose. Position: rotate the local
        # delta by the live orientation and add to the current position.
        # Orientation: the local rotation delta is identity, so the target
        # orientation is simply the live orientation itself -- this is what
        # holds the flange at whatever orientation it had on startup
        # without ever specifying rx/ry/rz manually (see class docstring
        # for the passive-vs-active hold trade-off).
        target_pos_base = current_flange_pos_base + R_B_G_live @ delta_g
        target_vel_base = R_B_G_live @ velocity_g
        target_orient_deg = R.from_matrix(R_B_G_live).as_euler('xyz', degrees=True)

        req = PVTCommand.Request()
        req.header.stamp = self.get_clock().now().to_msg()
        req.client_id = self.PVT_SERVER_CLIENT_ID
        req.tick = self.tick
        req.is_last = (self.tick == self.total_points - 1) or self._shutdown_requested

        req.x_m, req.y_m, req.z_m = target_pos_base.tolist()
        req.rx_deg, req.ry_deg, req.rz_deg = target_orient_deg.tolist()

        req.vx_mps, req.vy_mps, req.vz_mps = target_vel_base.tolist()
        req.wx_dps = 0.0
        req.wy_dps = 0.0
        req.wz_dps = 0.0

        req.point_time_s = self.ctrl_dt * self.pvt_point_time_ratio

        # Safty check (Stop robot if force > max_force_n)
        # Hold in place: zero delta, zero velocity, straight from wherever
        # the flange currently is (orientation unchanged, same as above).
        if self.force_observe.z > self.max_force_n:
            req.x_m, req.y_m, req.z_m = current_flange_pos_base.tolist()
            req.rx_deg, req.ry_deg, req.rz_deg = target_orient_deg.tolist()
            req.vx_mps, req.vy_mps, req.vz_mps = 0.0, 0.0, 0.0
            self.ctrl_timer.cancel()
            return

        this_tick = self.tick
        future = self.cmd_client.call_async(req)
        self._inflight += 1

        if req.is_last:
            self.ctrl_timer.cancel()

        def _done_cb(fut, tick=this_tick):
            self._inflight -= 1
            try:
                resp = fut.result()
            except Exception as exc:
                self._error_count += 1
                self.get_logger().error(f"[GEN {tick:03d}] service call failed: {exc}")
                return
            if resp.result != PVTCommand.Response.ROBOT_OK:
                self._error_count += 1
                self.get_logger().warn(
                    f"[GEN {tick:03d}] ROBOT_ERROR from control node: {resp.message}"
                )

        future.add_done_callback(_done_cb)

        if self.tick % 10 == 0:
            if self.state == SearchState.INSERTING:
                status = f"INSERTING (target={self.insert_force:.2f}N)"
            elif self.state == SearchState.HOLE_TESTING:
                status = f"HOLE_TESTING (surface_g={self.surface_height_g:.5f}m)"
            elif self.state == SearchState.SEARCHING:
                status = f"SEARCHING (spiral_idx={self.spiral_idx}/{len(self.spiral_traj)})"
            else:
                status = "WAITING_TOUCH"
            self.get_logger().debug(
                f"[GEN {self.tick:03d}] z_g={current_z_g:+.5f} vz_g={vz_g:+.3f} "
                f"base=({target_pos_base[0]:.4f},{target_pos_base[1]:.4f},{target_pos_base[2]:.4f}) "
                f"inflight={self._inflight} ({status})"
            )

        self.tick += 1

        if self._shutdown_requested:
            self.get_logger().info("[STOP] insertion complete -- shutting down node")
            rclpy.shutdown()


def main():
    rclpy.init()
    node = SpiralSearchControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()