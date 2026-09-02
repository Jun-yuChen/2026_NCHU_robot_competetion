#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-effector-frame force control node for TM PVT streaming control.

All control math -- the force servo and the spiral search -- is done in the
end-effector (tool) frame, E. Getting an E-frame point p_E into the base
frame goes through the same two transforms used elsewhere in this
workspace:

    T_B_G : base <- flange (G). tool_pose from feedback_states is the ATC
            flange pose, so this is LIVE -- rebuilt every time feedback
            arrives (see _fb_cb / _make_transform).
    T_G_E : flange <- end-effector (E). A fixed hand-eye-style calibration
            loaded once from robot_wrist_hand_eye_config_path.

    p_B = T_B_G @ T_G_E @ p_E

Two different things need this transform, and they use it differently:

    * The force servo moves the tool along E's own z-axis, correcting off
      whatever the sensor reads *right now*. This is naturally a LIVE,
      closed-loop use of T_B_G: "current TCP position (from the live
      transform) + a small rotated step".

    * The spiral search needs an offset that stays anchored to wherever the
      search started, not to wherever the tool happens to be this tick --
      re-deriving it from the live T_B_G every tick would compound each
      tick's own offset on top of the last and the pattern would run away.
      So a FROZEN snapshot of T_B_G, taken when the spiral (re)starts, is
      used for that half instead.

Both halves are combined into a single E-frame offset vector and rotated
into the base frame in one shot every tick -- see _ctrl_tick.

Control points are calculated in end-effector frame with respect to a fixed
anchored point. Those control points has postfix _ee.

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
import yaml
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

    All control below is expressed in the end-effector frame E, then carried
    into the base frame via p_B = T_B_G @ T_G_E @ p_E:

        * Fz is read directly off the F/T sensor (mounted at E) -- no
          rotation needed to get it into E-frame terms.
        * The PI controller's output moves the tool along E's z-axis (the
          insertion direction), self-correcting off the LIVE T_B_G every
          tick.
        * The spiral search offsets move the tool in E's x-y plane, anchored
          to a FROZEN snapshot of T_B_G taken when the search (re)starts.

    WAITING_TOUCH : ez-force target = touch_force, ee-XY held at the anchor
                    (zero offset). -> SEARCHING when |touch_force-Fz|<=tol
    SEARCHING     : ez-force target = touch_force, ee-XY streams the spiral.
                    -> HOLE_TESTING when Fz < align_force
    HOLE_TESTING  : ez-force target = insert_force. If the ee-z position
                    settles within position_steady_state_tolerance_mm of
                    the recorded surface_height, it wasn't a real hole ->
                    back to SEARCHING, anchor (and spiral) re-centered here.
                    If it sinks past that tolerance, it's a real hole ->
                    INSERTING.
    INSERTING     : ez-force target = insert_force, ee-XY frozen at the last
                    spiral point reached. Terminal: stops once ee-z is
                    steady within tolerance.
    """
    WAITING_TOUCH = auto()
    SEARCHING = auto()
    HOLE_TESTING = auto()
    INSERTING = auto()


class SpiralSearchControllerNode(Node):
    def __init__(self):
        super().__init__("spiral_search_controller")

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

        # ----- hand-eye / tool calibration -----
        # T_G_E: flange (G) <- end-effector (E), fixed, expressed in metres.
        # T_B_G (base <- flange) is NOT declared here -- it is live, rebuilt
        # every time feedback_states arrives (see _fb_cb).
        self.declare_parameter('robot_wrist_hand_eye_config_path', '/path/to/hand/eye/calib.yaml')
        config_path = self.get_parameter('robot_wrist_hand_eye_config_path').get_parameter_value().string_value
        self.get_logger().info(f"config path: {config_path}")

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        # Expected: 4x4 nested lists (metres), extrinsics expressed into the flange frame G
        self.T_G_E = np.array(config_data['T_G_E'], dtype=float)
        if self.T_G_E.shape != (4, 4):
            raise ValueError(f"T_G_E in {config_path} must be a 4x4 matrix, got shape {self.T_G_E.shape}")

        # ----- fixed commanded tool orientation -----
        # The arm always holds this orientation (radians, TM's tool_pose
        # convention). Sent on every outgoing PVTCommand as the target
        # flange orientation -- unrelated to T_G_E, which only maps flange
        # axes onto end-effector axes.
        self.declare_parameter("orientation_rx_rad", 1.57)
        self.declare_parameter("orientation_ry_rad", 0.0)
        self.declare_parameter("orientation_rz_rad", 1.57)
        self.orientation_deg = np.degrees([
            float(self.get_parameter("orientation_rx_rad").value),
            float(self.get_parameter("orientation_ry_rad").value),
            float(self.get_parameter("orientation_rz_rad").value),
        ])

        # ----- spiral search parameters (in E's x-y plane) -----
        # Archimedean spiral: r(theta) = (spiral_pitch_mm / 2*pi) * theta
        self.declare_parameter("spiral_pitch_mm", 2.0)          # radial growth per revolution
        self.declare_parameter("spiral_max_radius_mm", 15.0)    # stop generating past this radius
        self.declare_parameter("spiral_search_speed_mm_s", 3.0) # constant speed along the spiral path

        # ----- touch detection -----
        self.declare_parameter("touch_force_tol_n", 0.5)  # |Fz - touch_force| <= tol => "touched"

        # ----- alignment / insertion -----
        self.declare_parameter('position_steady_state_tolerance_mm', 0.05)
        self.declare_parameter('steady_state_window_s', 0.3)
        self.declare_parameter("align_force", 0.4)   # Fz drops below this => hole/peg aligned
        self.declare_parameter("insert_force", 5.0)  # ez-force target once aligned (the "peg" phase)
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
        steady_state_window_s = self.get_parameter('steady_state_window_s').value
        # ctrl_hz is fixed by the timer period, so a time window converts directly
        # to a sample count for the rolling buffers below.
        self._steady_state_len = max(2, round(steady_state_window_s * self.ctrl_hz))

        self._hole_test_z_history = deque(maxlen=self._steady_state_len)
        self._insert_z_history = deque(maxlen=self._steady_state_len)
        self.surface_height_ee = None
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

        # The anchor is a FROZEN snapshot of T_B_G, taken when the spiral
        # (re)starts (see _set_anchor). anchor_pos_base_m / R_B_E_anchor are
        # cached from it -- (T_B_G_anchor @ T_G_E)'s translation/rotation --
        # so the per-tick math below is just a matmul, not a fresh 4x4
        # compose every cycle.
        self.T_B_G_anchor: Optional[np.ndarray] = None
        self.anchor_pos_base_m: Optional[np.ndarray] = None
        self.R_B_E_anchor: Optional[np.ndarray] = None

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
            f"fixed orientation (deg)={np.round(self.orientation_deg, 2).tolist()}"
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
        """Pre-generate an Archimedean spiral in E's x-y plane, centered at
        the origin (offsets are added to the anchor, rotated into base,
        later).

        r(theta) = b * theta, with b = spiral_pitch_mm / (2*pi), so the
        radius grows by spiral_pitch_mm every full revolution.

        Points are sampled at ctrl_dt such that the end effector travels
        along the spiral at a constant spiral_search_speed_mm_s.

        Returns a list of (dx_ee_m, dy_ee_m, vx_ee_mps, vy_ee_mps) tuples.
        """
        b = self.spiral_pitch_mm / (2.0 * math.pi)
        v = self.spiral_search_speed_mm_s
        dt = self.ctrl_dt
        max_r = self.spiral_max_radius_mm

        traj: List[tuple] = []
        theta = 1e-6  # avoid the singularity at theta = 0

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

            traj.append((
                x_mm / 1000.0,
                y_mm / 1000.0,
                vx_mm_s / 1000.0,
                vy_mm_s / 1000.0,
            ))

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

    def _set_anchor(self, T_B_G_snapshot: np.ndarray):
        """Freeze T_B_G_snapshot as the search anchor and cache the base-frame
        position/rotation of T_B_E = T_B_G_anchor @ T_G_E derived from it.

        This is the ONE place a fixed reference for the spiral gets taken --
        everywhere else uses either this cache or the live T_B_G, never a
        fresh anchor of its own, so there is exactly one anchor in play at
        any time.
        """
        self.T_B_G_anchor = T_B_G_snapshot.copy()
        T_B_E_anchor = self.T_B_G_anchor @ self.T_G_E
        self.anchor_pos_base_m = T_B_E_anchor[:3, 3]
        self.R_B_E_anchor = T_B_E_anchor[:3, :3]

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

        # freeze the starting pose as the anchor (all E-frame offsets --
        # spiral xy and the force-driven z -- are relative to this point)
        # and pre-generate the search spiral in E's x-y plane
        self._set_anchor(self.T_B_G)
        self.spiral_traj = self._generate_spiral_trajectory()

        self.get_logger().info(
            f"Generated spiral trajectory: {len(self.spiral_traj)} points "
            f"(pitch={self.spiral_pitch_mm:.2f}mm, max_r={self.spiral_max_radius_mm:.2f}mm, "
            f"speed={self.spiral_search_speed_mm_s:.2f}mm/s), "
            f"anchor(base)={np.round(self.anchor_pos_base_m, 4).tolist()}"
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

        # live TCP position in base frame: T_B_G (live) @ T_G_E (fixed)
        current_tcp_pos_base = (self.T_B_G @ self.T_G_E)[:3, 3]
        # ... expressed in E coordinates relative to the (frozen) anchor
        rel_ee_now = self.R_B_E_anchor.T @ (current_tcp_pos_base - self.anchor_pos_base_m)
        current_z_ee = rel_ee_now[2]   # insertion-axis position, relative to anchor
        fz = self.force_observe.z      # F/T sensor is mounted at E -- already an E-frame reading

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
                self.surface_height_ee = current_z_ee
                self._hole_test_z_history.clear()
                self.get_logger().info(
                    f"[CANDIDATE] possible alignment (Fz={fz:.3f}N < "
                    f"align_force={self.align_force:.3f}N) -- testing for a real "
                    f"hole from surface_height(ee)={self.surface_height_ee:.5f}m"
                )

        elif self.state == SearchState.HOLE_TESTING:
            if self._check_steady_state(self._hole_test_z_history, current_z_ee):
                moved_m = abs(current_z_ee - self.surface_height_ee)
                if moved_m <= self.position_steady_state_tolerance_m:
                    # settled without sinking -- resting on the surface, not a hole
                    self.get_logger().info(
                        f"[NOT A HOLE] settled {moved_m*1000:.3f}mm from surface "
                        f"(tol={self.position_steady_state_tolerance_m*1000:.3f}mm) "
                        f"-- resuming spiral, anchor re-centered here"
                    )
                    self._set_anchor(self.T_B_G)
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
            if self._check_steady_state(self._insert_z_history, current_z_ee):
                self.get_logger().info(
                    f"[INSERT COMPLETE] ee-z steady within "
                    f"{self.position_steady_state_tolerance_m*1000:.3f}mm over "
                    f"{self._steady_state_len} ticks -- holding and stopping"
                )
                self._shutdown_requested = True

        # ============ 2) EE-Z FORCE OUTPUT (insertion axis) ============
        z_target = (self.insert_force
                    if self.state in (SearchState.HOLE_TESTING, SearchState.INSERTING)
                    else self.touch_force)

        error = z_target - fz
        delta_z = self.force_controller.update(error, dt)

        new_z_ee = current_z_ee - delta_z
        vz_ee = delta_z * self.ctrl_hz

        # ============ 3) EE-XY OUTPUT (spiral search plane) ============
        if self.state == SearchState.WAITING_TOUCH:
            dx_ee, dy_ee = 0.0, 0.0
            vx_ee, vy_ee = 0.0, 0.0

        elif self.state == SearchState.SEARCHING:
            if self.spiral_traj:
                idx = min(self.spiral_idx, len(self.spiral_traj) - 1)
                dx_ee, dy_ee, vx_ee, vy_ee = self.spiral_traj[idx]
                if self.spiral_idx < len(self.spiral_traj) - 1:
                    self.spiral_idx += 1
            else:
                dx_ee, dy_ee, vx_ee, vy_ee = 0.0, 0.0, 0.0, 0.0

        else:  # HOLE_TESTING or INSERTING -- freeze ee-XY at the test/insert point
            if self.spiral_traj:
                idx = min(self.spiral_idx, len(self.spiral_traj) - 1)
                dx_ee, dy_ee = self.spiral_traj[idx][0], self.spiral_traj[idx][1]
            else:
                dx_ee, dy_ee = 0.0, 0.0
            vx_ee, vy_ee = 0.0, 0.0

        # Insertion just concluded: hold ee-z at its current position rather
        # than trusting delta_z. Position has stopped moving but the force
        # error may not have (e.g. mechanically bottomed out), so applying
        # the PI output here would keep commanding further push against a
        # hard stop.
        if self._shutdown_requested:
            new_z_ee = current_z_ee
            vz_ee = 0.0

        offset_ee = np.array([dx_ee, dy_ee, new_z_ee])
        velocity_ee = np.array([vx_ee, vy_ee, vz_ee])

        # p_B = T_B_G @ T_G_E @ p_E, applied via the frozen anchor's cached
        # translation/rotation rather than re-composing the 4x4 every tick.
        target_pos_base = self.anchor_pos_base_m + self.R_B_E_anchor @ offset_ee
        target_vel_base = self.R_B_E_anchor @ velocity_ee

        req = PVTCommand.Request()
        req.header.stamp = self.get_clock().now().to_msg()
        req.tick = self.tick
        req.is_last = (self.tick == self.total_points - 1) or self._shutdown_requested

        req.x_m, req.y_m, req.z_m = target_pos_base.tolist()
        req.rx_deg, req.ry_deg, req.rz_deg = self.orientation_deg.tolist()

        req.vx_mps, req.vy_mps, req.vz_mps = target_vel_base.tolist()
        req.wx_dps = 0.0
        req.wy_dps = 0.0
        req.wz_dps = 0.0

        req.point_time_s = self.ctrl_dt * self.pvt_point_time_ratio

        # Safty check (Stop robot if force > max_force_n)
        # Only the ee-z (force/insertion) component is held here, same as
        # before -- the ee-xy spiral offset is left as computed above.
        if self.force_observe.z > self.max_force_n:
            hold_offset_ee = np.array([dx_ee, dy_ee, current_z_ee])
            hold_pos_base = self.anchor_pos_base_m + self.R_B_E_anchor @ hold_offset_ee
            hold_vel_base = self.R_B_E_anchor @ np.array([vx_ee, vy_ee, 0.0])
            req.x_m, req.y_m, req.z_m = hold_pos_base.tolist()
            req.vx_mps, req.vy_mps, req.vz_mps = hold_vel_base.tolist()
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
                self.get_logger().error(
                    f"[GEN {tick:03d}] ROBOT_ERROR from control node: {resp.message}"
                )

        future.add_done_callback(_done_cb)

        if self.tick % 10 == 0:
            if self.state == SearchState.INSERTING:
                status = f"INSERTING (target={self.insert_force:.2f}N)"
            elif self.state == SearchState.HOLE_TESTING:
                status = f"HOLE_TESTING (surface_ee={self.surface_height_ee:.5f}m)"
            elif self.state == SearchState.SEARCHING:
                status = f"SEARCHING (spiral_idx={self.spiral_idx}/{len(self.spiral_traj)})"
            else:
                status = "WAITING_TOUCH"
            self.get_logger().info(
                f"[GEN {self.tick:03d}] z_ee={current_z_ee:+.5f} vz_ee={vz_ee:+.3f} "
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
