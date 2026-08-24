#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z-axis force control node node for TM PVT streaming control.

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

import rclpy
from rclpy.node import Node

from stream_control.PI_controller import PI_controller

from geometry_msgs.msg import WrenchStamped
from tm_msgs.msg import FeedbackState
from custom_interface.srv import PVTCommand


class SearchState(Enum):
    """
    WAITING_TOUCH -> SEARCHING -> INSERTING

    WAITING_TOUCH : z-force target = fz_desire, XY held at the spiral center.
                    -> SEARCHING when |fz_desire - Fz| <= touch_force_tol_n
    SEARCHING     : z-force target = fz_desire, XY streams the spiral.
                    -> INSERTING when Fz < align_force
    INSERTING     : z-force target = insert_force, XY frozen at the last
                    spiral point reached. Terminal state.
    """
    WAITING_TOUCH = auto()
    SEARCHING = auto()
    INSERTING = auto()


class SpiralSearchControllerNode(Node):
    def __init__(self):
        super().__init__("spiral_search_controller")

        self.fz_desire = 1  # 1N

        # ----- parameters -----
        self.declare_parameter("PI_controller_Kp", 2e-4)
        self.declare_parameter("PI_controller_Ki", 1e-4)
        self.declare_parameter("PI_controller_integral_limit", 0.05)
        
        self.declare_parameter("ctrl_hz", 100.0)
        self.declare_parameter("duration_s", 10.0)
        self.declare_parameter("pvt_point_time_ratio", 0.9)  # Don't touch this
        self.declare_parameter("command_service", "pvt_command")
        self.declare_parameter("service_wait_log_period_s", 2.0)

        # ----- spiral search parameters -----
        # Archimedean spiral: r(theta) = (spiral_pitch_mm / 2*pi) * theta
        self.declare_parameter("spiral_pitch_mm", 2.0)          # radial growth per revolution
        self.declare_parameter("spiral_max_radius_mm", 15.0)    # stop generating past this radius
        self.declare_parameter("spiral_search_speed_mm_s", 3.0) # constant speed along the spiral path

        # ----- touch detection -----
        self.declare_parameter("touch_force_tol_n", 0.5)  # |Fz - fz_desire| <= tol => "touched"

        # ----- alignment / insertion -----
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

        # spiral center (recorded from start pose) and pre-generated trajectory,
        # each entry is (dx_m, dy_m, vx_mps, vy_mps) relative to the center
        self.spiral_center_x_m: Optional[float] = None
        self.spiral_center_y_m: Optional[float] = None
        self.spiral_traj: List[tuple] = []
        self.spiral_idx = 0

        command_service = self.get_parameter("command_service").value
        self.service_wait_log_period_s = float(self.get_parameter("service_wait_log_period_s").value)
        self.cmd_client = self.create_client(PVTCommand, command_service)

        self.create_subscription(FeedbackState, "feedback_states", self._fb_cb, 10)
        self.create_subscription(WrenchStamped, 'optoforce/wrench', self._FTsensor_cb, 10)

        self.feedback_is_avaliable = False
        self.current_pose_6d: Optional[List[float]] = None

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
            f"calling PVTCommand service '{command_service}'"
        )

    # ---------- ROS callbacks ----------
    def _fb_cb(self, msg: FeedbackState):
        if msg.tool_pose and len(msg.tool_pose) >= 6:
            self.current_pose_6d = [
                float(msg.tool_pose[0]),
                float(msg.tool_pose[1]),
                float(msg.tool_pose[2]),
                math.degrees(float(msg.tool_pose[3])),
                math.degrees(float(msg.tool_pose[4])),
                math.degrees(float(msg.tool_pose[5])),
            ]
            self.feedback_is_avaliable = True
            #self.get_logger().info(f"pose captured: Z={self.current_pose_6d[2]:.4f}")

    def _FTsensor_cb(self, msg: WrenchStamped):
        self.force_observe = msg.wrench.force
        self.torque_observe = msg.wrench.torque

        self.FT_is_avaliable = True
        #self.get_logger().info(f"Force-Torque capture")

    # ---------- spiral trajectory generation ----------
    def _generate_spiral_trajectory(self) -> List[tuple]:
        """Pre-generate an Archimedean spiral in the XY plane, centered at the
        origin (offsets are added to the recorded start pose later).

        r(theta) = b * theta, with b = spiral_pitch_mm / (2*pi), so the
        radius grows by spiral_pitch_mm every full revolution.

        Points are sampled at ctrl_dt such that the end effector travels
        along the spiral at a constant spiral_search_speed_mm_s.

        Returns a list of (dx_m, dy_m, vx_mps, vy_mps) tuples.
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

    def _startup_tick(self):
        if (not self.feedback_is_avaliable) or (self.current_pose_6d is None) or (not self.FT_is_avaliable):
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

        # record the starting XY as the spiral center and pre-generate the
        # search spiral (only XY -- Z is driven by the force controller)
        self.spiral_center_x_m = self.current_pose_6d[0]
        self.spiral_center_y_m = self.current_pose_6d[1]
        self.spiral_traj = self._generate_spiral_trajectory()

        self.get_logger().info(
            f"Generated spiral trajectory: {len(self.spiral_traj)} points "
            f"(pitch={self.spiral_pitch_mm:.2f}mm, max_r={self.spiral_max_radius_mm:.2f}mm, "
            f"speed={self.spiral_search_speed_mm_s:.2f}mm/s), "
            f"center=({self.spiral_center_x_m:.4f}, {self.spiral_center_y_m:.4f})"
        )

        self.ctrl_timer = self.create_timer(self.ctrl_dt, self._ctrl_tick)

    def _ctrl_tick(self):
        if (not self.feedback_is_avaliable) or (not self.FT_is_avaliable):
            return
        
        now = self.get_clock().now()

        if self.prev_time is None:
            self.prev_time = now
            return

        dt = (now - self.prev_time).nanoseconds * 1e-9
        self.prev_time = now

        # Guard against bad/zero dt
        if dt <= 0.0:
            return
    
        z_current = self.current_pose_6d[2]
        fz = self.force_observe.z

        # ================= 1) STATE TRANSITIONS =================
        # Evaluate the current state's exit condition against the latest
        # sensor reading and advance self.state at most once per tick.
        if self.state == SearchState.WAITING_TOUCH:
            if abs(self.fz_desire - fz) <= self.touch_force_tol_n:
                self.state = SearchState.SEARCHING
                self.spiral_idx = 0
                self.get_logger().info(
                    f"[TOUCH] contact detected (Fz={fz:.3f}N) -- starting spiral search"
                )

        elif self.state == SearchState.SEARCHING:
            if fz < self.align_force:
                self.state = SearchState.INSERTING
                self.get_logger().info(
                    f"[ALIGNED] hole/peg aligned (Fz={fz:.3f}N < "
                    f"align_force={self.align_force:.3f}N) -- stopping spiral, "
                    f"inserting to {self.insert_force:.2f}N"
                )

        elif self.state == SearchState.INSERTING:
            pass  # terminal state, no exit condition (yet)

        # ================= 2) Z-FORCE OUTPUT (per state) =================
        z_target = self.insert_force if self.state == SearchState.INSERTING else self.fz_desire

        error = z_target - fz
        delta_z = self.force_controller.update(error, dt)

        z_command = z_current - delta_z
        vz = delta_z * self.ctrl_hz

        # ================= 3) XY OUTPUT (per state) =================
        if self.state == SearchState.WAITING_TOUCH:
            x_command = self.spiral_center_x_m
            y_command = self.spiral_center_y_m
            vx_mps = 0.0
            vy_mps = 0.0

        elif self.state == SearchState.SEARCHING:
            if self.spiral_traj:
                idx = min(self.spiral_idx, len(self.spiral_traj) - 1)
                dx_m, dy_m, vx_mps, vy_mps = self.spiral_traj[idx]
                if self.spiral_idx < len(self.spiral_traj) - 1:
                    self.spiral_idx += 1
            else:
                dx_m, dy_m, vx_mps, vy_mps = 0.0, 0.0, 0.0, 0.0
            x_command = self.spiral_center_x_m + dx_m
            y_command = self.spiral_center_y_m + dy_m

        else:  # INSERTING -- freeze XY at the point the spiral stopped
            if self.spiral_traj:
                idx = min(self.spiral_idx, len(self.spiral_traj) - 1)
                dx_m, dy_m = self.spiral_traj[idx][0], self.spiral_traj[idx][1]
            else:
                dx_m, dy_m = 0.0, 0.0
            x_command = self.spiral_center_x_m + dx_m
            y_command = self.spiral_center_y_m + dy_m
            vx_mps = 0.0
            vy_mps = 0.0

        req = PVTCommand.Request()
        req.header.stamp = self.get_clock().now().to_msg()
        req.tick = self.tick
        req.is_last = (self.tick == self.total_points - 1)

        req.x_m = x_command
        req.y_m = y_command
        req.z_m = z_command
        req.rx_deg = self.current_pose_6d[3]
        req.ry_deg = self.current_pose_6d[4]
        req.rz_deg = self.current_pose_6d[5]

        req.vx_mps = vx_mps
        req.vy_mps = vy_mps
        req.vz_mps = vz
        req.wx_dps = 0.0
        req.wy_dps = 0.0
        req.wz_dps = 0.0

        req.point_time_s = self.ctrl_dt * self.pvt_point_time_ratio

        # Safty check (Stop robot if force > max_force_n)
        if self.force_observe.z > self.max_force_n:          
            req.z_m = self.current_pose_6d[2]
            req.vz_mps = 0.0
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
            elif self.state == SearchState.SEARCHING:
                status = f"SEARCHING (spiral_idx={self.spiral_idx}/{len(self.spiral_traj)})"
            else:
                status = "WAITING_TOUCH"
            self.get_logger().info(f"[GEN {self.tick:03d}] z={z_current:.4f} vz={vz:+.3f} inflight={self._inflight} ({status})")

        self.tick += 1


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