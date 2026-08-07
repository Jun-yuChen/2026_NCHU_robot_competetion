#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory generator node for TM PVT streaming control.

Responsibilities
-----------------
- Wait for one FeedbackState message to capture the starting tool pose.
- Pre-generate the complete Z trajectory (all positions) from a
  ZTrajectoryProvider (e.g. SineZTrajectory), exactly like the original
  pvt_sine_wave_online.py.
- Stream the trajectory one point at a time, at ctrl_hz, as PVTCommand
  *service* calls to the robot control node. Velocity is still computed
  online from the position difference between the current and previous
  tick (v = (z_curr - z_prev) / dt) -- this node "only knows current &
  previous point" the same way the original single-node version did.

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
from typing import List, Optional

import rclpy
from rclpy.node import Node

from tm_msgs.msg import FeedbackState
from custom_interface.srv import PVTCommand


# ---------------------------------------------------------------------------
# Trajectory providers (same interface/logic as pvt_sine_wave_online.py)
# ---------------------------------------------------------------------------
class ZTrajectoryProvider:
    """Interface for reusable Z-axis stream trajectories."""
    name = "z_trajectory"

    def build(self, start_pose_6d: List[float], ctrl_dt: float, total_points: int) -> List[float]:
        raise NotImplementedError


@dataclass
class SineZTrajectory(ZTrajectoryProvider):
    """Default sine trajectory used by the original online script."""
    amp_m: float = 0.05
    period_s: float = 2.0
    phase_rad: float = 0.0
    name: str = "sine_z"

    def build(self, start_pose_6d: List[float], ctrl_dt: float, total_points: int) -> List[float]:
        omega = 2.0 * math.pi / self.period_s
        z0 = start_pose_6d[2]
        return [
            z0 + self.amp_m * math.sin(omega * k * ctrl_dt + self.phase_rad)
            for k in range(total_points)
        ]


@dataclass
class ListZTrajectory(ZTrajectoryProvider):
    """Trajectory provider for a user-supplied list of relative or absolute Z points."""
    z_points_m: List[float]
    relative_to_start: bool = True
    name: str = "list_z"

    def build(self, start_pose_6d: List[float], ctrl_dt: float, total_points: int) -> List[float]:
        if not self.z_points_m:
            raise ValueError("z_points_m must contain at least one point")
        z0 = start_pose_6d[2] if self.relative_to_start else 0.0
        out = [z0 + z for z in self.z_points_m[:total_points]]
        if len(out) < total_points:
            out.extend([out[-1]] * (total_points - len(out)))
        return out


class TrajectoryGeneratorNode(Node):
    def __init__(self):
        super().__init__("pvt_trajectory_generator")

        # ----- parameters -----
        self.declare_parameter("ctrl_hz", 100.0)
        self.declare_parameter("duration_s", 6.0)
        self.declare_parameter("pvt_point_time_ratio", 0.9)
        self.declare_parameter("amp_m", 0.05)
        self.declare_parameter("period_s", 2.0)
        self.declare_parameter("phase_rad", 0.0)
        self.declare_parameter("command_service", "pvt_command")
        self.declare_parameter("service_wait_log_period_s", 2.0)

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

        # Swap this out (or make it a plugin param) for ListZTrajectory etc.
        self.trajectory: ZTrajectoryProvider = SineZTrajectory(
            amp_m=float(self.get_parameter("amp_m").value),
            period_s=float(self.get_parameter("period_s").value),
            phase_rad=float(self.get_parameter("phase_rad").value),
        )

        command_service = self.get_parameter("command_service").value
        self.service_wait_log_period_s = float(self.get_parameter("service_wait_log_period_s").value)
        self.cmd_client = self.create_client(PVTCommand, command_service)
        self.create_subscription(FeedbackState, "feedback_states", self._fb_cb, 10)

        self.has_feedback = False
        self.start_pose_6d: Optional[List[float]] = None
        self.trajectory_z: Optional[List[float]] = None
        self.z_prev: Optional[float] = None
        self.tick = 0
        self.done = False

        self._inflight = 0
        self._error_count = 0
        self._last_wait_log_wall = 0.0

        self.startup_timer = self.create_timer(0.05, self._startup_tick)
        self.ctrl_timer = None

        self.get_logger().info(
            f"Trajectory generator init: {self.trajectory.name}, "
            f"{self.ctrl_hz:.1f}Hz, {self.total_points} points, "
            f"calling PVTCommand service '{command_service}'"
        )

    # ---------- ROS callbacks ----------
    def _fb_cb(self, msg: FeedbackState):
        if self.has_feedback:
            return  # only need the very first pose to anchor the trajectory
        if msg.tool_pose and len(msg.tool_pose) >= 6:
            self.start_pose_6d = [
                float(msg.tool_pose[0]),
                float(msg.tool_pose[1]),
                float(msg.tool_pose[2]),
                math.degrees(float(msg.tool_pose[3])),
                math.degrees(float(msg.tool_pose[4])),
                math.degrees(float(msg.tool_pose[5])),
            ]
            self.has_feedback = True
            self.get_logger().info(f"✓ start pose captured: Z={self.start_pose_6d[2]:.4f}")

    def _startup_tick(self):
        if not self.has_feedback or self.start_pose_6d is None:
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

        self.trajectory_z = self.trajectory.build(
            self.start_pose_6d, self.ctrl_dt, self.total_points
        )
        self.get_logger().info(f"✓ pre-generated {len(self.trajectory_z)} trajectory points")
        self.get_logger().info("✓ PVTCommand service is up")

        self.ctrl_timer = self.create_timer(self.ctrl_dt, self._ctrl_tick)
        self.get_logger().info("=== TRAJECTORY STREAM START ===")

    def _ctrl_tick(self):
        if self.tick >= self.total_points:
            if self.ctrl_timer:
                self.ctrl_timer.cancel()
            if not self.done:
                self.done = True
                self.get_logger().info("=== TRAJECTORY STREAM END ===")
            return

        z_current = self.trajectory_z[self.tick]

        # Velocity from position difference (not analytical formula) --
        # this node only ever looks at "current & previous point".
        if self.tick == 0:
            vz = 0.0
        else:
            vz = (z_current - self.z_prev) / self.ctrl_dt
        self.z_prev = z_current

        req = PVTCommand.Request()
        req.header.stamp = self.get_clock().now().to_msg()
        req.tick = self.tick
        req.total_points = self.total_points
        req.is_last = (self.tick == self.total_points - 1)

        req.x_m = self.start_pose_6d[0]
        req.y_m = self.start_pose_6d[1]
        req.z_m = z_current
        req.rx_deg = self.start_pose_6d[3]
        req.ry_deg = self.start_pose_6d[4]
        req.rz_deg = self.start_pose_6d[5]

        req.vx_mps = 0.0
        req.vy_mps = 0.0
        req.vz_mps = vz
        req.wx_dps = 0.0
        req.wy_dps = 0.0
        req.wz_dps = 0.0

        req.point_time_s = self.ctrl_dt * self.pvt_point_time_ratio

        this_tick = self.tick
        future = self.cmd_client.call_async(req)
        self._inflight += 1

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
            self.get_logger().info(
                f"[GEN {self.tick:03d}] z={z_current:.4f} vz={vz:+.3f} inflight={self._inflight}"
            )

        self.tick += 1


def main():
    rclpy.init()
    node = TrajectoryGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
