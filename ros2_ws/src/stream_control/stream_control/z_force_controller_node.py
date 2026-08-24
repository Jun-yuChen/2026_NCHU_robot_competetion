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
from typing import List, Optional

import rclpy
from rclpy.node import Node

from stream_control.PI_controller import PI_controller

from geometry_msgs.msg import WrenchStamped
from tm_msgs.msg import FeedbackState
from custom_interface.srv import PVTCommand

class ZForceControllerNode(Node):
    def __init__(self):
        super().__init__("z_force_controller")

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
            # self.get_logger().info(f"Pose captured: Z={self.current_pose_6d[2]:.4f}")

    def _FTsensor_cb(self, msg: WrenchStamped):
        self.force_observe = msg.wrench.force
        self.torque_observe = msg.wrench.torque

        self.FT_is_avaliable = True
        # self.get_logger().info(f"Force-Torque capture")

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
        error = self.fz_desire - self.force_observe.z

        delta_z = self.force_controller.update(error, dt)

        #self.get_logger().info(f"delta_z: {delta_z}")

        # TODO: Check FT sensor corrdinate and robot coordinate
        z_command = z_current - delta_z
        vz = delta_z * self.ctrl_hz

        self.get_logger().info(f"delta_z: {delta_z}    z_command: {z_command}")

        req = PVTCommand.Request()
        req.header.stamp = self.get_clock().now().to_msg()
        req.tick = self.tick
        req.is_last = (self.tick == self.total_points - 1)

        req.x_m = self.current_pose_6d[0]
        req.y_m = self.current_pose_6d[1]
        req.z_m = z_command
        req.rx_deg = self.current_pose_6d[3]
        req.ry_deg = self.current_pose_6d[4]
        req.rz_deg = self.current_pose_6d[5]

        req.vx_mps = 0.0
        req.vy_mps = 0.0
        req.vz_mps = vz
        req.wx_dps = 0.0
        req.wy_dps = 0.0
        req.wz_dps = 0.0

        req.point_time_s = self.ctrl_dt * self.pvt_point_time_ratio

        # Safty check (Stop robot if force > 5N)
        if self.force_observe.z > 10:          
            req.z_m = self.current_pose_6d[2]
            req.vz_mps = 0.0
            self.get_logger().error(f"Z force > 10 N. Control end.")
            self.ctrl_timer.cancel()
            return

        this_tick = self.tick
        future = self.cmd_client.call_async(req)
        self._inflight += 1

        if req.is_last:
            self.get_logger().error(f"Duration end.")
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
            self.get_logger().info(
                f"[GEN {self.tick:03d}] z={z_current:.4f} vz={vz:+.3f} inflight={self._inflight}"
            )

        self.tick += 1


def main():
    rclpy.init()
    node = ZForceControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
