#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot control node for TM PVT streaming control.

Responsibilities
-----------------
- Serve the PVTCommand service that trajectory_generator_node calls.
  Because this is a service rather than a topic, the generator's client
  will not report "ready" (and therefore will not start streaming) until
  this node's server has actually been discovered -- so no points get
  silently dropped during startup the way they could with a plain
  publish/subscribe topic.
- On the first request received: send PVTEnter(1), then forward every
  request to the robot as a PVTPoint(...) SendScript call (async, with
  ack-latency / inflight tracking, same as pvt_sine_wave_online.py), and
  reply ROBOT_OK once it has been handed off, or ROBOT_ERROR if it could
  not be (e.g. send_script isn't available).
- On the last request (is_last=True): forward the final point, then send
  PVTExit().
- Independently, at obs_hz, read feedback_states to log measured 6-DOF pose / velocity,
  instantaneous tracking error vs. the most recently received reference,
  and an estimated lag -- then auto-stop once motion settles (or times
  out), saving CSV + PNG, exactly like the original combined script.

This node has NO knowledge of trajectory generation (sine / list / etc.)
-- it only knows how to relay whatever PVTCommand stream it receives to
the arm and log the result. It never reproduces the generator's math; the
"ref_z" it plots/logs is just whatever was most recently received.

Note on ROBOT_OK semantics: it means the point was accepted and handed
off to send_script -- NOT that the arm has physically reached it. Actual
tracking is what the obs loop / CSV / PNG are for.
"""

import os
import time
import csv
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tm_msgs.srv import SendScript
from tm_msgs.msg import FeedbackState
from custom_interface.srv import PVTCommand


@dataclass
class CtrlSample:
    t_wall: float
    tick: int
    ref_pose_6d: Tuple[float, float, float, float, float, float]
    ref_vel_6d: Tuple[float, float, float, float, float, float]
    jitter_s: float
    inflight: int
    ack_p50_ms: float
    ack_p95_ms: float
    ack_max_ms: float
    backlog_flag: int


@dataclass
class ObsSample:
    t_wall: float
    meas_pose_6d: Tuple[float, float, float, float, float, float]
    ref_pose_6d: Tuple[float, float, float, float, float, float]
    err_pose_6d: Tuple[float, float, float, float, float, float]
    meas_vel_6d: Tuple[float, float, float, float, float, float]
    lag_est_s: float
    inflight: int
    ack_last_ms: float


class StreamControlNode(Node):
    def __init__(self):
        super().__init__("pvt_robot_control")

        # ----- parameters -----
        self.declare_parameter("command_service", "pvt_command")
        self.declare_parameter("obs_hz", 1000.0)
        self.declare_parameter("obs_log_period_s", 0.2)
        self.declare_parameter("settle_vz_thr", 0.002)
        self.declare_parameter("settle_hold_s", 0.6)
        self.declare_parameter("settle_timeout_s", 12.0)
        self.declare_parameter("max_inflight", 5)
        self.declare_parameter("ack_over_period_ratio", 1.0)
        self.declare_parameter("overload_consecutive", 3)
        self.declare_parameter("output_dir", "pvt_out")
        self.declare_parameter("output_prefix", "pvt_split")
        self.declare_parameter("cmd_history_window_s", 2.0)

        # Safty limits
        self.declare_parameter("translation_speed_limit_mps", 0.01)
        self.declare_parameter("rotation_speed_limit_dps", 2.0)
        self.declare_parameter("translation_step_limit_m", 0.01)  # distance between two adjacent PVT points.
        self.declare_parameter("rotation_step_limit_deg", 2.0)
        self.declare_parameter("stall_timeout_s", 1.0)

        self.obs_hz = float(self.get_parameter("obs_hz").value)
        self.obs_log_period_s = float(self.get_parameter("obs_log_period_s").value)
        self.settle_vz_thr = float(self.get_parameter("settle_vz_thr").value)
        self.settle_hold_s = float(self.get_parameter("settle_hold_s").value)
        self.settle_timeout_s = float(self.get_parameter("settle_timeout_s").value)
        self.max_inflight = int(self.get_parameter("max_inflight").value)
        self.ack_over_period_ratio = float(self.get_parameter("ack_over_period_ratio").value)
        self.overload_consecutive = int(self.get_parameter("overload_consecutive").value)
        self.output_dir = self.get_parameter("output_dir").value
        self.output_prefix = self.get_parameter("output_prefix").value
        self.cmd_history_window_s = float(self.get_parameter("cmd_history_window_s").value)
        self.translation_speed_limit_mps = float(self.get_parameter("translation_speed_limit_mps").value)
        self.rotation_speed_limit_dps = float(self.get_parameter("rotation_speed_limit_dps").value)
        self.translation_step_limit_m = float(self.get_parameter("translation_step_limit_m").value)
        self.rotation_step_limit_deg = float(self.get_parameter("rotation_step_limit_deg").value)
        self.stall_timeout_s = float(self.get_parameter("stall_timeout_s").value)

        command_service = self.get_parameter("command_service").value

        self.send_script = self.create_client(SendScript, "send_script")
        self.create_subscription(FeedbackState, "feedback_states", self._fb_cb, 10)
        
        self.create_service(PVTCommand, command_service, self._cmd_srv_cb)

        # feedback
        self.has_feedback = False
        self.meas_pose_6d: Optional[List[float]] = None

        # stream state
        self.entered_pvt = False
        self.streaming = False
        self.stream_start_wall: Optional[float] = None
        self.stream_end_wall: Optional[float] = None
        self.done = False
        self._last_cmd_recv_wall: Optional[float] = None

        # ack tracking
        self._inflight = 0
        self._ack_lat_ms_hist: List[float] = []
        self._ack_last_ms: float = 0.0
        self._overload_streak = 0

        # reference-command history, used for obs-loop comparison / lag estimate
        self._cmd_history: deque = deque()  # (t_wall, ref_pose_6d)
        self._ref_pose_latest: Optional[Tuple[float, float, float, float, float, float]] = None

        # safety: previous (possibly-truncated) target pose, for step-limit checks
        self._prev_target: Optional[Tuple[float, float, float, float, float, float]] = None

        # obs tracking
        self._meas_z_prev: Optional[float] = None
        self._meas_vz_lpf: float = 0.0
        self._vz_alpha = 0.25
        self._obs_last_log_wall = 0.0

        self.ctrl_log: List[CtrlSample] = []
        self.obs_log: List[ObsSample] = []

        self.obs_timer = self.create_timer(1.0 / self.obs_hz, self._obs_tick)
        self.watchdog_timer = self.create_timer(0.5, self._watchdog_tick)

        self.get_logger().info("Robot control node init, waiting for send_script...")
        if not self.send_script.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("send_script not available")
        else:
            self.get_logger().info("send_script ready")
        self.get_logger().info(f"Serving PVTCommand on service '{command_service}'")

    # ---------- ROS callbacks ----------
    def _fb_cb(self, msg: FeedbackState):
        if not self.has_feedback:
            self.has_feedback = True
            self.get_logger().info("✓ feedback received")
        if msg.tool_pose and len(msg.tool_pose) >= 6:
            self.meas_pose_6d = [
                float(msg.tool_pose[0]),
                float(msg.tool_pose[1]),
                float(msg.tool_pose[2]),
                math.degrees(float(msg.tool_pose[3])),
                math.degrees(float(msg.tool_pose[4])),
                math.degrees(float(msg.tool_pose[5])),
            ]

    def _cmd_srv_cb(self, request: PVTCommand.Request, response: PVTCommand.Response):
        if self.done:
            response.result = PVTCommand.Response.ROBOT_ERROR
            response.message = "control node already finalized"
            return response

        if not self.send_script.service_is_ready():
            response.result = PVTCommand.Response.ROBOT_ERROR
            response.message = "send_script service unavailable"
            self.get_logger().error(
                f"[CTRL {request.tick:03d}] rejecting point: send_script unavailable"
            )
            return response

        recv_wall = time.time()
        self._last_cmd_recv_wall = recv_wall

        if not self.entered_pvt:
            self.entered_pvt = True
            self._send_async("E001", "PVTEnter(1)")
            self.stream_start_wall = recv_wall
            self.streaming = True
            self.get_logger().info("=== CTRL STREAM START (PVTEnter sent) ===")

        # Jitter here = generator's call time -> this node's receipt time,
        # i.e. it captures pipeline (generator timer + DDS/service round
        # trip) latency.
        stamp_s = request.header.stamp.sec + request.header.stamp.nanosec * 1e-9
        jitter_s = (recv_wall - stamp_s) if stamp_s > 0 else 0.0

        # ---------- safety checks: truncate, never pass raw values through ----------
        # 1) per-axis speed limit -- each of vx/vy/vz/wx/wy/wz is clamped to
        #    its own +-limit, NOT the vector norm.
        vx, vx_over = self._clamp_magnitude(request.vx_mps, self.translation_speed_limit_mps)
        vy, vy_over = self._clamp_magnitude(request.vy_mps, self.translation_speed_limit_mps)
        vz, vz_over = self._clamp_magnitude(request.vz_mps, self.translation_speed_limit_mps)
        wx, wx_over = self._clamp_magnitude(request.wx_dps, self.rotation_speed_limit_dps)
        wy, wy_over = self._clamp_magnitude(request.wy_dps, self.rotation_speed_limit_dps)
        wz, wz_over = self._clamp_magnitude(request.wz_dps, self.rotation_speed_limit_dps)

        # 2) per-axis step limit -- how far this target may move from the
        #    PREVIOUS command's (already-truncated) target on each axis.
        #    No previous target yet -> nothing to compare against, skip.
        #    Translation axes use plain delta; rotation axes use the
        #    shortest signed angular delta so a target that legitimately
        #    crosses +-180 deg isn't misread as a huge jump.
        pt = self._prev_target
        x, x_over = self._clamp_step(request.x_m, pt[0] if pt else None, self.translation_step_limit_m)
        y, y_over = self._clamp_step(request.y_m, pt[1] if pt else None, self.translation_step_limit_m)
        z, z_over = self._clamp_step(request.z_m, pt[2] if pt else None, self.translation_step_limit_m)
        rx, rx_over = self._clamp_step_angular(request.rx_deg, pt[3] if pt else None, self.rotation_step_limit_deg)
        ry, ry_over = self._clamp_step_angular(request.ry_deg, pt[4] if pt else None, self.rotation_step_limit_deg)
        rz, rz_over = self._clamp_step_angular(request.rz_deg, pt[5] if pt else None, self.rotation_step_limit_deg)
        self._prev_target = (x, y, z, rx, ry, rz)

        violations: List[str] = []
        if vx_over: violations.append(f"vx {request.vx_mps:+.4f}->{vx:+.4f} m/s")
        if vy_over: violations.append(f"vy {request.vy_mps:+.4f}->{vy:+.4f} m/s")
        if vz_over: violations.append(f"vz {request.vz_mps:+.4f}->{vz:+.4f} m/s")
        if wx_over: violations.append(f"wx {request.wx_dps:+.3f}->{wx:+.3f} dps")
        if wy_over: violations.append(f"wy {request.wy_dps:+.3f}->{wy:+.3f} dps")
        if wz_over: violations.append(f"wz {request.wz_dps:+.3f}->{wz:+.3f} dps")
        if x_over: violations.append(f"x-step {request.x_m:+.4f}->{x:+.4f} m")
        if y_over: violations.append(f"y-step {request.y_m:+.4f}->{y:+.4f} m")
        if z_over: violations.append(f"z-step {request.z_m:+.4f}->{z:+.4f} m")
        if rx_over: violations.append(f"rx-step {request.rx_deg:+.3f}->{rx:+.3f} deg")
        if ry_over: violations.append(f"ry-step {request.ry_deg:+.3f}->{ry:+.3f} deg")
        if rz_over: violations.append(f"rz-step {request.rz_deg:+.3f}->{rz:+.3f} deg")

        limit_exceeded = len(violations) > 0
        if limit_exceeded:
            self.get_logger().warn(
                f"[CTRL {request.tick:03d}] SAFETY LIMIT exceeded, truncating -> "
                + "; ".join(violations)
            )
        # ---------- end safety checks ----------

        cmd_script = self._pvt_point_cmd(
            x, y, z,
            rx, ry, rz,
            vx, vy, vz,
            wx, wy, wz,
            request.point_time_s,
        )
        self._send_async(f"P{request.tick % 90 + 10:03d}", cmd_script)

        ref_pose_6d = (x, y, z, rx, ry, rz)
        ref_vel_6d = (vx, vy, vz, wx, wy, wz)
        self._ref_pose_latest = ref_pose_6d
        self._cmd_history.append((recv_wall, ref_pose_6d))
        cutoff = recv_wall - self.cmd_history_window_s
        while self._cmd_history and self._cmd_history[0][0] < cutoff:
            self._cmd_history.popleft()

        p50, p95, mx = self._ack_stats()
        overload = 0
        if self._inflight > self.max_inflight:
            overload = 1
        expected_period_s = request.point_time_s if request.point_time_s > 0 else 0.01
        if self._ack_last_ms > (expected_period_s * 1000.0 * self.ack_over_period_ratio):
            overload = 1

        if overload:
            self._overload_streak += 1
        else:
            self._overload_streak = 0
        backlog_flag = 1 if self._overload_streak >= self.overload_consecutive else 0

        self.ctrl_log.append(
            CtrlSample(
                t_wall=recv_wall,
                tick=request.tick,
                ref_pose_6d=ref_pose_6d,
                ref_vel_6d=ref_vel_6d,
                jitter_s=jitter_s,
                inflight=self._inflight,
                ack_p50_ms=p50,
                ack_p95_ms=p95,
                ack_max_ms=mx,
                backlog_flag=backlog_flag,
            )
        )

        '''
        if request.tick % 10 == 0:
            self.get_logger().info(
                f"[CTRL {request.tick:03d}] jitter={jitter_s*1000:+.1f}ms "
                f"inflight={self._inflight} ack_last={self._ack_last_ms:.1f}ms "
                f"ref_xyz=({x:.4f},{y:.4f},{z:.4f}) "
                f"ref_rpy=({rx:.2f},{ry:.2f},{rz:.2f})"
            )
        '''

        if request.is_last:
            self.streaming = False
            self.stream_end_wall = time.time()
            self._send_async("E005", "PVTExit()")
            self.get_logger().info("=== CTRL STREAM END (PVTExit sent) ===")

        if limit_exceeded:
            response.result = PVTCommand.Response.ROBOT_ERROR
            response.message = "safety limit exceeded: " + "; ".join(violations)
        else:
            response.result = PVTCommand.Response.ROBOT_OK
            response.message = ""
        return response

    # ---------- helpers ----------
    @staticmethod
    def _clamp_magnitude(value: float, limit: float) -> Tuple[float, bool]:
        """Truncate a signed value to +-limit (per-axis, not vector norm).
        limit <= 0 disables the check. Returns (value, was_clamped)."""
        if limit <= 0:
            return value, False
        if abs(value) > limit:
            return math.copysign(limit, value), True
        return value, False

    @staticmethod
    def _angle_diff_deg(curr: float, prev: float) -> float:
        """Shortest signed delta curr-prev, wrapped to (-180, 180]."""
        return (curr - prev + 180.0) % 360.0 - 180.0

    @staticmethod
    def _clamp_step(curr: float, prev: Optional[float], limit: float) -> Tuple[float, bool]:
        """Truncate curr so |curr - prev| <= limit on this axis (plain
        subtraction -- use for translation axes, which don't wrap).
        limit <= 0 or prev is None (no previous command yet) disables the
        check. Returns (value, was_clamped)."""
        if limit <= 0 or prev is None:
            return curr, False
        delta = curr - prev
        if abs(delta) > limit:
            return prev + math.copysign(limit, delta), True
        return curr, False

    @classmethod
    def _clamp_step_angular(cls, curr: float, prev: Optional[float], limit: float) -> Tuple[float, bool]:
        """Truncate curr so the SHORTEST-PATH delta from prev is within
        +-limit degrees. Wrap-safe: a target crossing +-180 deg is measured
        by its true angular distance, not a raw subtraction, so it isn't
        misread as a huge jump. Use for rx/ry/rz. limit <= 0 or prev is
        None disables the check. Returns (value, was_clamped)."""
        if limit <= 0 or prev is None:
            return curr, False
        delta = cls._angle_diff_deg(curr, prev)
        if abs(delta) > limit:
            clamped_delta = math.copysign(limit, delta)
            new_val = prev + clamped_delta
            new_val = ((new_val + 180.0) % 360.0) - 180.0  # normalize back to (-180,180]
            return new_val, True
        return curr, False

    def _send_async(self, sid: str, script: str):
        """SendScript async with ack latency tracking."""
        req = SendScript.Request()
        req.id = sid
        req.script = script

        send_wall = time.time()
        future = self.send_script.call_async(req)
        self._inflight += 1

        def _done_cb(fut):
            done_wall = time.time()
            lat_ms = (done_wall - send_wall) * 1000.0
            self._ack_last_ms = lat_ms
            self._ack_lat_ms_hist.append(lat_ms)
            self._inflight -= 1
            try:
                _ = fut.result()
            except Exception:
                pass

        future.add_done_callback(_done_cb)

    @staticmethod
    def _pvt_point_cmd(x_m, y_m, z_m, rx_deg, ry_deg, rz_deg, vx, vy, vz, wx, wy, wz, t_s) -> str:
        return (
            f"PVTPoint({x_m*1000:.2f},{y_m*1000:.2f},{z_m*1000:.2f},"
            f"{rx_deg:.2f},{ry_deg:.2f},{rz_deg:.2f},"
            f"{vx*1000:.2f},{vy*1000:.2f},{vz*1000:.2f},"
            f"{wx:.2f},{wy:.2f},{wz:.2f},"
            f"{t_s:.3f})"
        )

    def _ack_stats(self) -> Tuple[float, float, float]:
        """p50/p95/max of ack latency"""
        if not self._ack_lat_ms_hist:
            return 0.0, 0.0, 0.0
        N = min(200, len(self._ack_lat_ms_hist))
        arr = sorted(self._ack_lat_ms_hist[-N:])
        p50 = arr[int(0.50 * (len(arr) - 1))]
        p95 = arr[int(0.95 * (len(arr) - 1))]
        mx = arr[-1]
        return p50, p95, mx

    def _estimate_lag(self, meas_pose_6d: List[float]) -> float:
        """Estimate lag from the recent 6-DOF command history."""
        if not self._cmd_history:
            return 0.0

        now = time.time()
        best_t, best_err = None, float("inf")
        for t_wall, ref_pose in self._cmd_history:
            pos_err = math.sqrt(sum(
                (meas_pose_6d[i] - ref_pose[i]) ** 2 for i in range(3)
            ))
            rot_err = math.sqrt(sum(
                self._angle_diff_deg(meas_pose_6d[i], ref_pose[i]) ** 2
                for i in range(3, 6)
            ))
            err = pos_err + 0.001 * rot_err
            if err < best_err:
                best_err = err
                best_t = t_wall

        return 0.0 if best_t is None else now - best_t

    # ---------- observation loop ----------
    def _obs_tick(self):
        if not self.has_feedback or self.meas_pose_6d is None:
            return

        now = time.time()
        meas = list(self.meas_pose_6d)

        if not hasattr(self, "_meas_pose_prev"):
            self._meas_pose_prev: Optional[List[float]] = None
            self._meas_vel_6d_lpf = [0.0] * 6

        if self._meas_pose_prev is None:
            meas_vel = [0.0] * 6
        else:
            dt = max(1e-6, 1.0 / self.obs_hz)
            meas_vel = [
                (meas[i] - self._meas_pose_prev[i]) / dt for i in range(6)
            ]
            for i in range(3, 6):
                meas_vel[i] = self._angle_diff_deg(
                    meas[i], self._meas_pose_prev[i]
                ) / dt

        self._meas_pose_prev = meas
        self._meas_vel_6d_lpf = [
            (1.0 - self._vz_alpha) * self._meas_vel_6d_lpf[i]
            + self._vz_alpha * meas_vel[i]
            for i in range(6)
        ]

        if self._ref_pose_latest is None:
            ref_pose = tuple(meas)
            lag_est = 0.0
        else:
            ref_pose = self._ref_pose_latest
            lag_est = self._estimate_lag(meas)

        err_pose = (
            meas[0] - ref_pose[0],
            meas[1] - ref_pose[1],
            meas[2] - ref_pose[2],
            self._angle_diff_deg(meas[3], ref_pose[3]),
            self._angle_diff_deg(meas[4], ref_pose[4]),
            self._angle_diff_deg(meas[5], ref_pose[5]),
        )

        self.obs_log.append(
            ObsSample(
                t_wall=now,
                meas_pose_6d=tuple(meas),
                ref_pose_6d=tuple(ref_pose),
                err_pose_6d=err_pose,
                meas_vel_6d=tuple(self._meas_vel_6d_lpf),
                lag_est_s=lag_est,
                inflight=self._inflight,
                ack_last_ms=self._ack_last_ms,
            )
        )

        '''
        if (now - self._obs_last_log_wall) >= self.obs_log_period_s:
            self._obs_last_log_wall = now
            self.get_logger().info(
                f"[OBS] xyz=({meas[0]:.4f},{meas[1]:.4f},{meas[2]:.4f}) "
                f"rpy=({meas[3]:.2f},{meas[4]:.2f},{meas[5]:.2f}) "
                f"err_xyz=({err_pose[0]*1000:+.1f},{err_pose[1]*1000:+.1f},"
                f"{err_pose[2]*1000:+.1f})mm "
                f"err_rpy=({err_pose[3]:+.2f},{err_pose[4]:+.2f},{err_pose[5]:+.2f})deg "
                f"lag~{lag_est:+.2f}s inflight={self._inflight}"
            )
        '''

        # Keep the original settle criterion: Z velocity must remain small.
        if self.stream_end_wall is not None and not self.done:
            t_after = now - self.stream_end_wall
            if t_after > self.settle_timeout_s:
                self.get_logger().warn("Settle timeout -> stop")
                self._finalize("settle_timeout")
                return

            if abs(self._meas_vel_6d_lpf[2]) < self.settle_vz_thr:
                N = int(max(1, round(self.settle_hold_s * self.obs_hz)))
                if len(self.obs_log) >= N:
                    ok = all(
                        abs(s.meas_vel_6d[2]) < self.settle_vz_thr
                        for s in self.obs_log[-N:]
                    )
                    if ok:
                        self.get_logger().info("Motion settled -> stop")
                        self._finalize("settled")
                        return

    def _watchdog_tick(self):
        """If we're mid-stream but haven't heard a PVTCommand in over
        stall_timeout_s, the generator is presumed dead/stalled: send
        PVTExit, finalize (saving CSV/PNG), and reset stream state so a
        fresh generator run starts clean instead of inheriting stale
        entered_pvt / _prev_target."""
        if not self.streaming or self.done or self._last_cmd_recv_wall is None:
            return

        gap_s = time.time() - self._last_cmd_recv_wall
        if gap_s > self.stall_timeout_s:
            self.get_logger().warn(
                f"Watchdog: no PVTCommand received for {gap_s:.2f}s "
                f"(stall_timeout_s={self.stall_timeout_s:.2f}) -> aborting stream"
            )
            self._send_async("E005", "PVTExit()")
            self.streaming = False
            self.stream_end_wall = time.time()
            self._finalize("stream_stalled")
            self.entered_pvt = False
            self._prev_target = None

    # ---------- finalize / logging ----------
    def _finalize(self, reason: str):
        if self.done:
            return
        self.done = True

        if self.obs_timer is not None:
            try:
                self.obs_timer.cancel()
            except Exception:
                pass
        if self.watchdog_timer is not None:
            try:
                self.watchdog_timer.cancel()
            except Exception:
                pass

        out_dir = self.output_dir
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        obs_csv_path = os.path.join(out_dir, f"{self.output_prefix}_{ts}.csv")
        control_csv_path = os.path.join(out_dir, f"{self.output_prefix}_control_{ts}.csv")
        png_path = os.path.join(out_dir, f"{self.output_prefix}_{ts}.png")

        # Observation/feedback CSV, plus a separate CSV containing every
        # PVTCommand control sample that was received by this node.
        self._save_obs_samples_csv(obs_csv_path)
        self._save_control_samples_csv(control_csv_path)
        self._save_plot(png_path, reason)

        self.get_logger().info(f"Saved: {obs_csv_path}")
        self.get_logger().info(f"Saved: {control_csv_path}")
        self.get_logger().info(f"Saved: {png_path}")
        self.get_logger().info(f"Reason: {reason}")

    def _save_obs_samples_csv(self, path: str):
        t0 = self.stream_start_wall if self.stream_start_wall is not None else time.time()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "t_s",
                "meas_x_m", "meas_y_m", "meas_z_m",
                "meas_rx_deg", "meas_ry_deg", "meas_rz_deg",
                "ref_x_m", "ref_y_m", "ref_z_m",
                "ref_rx_deg", "ref_ry_deg", "ref_rz_deg",
                "err_x_m", "err_y_m", "err_z_m",
                "err_rx_deg", "err_ry_deg", "err_rz_deg",
                "meas_vx_mps", "meas_vy_mps", "meas_vz_mps",
                "meas_wx_dps", "meas_wy_dps", "meas_wz_dps",
                "lag_est_s", "inflight", "ack_last_ms",
            ])

            for s in self.obs_log:
                mx, my, mz, mrx, mry, mrz = s.meas_pose_6d
                rx, ry, rz, rrx, rry, rrz = s.ref_pose_6d
                ex, ey, ez, erx, ery, erz = s.err_pose_6d
                vx, vy, vz, wx, wy, wz = s.meas_vel_6d

                w.writerow([
                    f"{s.t_wall - t0:.6f}",
                    f"{mx:.6f}", f"{my:.6f}", f"{mz:.6f}",
                    f"{mrx:.6f}", f"{mry:.6f}", f"{mrz:.6f}",
                    f"{rx:.6f}", f"{ry:.6f}", f"{rz:.6f}",
                    f"{rrx:.6f}", f"{rry:.6f}", f"{rrz:.6f}",
                    f"{ex:.6f}", f"{ey:.6f}", f"{ez:.6f}",
                    f"{erx:.6f}", f"{ery:.6f}", f"{erz:.6f}",
                    f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}",
                    f"{wx:.6f}", f"{wy:.6f}", f"{wz:.6f}",
                    f"{s.lag_est_s:.3f}",
                    f"{s.inflight}",
                    f"{s.ack_last_ms:.3f}",
                ])

    def _save_control_samples_csv(self, path: str):
        """Save the received control/PVTCommand samples to a separate CSV.

        One row is written for every CtrlSample recorded during the stream.
        This contains the commanded 6-DOF pose and velocity, plus command
        timing/jitter and acknowledgement/backlog information.
        """
        t0 = self.stream_start_wall if self.stream_start_wall is not None else time.time()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "t_s",
                "tick",
                "ref_x_m", "ref_y_m", "ref_z_m",
                "ref_rx_deg", "ref_ry_deg", "ref_rz_deg",
                "ref_vx_mps", "ref_vy_mps", "ref_vz_mps",
                "ref_wx_dps", "ref_wy_dps", "ref_wz_dps",
                "jitter_s",
                "inflight",
                "ack_p50_ms", "ack_p95_ms", "ack_max_ms",
                "backlog_flag",
            ])

            for s in self.ctrl_log:
                x, y, z, rx, ry, rz = s.ref_pose_6d
                vx, vy, vz, wx, wy, wz = s.ref_vel_6d

                w.writerow([
                    f"{s.t_wall - t0:.6f}",
                    f"{s.tick}",
                    f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                    f"{rx:.6f}", f"{ry:.6f}", f"{rz:.6f}",
                    f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}",
                    f"{wx:.6f}", f"{wy:.6f}", f"{wz:.6f}",
                    f"{s.jitter_s:.6f}",
                    f"{s.inflight}",
                    f"{s.ack_p50_ms:.3f}",
                    f"{s.ack_p95_ms:.3f}",
                    f"{s.ack_max_ms:.3f}",
                    f"{s.backlog_flag}",
                ])

    def _save_plot(self, path: str, reason: str):
        if not self.obs_log:
            return

        t0 = self.stream_start_wall if self.stream_start_wall is not None else self.obs_log[0].t_wall
        t = [s.t_wall - t0 for s in self.obs_log]

        meas_x = [s.meas_pose_6d[0] for s in self.obs_log]
        meas_y = [s.meas_pose_6d[1] for s in self.obs_log]
        meas_z = [s.meas_pose_6d[2] for s in self.obs_log]
        ref_x = [s.ref_pose_6d[0] for s in self.obs_log]
        ref_y = [s.ref_pose_6d[1] for s in self.obs_log]
        ref_z = [s.ref_pose_6d[2] for s in self.obs_log]

        plt.figure(figsize=(12, 9))

        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(t, ref_x, label="ref_x (m)")
        ax1.plot(t, meas_x, label="meas_x (m)")
        ax1.set_ylabel("X (m)")
        ax1.set_title(f"PVT Split Nodes: Generator -> Control | reason={reason}")
        ax1.legend(loc="upper right")
        ax1.grid(True)

        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(t, ref_y, label="ref_y (m)")
        ax2.plot(t, meas_y, label="meas_y (m)")
        ax2.set_ylabel("Y (m)")
        ax2.legend(loc="upper right")
        ax2.grid(True)

        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(t, ref_z, label="ref_z (m)")
        ax3.plot(t, meas_z, label="meas_z (m)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Z (m)")
        ax3.legend(loc="upper right")
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()


def main():
    rclpy.init()
    node = StreamControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt")
        node._finalize("keyboard_interrupt")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
