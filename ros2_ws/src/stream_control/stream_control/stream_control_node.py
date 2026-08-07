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
- Independently, at obs_hz, read feedback_states to log measured Z / Vz,
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
    ref_z: float
    ref_vz: float
    jitter_s: float
    inflight: int
    ack_p50_ms: float
    ack_p95_ms: float
    ack_max_ms: float
    backlog_flag: int


@dataclass
class ObsSample:
    t_wall: float
    meas_z: float
    meas_vz: float
    ref_z_latest: float
    inst_err_m: float
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

        # ack tracking
        self._inflight = 0
        self._ack_lat_ms_hist: List[float] = []
        self._ack_last_ms: float = 0.0
        self._overload_streak = 0

        # reference-command history, used for obs-loop comparison / lag estimate
        self._cmd_history: deque = deque()  # (t_wall, ref_z)
        self._ref_z_latest: Optional[float] = None

        # obs tracking
        self._meas_z_prev: Optional[float] = None
        self._meas_vz_lpf: float = 0.0
        self._vz_alpha = 0.25
        self._obs_last_log_wall = 0.0

        self.ctrl_log: List[CtrlSample] = []
        self.obs_log: List[ObsSample] = []

        self.obs_timer = self.create_timer(1.0 / self.obs_hz, self._obs_tick)

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

        cmd_script = self._pvt_point_cmd(
            request.x_m, request.y_m, request.z_m,
            request.rx_deg, request.ry_deg, request.rz_deg,
            request.vx_mps, request.vy_mps, request.vz_mps,
            request.wx_dps, request.wy_dps, request.wz_dps,
            request.point_time_s,
        )
        self._send_async(f"P{request.tick % 90 + 10:03d}", cmd_script)

        self._ref_z_latest = request.z_m
        self._cmd_history.append((recv_wall, request.z_m))
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
                ref_z=request.z_m,
                ref_vz=request.vz_mps,
                jitter_s=jitter_s,
                inflight=self._inflight,
                ack_p50_ms=p50,
                ack_p95_ms=p95,
                ack_max_ms=mx,
                backlog_flag=backlog_flag,
            )
        )

        if request.tick % 10 == 0:
            self.get_logger().info(
                f"[CTRL {request.tick:03d}] jitter={jitter_s*1000:+.1f}ms "
                f"inflight={self._inflight} ack_last={self._ack_last_ms:.1f}ms "
                f"ref_z={request.z_m:.4f} ref_vz={request.vz_mps:+.3f}"
            )

        if request.is_last:
            self.streaming = False
            self.stream_end_wall = time.time()
            self._send_async("E005", "PVTExit()")
            self.get_logger().info("=== CTRL STREAM END (PVTExit sent) ===")

        response.result = PVTCommand.Response.ROBOT_OK
        response.message = ""
        return response

    # ---------- helpers ----------
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

    def _estimate_lag(self, meas_z: float) -> float:
        """Find the recorded command whose ref_z is closest to meas_z within
        the recent history window, and return how long ago it was received.
        This is the discrete-stream equivalent of the original analytic
        lag search -- this node has no trajectory formula to search over,
        only the finite window of commands it has actually received."""
        if not self._cmd_history:
            return 0.0
        now = time.time()
        best_t, best_err = None, float("inf")
        for t_wall, ref_z in self._cmd_history:
            err = abs(meas_z - ref_z)
            if err < best_err:
                best_err = err
                best_t = t_wall
        if best_t is None:
            return 0.0
        return now - best_t

    # ---------- observation loop ----------
    def _obs_tick(self):
        if not self.has_feedback or self.meas_pose_6d is None:
            return

        now = time.time()
        meas_z = self.meas_pose_6d[2]

        if self._meas_z_prev is None:
            vz = 0.0
        else:
            dt = 1.0 / self.obs_hz
            vz = (meas_z - self._meas_z_prev) / max(1e-6, dt)
        self._meas_z_prev = meas_z
        self._meas_vz_lpf = (1.0 - self._vz_alpha) * self._meas_vz_lpf + self._vz_alpha * vz

        if self._ref_z_latest is None:
            ref_z_latest = meas_z
            lag_est = 0.0
        else:
            ref_z_latest = self._ref_z_latest
            lag_est = self._estimate_lag(meas_z)

        inst_err = meas_z - ref_z_latest

        self.obs_log.append(
            ObsSample(
                t_wall=now,
                meas_z=meas_z,
                meas_vz=self._meas_vz_lpf,
                ref_z_latest=ref_z_latest,
                inst_err_m=inst_err,
                lag_est_s=lag_est,
                inflight=self._inflight,
                ack_last_ms=self._ack_last_ms,
            )
        )

        if (now - self._obs_last_log_wall) >= self.obs_log_period_s:
            self._obs_last_log_wall = now
            self.get_logger().info(
                f"[OBS] z={meas_z:.4f} vz={self._meas_vz_lpf:+.3f} "
                f"ref_z={ref_z_latest:.4f} err={inst_err*1000:+.1f}mm "
                f"lag~{lag_est:+.2f}s inflight={self._inflight}"
            )

        # Auto-stop after stream end
        if self.stream_end_wall is not None and not self.done:
            t_after = now - self.stream_end_wall
            if t_after > self.settle_timeout_s:
                self.get_logger().warn("Settle timeout -> stop")
                self._finalize("settle_timeout")
                return

            if abs(self._meas_vz_lpf) < self.settle_vz_thr:
                N = int(max(1, round(self.settle_hold_s * self.obs_hz)))
                if len(self.obs_log) >= N:
                    ok = all(abs(s.meas_vz) < self.settle_vz_thr for s in self.obs_log[-N:])
                    if ok:
                        self.get_logger().info("Motion settled -> stop")
                        self._finalize("settled")
                        return

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

        out_dir = self.output_dir
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"{self.output_prefix}_{ts}.csv")
        png_path = os.path.join(out_dir, f"{self.output_prefix}_{ts}.png")

        self._save_csv(csv_path)
        self._save_plot(png_path, reason)

        self.get_logger().info(f"Saved: {csv_path}")
        self.get_logger().info(f"Saved: {png_path}")
        self.get_logger().info(f"Reason: {reason}")

    def _save_csv(self, path: str):
        t0 = self.stream_start_wall if self.stream_start_wall is not None else time.time()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "t_s",
                "meas_z_m", "meas_vz_mps",
                "ref_z_m", "inst_err_m",
                "lag_est_s",
                "inflight",
                "ack_last_ms",
            ])
            for s in self.obs_log:
                w.writerow([
                    f"{(s.t_wall - t0):.6f}",
                    f"{s.meas_z:.6f}",
                    f"{s.meas_vz:.6f}",
                    f"{s.ref_z_latest:.6f}",
                    f"{s.inst_err_m:.6f}",
                    f"{s.lag_est_s:.3f}",
                    f"{s.inflight}",
                    f"{s.ack_last_ms:.3f}",
                ])

    def _save_plot(self, path: str, reason: str):
        if not self.obs_log:
            return
        t0 = self.stream_start_wall if self.stream_start_wall is not None else self.obs_log[0].t_wall
        t = [(s.t_wall - t0) for s in self.obs_log]
        meas_z = [s.meas_z for s in self.obs_log]
        ref_z = [s.ref_z_latest for s in self.obs_log]
        err_mm = [s.inst_err_m * 1000.0 for s in self.obs_log]
        lag = [s.lag_est_s for s in self.obs_log]
        ack = [s.ack_last_ms for s in self.obs_log]
        inflight = [s.inflight for s in self.obs_log]

        plt.figure(figsize=(12, 8))

        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(t, ref_z, label="ref_z (m)")
        ax1.plot(t, meas_z, label="meas_z (m)")
        ax1.set_ylabel("Z (m)")
        ax1.set_title(f"PVT Split Nodes: Generator -> Control | reason={reason}")
        ax1.legend(loc="upper right")
        ax1.grid(True)

        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(t, err_mm, label="inst_err (mm)")
        ax2.plot(t, lag, label="lag_est (s)")
        ax2.set_ylabel("Error / Lag")
        ax2.legend(loc="upper right")
        ax2.grid(True)

        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(t, ack, label="ack_last (ms)")
        ax3.plot(t, inflight, label="inflight (count)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Ack / inflight")
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
