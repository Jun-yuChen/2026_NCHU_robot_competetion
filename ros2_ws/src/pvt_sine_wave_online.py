#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reusable TM PVT stream-control component with PRE-GENERATED trajectory + ONLINE simulation
- Pre-generates complete trajectory (all positions) at startup
- Simulates realtime control: each tick only knows current & previous point
- Velocity computed from position difference: v = (z_current - z_prev) / dt
- Same control/observation structure as pvt_sine_wave.py

Run directly:
    python3 pvt_sine_wave_online.py

Use as a component:
    from pvt_sine_wave_online import PVTStreamConfig, SineZTrajectory, run_stream_control

    config = PVTStreamConfig(ctrl_hz=100.0, duration_s=6.0)
    trajectory = SineZTrajectory(amp_m=0.05, period_s=2.0)
    run_stream_control(config=config, trajectory=trajectory)

Use pre-generated points:
    from pvt_sine_wave_online import PVTStreamConfig, ListZTrajectory, run_stream_control

    points = [0.001 * i for i in range(100)]
    run_stream_control(PVTStreamConfig(ctrl_hz=100.0), ListZTrajectory(points))

Author: Y.F.Chou
"""

import os
import time
import math
import csv
from dataclasses import dataclass
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from tm_msgs.srv import SendScript
from tm_msgs.msg import FeedbackState

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class PVTStreamConfig:
    """Reusable stream-control settings."""
    ctrl_hz: float = 100.0
    duration_s: float = 6.0
    obs_hz: Optional[float] = None
    obs_log_period_s: float = 0.2
    pvt_point_time_ratio: float = 0.9
    stream_start_delay_s: float = 0.30
    settle_vz_thr: float = 0.002
    settle_hold_s: float = 0.6
    settle_timeout_s: float = 12.0
    max_inflight: int = 5
    ack_over_period_ratio: float = 1.0
    overload_consecutive: int = 3
    output_dir: str = "pvt_out"
    output_prefix: str = "pvt_online"


@dataclass
class StreamPoint:
    """Cartesian PVT target in SI units plus orientation in degrees."""
    x_m: float
    y_m: float
    z_m: float
    rx_deg: float
    ry_deg: float
    rz_deg: float
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    vz_mps: float = 0.0
    wx_dps: float = 0.0
    wy_dps: float = 0.0
    wz_dps: float = 0.0


class ZTrajectoryProvider:
    """Interface for reusable Z-axis stream trajectories."""
    name = "z_trajectory"

    def build(self, start_pose_6d: List[float], ctrl_dt: float, total_points: int) -> List[float]:
        raise NotImplementedError

    def ref_z_at(self, start_pose_6d: List[float], t_s: float) -> float:
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

    def ref_z_at(self, start_pose_6d: List[float], t_s: float) -> float:
        omega = 2.0 * math.pi / self.period_s
        return start_pose_6d[2] + self.amp_m * math.sin(omega * t_s + self.phase_rad)


@dataclass
class ListZTrajectory(ZTrajectoryProvider):
    """Trajectory provider for a user-supplied list of relative or absolute Z points."""
    z_points_m: List[float]
    relative_to_start: bool = True
    name: str = "list_z"

    def build(self, start_pose_6d: List[float], ctrl_dt: float, total_points: int) -> List[float]:
        if not self.z_points_m:
            raise ValueError("z_points_m must contain at least one point")

        self._ctrl_dt = ctrl_dt
        z0 = start_pose_6d[2] if self.relative_to_start else 0.0
        out = [z0 + z for z in self.z_points_m[:total_points]]
        if len(out) < total_points:
            out.extend([out[-1]] * (total_points - len(out)))
        return out

    def ref_z_at(self, start_pose_6d: List[float], t_s: float) -> float:
        if not self.z_points_m:
            return start_pose_6d[2]

        z0 = start_pose_6d[2] if self.relative_to_start else 0.0
        ctrl_dt = getattr(self, "_ctrl_dt", 1.0)
        idx = min(len(self.z_points_m) - 1, max(0, int(round(t_s / ctrl_dt))))
        return z0 + self.z_points_m[idx]


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


class PVTStreamControlComponent(Node):
    """
    Reusable ROS2 component for online TM PVT stream control.

    Import this class when you want to plug in another trajectory provider:

        config = PVTStreamConfig(ctrl_hz=100.0, duration_s=6.0)
        trajectory = SineZTrajectory(amp_m=0.05, period_s=2.0)
        node = PVTStreamControlComponent(config=config, trajectory=trajectory)
    """
    def __init__(
        self,
        config: Optional[PVTStreamConfig] = None,
        trajectory: Optional[ZTrajectoryProvider] = None,
        node_name: str = "pvt_stream_control",
    ):
        super().__init__(node_name)

        self.config = config or PVTStreamConfig()
        self.trajectory = trajectory or SineZTrajectory()
        if self.config.ctrl_hz <= 0.0:
            raise ValueError("ctrl_hz must be > 0")
        if self.config.duration_s <= 0.0:
            raise ValueError("duration_s must be > 0")
        if self.config.pvt_point_time_ratio <= 0.0:
            raise ValueError("pvt_point_time_ratio must be > 0")

        # ===== user-config =====
        self.ctrl_hz = self.config.ctrl_hz
        self.duration_s = self.config.duration_s

        self.obs_hz = self.config.obs_hz or max(50.0, self.ctrl_hz * 10.0)
        self.obs_log_period_s = self.config.obs_log_period_s

        self.settle_vz_thr = self.config.settle_vz_thr
        self.settle_hold_s = self.config.settle_hold_s
        self.settle_timeout_s = self.config.settle_timeout_s

        self.max_inflight = self.config.max_inflight
        self.ack_over_period_ratio = self.config.ack_over_period_ratio
        self.overload_consecutive = self.config.overload_consecutive

        # ===== internal =====
        self.ctrl_dt = 1.0 / self.ctrl_hz
        self.total_points = int(round(self.duration_s * self.ctrl_hz))

        self.send_script = self.create_client(SendScript, "send_script")
        self.create_subscription(FeedbackState, "feedback_states", self._fb_cb, 10)

        self.start_pose_6d: Optional[List[float]] = None
        self.meas_pose_6d: Optional[List[float]] = None
        self.has_feedback = False

        self.ctrl_timer = None
        self.obs_timer = None
        self.startup_timer = None

        self.t0_wall: Optional[float] = None
        self.stream_start_wall: Optional[float] = None
        self.tick = 0
        self.streaming = False
        self.stream_end_wall: Optional[float] = None
        self.done = False

        # ack tracking
        self._inflight = 0
        self._ack_lat_ms_hist: List[float] = []
        self._ack_last_ms: float = 0.0
        self._overload_streak = 0

        # obs tracking
        self._meas_z_prev: Optional[float] = None
        self._meas_vz_lpf: float = 0.0
        self._vz_alpha = 0.25
        self._obs_last_log_wall = 0.0

        # logs
        self.ctrl_log: List[CtrlSample] = []
        self.obs_log: List[ObsSample] = []

        # ===== PRE-GENERATED TRAJECTORY =====
        # Trajectory will be generated at startup (after getting start pose)
        # During control, we only access one point at a time
        self.trajectory_z: Optional[List[float]] = None  # pre-generated z positions
        self.z_prev: Optional[float] = None              # previous z (for velocity calc)

        self.get_logger().info("PVT stream-control component init")
        self.get_logger().info("Mode: pre-generate trajectory, simulate realtime control")
        self.get_logger().info(
            f"CTRL {self.ctrl_hz:.1f}Hz, OBS {self.obs_hz:.1f}Hz, "
            f"points={self.total_points}, trajectory={self.trajectory.name}"
        )

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

    # ---------- helper ----------
    def _send_async(self, sid: str, script: str):
        """SendScript async with ack latency tracking"""
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

    def _pvt_point_cmd(self, x_m, y_m, z_m, rx_deg, ry_deg, rz_deg, vx, vy, vz, wx, wy, wz, t_s) -> str:
        return (
            f"PVTPoint({x_m*1000:.2f},{y_m*1000:.2f},{z_m*1000:.2f},"
            f"{rx_deg:.2f},{ry_deg:.2f},{rz_deg:.2f},"
            f"{vx*1000:.2f},{vy*1000:.2f},{vz*1000:.2f},"
            f"{wx:.2f},{wy:.2f},{wz:.2f},"
            f"{t_s:.3f})"
        )

    def _stream_point_cmd(self, point: StreamPoint, t_s: float) -> str:
        return self._pvt_point_cmd(
            point.x_m, point.y_m, point.z_m,
            point.rx_deg, point.ry_deg, point.rz_deg,
            point.vx_mps, point.vy_mps, point.vz_mps,
            point.wx_dps, point.wy_dps, point.wz_dps,
            t_s,
        )

    def _generate_trajectory(self):
        """
        Pre-generate complete trajectory (all position points).
        Called once at startup after getting start pose.
        """
        self.trajectory_z = self.trajectory.build(self.start_pose_6d, self.ctrl_dt, self.total_points)
        self.get_logger().info(f"✓ Pre-generated {len(self.trajectory_z)} trajectory points")

    def _get_next_point(self, tick: int) -> Tuple[float, float]:
        """
        Get next point from pre-generated trajectory.
        Simulates realtime control: only knows current & previous point.
        
        Velocity is computed from position difference (NOT analytical formula).
        
        Returns: (z, vz) for this tick
        """
        if tick >= len(self.trajectory_z):
            # Should not happen, but handle gracefully
            return self.trajectory_z[-1], 0.0
        
        z_current = self.trajectory_z[tick]
        
        # Compute velocity from position difference
        if tick == 0:
            vz = 0.0  # first point, no previous point
            self.z_prev = z_current
        else:
            vz = (z_current - self.z_prev) / self.ctrl_dt
            self.z_prev = z_current
        
        return z_current, vz

    def _get_stream_point(self, tick: int) -> StreamPoint:
        ref_z, ref_vz = self._get_next_point(tick)
        x, y = self.start_pose_6d[0], self.start_pose_6d[1]
        rx, ry, rz = self.start_pose_6d[3], self.start_pose_6d[4], self.start_pose_6d[5]
        return StreamPoint(
            x_m=x,
            y_m=y,
            z_m=ref_z,
            rx_deg=rx,
            ry_deg=ry,
            rz_deg=rz,
            vz_mps=ref_vz,
        )

    def _ref_z_at_elapsed(self, t_s: float) -> float:
        return self.trajectory.ref_z_at(self.start_pose_6d, t_s)

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

    def _estimate_lag(self, meas_z: float, now_wall: float) -> float:
        """Estimate lag between measured z and reference z"""
        if self.stream_start_wall is None:
            return 0.0
        t_plan = now_wall - self.stream_start_wall
        best = 0.0
        best_err = 1e9
        for dt_s in [i * 0.01 for i in range(-300, 301)]:
            t = max(0.0, t_plan - dt_s)
            ref_z = self._ref_z_at_elapsed(t)
            err = abs(meas_z - ref_z)
            if err < best_err:
                best_err = err
                best = dt_s
        return best

    # ---------- lifecycle ----------
    def start(self):
        self.get_logger().info("Waiting for send_script service...")
        if not self.send_script.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("send_script not available")
            self.done = True
            return
        self.get_logger().info("send_script ready")
        self.get_logger().info("Waiting for feedback...")
        self.startup_timer = self.create_timer(0.05, self._startup_tick)

    def _startup_tick(self):
        if not self.has_feedback or self.meas_pose_6d is None:
            return
        self.startup_timer.cancel()
        self.start_pose_6d = self.meas_pose_6d.copy()
        self.get_logger().info(
            f"Start pose Z={self.start_pose_6d[2]:.4f} RPY=({self.start_pose_6d[3]:.1f},{self.start_pose_6d[4]:.1f},{self.start_pose_6d[5]:.1f})"
        )
        
        # Pre-generate complete trajectory
        self._generate_trajectory()
        
        # Enter PVT
        self._send_async("E001", "PVTEnter(1)")
        
        self.t0_wall = time.time()
        self.stream_start_wall = self.t0_wall + self.config.stream_start_delay_s
        
        self.get_logger().info("=== REALTIME CONTROL START (with pre-gen trajectory) ===")
        self.get_logger().info(f"Control loop: read one point at a time, v = (z_curr - z_prev) / dt")
        
        # Start timers
        self.ctrl_timer = self.create_timer(self.ctrl_dt, self._ctrl_tick)
        self.obs_timer = self.create_timer(1.0 / self.obs_hz, self._obs_tick)
        
        self.streaming = True

    def _ctrl_tick(self):
        now = time.time()
        if self.stream_start_wall is None:
            return
        if now < self.stream_start_wall:
            return

        # Finish dispatch
        if self.tick >= self.total_points:
            if self.ctrl_timer:
                self.ctrl_timer.cancel()
            if self.streaming:
                self.streaming = False
                self.stream_end_wall = time.time()
                self._send_async("E005", "PVTExit()")
                self.get_logger().info("=== CTRL STREAM END (PVTExit sent) ===")
            return

        # Strict beat jitter
        ideal = self.stream_start_wall + self.tick * self.ctrl_dt
        jitter_s = now - ideal

        # ===== GET NEXT POINT FROM PRE-GENERATED TRAJECTORY =====
        # Read one point at a time from trajectory
        # Velocity computed from position difference (not formula)
        point = self._get_stream_point(self.tick)
        ref_z = point.z_m
        ref_vz = point.vz_mps
        cmd = self._stream_point_cmd(point, self.ctrl_dt * self.config.pvt_point_time_ratio)

        # Dispatch
        self._send_async(f"P{self.tick % 90 + 10:03d}", cmd)

        # Ack stats / overload detection
        p50, p95, mx = self._ack_stats()
        overload = 0
        if self._inflight > self.max_inflight:
            overload = 1
        if self._ack_last_ms > (self.ctrl_dt * 1000.0 * self.ack_over_period_ratio):
            overload = 1

        if overload:
            self._overload_streak += 1
        else:
            self._overload_streak = 0

        backlog_flag = 1 if self._overload_streak >= self.overload_consecutive else 0

        self.ctrl_log.append(
            CtrlSample(
                t_wall=now,
                tick=self.tick,
                ref_z=ref_z,
                ref_vz=ref_vz,
                jitter_s=jitter_s,
                inflight=self._inflight,
                ack_p50_ms=p50,
                ack_p95_ms=p95,
                ack_max_ms=mx,
                backlog_flag=backlog_flag,
            )
        )

        if self.tick % 10 == 0:
            self.get_logger().info(
                f"[CTRL {self.tick:03d}] jitter={jitter_s*1000:+.1f}ms "
                f"inflight={self._inflight} ack_last={self._ack_last_ms:.1f}ms "
                f"ref_z={ref_z:.4f} ref_vz={ref_vz:+.3f}"
            )

        self.tick += 1

    def _obs_tick(self):
        if not self.has_feedback or self.meas_pose_6d is None or self.start_pose_6d is None:
            return

        now = time.time()
        meas_z = self.meas_pose_6d[2]

        # Estimate vz
        if self._meas_z_prev is None:
            vz = 0.0
        else:
            dt = 1.0 / self.obs_hz
            vz = (meas_z - self._meas_z_prev) / max(1e-6, dt)
        self._meas_z_prev = meas_z

        self._meas_vz_lpf = (1.0 - self._vz_alpha) * self._meas_vz_lpf + self._vz_alpha * vz

        # Latest ref for comparison
        if self.stream_start_wall is None:
            ref_z_latest = self.start_pose_6d[2]
            lag_est = 0.0
        else:
            # For observation, we still use formula for comparison
            t_plan = max(0.0, now - self.stream_start_wall)
            ref_z_latest = self._ref_z_at_elapsed(t_plan)
            lag_est = self._estimate_lag(meas_z, now)

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

        # Periodic log
        if (now - self._obs_last_log_wall) >= self.obs_log_period_s:
            self._obs_last_log_wall = now
            self.get_logger().info(
                f"[OBS] z={meas_z:.4f} vz={self._meas_vz_lpf:+.3f} "
                f"ref_z={ref_z_latest:.4f} err={inst_err*1000:+.1f}mm "
                f"lag~{lag_est:+.2f}s inflight={self._inflight}"
            )

        # Auto-stop after stream end
        if self.stream_end_wall is not None:
            t_after = now - self.stream_end_wall
            if t_after > self.settle_timeout_s:
                self.get_logger().warn("Settle timeout -> stop")
                self._finalize("settle_timeout")
                return

            if abs(self._meas_vz_lpf) < self.settle_vz_thr:
                N = int(max(1, round(self.settle_hold_s * self.obs_hz)))
                if len(self.obs_log) >= N:
                    ok = True
                    for s in self.obs_log[-N:]:
                        if abs(s.meas_vz) >= self.settle_vz_thr:
                            ok = False
                            break
                    if ok:
                        self.get_logger().info("Motion settled -> stop")
                        self._finalize("settled")
                        return

    def _finalize(self, reason: str):
        if self.done:
            return
        self.done = True

        for t in [self.ctrl_timer, self.obs_timer, self.startup_timer]:
            if t is not None:
                try:
                    t.cancel()
                except Exception:
                    pass

        out_dir = self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"{self.config.output_prefix}_{ts}.csv")
        png_path = os.path.join(out_dir, f"{self.config.output_prefix}_{ts}.png")

        self._save_csv(csv_path)
        self._save_plot(png_path, reason)

        self.get_logger().info(f"Saved: {csv_path}")
        self.get_logger().info(f"Saved: {png_path}")
        self.get_logger().info(f"Reason: {reason}")

    def _save_csv(self, path: str):
        t0 = self.stream_start_wall if self.stream_start_wall is not None else (self.t0_wall or time.time())
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
        ax1.set_title(f"PVT Pre-gen Traj + Realtime Sim (v from diff) | reason={reason}")
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


PVTOnlineGenerator = PVTStreamControlComponent


def run_stream_control(
    config: Optional[PVTStreamConfig] = None,
    trajectory: Optional[ZTrajectoryProvider] = None,
    node_name: str = "pvt_stream_control",
):
    """Run the component with a SingleThreadedExecutor until it finalizes."""
    rclpy.init()
    node = PVTStreamControlComponent(
        config=config,
        trajectory=trajectory,
        node_name=node_name,
    )
    node.start()

    exec_ = SingleThreadedExecutor()
    exec_.add_node(node)

    try:
        while rclpy.ok() and (not node.done):
            exec_.spin_once(timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt")
        node._finalize("keyboard_interrupt")
    finally:
        try:
            exec_.remove_node(node)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


def main():
    run_stream_control()


if __name__ == "__main__":
    main()
