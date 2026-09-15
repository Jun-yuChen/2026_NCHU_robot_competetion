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

Action server
-------------
This node also serves the FineAlignment action ('fine_alignment'). It
persists across goals rather than exiting after one insertion: the FSM
above (state, tick, spiral_idx, the steady-state histories, ...) is now
per-goal state that _reset_fsm_state() puts back to scratch at the start
of every execute_callback, instead of per-process state set up once in
__init__. optoforce_node and stream_control_node are unaffected by any of
this -- they're started once (same spiral_search_launch.py as before) and
just sit there serving sensor data / the PVTCommand service across
however many goals this node runs.

_ctrl_tick still runs on a plain ROS timer, same as before, in the node's
default (mutually-exclusive) callback group -- so it's still effectively
single-threaded with respect to itself and the feedback/wrench
subscriptions, exactly as when this was a run-once script. execute_callback
runs in a separate ReentrantCallbackGroup on a MultiThreadedExecutor
thread and never touches FSM state directly: it only blocks on
_feedback_q, which _ctrl_tick pushes to on every transition (and a
heartbeat), and turns that into action feedback / the final Result. The
one thing _ctrl_tick does need from execute_callback's side is
_active_goal_handle, so it can notice a cancellation itself and unwind
the stream the same way it already unwinds for a completion, a max-force
trip, or a duration_s timeout -- see the class docstring for SearchState
and _ctrl_tick's cancellation check for how that mirrors the other three.
"""

import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional
from collections import deque
from queue import Empty, Queue

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from stream_control.PI_controller import PI_controller

from geometry_msgs.msg import WrenchStamped
from tm_msgs.srv import SetPositions, SetEvent, SetIO
from tm_msgs.msg import FeedbackState
from custom_interface.srv import PVTCommand
from action_interface.action import FineAlignment


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


@dataclass
class _TickUpdate:
    """One item pushed onto self._feedback_q by _ctrl_tick, drained by
    execute_callback (a different thread) and turned into either action
    Feedback or the final Result. Purely internal -- never leaves this
    process, so it's a plain dataclass rather than a message type.

    terminal=False -> an in-progress update: `state` is a
      FineAlignment.Feedback.* constant, forwarded as-is via
      goal_handle.publish_feedback().
    terminal=True  -> the goal is over: `result_status` is a
      FineAlignment.Result.* constant, forwarded via goal_handle.succeed()
      /abort()/canceled() (execute_callback picks which based on
      result_status) plus the Result message itself.
    """
    terminal: bool
    state: int = FineAlignment.Feedback.INITIALIZING
    message: str = ""
    progress: float = 0.0
    result_status: int = FineAlignment.Result.SUCCESS


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
        # How long execute_callback will wait, per goal, for feedback_states
        # / optoforce/wrench / the PVTCommand service before aborting with
        # ERROR. Separate from service_wait_log_period_s above (that's just
        # the log throttle within this wait).
        self.declare_parameter("goal_ready_timeout_s", 10.0)

        # ----- spiral search parameters (in the flange's x-y plane) -----
        # Archimedean spiral: r(theta) = (spiral_pitch_mm / 2*pi) * theta
        self.declare_parameter("spiral_pitch_mm", 2.0)          # radial growth per revolution
        self.declare_parameter("spiral_max_radius_mm", 15.0)    # stop generating past this radius
        self.declare_parameter("spiral_search_speed_mm_s", 3.0) # constant speed along the spiral path

        # ----- touch detection -----
        self.declare_parameter("touch_force_tol_n", 0.5)  # |Fz - touch_force| <= tol => "touched"

        # ----- alignment / insertion -----
        self.declare_parameter('position_steady_state_tolerance_mm', 0.05)
        self.declare_parameter('hole_force_delta_N', 0.5)
        self.declare_parameter('hole_testing_threshold_mm', 5)
        self.declare_parameter('steady_state_window_s', 0.3)
        self.declare_parameter("align_force", 0.4)   # Fz drops below this => hole/peg aligned
        self.declare_parameter("insert_force", 5.0)  # z-force target once aligned (the "peg" phase)
        self.declare_parameter("max_force_n", 8.0)   # hard safety cutoff, must stay above insert_force

        self.force_controller = PI_controller(
            Kp=self.get_parameter("PI_controller_Kp").value,
            Ki=self.get_parameter("PI_controller_Ki").value,
            integral_limit=self.get_parameter("PI_controller_integral_limit").value,
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

        self.hole_force_delta_N = (
            self.get_parameter('hole_force_delta_N').value
        )

        self.hole_testing_threshold_m = (
            self.get_parameter('hole_testing_threshold_mm').value / 1000.0
        )

        steady_state_window_s = self.get_parameter('steady_state_window_s').value
        # ctrl_hz is fixed by the timer period, so a time window converts directly
        # to a sample count for the rolling buffers below.
        self._steady_state_len = max(2, round(steady_state_window_s * self.ctrl_hz))

        self._search_z_history = deque(maxlen=self._steady_state_len)
        self._hole_test_z_history = deque(maxlen=self._steady_state_len)
        self._insert_z_history = deque(maxlen=self._steady_state_len)
        self.surface_height_g = None
        self._shutdown_requested = False
        self._pending_terminal: Optional["_TickUpdate"] = None

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

        # Purely a function of the static spiral_* params above, so
        # generated once here and reused for every goal (only spiral_idx,
        # reset per-goal in _reset_fsm_state, tracks where a given run is
        # along it).
        self.spiral_traj: List[tuple] = self._generate_spiral_trajectory()
        self.spiral_idx = 0
        self.get_logger().info(
            f"Generated spiral trajectory: {len(self.spiral_traj)} points "
            f"(pitch={self.spiral_pitch_mm:.2f}mm, max_r={self.spiral_max_radius_mm:.2f}mm, "
            f"speed={self.spiral_search_speed_mm_s:.2f}mm/s)"
        )

        command_service = self.get_parameter("command_service").value
        self._command_service_name = command_service
        self.service_wait_log_period_s = float(self.get_parameter("service_wait_log_period_s").value)
        self.goal_ready_timeout_s = float(self.get_parameter("goal_ready_timeout_s").value)
        self.cmd_client = self.create_client(PVTCommand, command_service)

        # Subscriptions live for the node's whole lifetime, not per-goal --
        # optoforce_node and stream_control_node's feedback publisher are
        # background nodes now (see spiral_search_launch.py), so there's no
        # reason this node's picture of the arm/sensor should go stale
        # between goals. These stay on the node's default (mutually
        # exclusive) callback group, same group as _ctrl_tick and the
        # PVTCommand response callback -- see the class docstring for why
        # that matters (it's what keeps the FSM effectively single-threaded
        # even though this is now a MultiThreadedExecutor node).
        self.create_subscription(FeedbackState, "feedback_states", self._fb_cb, 10)
        self.create_subscription(WrenchStamped, 'optoforce/wrench', self._FTsensor_cb, 10)


        # Create client for control the gripper
        self.io_cli = self.create_client(SetIO, '/set_io')

        self.feedback_is_avaliable = False

        self.FT_is_avaliable = False
        self.force_observe = None
        self.torque_observe = None

        self.tick = 0
        self.done = False

        self._inflight = 0
        self._error_count = 0

        self.ctrl_timer = None
        self.prev_time = None  # For control loop

        # ----- action server -----
        # execute_callback blocks (queue.get / time.sleep) for the whole
        # goal, so it needs its own callback group, separate from the
        # default one _ctrl_tick/_fb_cb/_FTsensor_cb/the PVTCommand
        # response callback all sit in -- otherwise it would starve them
        # for the entire run. See main() for the matching
        # MultiThreadedExecutor.
        self._action_cb_group = ReentrantCallbackGroup()
        self._goal_active = False
        self._active_goal_handle = None
        self._feedback_q: "Queue[_TickUpdate]" = Queue()

        self._action_server = ActionServer(
            self,
            FineAlignment,
            "fine_alignment",
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._action_cb_group,
        )

        self.get_logger().info(
            f"{self.ctrl_hz:.1f}Hz, {self.total_points} points/goal, "
            f"calling PVTCommand service '{command_service}', "
            f"orientation held passively at whatever it is when each goal starts -- "
            f"serving FineAlignment action 'fine_alignment'"
        )

    # ---------- action server callbacks ----------
    def _goal_callback(self, goal_request):
        if self._goal_active:
            self.get_logger().warn("Rejecting goal: fine alignment already running")
            return GoalResponse.REJECT
        if not self.cmd_client.service_is_ready():
            self.get_logger().warn(
                f"Rejecting goal: PVTCommand service '{self._command_service_name}' "
                f"not available (is stream_control_node up?)"
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        # Actually stopping the robot happens in _ctrl_tick, which notices
        # self._active_goal_handle.is_cancel_requested -- see there.
        self.get_logger().info("Cancel requested")
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        self._goal_active = True
        self._active_goal_handle = goal_handle
        try:
            return self._run_goal(goal_handle)
        finally:
            self._active_goal_handle = None
            self._goal_active = False

    def _run_goal(self, goal_handle):
        # Drop anything left in the queue from a previous goal (there
        # shouldn't be any -- every terminal path drains/returns cleanly --
        # but a goal is expensive enough on real hardware that starting
        # clean is worth the one extra check).
        while not self._feedback_q.empty():
            try:
                self._feedback_q.get_nowait()
            except Empty:
                break

        self._reset_fsm_state()
        self._publish_feedback(goal_handle, FineAlignment.Feedback.INITIALIZING,
                                "waiting for feedback_states / optoforce/wrench", 0.0)

        if not self._wait_until_ready(goal_handle):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return self._make_result(FineAlignment.Result.CANCELLED, "cancelled while waiting to start")
            goal_handle.abort()
            return self._make_result(
                FineAlignment.Result.ERROR,
                f"feedback_states/optoforce/wrench not available within "
                f"goal_ready_timeout_s={self.goal_ready_timeout_s:.1f}s",
            )

        # Reference point for insertion-axis travel monitoring -- see
        # class docstring / _ctrl_tick. Recorded fresh for every goal,
        # from wherever the flange happens to be right now (spiral_traj
        # itself was already generated once in __init__ -- see there).
        self._monitor_ref_pos_base = self.T_B_G[:3, 3].copy()
        self.get_logger().info(
            f"monitor ref(base)={np.round(self._monitor_ref_pos_base, 4).tolist()}"
        )
        self._publish_feedback(
            goal_handle, FineAlignment.Feedback.WAITING_TOUCH,
            f"spiral trajectory ready ({len(self.spiral_traj)} points) -- waiting for touch",
            0.0,
        )

        self.ctrl_timer = self.create_timer(self.ctrl_dt, self._ctrl_tick)

        while True:
            try:
                item = self._feedback_q.get(timeout=0.2)
            except Empty:
                continue

            if not item.terminal:
                self._publish_feedback(goal_handle, item.state, item.message, item.progress)
                continue

            self._cleanup_timer()

            time.sleep(1.5) # Wait PVT to exit
            self.get_logger().info("PVT should END now")

            if item.result_status == FineAlignment.Result.SUCCESS:
                goal_handle.succeed()
                self._set_gripper(0.0)

            elif item.result_status == FineAlignment.Result.CANCELLED:
                goal_handle.canceled()
            else:
                goal_handle.abort()
            return self._make_result(item.result_status, item.message)

    def _reset_fsm_state(self):
        """Put every piece of per-goal FSM state back to scratch. Static,
        parameter-derived things (spiral_traj, PI controller gains, ctrl_dt,
        total_points, ...) are NOT touched here -- they don't change
        between goals."""
        self.state = SearchState.WAITING_TOUCH
        self.tick = 0
        self.spiral_idx = 0
        self._shutdown_requested = False
        self._pending_terminal = None
        self.surface_height_g = None
        self._search_z_history.clear()
        self._hole_test_z_history.clear()
        self._insert_z_history.clear()
        self.prev_time = None
        self._inflight = 0
        self._error_count = 0
        self.force_controller.reset()

    def _wait_until_ready(self, goal_handle) -> bool:
        """Block (this is execute_callback's own thread, blocking here is
        fine) until feedback_states and optoforce/wrench have both
        delivered at least one message, or goal_ready_timeout_s elapses,
        or the goal is cancelled. Returns False on timeout or cancel."""
        deadline = time.monotonic() + self.goal_ready_timeout_s
        last_log = 0.0
        while not (self.feedback_is_avaliable and self.FT_is_avaliable):
            if goal_handle.is_cancel_requested:
                return False
            if time.monotonic() >= deadline:
                return False
            now = time.monotonic()
            if (now - last_log) >= self.service_wait_log_period_s:
                last_log = now
                self.get_logger().info(
                    "Waiting for feedback_states / optoforce/wrench "
                    f"(feedback={self.feedback_is_avaliable}, FT={self.FT_is_avaliable})..."
                )
            time.sleep(0.05)
        return True

    def _publish_feedback(self, goal_handle, state: int, message: str, progress: float):
        fb = FineAlignment.Feedback()
        fb.state = state
        fb.message = message
        fb.progress = float(progress)
        goal_handle.publish_feedback(fb)

    @staticmethod
    def _make_result(status: int, message: str):
        result = FineAlignment.Result()
        result.status = status
        result.message = message
        return result

    def _cleanup_timer(self):
        if self.ctrl_timer is None:
            return
        try:
            self.ctrl_timer.cancel()
        except Exception:
            pass
        try:
            self.destroy_timer(self.ctrl_timer)
        except Exception:
            pass
        self.ctrl_timer = None

    def _send_final_hold(self, is_last: bool = True):
        """Send one PVTCommand holding the flange exactly where it is right
        now. Used to end a stream cleanly (cancellation) without falling
        through the rest of a normal tick's force/spiral computation --
        there's nothing to compute, we just want to stop exactly here."""
        current_flange_pos_base = self.T_B_G[:3, 3]
        R_B_G_live = self.T_B_G[:3, :3]
        target_orient_deg = R.from_matrix(R_B_G_live).as_euler('xyz', degrees=True)

        req = PVTCommand.Request()
        req.header.stamp = self.get_clock().now().to_msg()
        req.tick = self.tick
        req.is_last = is_last
        req.x_m, req.y_m, req.z_m = current_flange_pos_base.tolist()
        req.rx_deg, req.ry_deg, req.rz_deg = target_orient_deg.tolist()
        req.vx_mps, req.vy_mps, req.vz_mps = 0.0, 0.0, 0.0
        req.wx_dps, req.wy_dps, req.wz_dps = 0.0, 0.0, 0.0
        req.point_time_s = self.ctrl_dt * self.pvt_point_time_ratio
        self.cmd_client.call_async(req)
        self.tick += 1

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

    def _check_steady_state(self, history: deque, z: float, tol=None) -> bool:
        """Push z into a rolling window; True once the window is full and its
        spread (max-min) is within position_steady_state_tolerance_m.

        Same discard-until-full behaviour as the OptoForce moving-average filter:
        a window that isn't full yet says nothing about steadiness, so it must
        not report True just because the spread-so-far happens to be small.
        """
        if tol is None:
            tol = self.position_steady_state_tolerance_m

        history.append(z)
        if len(history) < history.maxlen:
            return False
        return (max(history) - min(history)) <= tol

    def _set_gripper(self, state=0):
        """
        state = 0.0 open the gripper
        state = 1.0 close the gripper
        """
        res = SetIO.Request()
        res.module = 1  # IO module 1
        res.type = 1    # digital IO
        res.pin = 0     # pin 0
        res.state = float(state)
        self.io_cli.call_async(res)

    #=======================================================================================================================
    #===================== MAIN CONTROL LOOP ===============================================================================
    #=======================================================================================================================

    def _ctrl_tick(self):
        if (not self.feedback_is_avaliable) or (not self.FT_is_avaliable):
            return
        if self._shutdown_requested:
            return

        # Mirrors the max-force safety branch further down: notice it,
        # send one hold-in-place point with is_last=True so
        # stream_control_node gets a clean PVTExit instead of waiting on
        # its stall watchdog, stop the timer, report it, and stop -- all
        # in this one tick, same pattern as every other terminal path here.
        if self._active_goal_handle is not None and self._active_goal_handle.is_cancel_requested:
            self.get_logger().info("[CANCEL] cancel requested -- sending final hold point and stopping")
            self._send_final_hold(is_last=True)
            self.ctrl_timer.cancel()
            self._shutdown_requested = True
            self._feedback_q.put(_TickUpdate(
                terminal=True, result_status=FineAlignment.Result.CANCELLED,
                message="cancelled by client",
            ))
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
                self._search_z_history.clear()
                self.get_logger().info(
                    f"[TOUCH] contact detected (Fz={fz:.3f}N) -- starting spiral search"
                )
                self._feedback_q.put(_TickUpdate(
                    terminal=False, state=FineAlignment.Feedback.SEARCHING,
                    message=f"touch detected (Fz={fz:.3f}N) -- starting spiral search",
                ))

        elif self.state == SearchState.SEARCHING:
            if (self.align_force - fz) > self.hole_force_delta_N and self._check_steady_state(self._search_z_history, current_z_g):
                self.state = SearchState.HOLE_TESTING
                self.surface_height_g = current_z_g
                self._hole_test_z_history.clear()
                self.get_logger().info(
                    f"[CANDIDATE] possible alignment (Fz={fz:.3f}N < "
                    f"align_force={self.align_force:.3f}N) -- testing for a real "
                    f"hole from surface_height(g)={self.surface_height_g:.5f}m"
                )
                self._feedback_q.put(_TickUpdate(
                    terminal=False, state=FineAlignment.Feedback.HOLE_TESTING,
                    message=f"candidate alignment (Fz={fz:.3f}N) -- testing for a real hole",
                ))

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
                    self._search_z_history.clear()
                    self._hole_test_z_history.clear()
                    self.state = SearchState.SEARCHING
                    self._feedback_q.put(_TickUpdate(
                        terminal=False, state=FineAlignment.Feedback.SEARCHING,
                        message=f"not a hole (settled {moved_m*1000:.3f}mm from surface) -- resuming spiral",
                    ))
                else:
                    # sank in past tolerance -- this is the hole
                    self.get_logger().info(
                        f"[HOLE CONFIRMED] settled {moved_m*1000:.3f}mm from "
                        f"surface -- inserting to {self.insert_force:.2f}N"
                    )
                    self._insert_z_history.clear()
                    self.state = SearchState.INSERTING
                    self._feedback_q.put(_TickUpdate(
                        terminal=False, state=FineAlignment.Feedback.INSERTING,
                        message=f"hole confirmed (settled {moved_m*1000:.3f}mm from surface) -- inserting",
                    ))

        elif self.state == SearchState.INSERTING:
            if self._check_steady_state(self._insert_z_history, current_z_g):
                self.get_logger().info(
                    f"[INSERT COMPLETE] z steady within "
                    f"{self.position_steady_state_tolerance_m*1000:.3f}mm over "
                    f"{self._steady_state_len} ticks -- holding and stopping"
                )
                self._shutdown_requested = True
                # NOT pushed to self._feedback_q here -- execute_callback's
                # thread would be free to pop it and call _cleanup_timer()
                # (which nulls self.ctrl_timer) while THIS tick is still
                # running and hasn't reached its own
                # `self.ctrl_timer.cancel()` yet (below, in the shared
                # `if req.is_last:` block) -- that's the
                # AttributeError: 'NoneType' object has no attribute
                # 'cancel' race. Stash it and push only once this tick is
                # done touching self.ctrl_timer, same as every other
                # terminal path already does.
                self._pending_terminal = _TickUpdate(
                    terminal=True, result_status=FineAlignment.Result.SUCCESS,
                    message=(
                        f"insertion complete, z steady within "
                        f"{self.position_steady_state_tolerance_m*1000:.3f}mm"
                    ),
                )

        # ============ 2) Z FORCE OUTPUT (insertion axis) ============
        # A step from wherever the flange is *right now* -- no reference
        # point needed, this is a pure delta.
        z_target = (self.insert_force
                            if self.state in (SearchState.HOLE_TESTING, SearchState.INSERTING)
                            else self.touch_force
                    )

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
        req.tick = self.tick
        # Distinguish "ran out of duration_s without completing" from a
        # real completion, so it can be reported as TIMEOUT instead of
        # silently looking identical to success.
        reached_time_limit = (self.tick == self.total_points - 1) and not self._shutdown_requested
        req.is_last = reached_time_limit or self._shutdown_requested

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
            req.is_last = True
            self.cmd_client.call_async(req)
            self.get_logger().error(
                f"[SAFETY] max_force_n exceeded (Fz={self.force_observe.z:.2f}N > "
                f"{self.max_force_n:.2f}N) -- holding in place and ending stream"
            )
            self.ctrl_timer.cancel()
            self._shutdown_requested = True
            self._feedback_q.put(_TickUpdate(
                terminal=True, result_status=FineAlignment.Result.FAILURE,
                message=(
                    f"max_force_n exceeded (Fz={self.force_observe.z:.2f}N > "
                    f"{self.max_force_n:.2f}N)"
                ),
            ))
            return

        this_tick = self.tick
        future = self.cmd_client.call_async(req)
        self._inflight += 1

        if req.is_last:
            self.ctrl_timer.cancel()
            if reached_time_limit:
                self.get_logger().warn(
                    f"[TIMEOUT] duration_s={self.duration_s:.1f}s elapsed while in "
                    f"{self.state.name} without completing insertion"
                )
                self._shutdown_requested = True
                self._feedback_q.put(_TickUpdate(
                    terminal=True, result_status=FineAlignment.Result.TIMEOUT,
                    message=(
                        f"duration_s={self.duration_s:.1f}s elapsed while in "
                        f"{self.state.name} without completing insertion"
                    ),
                ))
            elif self._pending_terminal is not None:
                # The INSERTING-complete case from above -- safe to hand
                # off now that self.ctrl_timer.cancel() has already run on
                # this thread.
                self._feedback_q.put(self._pending_terminal)
                self._pending_terminal = None

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
                heartbeat_state, heartbeat_progress = FineAlignment.Feedback.INSERTING, 0.0
            elif self.state == SearchState.HOLE_TESTING:
                status = f"HOLE_TESTING (surface_g={self.surface_height_g:.5f}m)"
                heartbeat_state, heartbeat_progress = FineAlignment.Feedback.HOLE_TESTING, 0.0
            elif self.state == SearchState.SEARCHING:
                status = f"SEARCHING (spiral_idx={self.spiral_idx}/{len(self.spiral_traj)})"
                heartbeat_state = FineAlignment.Feedback.SEARCHING
                heartbeat_progress = (
                    self.spiral_idx / len(self.spiral_traj) if self.spiral_traj else 0.0
                )
            else:
                status = "WAITING_TOUCH"
                heartbeat_state, heartbeat_progress = FineAlignment.Feedback.WAITING_TOUCH, 0.0
            self.get_logger().debug(
                f"[GEN {self.tick:03d}] z_g={current_z_g:+.5f} vz_g={vz_g:+.3f} "
                f"base=({target_pos_base[0]:.4f},{target_pos_base[1]:.4f},{target_pos_base[2]:.4f}) "
                f"inflight={self._inflight} ({status})"
            )
            # Heartbeat, not a transition -- skip it on a tick that already
            # pushed a terminal update above (COMPLETED/FAILURE/TIMEOUT),
            # so it doesn't immediately follow that with a stale
            # in-progress one. (CANCELLED returns before reaching here.)
            if not self._shutdown_requested:
                self._feedback_q.put(_TickUpdate(
                    terminal=False, state=heartbeat_state,
                    message=status, progress=heartbeat_progress,
                ))

        self.tick += 1


def main():
    rclpy.init()
    node = SpiralSearchControllerNode()
    # Needs >=2 threads: execute_callback blocks for the whole goal on its
    # own ReentrantCallbackGroup, while _ctrl_tick / _fb_cb / _FTsensor_cb /
    # the PVTCommand response callback (the node's default callback group)
    # need to keep running concurrently -- see the class docstring.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()