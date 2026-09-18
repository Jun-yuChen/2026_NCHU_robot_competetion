# FineAlignment action -- integration guide (v2: merged into spiral_search_control_node)

This supersedes the previous version of this guide, which had a separate
`fine_alignment_action_server.py` relaunching all 3 nodes per goal via
`ros2 launch` + an `AlignmentStatus` status topic. That's gone now --
`spiral_search_control_node.py` **is** the action server, and
`optoforce_node` / `stream_control_node` are just background nodes it
depends on, started once.

What's in this patch:

```
custom_interface/
  action/FineAlignment.action      (unchanged)

stream_control/
  stream_control/
    spiral_search_control_node.py  (rewritten: FSM + action server, same
                                     module/executable name as before)
```

`spiral_search_launch.py`, `stream_control_node.py`, and `optoforce_node.py`
are **untouched** and still launched exactly the way they are today.

## Architecture

- `ros2 launch stream_control spiral_search_launch.py` starts all 3 nodes
  **once**, and they stay up for as many fine-alignment attempts as you
  want to run -- not per-goal anymore.
- `spiral_search_control_node.py` now persists across goals.
  `feedback_states` / `optoforce/wrench` subscriptions and the
  `PVTCommand` client are created once in `__init__`; the FSM state
  (`state`, `tick`, `spiral_idx`, the steady-state histories, the PI
  controller...) is reset at the start of every goal by
  `_reset_fsm_state()`.
- `_ctrl_tick` still runs on a plain ROS timer created fresh per goal,
  in the node's default (mutually-exclusive) callback group -- same
  effectively-single-threaded execution as before. `execute_callback`
  runs on a separate `ReentrantCallbackGroup`, blocks on an internal
  queue that `_ctrl_tick` pushes to on every transition, and turns that
  into action feedback / the final `Result`. This needs a
  `MultiThreadedExecutor` (already set up in `main()`) -- a
  `SingleThreadedExecutor` would deadlock the first time a goal is
  active, since `execute_callback` would block the only thread and
  `_ctrl_tick` would never get to run.
- No new message type, no subprocess management, no process-group
  signals. Cancellation and every terminal outcome (success / max-force
  trip / `duration_s` timeout) all funnel through the same
  `_ctrl_tick` -> queue -> `execute_callback` path.

## Build

No new interfaces beyond `FineAlignment.action` itself, so the
`CMakeLists.txt` / `package.xml` edits are exactly what they'd have been
for the action alone:

```cmake
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PVTCommand.srv"
  "action/FineAlignment.action"  # add
  DEPENDENCIES geometry_msgs std_msgs
)
```

```xml
<depend>action_msgs</depend>
```

No `setup.py` changes in `stream_control` -- it's the same executable
name (`spiral_search_control_node`) as before, so
`spiral_search_launch.py`'s `Node(...executable='spiral_search_control_node'...)`
entry needs no changes either.

```bash
colcon build --packages-select custom_interface stream_control
source install/setup.bash
```

## Run

```bash
ros2 launch stream_control spiral_search_launch.py
```

Then, in another terminal, as many times as you like:

```bash
ros2 action send_goal /fine_alignment custom_interface/action/FineAlignment {} --feedback
```

A second `send_goal` while one is already running gets rejected
(`GoalResponse.REJECT`, logged as "fine alignment already running") --
fine alignment owns the arm exclusively.

## Calling it from your own skill-sequencing code

```python
from rclpy.action import ActionClient
from custom_interface.action import FineAlignment

client = ActionClient(node, FineAlignment, 'fine_alignment')
client.wait_for_server()

goal_future = client.send_goal_async(
    FineAlignment.Goal(),
    feedback_callback=lambda fb: node.get_logger().info(
        f"[fine_alignment] state={fb.feedback.state} "
        f"progress={fb.feedback.progress:.2f} {fb.feedback.message}"
    ),
)
goal_handle = await goal_future
result = await goal_handle.get_result_async()

if result.result.status == FineAlignment.Result.SUCCESS:
    ...  # next skill
else:
    ...  # FAILURE / TIMEOUT / CANCELLED / ERROR -- result.result.message has why
```

## What changed in `spiral_search_control_node.py` -- please read before running on hardware

- **A real bug fix in the max-force safety branch.** The original branch
  built a hold-in-place `req` and then `return`ed *without ever calling
  `cmd_client.call_async(req)`* -- the mutated request was discarded, so
  a `max_force_n` trip sent **no further command at all**; the arm's last
  commanded point stayed whatever the previous (still-pushing-harder)
  tick had sent, and the stream only actually ended once
  `stream_control_node`'s `stall_timeout_s` watchdog (1.0s default)
  noticed and sent `PVTExit()` on its own. This patch actually sends the
  hold point now, with `is_last=True` for an immediate clean `PVTExit`
  instead of waiting out the watchdog. Worth testing this path
  specifically before trusting it on hardware -- I did not have a way to
  run this against the real arm.
- **Cancellation** sends one final hold-in-place point (same idea as the
  fix above) and stops, rather than relying on the stall watchdog.
- **The node no longer calls `rclpy.shutdown()` on success.** It used to
  exit the whole process after one insertion; now it just tears down that
  goal's timer and waits for the next goal.
- **Per-goal PI controller reset.** `force_controller` is rebuilt fresh
  (not `.reset()`, since I don't have `PI_controller`'s source to know if
  that exists) at the start of every goal, so one goal's accumulated
  integral term can't leak into the next one's.
- **`spiral_traj` is generated once in `__init__`**, not per-goal -- it's
  a pure function of the static `spiral_*` parameters. Only `spiral_idx`
  (which position along it) is per-goal state.
- **Progress is best-effort**, same as before: only really informative
  during `SEARCHING` (fraction of `spiral_max_radius_mm` covered), `0.0`
  elsewhere.
- New parameter `goal_ready_timeout_s` (default 10.0s): how long a goal
  will wait for `feedback_states`/`optoforce/wrench` before aborting with
  `ERROR`. In normal operation this should resolve almost instantly,
  since the sensors are already running by the time you send a goal.
