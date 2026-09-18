"""Cycle through a sequence of ports: align, hand off to fine insertion,
withdraw to a recorded pose, hand back for the next connector, repeat.

No robot-script JSON exists yet (see load_sequence) -- the order is
hardcoded until one does. usb1 is assumed already grasped (gripper closed)
before this starts; every later port's own grasp is a handoff to a program
that does not exist yet either (see fetch_connector) -- both handoffs are
stubbed as a timed pause for now, structured so the real programs can drop
in without anything here changing.

    ros2 run py_gripper arm_cmd_cycle
"""
import threading
import time

import rclpy

from py_gripper.arm_cmd import ArmCmd

# Placeholder until a real robot-script JSON exists.
DEFAULT_SEQUENCE = ['usb1', 'usb4', 'hdmi1']

# hover_at()'s own unit is centimetres, not millimetres.
STANDOFF_CM = 2.0   # 2cm before the target

# Measured with a ruler FROM THE FLANGE (the wrist's own mounting face,
# not the camera) to the tip of whatever the gripper is currently holding
# -- hover_at() adds this straight onto STANDOFF_CM in world frame. Must
# be flange-to-tip, not camera-to-tip: the camera's own optical axis is
# not parallel to the flange's Z axis (about 23deg apart on this rig, per
# T_G_C), so a camera-frame distance cannot be added to a flange-frame
# move without introducing that angular error -- confirmed live
# 2026-09-10 (usb1 vs usb3 depths disagreed by 3.2cm from a single
# stationary, already-squared pose, matching the 23deg offset rather than
# panel tilt). Different connectors (usb vs hdmi) may not be the same
# length -- re-measure and update this if the held connector changes
# partway through a real sequence; this cycle currently uses one value
# for all three ports.
TIP_OFFSET_CM = 17.2  # measured flange-to-tip, 2026-09-10

TILT_TOLERANCE_DEG = 2.0
MAX_SQUARE_ATTEMPTS = 3

# wait_until_arrived()'s own default (5mm) considers a move "done" for
# ordinary software polling -- fine for transit moves, too loose for the
# pose this cycle records as e1/e2/e3 and will later retrace to pull each
# connector back out, so those specific moves wait to a tighter tolerance.
# Neither one changes how precisely the arm itself physically stops --
# that is send_request's own blend/fine_goal, untouched here -- only when
# this script considers the move finished and moves on.
ARRIVED_TOL_M = 0.005
ARRIVED_TOL_TIGHT_M = 0.001

# wait_until_arrived()'s own default timeout (15s) assumes something close
# to send_request()'s commanded velocity (0.1 m/s) actually happens. With
# the teach pendant's own speed override turned way down for this first
# round of real-hardware testing, a transit of a few tens of cm can
# genuinely take longer than that with nothing wrong -- confirmed live
# 2026-09-10 (usb1's own ~35cm approach only covered ~11cm in 15s, matching
# the reduced override, not a stall). Generous on purpose; turn the
# override back up and this can come back down once motion is trusted.
MOVE_TIMEOUT_S = 90.0

GRIPPER_SETTLE_S = 1.5
HANDOFF_STUB_S = 2.0


def load_sequence(path=None):
    """-> ordered list of port names to insert, in order.

    No robot-script JSON exists yet -- this returns the hardcoded default.
    Once one does, point `path` at it: same idea as depth_pose_node's own
    task_file.py, extended to a list rather than a single target.
    """
    if path is None:
        return list(DEFAULT_SEQUENCE)
    import json
    with open(path) as f:
        doc = json.load(f)
    return doc['sequence']


def fine_insert(arm, name):
    """Hand off to the (not yet written) fine-insertion program.

    Stubbed as a fixed pause; does not touch the gripper -- run_cycle
    opens it itself right after this returns, before retracting to e_i
    (see run_cycle), so a plug this just seated is released before the
    arm pulls back rather than getting dragged out with it.
    """
    arm.get_logger().info(
        f'-- handing off to fine insertion for {name} '
        f'(stub: {HANDOFF_STUB_S}s pause, gripper untouched) --')
    time.sleep(HANDOFF_STUB_S)
    return True


def fetch_connector(arm, name):
    """Hand off to the (not yet written) connector-grasp program.

    Stubbed as a fixed pause, gripper untouched here too -- the real
    program is expected to close the gripper on `name`'s connector itself.
    Wherever it leaves the arm is not trusted: run_cycle re-homes to the
    save0 pose right after this returns, rather than assuming it.
    """
    arm.get_logger().info(
        f'-- handing off to grasp program for {name} '
        f'(stub: {HANDOFF_STUB_S}s pause) --')
    time.sleep(HANDOFF_STUB_S)
    return True


def square_within(arm, tolerance_deg=TILT_TOLERANCE_DEG,
                   max_attempts=MAX_SQUARE_ATTEMPTS):
    """Square up, re-checking until under tolerance_deg or attempts run out.

    One square_up() is not always enough -- 2deg was the practical floor
    even jogging by hand (see README) -- so this is that same "check,
    correct, recheck" loop, done for real instead of by eye.
    """
    for attempt in range(max_attempts):
        tilt = arm.check_tilt()
        if tilt is None:
            arm.get_logger().error('no tilt reading -- aborting square')
            return False
        if tilt <= tolerance_deg:
            arm.get_logger().info(
                f'square within {tolerance_deg} deg ({tilt:.1f}) '
                f'after {attempt} correction(s)')
            return True
        if arm.square_up(move=True) is None:
            arm.get_logger().error('square_up refused -- aborting square')
            return False
        if not arm.wait_until_arrived(error=ARRIVED_TOL_M):
            arm.get_logger().error('square_up move did not complete -- aborting')
            return False
    tilt = arm.check_tilt()
    ok = tilt is not None and tilt <= tolerance_deg
    if not ok:
        arm.get_logger().error(
            f'could not reach {tolerance_deg} deg after {max_attempts} '
            f'attempts (last reading {tilt})')
    return ok


def goto_pose(arm, pose, tight=False):
    if not arm.target_is_sane(pose):
        return False
    if arm.send_request(list(pose)) is None:
        return False
    return arm.wait_until_arrived(
        timeout=MOVE_TIMEOUT_S,
        error=ARRIVED_TOL_TIGHT_M if tight else ARRIVED_TOL_M)


def approach_target(arm, standoff_cm=STANDOFF_CM, tip_offset_cm=TIP_OFFSET_CM):
    if arm.hover_at(standoff_cm, tip_offset_cm, move=True) is None:
        return False
    return arm.wait_until_arrived(timeout=MOVE_TIMEOUT_S, error=ARRIVED_TOL_TIGHT_M)


def open_gripper(arm):
    arm.send_gripper(0.085)
    time.sleep(GRIPPER_SETTLE_S)


def run_cycle(arm, sequence):
    if not sequence:
        arm.get_logger().error('empty sequence -- nothing to do')
        return False

    # usb1 is assumed already grasped (gripper closed) before this runs --
    # not checked here, there is no gripper feedback to check it against.
    first = sequence[0]
    if not arm.set_target_port(first):
        arm.get_logger().error(f'no confirmed reading for {first} -- aborting before moving')
        return False

    if not square_within(arm):
        return False

    arm.freeze(True)
    home_pose = list(arm.current_positions)
    arm.get_logger().info(
        f'save0: frame frozen, home pose recorded -> {[round(v, 4) for v in home_pose]}')

    for i, name in enumerate(sequence):
        if i > 0:
            # re-sync before trusting "above" -- fetch_connector() just ran
            # and its own resting pose is not assumed
            if not goto_pose(arm, home_pose):
                arm.get_logger().error(f'could not return to save0 pose before {name} -- aborting')
                return False
            if not arm.set_target_port(name):
                arm.get_logger().error(f'no confirmed reading for {name} -- aborting')
                return False

        if not approach_target(arm):
            arm.get_logger().error(f'could not reach the {STANDOFF_CM * 10:.0f}mm standoff over {name} -- aborting')
            return False
        e_i = list(arm.current_positions)
        arm.get_logger().info(f'{name}: standoff reached, recorded as e{i+1} -> {[round(v, 4) for v in e_i]}')

        if not fine_insert(arm, name):
            arm.get_logger().error(f'fine insertion reported failure on {name} -- aborting')
            return False

        # release before retracting -- pulling back with the gripper still
        # closed on a plug that is now seated would drag it back out
        open_gripper(arm)

        if not goto_pose(arm, e_i, tight=True):
            arm.get_logger().error(f'could not retract to e{i+1} after {name} -- aborting')
            return False
        arm.get_logger().info(f'{name}: retracted to e{i+1}, gripper open')

        last = (i == len(sequence) - 1)
        if last:
            break

        if not goto_pose(arm, home_pose):
            arm.get_logger().error(f'could not return to save0 pose after {name} -- aborting')
            return False

        next_name = sequence[i + 1]
        if not fetch_connector(arm, next_name):
            arm.get_logger().error(f'grasp handoff reported failure on {next_name} -- aborting')
            return False

    arm.get_logger().info('cycle complete')
    return True


def main(args=None):
    rclpy.init(args=args)
    arm = ArmCmd()
    # An explicit executor, not the rclpy.spin(arm) convenience function --
    # shutting it down (below) is what lets the background thread's spin()
    # call return on its own before rclpy.shutdown() runs. Calling
    # rclpy.shutdown() while another thread is still blocked inside spin()
    # raced the rclpy/rcl C bindings and aborted the whole process outright
    # (confirmed live: "terminate called without an active exception").
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(arm)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)  # let feedback_states populate before the first read

    sequence = load_sequence()
    arm.get_logger().info(f'starting cycle: {sequence}')
    try:
        ok = run_cycle(arm, sequence)
    finally:
        # end0 -- whether the cycle finished or aborted partway, nothing
        # after this run should keep reading the frame it froze on.
        arm.freeze(False)
    arm.get_logger().info('cycle finished OK' if ok else 'cycle ABORTED (see errors above)')

    executor.shutdown()
    spin_thread.join(timeout=2.0)
    arm.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
