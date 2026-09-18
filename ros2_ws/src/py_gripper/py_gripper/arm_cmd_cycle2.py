"""One port's worth of arm_cmd_cycle.py, stopped early: square up, approach
the standoff over TARGET_PORT, record e1, then stop -- no insertion, no
retract, no handoff. For checking alignment/standoff accuracy on hardware
without committing to an actual insertion, and (see align_port_action_server)
the first of two actions an upstream orchestrator will call in sequence --
the second, a separately-built fine-insertion action, is not part of this
repo.

Reuses arm_cmd_cycle's own building blocks (square_within, approach_target)
rather than duplicating them, so a fix made there (e.g. the timeout/
tolerance tuning) does not have to be made twice.

    ros2 run py_gripper arm_cmd_cycle2
"""
import threading
import time

import rclpy

from py_gripper.arm_cmd import ArmCmd
from py_gripper.arm_cmd_cycle import square_within, approach_target

# Hardcoded until either this or the action wrapping it takes the port as
# a goal/argument instead -- there is no real port-sequencing story feeding
# this yet (see arm_cmd_cycle.py's own load_sequence for that, unused here
# on purpose).
TARGET_PORT = 'usb3'


def run_cycle2(arm, name=TARGET_PORT):
    # name is assumed already grasped (gripper closed) before this runs --
    # not checked here, there is no gripper feedback to check it against.
    if not arm.set_target_port(name):
        arm.get_logger().error(f'no confirmed reading for {name} -- aborting before moving')
        return False

    if not square_within(arm):
        return False

    arm.freeze(True)
    home_pose = list(arm.current_positions)
    arm.get_logger().info(
        f'save0: frame frozen, home pose recorded -> {[round(v, 4) for v in home_pose]}')

    if not approach_target(arm):
        arm.get_logger().error(f'could not reach the standoff over {name} -- aborting')
        return False
    e1 = list(arm.current_positions)
    arm.get_logger().info(f'{name}: standoff reached, recorded as e1 -> {[round(v, 4) for v in e1]}')

    arm.get_logger().info(f'cycle2 complete -- holding at e1 over {name}, stopping here')
    return True


def main(args=None):
    rclpy.init(args=args)
    arm = ArmCmd()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(arm)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    time.sleep(1.0)  # let feedback_states populate before the first read

    arm.get_logger().info(f'starting cycle2 on {TARGET_PORT!r}')
    try:
        ok = run_cycle2(arm)
    finally:
        arm.freeze(False)
    arm.get_logger().info('cycle2 finished OK' if ok else 'cycle2 ABORTED (see errors above)')

    executor.shutdown()
    spin_thread.join(timeout=2.0)
    arm.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
