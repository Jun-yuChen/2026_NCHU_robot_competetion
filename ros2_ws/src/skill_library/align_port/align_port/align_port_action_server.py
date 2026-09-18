"""ROS2 action wrapper around arm_cmd_cycle2's run_cycle2 -- squares up,
approaches the standoff over arm_cmd_cycle2.TARGET_PORT (hardcoded to usb3
for now), and holds there. Meant to be the first of two actions an
upstream orchestrator calls in sequence -- the second, a separately-built
fine-insertion action, is not part of this repo.

    ros2 run align_port align_port_action_server
"""
import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from action_interface.action import AlignPort
from py_gripper.arm_cmd import ArmCmd
from py_gripper.arm_cmd_cycle2 import run_cycle2, TARGET_PORT


class AlignPortActionServer(ArmCmd):
    def __init__(self):
        super().__init__(node_name='align_port_action_server')
        # Reentrant, and paired with the MultiThreadedExecutor in main()
        # below -- run_cycle2 blocks on sleeps/polling for the whole
        # length of the goal, same as it does in arm_cmd_cycle2's own
        # plain script, and this node's other callbacks (feedback_states,
        # world_frame/object_pose) have to keep arriving throughout, since
        # run_cycle2 reads self.current_positions/latest_object_pose as it
        # goes. On the default (mutually-exclusive) callback group those
        # would queue up behind the running goal and never update.
        self._action_server = ActionServer(
            self, AlignPort, 'align_port', self._execute,
            callback_group=ReentrantCallbackGroup())

    def _execute(self, goal_handle):
        feedback = AlignPort.Feedback()

        def status(msg):
            feedback.status = msg
            goal_handle.publish_feedback(feedback)
            self.get_logger().info(msg)

        status(f"aligning to '{TARGET_PORT}'")
        try:
            ok = run_cycle2(self, TARGET_PORT)
        finally:
            # end0 -- whether the goal succeeded or aborted, nothing after
            # this should keep reading the frame it froze on.
            self.freeze(False)

        if not ok:
            goal_handle.abort()
            return AlignPort.Result(
                success=False, message=f'could not align to {TARGET_PORT} -- see log above')
        goal_handle.succeed()
        return AlignPort.Result(
            success=True, message=f'holding at standoff over {TARGET_PORT}')


def main(args=None):
    rclpy.init(args=args)
    node = AlignPortActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
