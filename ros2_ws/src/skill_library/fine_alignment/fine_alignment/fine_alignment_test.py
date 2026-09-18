import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_interface.action import FineAlignment


def feedback_cb(node):
    def _cb(fb):
        node.get_logger().info(
            f"[fine_alignment] state={fb.feedback.state}"
            f"progress={fb.feedback.progress:.2f} {fb.feedback.message}"
        )
    return _cb


def main():
    rclpy.init()
    node = Node('fine_alignment_test_client')
    client = ActionClient(node, FineAlignment, 'fine_alignment')

    if not client.wait_for_server(timeout_sec=5.0):
        node.get_logger().error('Action server not available')
        rclpy.shutdown()
        return

    goal_future = client.send_goal_async(
        FineAlignment.Goal(),
        feedback_callback=feedback_cb(node),
    )
    rclpy.spin_until_future_complete(node, goal_future)
    goal_handle = goal_future.result()

    if not goal_handle.accepted:
        node.get_logger().error('Goal rejected')
        rclpy.shutdown()
        return

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result()

    if result.result.status == FineAlignment.Result.SUCCESS:
        print("Action success")
        if not result.result.pose_valid:
            print("Fail to get final pose")
        else:
            x = result.result.x_m
            y = result.result.y_m
            z = result.result.z_m
            rx = result.result.rx_deg
            ry = result.result.ry_deg
            rz = result.result.rz_deg
            print(f"Final pose: {x}, {y}, {z}, {rx}, {ry}, {rz}")
    else:
        print("Action fail")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()