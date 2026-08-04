import rclpy
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from custom_interface.msg import Detection3DArray
from tm_msgs.msg  import FeedbackState

from py_gripper.arm_cmd import ArmCmd


class ArmMoveNode(ArmCmd):
    """
    Extends ArmCmd with:
      - a subscription to yolo/detections_3d (highest-confidence detection -> p_C)
      - a subscription to feedback_states (already present in ArmCmd) used here
        to additionally maintain T_B_G, the live base->flange transform
      - T_G_C (camera->flange) and T_G_E (end-effector->flange) loaded from a
        yaml file (path given as a ROS 2 parameter)
      - pipeline to hover above the detected point:
          p_E               = T_E_G @ T_G_C @ p_C   (detection, in end-effector frame)
          hover_point_E     = p_E with z += hover_height (offset along the tool's own approach axis)
          hover_point_B     = T_B_G @ T_G_E @ hover_point_E
        then commands the arm to hover_point_B via set_position()
    """


    def __init__(self, node_name='arm_move_node'):
        super().__init__(node_name=node_name)  # sets up pos_cli, event_cli, io_cli, pos_sub, etc.

        #self.declare_parameter('robot_wrist_hand_eye_config_path', '/ros2_ws/src/configs/robot_config/ICA_Lab_UMI_Config_0803_1.yaml')
        self.declare_parameter('robot_wrist_hand_eye_config_path', '/path/to/hand/eye/calib.yaml')

        config_path = self.get_parameter('robot_wrist_hand_eye_config_path').get_parameter_value().string_value
        self.get_logger().info(f"config path: {config_path}")


        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        # Expected: 4x4 nested lists (meters), extrinsics expressed into the flange frame G
        self.T_G_C = np.array(config_data['T_G_C'], dtype=float)
        self.T_G_E = np.array(config_data['T_G_E'], dtype=float)
        self.T_E_G = np.array(config_data['T_E_G'], dtype=float)
        for name, T in (('T_G_C', self.T_G_C), ('T_G_E', self.T_G_E), ('T_E_G', self.T_E_G)):
            if T.shape != (4, 4):
                raise ValueError(f"{name} in {config_path} must be a 4x4 matrix, got shape {T.shape}")
 
        self.declare_parameter('hover_height', 0.2)
        self.hover_height = self.get_parameter('hover_height').get_parameter_value().double_value
        self.get_logger().info(f"Hover height: {self.hover_height}")

        self.T_B_G = np.eye(4)  # updated every time feedback_states arrives
        self.latest_detections = None

        self.det_sub = self.create_subscription(Detection3DArray, 'yolo/detections_3d', self.det_callback, 10)

    # --- feedback ---------------------------------------------------------
    def pos_callback(self, msg):
        # Keep ArmCmd's original behavior (updates self.current_positions)
        super().pos_callback(msg)

        x, y, z = msg.tool_pose[:3]  # tool_pose = ATC flange_pose
        quat = R.from_euler('xyz', msg.tool_pose[3:], degrees=False).as_quat()
        self.T_B_G = self._make_transform(x, y, z, quat)

    @staticmethod
    def _make_transform(x, y, z, quat_xyzw):
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        T[:3, 3] = [x, y, z]
        return T

    # --- detections ---------------------------------------------------------
    def det_callback(self, msg):
        self.latest_detections = msg

    def get_best_detection_p_C(self):
        """Returns the xyz position (camera optical frame, meters) of the
        highest-confidence detection, or None if there are no detections yet."""
        if self.latest_detections is None or len(self.latest_detections.detections) == 0:
            return None

        best_det = max(self.latest_detections.detections, key=lambda d: d.confidence)
        p = best_det.pose.position
        return np.array([p.x, p.y, p.z])

    # --- transform + command -------------------------------------------------
    def compute_hover_point_B(self, spin_timeout=0.5):
        """Spins briefly to pull in the freshest feedback_states + detections,
        then returns the hover position in base frame (xyz), or None."""
        rclpy.spin_once(self, timeout_sec=spin_timeout)
        rclpy.spin_once(self, timeout_sec=spin_timeout)  # one for each topic, in case they interleave
 
        p_C_xyz = self.get_best_detection_p_C()
        if p_C_xyz is None:
            self.get_logger().warn("No detections available yet")
            return None
 
        p_C = np.array([p_C_xyz[0], p_C_xyz[1], p_C_xyz[2], 1.0])
 
        # camera frame -> flange frame -> end-effector frame
        p_G = self.T_G_C @ p_C
        p_E = self.T_E_G @ p_G

        self.get_logger().info(f"p_C: {p_C}")
        self.get_logger().info(f"p_G: {p_G}")
        self.get_logger().info(f"p_E: {p_E}")
 
        # offset along the end-effector's own z-axis (its approach direction)
        hover_point_E = p_E.copy()
        hover_point_E[2] -= self.hover_height

        self.get_logger().info(f"hover_point_E: {hover_point_E[:3]}")
 
        # end-effector frame -> flange frame -> base frame (T_B_G is live, from feedback)
        hover_point_G = self.T_G_E @ hover_point_E
        self.get_logger().info(f"hover_point_G: {hover_point_G[:3]}")
        self.get_logger().info(f"T_B_G: \n{self.T_B_G}")

        hover_point_B = self.T_B_G @ hover_point_G
        return hover_point_B[:3]

    def move_to_best_detection(self, velocity=0.1, acc_time=0.5):
        """Computes the target in base frame, shows current vs. target position,
        waits for the user to confirm with 'a', then sends it to the arm."""

        target_xyz = self.compute_hover_point_B()
        if target_xyz is None:
            return None
 
        #orientation = list(self.target_positions[3:])  # keep current rx, ry, rz
        orientation = [3.14159, 0.0, 3.14] # Fix orientation to face downward

        positions = list(target_xyz) + orientation
 
        current_str = ", ".join(f"{v:.4f}" for v in self.current_positions)
        target_str = ", ".join(f"{v:.4f}" for v in positions)
        print(f"Current position (m, rad): [{current_str}]")
        print(f"Target  position (m, rad): [{target_str}]")
 
        key = input("Press 'a' to move, anything else to cancel: ").strip().lower()
        if key != 'a':
            print("Move cancelled.")
            return None
 
        return self.set_position(positions, velocity=velocity, acc_time=acc_time)



def main(args=None):
    rclpy.init(args=args)
    node = ArmMoveNode()

    try:
        while rclpy.ok():
            input("Press Enter to move to highest-confidence detection (Ctrl+C to quit)...")
            response = node.move_to_best_detection()
            node.get_logger().info("Response: %s" % response)
            node.wait_until_arrived()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()