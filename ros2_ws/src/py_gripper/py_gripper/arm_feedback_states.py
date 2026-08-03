import yaml
from tm_msgs.srv import SetPositions,SetEvent,SendScript
from tm_msgs.msg import FeedbackState
from robotiq_85_msgs.msg import GripperCmd
from py_gripper_interfaces.srv import Trajectory

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import time
import numpy as np
import queue
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R
import math
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped,PoseStamped

# Load from YAML file
with open('/home/lab606/ICA_Lab_UMI/ICA_Lab_UMI_Config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)
    T_G_C = np.array(config_data['T_G_C'])
    T_C_G = np.array(config_data['T_C_G'])
    T_G_E = np.array(config_data['T_G_E'])
    T_E_G = np.array(config_data['T_E_G'])
    T_W_a = np.array(config_data['T_W_a'])
    T_a_A = np.array(config_data['T_a_A'])
    T_A_a = np.array(config_data['T_A_a'])
    T_E_C = T_E_G @ T_G_C
    T_C_E = np.linalg.inv(T_E_C)
    print("T_G_C: \n",np.array2string(T_G_C,separator=',',precision=6))


class ArmFeedbackStates(Node):
    def __init__(self):
        super().__init__('arm_feedback_states')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pos_sub = self.create_subscription(FeedbackState, 'feedback_states', self.pos_callback, 10)
        self.get_logger().info("Start Arm Feedback States")
        self.pub = self.create_publisher(Float64MultiArray, 'arm_pose', 10)


    def pos_callback(self,msg):
        d = Float64MultiArray()
        d.data = msg.tool_pose
        self.pub.publish(d)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'arm'
        x,y,z = msg.tool_pose[:3]
        q = R.from_euler('xyz', msg.tool_pose[3:], degrees=False).as_quat()
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'arm'
        t.child_frame_id = 'end_effector'
        t.transform.translation.x = T_G_E[0,3]
        t.transform.translation.y = T_G_E[1,3]
        t.transform.translation.z = T_G_E[2,3]
        rot = R.from_matrix(T_G_E[:3,:3]).as_quat()
        t.transform.rotation.x = rot[0]
        t.transform.rotation.y = rot[1]
        t.transform.rotation.z = rot[2]
        t.transform.rotation.w = rot[3]
        self.tf_broadcaster.sendTransform(t)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'end_effector'
        t.child_frame_id = 'camera'
        t.transform.translation.x = T_E_C[0,3]
        t.transform.translation.y = T_E_C[1,3]
        t.transform.translation.z = T_E_C[2,3]
        rot = R.from_matrix(T_E_C[:3,:3]).as_quat()
        t.transform.rotation.x = rot[0]
        t.transform.rotation.y = rot[1]
        t.transform.rotation.z = rot[2]
        t.transform.rotation.w = rot[3]
        self.tf_broadcaster.sendTransform(t)

        # t = TransformStamped()
        # t.header.stamp = self.get_clock().now().to_msg()
        # t.header.frame_id = 'arm'
        # t.child_frame_id = 'camera'
        # t.transform.translation.x = T_G_C[0,3]
        # t.transform.translation.y = T_G_C[1,3]
        # t.transform.translation.z = T_G_C[2,3]
        # rot = R.from_matrix(T_G_C[:3,:3]).as_quat()
        # t.transform.rotation.x = rot[0]
        # t.transform.rotation.y = rot[1]
        # t.transform.rotation.z = rot[2]
        # t.transform.rotation.w = rot[3]
        # self.tf_broadcaster.sendTransform(t)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'apriltag'
        t.transform.translation.x = T_W_a[0,3]
        t.transform.translation.y = T_W_a[1,3]
        t.transform.translation.z = T_W_a[2,3]
        rot = R.from_matrix(T_W_a[:3,:3]).as_quat()
        t.transform.rotation.x = rot[0]
        t.transform.rotation.y = rot[1]
        t.transform.rotation.z = rot[2]
        t.transform.rotation.w = rot[3]
        self.tf_broadcaster.sendTransform(t)

        # t = TransformStamped()
        # t.header.stamp = self.get_clock().now().to_msg()
        # t.header.frame_id = 'apriltag'
        # t.child_frame_id = 'camera'
        # T_a_C = T_a_W @ T_W_G @ T_G_C
        # t.transform.translation.x = T_a_C[0,3]
        # t.transform.translation.y = T_a_C[1,3]
        # t.transform.translation.z = T_a_C[2,3]
        # rot = R.from_matrix(T_a_C[:3,:3]).as_quat()
        # t.transform.rotation.x = rot[0]
        # t.transform.rotation.y = rot[1]
        # t.transform.rotation.z = rot[2]
        # t.transform.rotation.w = rot[3]
        # self.tf_broadcaster.sendTransform(t)


        # T_a_E = T_a_W @ T_W_G @ T_G_E
        # x,y,z = T_a_E[:3,3]
        # rot = R.from_matrix(T_a_E[:3,:3]).as_euler('xyz', degrees=True)
        # self.get_logger().info(f"Current Position: {x:.3f} {y:.3f} {z:.3f} {rot[0]:.3f} {rot[1]:.3f} {rot[2]:.3f}")


    


    
def main(args=None):
    rclpy.init(args=args)
    arm = ArmFeedbackStates()
    rclpy.spin(arm)
    rclpy.shutdown()