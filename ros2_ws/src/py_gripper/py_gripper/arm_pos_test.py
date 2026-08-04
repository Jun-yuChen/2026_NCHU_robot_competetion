import yaml
from tm_msgs.srv import SetPositions,SetEvent,SendScript
from tm_msgs.msg import FeedbackState

import rclpy
from rclpy.node import Node
import time
import numpy as np
import queue
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R
import math
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped,PoseStamped

class ArmPosStates(Node):
    def __init__(self):
        super().__init__('arm_position_states')
        self.pos_sub = self.create_subscription(FeedbackState, 'feedback_states', self.pos_callback, 10)
        self.get_logger().info("Start Arm Position States")

    def pos_callback(self,msg):
        T_B_t = TransformStamped()
        T_B_t.header.stamp = self.get_clock().now().to_msg()
        T_B_t.header.frame_id = 'base'
        T_B_t.child_frame_id = 'tool'
        x,y,z = msg.tool_pose[:3]
        q = R.from_euler('xyz', msg.tool_pose[3:], degrees=False).as_quat()
        T_B_t.transform.translation.x = x
        T_B_t.transform.translation.y = y
        T_B_t.transform.translation.z = z
        T_B_t.transform.rotation.x = q[0]
        T_B_t.transform.rotation.y = q[1]
        T_B_t.transform.rotation.z = q[2]
        T_B_t.transform.rotation.w = q[3]
        self.get_logger().info(f"tool  pose: {x}, {y}, {z}")
    

        T_B_t0 = TransformStamped()
        T_B_t0.header.stamp = self.get_clock().now().to_msg()
        T_B_t0.header.frame_id = 'base'
        T_B_t0.child_frame_id = 'tool'
        x,y,z = msg.tool_pose[:3]
        q = R.from_euler('xyz', msg.tool_pose[3:], degrees=False).as_quat()
        T_B_t0.transform.translation.x = x
        T_B_t0.transform.translation.y = y
        T_B_t0.transform.translation.z = z
        T_B_t0.transform.rotation.x = q[0]
        T_B_t0.transform.rotation.y = q[1]
        T_B_t0.transform.rotation.z = q[2]
        T_B_t0.transform.rotation.w = q[3]
        self.get_logger().info(f"tool0 pose: {x}, {y}, {z}")




    


    
def main(args=None):
    rclpy.init(args=args)
    arm = ArmPosStates()
    rclpy.spin(arm)
    rclpy.shutdown()