from tm_msgs.srv import SetPositions,SetEvent,SendScript,SetIO
from tm_msgs.msg import FeedbackState
from robotiq_85_msgs.msg import GripperCmd
from py_gripper_interfaces.srv import Trajectory

from py_gripper_interfaces.msg import TrajState


from sensor_msgs.msg import Image

from std_msgs.msg import Float64MultiArray

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import time
import numpy as np
import queue

from scipy.spatial.transform import Rotation as R
import math

class Arm(Node):
    def __init__(self):
        super().__init__('arm')
        self.project_speed = 100
        self.delay_queue_size = 25
        
        self.color_queue = queue.Queue(2)
        self.depth_queue = queue.Queue(2)

        self.color_img_sub = self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, qos_profile_sensor_data)
        self.depth_img_sub = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, qos_profile_sensor_data)
        self.traj_state_pub = self.create_publisher(TrajState, '/traj_state', 10)
        self.gripList = []
        self.i = 0
        self.create_timer(0.01, self.run)

        self.create_service(Trajectory, 'trajectory', self.trajectory_callback)
        print('init done')
    
    def color_callback(self,msg):
        if self.color_queue.full():
            self.color_queue.get()
        self.color_queue.put(msg)

    def depth_callback(self,msg):
        if self.depth_queue.full():
            self.depth_queue.get()
        self.depth_queue.put(msg)
    
    def trajectory_callback(self,req:Trajectory.Request,res:Trajectory.Response):
        self.joy_queue.put(req)
        res.ok = True
        # self.get_logger().info("Get Trajectory Request: %s" % req)
        return res
    
    def pubTrajState(self,idx):
        msg = TrajState()
        msg.idx = idx
        self.traj_state_pub.publish(msg)
    
    def run(self):
        self.pubTrajState(0)
        self.get_logger().info(f"Publish TrajState: 0")
    
def main(args=None):
    rclpy.init(args=args)
    arm = Arm()
    rclpy.spin(arm)
    rclpy.shutdown()

if __name__ == '__main__':
    main()