from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tm_driver',
            executable='tm_driver',
            name='tm_driver_node',
            arguments=['robot_ip:=192.168.10.31']
        ),
        
        # Node(
        #     package='py_gripper',
        #     executable='arm_cmd',
        #     name='arm_cmd'
        # ),
    ])



