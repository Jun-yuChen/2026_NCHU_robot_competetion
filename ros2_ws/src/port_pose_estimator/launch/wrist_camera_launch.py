import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('port_pose_estimator'), 'config', 'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='port_pose_estimator',
            executable='wrist_camera_node',
            name='wrist_camera_node',
            output='screen',
            parameters=[params_file],
        ),

        Node(
                package='port_pose_estimator',
                executable='yolo_detector_node',
                name='yolo_detector_node',
                parameters=[params_file],
        ),
    ])
