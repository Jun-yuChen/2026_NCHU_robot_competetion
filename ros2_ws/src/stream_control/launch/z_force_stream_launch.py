from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package = 'stream_control',
            executable='z_force_controller_node',
            name = 'z_force_controller',
            output='screen',
            parameters=[{
                'PI_controller_Kp': 2e-4,
                'PI_controller_Ki': 1e-4,
                'PI_controller_integral_limit': 0.05,
                'ctrl_hz': 100.0,
                'duration_s': 120.0,
                'pvt_point_time_ratio': 0.9,
                'command_service': 'pvt_command',
                'service_wait_log_period_s': 2.0,
            }],
        ),
    ])
