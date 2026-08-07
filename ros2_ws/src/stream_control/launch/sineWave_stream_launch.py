from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='stream_control',
            executable='trajectory_generator_node',
            name='pvt_trajectory_generator',
            output='screen',
            parameters=[{
                'ctrl_hz': 100.0,
                'duration_s': 10.0,
                'pvt_point_time_ratio': 0.9,
                'amp_m': 0.05,
                'period_s': 5.0,
                'phase_rad': 0.0,
                'command_service': 'pvt_command',
                'service_wait_log_period_s': 2.0,
            }],
        ),
        Node(
            package='stream_control',
            executable='stream_control_node',
            name='pvt_stream_control',
            output='screen',
            parameters=[{
                'command_service': 'pvt_command',
                'obs_hz': 1000.0,
                'obs_log_period_s': 0.2,
                'settle_vz_thr': 0.002,
                'settle_hold_s': 0.6,
                'settle_timeout_s': 12.0,
                'max_inflight': 5,
                'ack_over_period_ratio': 1.0,
                'overload_consecutive': 3,
                'output_dir': 'pvt_out',
                'output_prefix': 'pvt_split',
                'cmd_history_window_s': 2.0,
            }],
        ),
    ])
