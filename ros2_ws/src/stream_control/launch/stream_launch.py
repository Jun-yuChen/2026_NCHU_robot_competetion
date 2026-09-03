from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
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
                'stall_timeout_s': 1.0,

                # Safty limits
                'translation_speed_limit_mps': 0.06,
                'rotation_speed_limit_dp': 10.0,
                'translation_step_limit_m': 0.0005,  # Effectively 0.05 m/s (100 Hz control frequency so *100)
                'rotation_step_limit_deg': 0.1,
            }],
        ),
    ])
