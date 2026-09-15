from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package = 'fine_alignment',
            executable='fine_alignment_server_node',
            name = 'fine_alignment_server',
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

                'spiral_pitch_mm': 3.0,
                'spiral_max_radius_mm': 50.0,
                'spiral_search_speed_mm_s': 30.0,

                'touch_force_tol_n': 0.3,

                'position_steady_state_tolerance_mm': 2.0,
                'hole_force_delta_N': 0.5,
                'hole_testing_threshold_mm': 3,
                'steady_state_window_s': 0.3,
                'align_force': 0.3,
                'insert_force': 6.0,
                'max_force_n': 12.0,

            }],
        ),
        
    ])