"""Bring-up for the depth-only pose route (no FoundationPose).

Deliberately a separate file from tmr_grip_launch.py rather than a flag on it,
because the two routes publish the same topics and must never run together --
having them in one launch invites exactly that mistake.

Differences from the FoundationPose bring-up:
  * align_depth is on. The hand-eye calibration is expressed against
    camera_color_optical_frame, so the pose has to be computed on depth aligned
    to colour or the depth-to-colour extrinsic shows up as a fixed offset.
  * the camera node stays running. FoundationPose needs the device to itself and
    the notes say to pkill it; this route consumes its topics instead.
  * fp_pose_bridge is not started.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    part = LaunchConfiguration('part')
    method = LaunchConfiguration('method')

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'),
                         'launch', 'rs_launch.py')),
        # 1280x720 rather than the 848x480 default. Measured on this rig at
        # ~18cm, the RJ45 mouth spans only 30x23 pixels and the latch slot that
        # tells the two 180-deg headings apart is a 13-vs-23 pixel difference --
        # too fine to survive depth noise. Every pixel counts here.
        # On the D405 both streams come off the one Stereo Module, so colour is
        # configured through depth_module.color_profile -- not rgb_camera.* ,
        # which is simply unset on this device. Both have to be raised together
        # or the driver falls back to the default, and since the pose is computed
        # on aligned_depth_to_color it is the *colour* profile that sets the
        # resolution the algorithm actually sees.
        launch_arguments={
            'align_depth.enable': 'true',
            'depth_module.depth_profile': '1280x720x30',
            'depth_module.color_profile': '1280x720x30',
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'part', default_value='rj45_test',
            description='key in config/opening_reference.json to track'),
        DeclareLaunchArgument(
            'method', default_value='mono',
            description="'mono' solves the pose from colour alone against the "
                        "CAD port table; 'depth' is the original measure-first "
                        "route, which is all a part without a port table can use"),

        realsense_launch,

        Node(
            package='port_pose_estimator', executable='depth_pose_node',
            name='depth_pose_node',
            parameters=[{'part': part, 'method': method}]
        ),
    ])
