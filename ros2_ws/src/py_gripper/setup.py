from setuptools import setup
import os
from glob import glob

package_name = 'py_gripper'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))
            + glob(os.path.join('config', '*.json'))),
        (os.path.join('share', package_name, 'calibration'), glob(os.path.join('calibration', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hsun91chen',
    maintainer_email='hsun91chen@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'arm_cmd = py_gripper.arm_cmd:main',
            'arm_cmd_simple = py_gripper.arm_cmd_simple:main',
            'arm_cmd_cycle = py_gripper.arm_cmd_cycle:main',
            'find_computer_target= py_gripper.find_computer_target:main',
            'find_computer_targetnew= py_gripper.find_computer_targetnew:main',
            'arm_cmd_cycle2 = py_gripper.arm_cmd_cycle2:main',
            'align_port_action_server = py_gripper.align_port_action_server:main',
            'arm_script = py_gripper.arm_script:main',
            'arm = py_gripper.arm:main',
            'joy = py_gripper.joy:main',
            'arm_feedback_states = py_gripper.arm_feedback_states:main',
            'arm_cmd_port = py_gripper.arm_cmd_port:main',
            'fp_pose_bridge = py_gripper.fp_pose_bridge:main',
            'depth_pose_node = py_gripper.depth_pose_node:main',
            'arm_goto_fixed_table = py_gripper.arm_goto_fixed_table:main',
            'tag_navigator = py_gripper.tag_navigator:main',
            'arm_unplug_host_ports = py_gripper.arm_unplug_host_ports:main',
            'verify_euler_convention = py_gripper.verify_euler_convention:main',
        ],
    },
)