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
            'arm_script = py_gripper.arm_script:main',
            'arm = py_gripper.arm:main',
            'joy = py_gripper.joy:main',
            'arm_feedback_states = py_gripper.arm_feedback_states:main',
            'arm_cmd_port = py_gripper.arm_cmd_port:main',
        ],
    },
)
