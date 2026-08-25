from setuptools import setup
import os
from glob import glob

package_name = 'port_pose_estimator'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hsun91chen',
    maintainer_email='hsun91chen@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'fp_pose_bridge = port_pose_estimator.fp_pose_bridge:main',
            'depth_pose_node = port_pose_estimator.depth_pose_node:main',
        ],
    },
)
