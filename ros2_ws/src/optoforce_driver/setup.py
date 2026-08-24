from setuptools import setup

package_name = 'optoforce_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='OptoForce HEX-70系列六軸力/扭矩感測器的ROS2驅動(非官方)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'optoforce_node = optoforce_driver.optoforce_node:main',
        ],
    },
)
