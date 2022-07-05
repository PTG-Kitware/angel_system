from setuptools import setup

package_name = 'angel_system_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kitware',
    maintainer_email='kitware@kitware.com',
    description='Contains python ROS nodes implementing the various Angel system nodes',
    license='BSD 3-Clause',
    tests_require=['pytest'],
    entry_points={
            'console_scripts': [
                'video_listener = angel_system_nodes.video_subscriber:main',
                'audio_player = angel_system_nodes.audio_player:main',
                'frame_publisher = angel_system_nodes.ros_publisher:main',
                'generate_images = angel_system_nodes.generate_images:main',
                'object_detector = angel_system_nodes.object_detector:main',
                'spatial_mapper = angel_system_nodes.spatial_mapper:main',
                'activity_detector = angel_system_nodes.activity_detector:main',
                'task_monitor = angel_system_nodes.task_monitor:main',
                'feedback_generator = angel_system_nodes.feedback_generator:main'
            ],
    },
)
