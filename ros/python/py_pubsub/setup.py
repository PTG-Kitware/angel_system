from setuptools import setup

package_name = 'py_pubsub'

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
    maintainer='josh.anderson',
    maintainer_email='josh.anderson@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
            'console_scripts': [
                'talker = py_pubsub.publisher_member_function:main',
                'video_listener = py_pubsub.video_subscriber:main',
                'audio_listener = py_pubsub.audio_subscriber:main',
                'frame_publisher = py_pubsub.ros_publisher:main',
                'generate_images = py_pubsub.generate_images:main',
                'object_detector = py_pubsub.object_detector:main',
                'spatial_mapper = py_pubsub.spatial_mapper:main',
                'activity_detector = py_pubsub.activity_detector:main',
            ],
    },
)
