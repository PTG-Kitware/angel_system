from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import GroupAction
from launch.actions import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace


def generate_launch_description():
    launch_arguments = [
        DeclareLaunchArgument("ns", default_value="offline_debug_demo"),
        DeclareLaunchArgument("source_image_topic", default_value="image_source"),
        DeclareLaunchArgument("detections_topic", default_value="detections"),
        DeclareLaunchArgument("debug_image_topic", default_value="image_debug/compressed"),
        DeclareLaunchArgument("threshold", default_value=""),
        DeclareLaunchArgument("use_gpu", default_value="true"),
    ]

    node_param_common = {
        "on_exit": Shutdown(),
    }
    node_list = [
        # Image feeder
        Node(
            **node_param_common,
            package="image_tools",
            executable="cam2image",
            name="image_feed",
            parameters=[{
                "width": 1920,
                "height": 1080,
                "frequency": 30.0,
            }],
            remappings=[
                ("image", LaunchConfiguration("source_image_topic"))
            ]
        ),
        # Object Detection
        Node(
            **node_param_common,
            package="py_pubsub",
            executable="object_detector",
            name="object_detector",
            parameters=[{
                "image_topic": LaunchConfiguration("source_image_topic"),
                "det_topic": LaunchConfiguration("detections_topic"),
                "use_cuda": LaunchConfiguration("use_gpu"),
            }]
        ),
        # Detection-on-image overlay
        Node(
            **node_param_common,
            package="py_pubsub",
            executable="object_detector_debug",
            name="object_detector_debug",
            parameters=[{
                "image_topic": LaunchConfiguration("source_image_topic"),
                "det_topic": LaunchConfiguration("detections_topic"),
                "out_image_topic": LaunchConfiguration("debug_image_topic"),
            }]
        ),
    ]
    grouped_actions = GroupAction(actions=[
        PushRosNamespace(LaunchConfiguration("ns")),
        *node_list,
    ])

    return LaunchDescription([
        *launch_arguments,
        grouped_actions,
        # Launch an RQT view to show the debug image stream
        ExecuteProcess(
            name="rqt_view_debug",
            cmd=[
                "rqt", "-s", "rqt_image_view/ImageView", "--args",
                ["/", LaunchConfiguration("ns"), "/", LaunchConfiguration("debug_image_topic")]
            ],
        )
    ])
