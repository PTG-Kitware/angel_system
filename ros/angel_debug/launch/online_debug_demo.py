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
        # Required arguments
        DeclareLaunchArgument(
            "hl2_ip",
            description="IP address of the Hololens2 device running the "
                        "sensor publishing package."
        ),
        # Arguments with defaults
        DeclareLaunchArgument(
            "ns", default_value="",
            description="Namespace nodes in this launch file should run under "
                        "(except rqt). This should always start with a slash "
                        "(/) unless no namespace is to be used."
        ),
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
        # Image feeder -- from
        Node(
            **node_param_common,
            package="cpp_pubsub",
            executable="talker",
            name="data_hub",
            parameters=[{
                "tcp_server_uri": LaunchConfiguration("hl2_ip"),
            }],
            remappings=[
                ("PVFrames", LaunchConfiguration("source_image_topic")),
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
        # This is down here because it doesn't seem to respect the namespace
        # push in the GroupAction.
        ExecuteProcess(
            name="rqt_view_debug",
            cmd=[
                "rqt", "-s", "rqt_image_view/ImageView", "--args",
                [LaunchConfiguration("ns"), "/", LaunchConfiguration("debug_image_topic")]
            ],
        )
    ])
