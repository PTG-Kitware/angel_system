from setuptools import setup, find_packages

package_name = "angel_system_nodes"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Kitware",
    maintainer_email="kitware@kitware.com",
    description="Contains python ROS nodes implementing the various Angel system nodes",
    license="BSD 3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "video_listener = angel_system_nodes.video_subscriber:main",
            "base_intent_detector = angel_system_nodes.base_intent_detector:main",
            "gpt_intent_detector = angel_system_nodes.gpt_intent_detector:main",
            "base_emotion_detector = angel_system_nodes.base_emotion_detector:main",
            "gpt_emotion_detector = angel_system_nodes.gpt_emotion_detector:main",
            "question_answerer = angel_system_nodes.question_answerer:main",
            "intent_detector = angel_system_nodes.intent_detector:main",
            "spatial_mapper = angel_system_nodes.spatial_mapper:main",
            "feedback_generator = angel_system_nodes.feedback_generator:main",
            "annotation_event_monitor = angel_system_nodes.annotation_event_monitor:main",
            "intent_to_command = angel_system_nodes.intent_to_command:main",
            # Data Publishers
            "frame_publisher = angel_system_nodes.data_publishers.ros_publisher:main",
            "generate_images = angel_system_nodes.data_publishers.generate_images:main",
            "hl2ss_ros_bridge = angel_system_nodes.data_publishers.hl2ss_ros_bridge:main",
            "image_timestamp_relay = angel_system_nodes.data_publishers.image_timestamp_relay:main",
            "redis_ros_bridge = angel_system_nodes.data_publishers.redis_ros_bridge:main",
            # Evaluation Components
            "eval_2_logger = angel_system_nodes.eval.mitll_eval_2_logger:main",
            # Object detection
            "berkeley_object_detector = angel_system_nodes.object_detection.berkeley_object_detector:main",
            "object_detector = angel_system_nodes.object_detection.object_detector:main",
            "object_detector_with_descriptors = angel_system_nodes.object_detection.object_detector_with_descriptors:main",
            "object_detector_with_descriptors_v2 = angel_system_nodes.object_detection.object_detector_with_descriptors_v2:main",
            "object_detection_yolo_v7 = angel_system_nodes.object_detection.yolov7_object_detector:main",
            "object_detection_filter = angel_system_nodes.object_detection.object_detection_filter:main",
            # Activity Classification
            "activity_classifier_tcn = angel_system_nodes.activity_classification.activity_classifier_tcn:main",
            "activity_detector = angel_system_nodes.activity_classification.activity_detector:main",
            "activity_from_obj_dets_classifier = angel_system_nodes.activity_classification.activity_from_obj_dets_classifier:main",
            "mm_activity_detector = angel_system_nodes.activity_classification.mm_activity_detector:main",
            "uho_activity_detector = angel_system_nodes.activity_classification.uho_activity_detector:main",
            # Audio related nodes
            "asr = angel_system_nodes.audio.asr:main",
            "audio_player = angel_system_nodes.audio.audio_player:main",
            "voice_activity_detector = angel_system_nodes.audio.voice_activity_detector:main",
            # Task Monitoring
            "berkeley_task_monitor = angel_system_nodes.task_monitoring.berkeley_task_monitor:main",
            "task_monitor_v1 = angel_system_nodes.task_monitoring.task_monitor_v1:main",
            "task_monitor_v2 = angel_system_nodes.task_monitoring.task_monitor_v2:main",
            "dummy_multi_task_monitor = angel_system_nodes.task_monitoring.dummy_multi_task_monitor:main",
            "global_step_predictor = angel_system_nodes.task_monitoring.global_step_predictor:main",
            "keyboard_to_sys_cmd = angel_system_nodes.task_monitoring.keyboard_to_sys_cmd:main",
            "transform_update_lod = angel_system_nodes.task_monitoring.transform_update_lod:main",
        ],
    },
)
