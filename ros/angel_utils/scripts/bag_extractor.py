#!/usr/bin/env python3
"""
Convert ROS bags containing Hololens 2 data to other suitable file formats.

This script should be run within a ROS environment, specifically one with
rosbag2_py.
"""
import json
import os
from typing import Dict, Optional

import cv2
import numpy as np
import scipy.io.wavfile

import rclpy.logging
import rclpy.parameter
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

from angel_msgs.msg import (
    AnnotationEvent,
    EyeGazeData,
    HandJointPosesUpdate,
    HeadsetAudioData,
    HeadsetPoseData,
    SpatialMesh,
    ActivityDetection,
    TaskUpdate,
)
from sensor_msgs.msg import Image
from angel_utils.conversion import convert_nv12_to_rgb

from cv_bridge import CvBridge

# Instantiate CvBridge
bridge = CvBridge()


def get_rosbag_options(path, serialization_format="cdr"):
    """
    Helper function taken from rosbag2_py repo:
    https://github.com/ros2/rosbag2/blob/master/rosbag2_py/test/common.py
    """
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3")

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )

    return storage_options, converter_options


class BagConverter(Node):
    """
    Manages the exploding of ROS2 bags.
    """

    def __init__(self):
        """
        Performs basic initialization and then parses the bag.
        """
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        self.bag_path = (
            self.declare_parameter("bag_path", "").get_parameter_value().string_value
        )
        self.extract_audio = (
            self.declare_parameter("extract_audio", True)
            .get_parameter_value()
            .bool_value
        )
        self.extract_images = (
            self.declare_parameter("extract_images", True)
            .get_parameter_value()
            .bool_value
        )
        self.extract_eye_gaze_data = (
            self.declare_parameter("extract_eye_gaze_data", True)
            .get_parameter_value()
            .bool_value
        )
        self.extract_head_pose_data = (
            self.declare_parameter("extract_head_pose_data", True)
            .get_parameter_value()
            .bool_value
        )
        self.extract_hand_pose_data = (
            self.declare_parameter("extract_hand_pose_data", True)
            .get_parameter_value()
            .bool_value
        )
        self.extract_spatial_map_data = (
            self.declare_parameter("extract_spatial_map_data", True)
            .get_parameter_value()
            .bool_value
        )
        self.extract_annotation_event_data = (
            self.declare_parameter("extract_annotation_event_data", True)
            .get_parameter_value()
            .bool_value
        )
        self.extract_activity_detection_data = (
            self.declare_parameter("extract_activity_detection_data", True)
            .get_parameter_value()
            .bool_value
        )
        self.extract_task_update_data = (
            self.declare_parameter("extract_task_update_data", True)
            .get_parameter_value()
            .bool_value
        )
        self.extract_depth_images = (
            self.declare_parameter("extract_depth_images", True)
            .get_parameter_value()
            .bool_value
        )
        self.extract_depth_head_pose_data = (
            self.declare_parameter("extract_depth_head_pose_data", True)
            .get_parameter_value()
            .bool_value
        )
        self.depth_image_frame_id = (
            self.declare_parameter("depth_image_frame_id", "shortThrowDepthMap")
            .get_parameter_value()
            .string_value
        )
        self.pv_image_frame_id = (
            self.declare_parameter("pv_image_frame_id", "/kitware/pv_image_detections_2d")#"PVFramesBGR")
            .get_parameter_value()
            .string_value
        )

        if self.bag_path == "":
            self.log.info("Please provide bag file to convert")
            self.log.info(
                "Usage: ros2 run angel_utils bag_extractor.py --ros-args -p bag_path:=`bag_name`"
            )
            raise ValueError("Bag file path not provided")

        # Build the message type to handler function map
        self.msg_type_to_handler_map = {}
        if self.extract_audio:
            self.msg_type_to_handler_map[HeadsetAudioData] = self.handle_audio_msg
        self.log.info(f"---------- images: {self.extract_images}")
        #if self.extract_images or self.extract_depth_images:

        self.msg_type_to_handler_map[Image] = self.handle_image_msg

        if self.extract_eye_gaze_data:
            self.msg_type_to_handler_map[EyeGazeData] = self.handle_eye_gaze_msg
        if self.extract_head_pose_data or self.extract_depth_head_pose_data:
            self.msg_type_to_handler_map[HeadsetPoseData] = self.handle_head_pose_msg
        if self.extract_hand_pose_data:
            self.msg_type_to_handler_map[
                HandJointPosesUpdate
            ] = self.handle_hand_pose_msg
        if self.extract_spatial_map_data:
            self.msg_type_to_handler_map[SpatialMesh] = self.handle_spatial_mesh_msg
        if self.extract_annotation_event_data:
            self.msg_type_to_handler_map[
                AnnotationEvent
            ] = self.handle_annotation_event_msg
        if self.extract_activity_detection_data:
            self.msg_type_to_handler_map[
                ActivityDetection
            ] = self.handle_activity_detection_msg
        if self.extract_task_update_data:
            self.msg_type_to_handler_map[TaskUpdate] = self.handle_task_update_msg

        # Top level data folder
        self.num_total_msgs = 0
        self.data_folder = self.bag_path + "_extracted/"
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            self.log.info(f"Created {self.data_folder} for data")

        # For extracking audio data
        self.num_audio_msgs = 0
        self.audio_data = []
        self.min_audio_stamp = None
        self.max_audio_stamp = None

        # For extracting images
        self.num_image_msgs = 0
        self.image_folder = self.data_folder + "images/"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
            self.log.info(f"Created {self.image_folder} for extracted images")

        # For extracting eye gaze data
        self.eye_gaze_msgs = []
        self.num_eye_gaze_msgs = 0

        # For extracting hand pose data
        self.hand_pose_msgs = []
        self.num_hand_pose_msgs = 0

        # For extracting head pose data
        self.head_pose_msgs = []
        self.num_head_pose_msgs = 0

        # For extracting spatial map data
        self.spatial_map_msgs = []
        self.num_spatial_map_msgs = 0

        # For extracting spatial map data
        self.annotation_event_msgs = []
        self.num_annotation_event_msgs = 0

        # For activity detection data
        self.activity_detection_msgs = []
        self.num_activity_detection_msgs = 0

        # For task update data
        self.task_update_msgs = []
        self.num_task_update_msgs = 0

        # For extracting depth images
        self.num_depth_image_msgs = 0
        self.depth_image_folder = self.data_folder + "depth_images/"
        if not os.path.exists(self.depth_image_folder):
            os.makedirs(self.depth_image_folder)
            self.log.info(f"Created {self.depth_image_folder} for extracted images")

        # For extracting depth head pose data
        self.depth_head_pose_msgs = []
        self.num_depth_head_pose_msgs = 0

        # Parse the bag
        self.parse_bag()

    def parse_bag(self) -> None:
        """
        Read the ROS2 bag, deserialize the messages, and extract the relevant
        data.

        TODO: add function to check that messages arrive in order
        """
        self.log.info(f"Reading bag: {self.bag_path}")
        storage_options, converter_options = get_rosbag_options(self.bag_path)

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()

        # Create a map for quicker lookup
        type_map = {
            topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
        }

        # Loop through the bag until there are no more messages
        while reader.has_next():
            (topic, data, _) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            self.num_total_msgs += 1

            if (self.num_total_msgs % 100) == 0:
                self.log.info(f"Parsing message: {self.num_total_msgs} type: {type(msg)}")

            # Attempt to call the message handler for this message type
            try:
                self.msg_type_to_handler_map[type(msg)](msg)
            except KeyError as e:
                pass

        # Print bag stats
        self.print_bag_info()

        # Save the audio file
        if self.extract_audio:
            self.create_wav_file()
            audio_stamp_file = self.data_folder + "audio_time_data.json"
            stamps = {
                "start_time": self.min_audio_stamp,
                "end_time": self.max_audio_stamp,
            }
            with open(audio_stamp_file, mode="w", encoding="utf-8") as f:
                json.dump(stamps, f)
            self.log.info(f"Created audio timestamp file: {audio_stamp_file}")
        else:
            self.log.info(f"Skipping audio extraction")

        # Write the json data to disk
        if self.extract_eye_gaze_data:
            eye_gaze_file = self.data_folder + "eye_gaze_data.json"
            with open(eye_gaze_file, mode="w", encoding="utf-8") as f:
                json.dump(self.eye_gaze_msgs, f)
            self.log.info(f"Created eye gaze file: {eye_gaze_file}")
        else:
            self.log.info(f"Skipping eye gaze file creation")

        if self.extract_hand_pose_data:
            hand_pose_file = self.data_folder + "hand_pose_data.json"
            with open(hand_pose_file, mode="w", encoding="utf-8") as f:
                json.dump(self.hand_pose_msgs, f)
            self.log.info(f"Created hand pose file: {hand_pose_file}")
        else:
            self.log.info(f"Skipping hand pose file creation")

        if self.extract_head_pose_data:
            head_pose_file = self.data_folder + "head_pose_data.json"
            with open(head_pose_file, mode="w", encoding="utf-8") as f:
                json.dump(self.head_pose_msgs, f)
            self.log.info(f"Created head pose file: {head_pose_file}")
        else:
            self.log.info(f"Skipping head pose file creation")

        if self.extract_spatial_map_data:
            spatial_map_file = self.data_folder + "spatial_map_data.json"
            with open(spatial_map_file, mode="w", encoding="utf-8") as f:
                json.dump(self.spatial_map_msgs, f)
            self.log.info(f"Created spatial map file: {spatial_map_file}")
        else:
            self.log.info(f"Skipping spatial map file creation")

        if self.extract_annotation_event_data:
            annotation_event_file = self.data_folder + "annotation_event_data.json"
            with open(annotation_event_file, mode="w", encoding="utf-8") as f:
                json.dump(self.annotation_event_msgs, f)
            self.log.info(f"Created annotation event file: {annotation_event_file}")
        else:
            self.log.info(f"Skipping annotation event file creation")

        if self.extract_activity_detection_data:
            activity_detection_file = self.data_folder + "activity_detection_data.json"
            with open(activity_detection_file, mode="w", encoding="utf-8") as f:
                json.dump(self.activity_detection_msgs, f)
            self.log.info(f"Created activity detection file: {activity_detection_file}")
        else:
            self.log.info(f"Skipping activity detection file creation")

        if self.extract_task_update_data:
            task_update_file = self.data_folder + "task_update_data.json"
            with open(task_update_file, mode="w", encoding="utf-8") as f:
                json.dump(self.task_update_msgs, f)
            self.log.info(f"Created task update file: {task_update_file}")
        else:
            self.log.info(f"Skipping task_update file creation")

        if self.extract_depth_head_pose_data:
            depth_head_pose_file = self.data_folder + "depth_head_pose_data.json"
            with open(depth_head_pose_file, mode="w", encoding="utf-8") as f:
                json.dump(self.depth_head_pose_msgs, f)
            self.log.info(f"Created depth head pose file: {depth_head_pose_file}")
        else:
            self.log.info(f"Skipping depth head pose file creation")

    def print_bag_info(self) -> None:
        """
        Print stats about the bag.
        """
        self.log.info(f"Bag info for {self.bag_path}:")
        self.log.info(f"Total ROS messages: {self.num_total_msgs}")
        self.log.info(f"Audio messages: {self.num_audio_msgs}")
        self.log.info(f"Eye gaze messages: {self.num_eye_gaze_msgs}")
        self.log.info(f"Hand pose messages: {self.num_hand_pose_msgs}")
        self.log.info(f"Head pose messages: {self.num_head_pose_msgs}")
        self.log.info(f"Image messages: {self.num_image_msgs}")
        self.log.info(f"Spatial map messages: {self.num_spatial_map_msgs}")
        self.log.info(f"Annotation event messages: {self.num_annotation_event_msgs}")
        self.log.info(
            f"Activity detection messages: {self.num_activity_detection_msgs}"
        )
        self.log.info(f"Task update messages: {self.num_task_update_msgs}")
        self.log.info(f"Depth image messages: {self.num_depth_image_msgs}")
        self.log.info(f"Depth head pose messages: {self.num_depth_head_pose_msgs}")

    def create_wav_file(self, filename: Optional[str] = None) -> None:
        """
        Creates a .wav file from the bag's audio data topic messages.
        """
        if filename is None:
            filename = self.data_folder + "audio.wav"

        if len(self.audio_data) == 0:
            self.log.info("No audio data found in bag")
            return

        # Split audio data into two channels
        # HL audio samples are interleaved like [CH1, CH2, CH1, CH2,...]
        audio_ch1 = np.array(self.audio_data)[::2]
        audio_ch2 = np.array(self.audio_data)[1::2]

        audio_both_channels = np.stack((audio_ch1, audio_ch2), axis=1)
        scipy.io.wavfile.write(filename, self.audio_sample_rate, audio_both_channels)

        self.log.info(f"Created audio file: {filename}")

    def convert_eye_gaze_msg_to_dict(self, msg) -> Dict:
        """
        Converts an eye gaze ROS message object to a dictionary.
        """
        d = {
            "time_sec": msg.header.stamp.sec,
            "time_nanosec": msg.header.stamp.nanosec,
            "gaze_origin": [msg.gaze_origin.x, msg.gaze_origin.y, msg.gaze_origin.z],
            "gaze_direction": [
                msg.gaze_direction.x,
                msg.gaze_direction.y,
                msg.gaze_direction.z,
            ],
            "head_movement_direction": [
                msg.head_movement_direction.x,
                msg.head_movement_direction.y,
                msg.head_movement_direction.z,
            ],
            "head_velocity": [
                msg.head_velocity.x,
                msg.head_velocity.y,
                msg.head_velocity.z,
            ],
            "is_object_hit": msg.is_object_hit,
            "hit_object_position": [
                msg.hit_object_position.x,
                msg.hit_object_position.y,
                msg.hit_object_position.z,
            ],
        }

        return d

    def convert_hand_pose_msg_to_dict(self, msg) -> Dict:
        """
        Converts a hand pose ROS message object to a dictionary.
        """
        d = {
            "time_sec": msg.header.stamp.sec,
            "time_nanosec": msg.header.stamp.nanosec,
            "hand": msg.hand,
            "joint_poses": [
                {
                    "joint": m.joint,
                    "position": [
                        m.pose.position.x,
                        m.pose.position.y,
                        m.pose.position.z,
                    ],
                    "rotation_xyzw": [
                        m.pose.orientation.x,
                        m.pose.orientation.y,
                        m.pose.orientation.z,
                        m.pose.orientation.w,
                    ],
                }
                for m in msg.joints
            ],
        }
        return d

    def convert_head_pose_msg_to_dict(self, msg) -> Dict:
        """
        Converts a head pose ROS message object to a dictionary.
        """
        d = {
            "time_sec": msg.header.stamp.sec,
            "time_nanosec": msg.header.stamp.nanosec,
            "world_matrix": list(msg.world_matrix),
            "projection_matrix": list(msg.projection_matrix),
        }
        return d

    def convert_spatial_map_msg_to_dict(self, msg) -> Dict:
        """
        Converts a head pose ROS message object to a dictionary.

        # TODO: should we add a std_msgs/Header to this message?
        """
        d = {
            "mesh_id": msg.mesh_id,
            "removal": msg.removal,
            "triangles": [
                [
                    int(t.vertex_indices[0]),
                    int(t.vertex_indices[1]),
                    int(t.vertex_indices[2]),
                ]
                for t in msg.mesh.triangles
            ],
            "vertices": [[v.x, v.y, v.z] for v in msg.mesh.vertices],
        }
        return d

    def convert_annotation_event_msg_to_dict(self, msg) -> Dict:
        """
        Converts a AnnotationEvent ROS message object to a dictionary.
        """
        d = {
            "time_sec": msg.header.stamp.sec,
            "time_nanosec": msg.header.stamp.nanosec,
            "description": msg.description,
        }
        return d

    def convert_activity_detection_msg_to_dict(self, msg) -> Dict:
        """
        Converts a ActivityDetection ROS message object to a dictionary.
        """
        d = {
            "header": {
                "time_sec": msg.header.stamp.sec,
                "time_nanosec": msg.header.stamp.nanosec,
                "frame_id": msg.header.frame_id,
            },
            "source_stamp_start_frame": msg.source_stamp_start_frame.sec
            + (msg.source_stamp_start_frame.nanosec * 1e-9),
            "source_stamp_end_frame": msg.source_stamp_end_frame.sec
            + (msg.source_stamp_end_frame.nanosec * 1e-9),
            "label_vec": list(msg.label_vec),
            "conf_vec": list(msg.conf_vec),
        }
        return d

    def convert_task_update_msg_to_dict(self, msg) -> Dict:
        """
        Converts a TaskUpdate ROS message object to a dictionary.
        """
        d = {
            "header": {
                "time_sec": msg.header.stamp.sec,
                "time_nanosec": msg.header.stamp.nanosec,
                "frame_id": msg.header.frame_id,
            },
            "task_name": msg.task_name,
            "task_description": msg.task_description,
            "latest_sensor_input_time": msg.latest_sensor_input_time.sec
            + (msg.latest_sensor_input_time.nanosec * 1e-9),
            "current_step_id": msg.current_step_id,
            "current_step": msg.current_step,
            "previous_step": msg.previous_step,
            "task_complete_confidence": msg.task_complete_confidence,
            "completed_steps": list(msg.completed_steps),
            "hmm_step_confidence": list(msg.hmm_step_confidence),
        }
        return d

    def handle_audio_msg(self, msg: HeadsetAudioData) -> None:
        """
        Handler for the audio messages in the ROS bag.
        Adds the audio data to the audio_data list.
        """
        time = (msg.header.stamp.sec + msg.header.stamp.nanosec * 10e-9,)
        if self.num_audio_msgs == 0:
            self.min_audio_stamp = time
            self.max_audio_stamp = time
        else:
            self.max_audio_stamp = max(time, self.max_audio_stamp)

        self.num_audio_msgs += 1
        self.audio_sample_rate = msg.sample_rate

        # Accumulate the audio data
        self.audio_data.extend(msg.data)

    def handle_image_msg(self, msg: Image) -> None:
        """
        Handler for the image messages in the ROS bag.
        Converts the PV images to RGB and saves them to disk.
        Converts the depth images to 16-bit grayscale and saves them to disk.
        """
        print(f"Image message received: {msg.encoding}")
        if msg.encoding in ["nv12", "rgb8", "rgba8", "bgr8"]: # and self.extract_images:
            self.num_image_msgs += 1

            if msg.encoding == "nv12":
                # Convert NV12 image to RGB
                rgb_image = convert_nv12_to_rgb(msg.data, msg.height, msg.width)
            else:
                rgb_image = bridge.imgmsg_to_cv2(msg, msg.encoding)

            if "rgb" in msg.encoding:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGBA)

            # Save image to disk
            timestamp_str = (
                f"{msg.header.stamp.sec:011d}_{msg.header.stamp.nanosec:09d}"
            )
            file_name = f"{self.image_folder}frame_{self.num_image_msgs:05d}_{timestamp_str}.png"
            cv2.imwrite(file_name, rgb_image)
        elif msg.encoding == "mono16" and self.extract_depth_images:
            self.num_depth_image_msgs += 1

            # Convert to 1D uint16 array
            data_uint16 = np.array(msg.data).view(np.uint16)

            # Convert to grayscale image
            depth_image = np.reshape(data_uint16, (msg.width, msg.height, 1))

            # Save image to disk
            timestamp_str = (
                f"{msg.header.stamp.sec:011d}_{msg.header.stamp.nanosec:09d}"
            )
            file_name = f"{self.depth_image_folder}depth_frame_{self.num_depth_image_msgs:05d}_{timestamp_str}.png"
            cv2.imwrite(file_name, depth_image)

    def handle_eye_gaze_msg(self, msg: EyeGazeData) -> None:
        """
        Handler for the eye gaze messages in the ROS bag.
        Converts the eye gaze data to a dictionary and then adds it to
        the eye gaze dictionary list.
        """
        self.num_eye_gaze_msgs += 1

        msg_dict = self.convert_eye_gaze_msg_to_dict(msg)
        self.eye_gaze_msgs.append(msg_dict)

    def handle_head_pose_msg(self, msg: HeadsetPoseData) -> None:
        """
        Handler for the head pose messages in the ROS bag.
        Converts the head pose data to a dictionary and then adds it to
        the head pose dictionary list for the corresponding frame type.
        """
        msg_dict = self.convert_head_pose_msg_to_dict(msg)

        # Check message header to see which frame this head pose is for
        if (
            msg.header.frame_id == self.pv_image_frame_id
            and self.extract_head_pose_data
        ):
            self.head_pose_msgs.append(msg_dict)
            self.num_head_pose_msgs += 1
        elif (
            msg.header.frame_id == self.depth_image_frame_id
            and self.extract_depth_head_pose_data
        ):
            self.depth_head_pose_msgs.append(msg_dict)
            self.num_depth_head_pose_msgs += 1

    def handle_hand_pose_msg(self, msg: HandJointPosesUpdate) -> None:
        """
        Handler for the hand pose messages in the ROS bag.
        Converts the hand pose data to a dictionary and then adds it to
        the hand pose dictionary list.
        """
        self.num_hand_pose_msgs += 1

        msg_dict = self.convert_hand_pose_msg_to_dict(msg)
        self.hand_pose_msgs.append(msg_dict)

    def handle_spatial_mesh_msg(self, msg: SpatialMesh) -> None:
        """
        Handler for the spatial mesh messages in the ROS bag.
        Converts the spatial mesh data to a dictionary and then adds it to
        the spatial mesh dictionary list.
        """
        self.num_spatial_map_msgs += 1

        msg_dict = self.convert_spatial_map_msg_to_dict(msg)
        self.spatial_map_msgs.append(msg_dict)

    def handle_annotation_event_msg(self, msg: AnnotationEvent) -> None:
        """
        Handler for the AnnotationEvent messages in the ROS bag.
        Converts the AnnotationEvent data to a dictionary and then adds it to
        the AnnotationEvent dictionary list.
        """
        self.num_annotation_event_msgs += 1

        msg_dict = self.convert_annotation_event_msg_to_dict(msg)
        self.annotation_event_msgs.append(msg_dict)

    def handle_activity_detection_msg(self, msg: ActivityDetection) -> None:
        """
        Handler for the activity detection messages in the ROS bag.
        Converts the ActivityDetection data to a dictionary and then adds it to
        the ActivityDetection dictionary list.
        """
        self.num_activity_detection_msgs += 1

        msg_dict = self.convert_activity_detection_msg_to_dict(msg)
        self.activity_detection_msgs.append(msg_dict)

    def handle_task_update_msg(self, msg: TaskUpdate):
        """
        Handler for the task update messages in the ROS bag.
        Converts the TaskUpdate data to a dictionary and then adds it to
        the TaskUpdate dictionary list.
        """
        self.num_task_update_msgs += 1

        msg_dict = self.convert_task_update_msg_to_dict(msg)
        self.task_update_msgs.append(msg_dict)


if __name__ == "__main__":
    rclpy.init()

    bag_converter = BagConverter()
