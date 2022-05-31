#!/usr/bin/env python3
"""
Convert ROS bags containing Hololens 2 data to other suitable file formats.

This script should be run within a ROS environment, specifically one with
rosbag2_py.
"""
import json
import os
import sys
from typing import Dict, Optional

from cv_bridge import CvBridge
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
    EyeGazeData,
    HandJointPosesUpdate,
    HeadsetAudioData,
    HeadsetPoseData,
    SpatialMesh,
)
from sensor_msgs.msg import Image


BRIDGE = CvBridge()


def get_rosbag_options(path, serialization_format='cdr'):
    """
    Helper function taken from rosbag2_py repo:
    https://github.com/ros2/rosbag2/blob/master/rosbag2_py/test/common.py
    """
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

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

        self.bag_path = self.declare_parameter("bag_path", "").get_parameter_value().string_value
        self.extract_audio = self.declare_parameter("extract_audio", True).get_parameter_value().bool_value
        self.extract_images = self.declare_parameter("extract_images", True).get_parameter_value().bool_value
        self.extract_eye_gaze_data = self.declare_parameter("extract_eye_gaze_data", True).get_parameter_value().bool_value
        self.extract_head_pose_data = self.declare_parameter("extract_head_pose_data", True).get_parameter_value().bool_value
        self.extract_hand_pose_data = self.declare_parameter("extract_hand_pose_data", True).get_parameter_value().bool_value
        self.extract_spatial_map_data = self.declare_parameter("extract_spatial_map_data", True).get_parameter_value().bool_value

        if self.bag_path == "":
            self.log.info("Please provide bag file to convert")
            self.log.info("Usage: ros2 run angel_utils bag_extractor.py --ros-args -p bag_path:=`bag_name`")
            raise ValueError("Bag file path not provided")
            return

        # Top level data folder
        self.num_total_msgs = 0
        self.data_folder = self.bag_path + "_extracted/"
        if not(os.path.exists(self.data_folder)):
            os.makedirs(self.data_folder)
            self.log.info(f"Created {self.data_folder} for data")

        # For extracking audio data
        self.num_audio_msgs = 0
        self.audio_data = []

        # For extracting images data
        self.num_image_msgs = 0
        self.images = []
        self.image_folder = self.data_folder + "images/"
        if not(os.path.exists(self.image_folder)):
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
        type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

        # Loop through the bag until there are no more messages
        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            self.num_total_msgs += 1

            if (self.num_total_msgs % 100) == 0:
                self.log.info(f"Parsing message: {self.num_total_msgs}")

            if isinstance(msg, EyeGazeData):
                self.num_eye_gaze_msgs += 1
                if not self.extract_eye_gaze_data:
                    continue

                msg_dict = self.convert_eye_gaze_msg_to_dict(msg)
                self.eye_gaze_msgs.append(msg_dict)
            elif isinstance(msg, HandJointPosesUpdate):
                self.num_hand_pose_msgs += 1
                if not self.extract_hand_pose_data:
                    continue

                msg_dict = self.convert_hand_pose_msg_to_dict(msg)
                self.hand_pose_msgs.append(msg_dict)
            elif isinstance(msg, HeadsetPoseData):
                self.num_head_pose_msgs += 1
                if not self.extract_head_pose_data:
                    continue

                msg_dict = self.convert_head_pose_msg_to_dict(msg)
                self.head_pose_msgs.append(msg_dict)
            elif isinstance(msg, HeadsetAudioData):
                self.num_audio_msgs += 1
                if not self.extract_audio:
                    continue

                self.audio_sample_rate = msg.sample_rate

                # Accumulate the audio data
                self.audio_data.extend(msg.data)
            elif isinstance(msg, Image):
                self.num_image_msgs += 1
                if not self.extract_images:
                    continue

                # Convert NV12 image to RGB
                yuv_image = np.frombuffer(msg.data, np.uint8).reshape(msg.height*3//2, msg.width)
                rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)

                # Save image to disk
                cv2.imwrite(f"{self.image_folder}frame_{self.num_image_msgs:05d}.png", rgb_image)
            elif isinstance(msg, SpatialMesh):
                self.num_spatial_map_msgs += 1
                if not self.extract_spatial_map_data:
                    continue

                msg_dict = self.convert_spatial_map_msg_to_dict(msg)
                self.spatial_map_msgs.append(msg_dict)

        # Print bag stats
        self.print_bag_info()

        # Save the audio file
        self.create_wav_file()

        # Write the json data to disk
        if self.extract_eye_gaze_data:
            eye_gaze_file = self.data_folder + "eye_gaze_data.txt"
            with open(eye_gaze_file, mode="w", encoding='utf-8') as f:
                json.dump(self.eye_gaze_msgs, f)
            self.log.info(f"Created eye gaze file: {eye_gaze_file}")
        else:
            self.log.info(f"Skipping eye gaze file creation")

        if self.extract_hand_pose_data:
            hand_pose_file = self.data_folder + "hand_pose_data.txt"
            with open(hand_pose_file, mode="w", encoding='utf-8') as f:
                json.dump(self.hand_pose_msgs, f)
            self.log.info(f"Created hand pose file: {hand_pose_file}")
        else:
            self.log.info(f"Skipping hand pose file creation")

        if self.extract_head_pose_data:
            head_pose_file = self.data_folder + "head_pose_data.txt"
            with open(head_pose_file, mode="w", encoding='utf-8') as f:
                json.dump(self.head_pose_msgs, f)
            self.log.info(f"Created head pose file: {head_pose_file}")
        else:
            self.log.info(f"Skipping head pose file creation")

        if self.extract_spatial_map_data:
            spatial_map_file = self.data_folder + "spatial_map_data.txt"
            with open(spatial_map_file, mode="w", encoding='utf-8') as f:
                json.dump(self.spatial_map_msgs, f)
            self.log.info(f"Created spatial map file: {spatial_map_file}")
        else:
            self.log.info(f"Skipping spatial map file creation")

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

    def create_wav_file(self, filename: Optional[str] = None) -> None:
        """
        Creates a .wav file from the bag's audio data topic messages.
        """
        if not self.extract_audio:
            self.log.info(f"Skipping audio extraction")
            return

        if filename is None:
            filename = self.data_folder + "audio.wav"

        # Split audio data into two channels
        # HL audio samples are interleaved like [CH1, CH2, CH1, CH2,...]
        audio_ch1 = np.array(self.audio_data)[::2]
        audio_ch2 = np.array(self.audio_data)[1::2]

        audio_both_channels = np.stack((audio_ch1, audio_ch2), axis=1)
        scipy.io.wavfile.write(filename,
                               self.audio_sample_rate,
                               audio_both_channels)

        self.log.info(f"Created audio file: {filename}")

    def convert_eye_gaze_msg_to_dict(self, msg) -> Dict:
        """
        Converts an eye gaze ROS message object to a dictionary.
        """
        d = {
            "time_sec": msg.header.stamp.sec,
            "time_nanosec": msg.header.stamp.nanosec,
            "gaze_origin": [msg.gaze_origin.x,
                            msg.gaze_origin.y,
                            msg.gaze_origin.z],
            "gaze_direction": [msg.gaze_direction.x,
                               msg.gaze_direction.y,
                               msg.gaze_direction.z],
            "head_movement_direction": [msg.head_movement_direction.x,
                                        msg.head_movement_direction.y,
                                        msg.head_movement_direction.z],
            "head_velocity": [msg.head_velocity.x,
                              msg.head_velocity.y,
                              msg.head_velocity.z],
            "is_object_hit": msg.is_object_hit,
            "hit_object_position": [msg.hit_object_position.x,
                                    msg.hit_object_position.y,
                                    msg.hit_object_position.z]
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
            "joint_poses": [{"joint": m.joint,
                             "position": [m.pose.position.x,
                                          m.pose.position.y,
                                          m.pose.position.z],
                             "rotation": [m.pose.orientation.x,
                                          m.pose.orientation.y,
                                          m.pose.orientation.z,
                                          m.pose.orientation.w],
                            } for m in msg.joints
                           ]
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
            "projection_matrix": list(msg.projection_matrix)
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
            "triangles": [[int(t.vertex_indices[0]),
                           int(t.vertex_indices[1]),
                           int(t.vertex_indices[2])] for t in msg.mesh.triangles],
            "vertices": [[v.x, v.y, v.z] for v in msg.mesh.vertices]
        }
        return d


if __name__ == "__main__":
    rclpy.init()

    bag_converter = BagConverter()
