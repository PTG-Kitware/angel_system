#!/usr/bin/env python3
"""
Convert ROS bags containing Hololens 2 data to other suitable file formats.

This script should be run within a ROS environment, specifically one with
rosbag2_py.
"""
import os
import sys
from typing import Optional

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
    HeadsetAudioData,
    SpatialMesh,
    HeadsetPoseData
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

        if self.bag_path == "":
            self.log.info("Please provide bag file to convert")
            self.log.info("Usage: ros2 run angel_utils bag_extractor.py --ros-args -p bag_path:=`bag_name`")
            raise ValueError("Bag file path not provided")
            return

        # Top level data folder
        self.total_messages = 0
        self.data_folder = self.bag_path + "_extracted/"
        if not(os.path.exists(self.data_folder)):
            os.makedirs(self.data_folder)
            self.log.info(f"Created {self.data_folder} for data")

        # For extracking audio data
        self.audio_messages = 0
        self.audio_data = []

        # For extracting images data
        self.image_messages = 0
        self.images = []
        self.image_folder = self.data_folder + "images/"
        if not(os.path.exists(self.image_folder)):
            os.makedirs(self.image_folder)
            self.log.info(f"Created {self.image_folder} for extracted images")

        # Parse the bag
        self.parse_bag()

    def parse_bag(self) -> None:
        """
        Read the ROS2 bag, deserialize the messages, and extract the relevant
        data.
        """
        self.log.info(f"Reading bag: {self.bag_path}")
        storage_options, converter_options = get_rosbag_options(self.bag_path)

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()

        # Create a map for quicker lookup
        type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            self.total_messages += 1

            if (self.total_messages % 100) == 0:
                self.log.info(f"Parsing message: {self.total_messages}")

            # TODO: add other message types
            if isinstance(msg, HeadsetPoseData):
                pass
            elif isinstance(msg, HeadsetAudioData):
                self.audio_sample_rate = msg.sample_rate
                self.audio_messages += 1

                # Accumulate the audio data
                self.audio_data.extend(msg.data)
            elif isinstance(msg, Image):
                self.image_messages += 1

                # Check if we should process the images
                if not self.extract_images:
                    continue

                # Convert NV12 image to RGB
                yuv_image = np.frombuffer(msg.data, np.uint8).reshape(msg.height*3//2, msg.width)
                rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)

                # Save image to disk
                cv2.imwrite(f"{self.image_folder}frame_{self.image_messages:05d}.png", rgb_image)

        self.print_bag_info()

        # Save the audio file
        self.create_wav_file()

    def print_bag_info(self) -> None:
        """
        Print stats about the bag.
        """
        self.log.info(f"Bag info for {self.bag_path}")
        self.log.info(f"Total ROS messages: {self.total_messages}")
        self.log.info(f"Audio messages: {self.audio_messages}")
        self.log.info(f"Image messages: {self.image_messages}")

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


if __name__ == "__main__":
    rclpy.init()

    bag_converter = BagConverter()
