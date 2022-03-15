import time
from typing import Dict

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image

from angel_system.impls.detect_activities.pytorchvideo_slow_fast_r50 import PytorchVideoSlowFastR50


BRIDGE = CvBridge()

class ActivityDetector(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        self._image_topic = self.declare_parameter("image_topic", "debug/PVFrames").get_parameter_value().string_value
        self._use_cuda = self.declare_parameter("use_cuda", True).get_parameter_value().bool_value
        self._det_topic = self.declare_parameter("det_topic", "ActivityDetections").get_parameter_value().string_value
        self._det_threshold = self.declare_parameter("det_threshold", 0.8).get_parameter_value().double_value
        self._frames_per_det = self.declare_parameter("frames_per_det", 32.0).get_parameter_value().double_value

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Use cuda? {self._use_cuda}")
        log.info(f"Detection threshold: {self._det_threshold}")
        log.info(f"Frames per detection: {self._frames_per_det}")

        self._subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.listener_callback,
            1
        )

        '''
        self._publisher = self.create_publisher(
            ActivityDetections,
            self._det_topic,
            1
        )
        '''

        self._frames_recvd = 0
        self._prev_time = -1

        # Stores the frames until we have enough to send to the detector
        self._frames = []
        
        self._detector = PytorchVideoSlowFastR50()


    def listener_callback(self, image):
        """
        Callback for when an image is received on the selected image topic. 
        The image is added to a list of images and if the list length
        exceeds the activity durations, it is passed to the activity detector.
        Any detected activities are published to the configured activity 
        topic as angel_msgs/Activity messages.
        """
        log = self.get_logger()
        self._frames_recvd += 1
        if self._prev_time == -1:
            self._prev_time = time.time()
        elif time.time() - self._prev_time > 1:
            log.info(f"Frames rcvd: {self._frames_recvd}")
            self._frames_recvd = 0
            self._prev_time = time.time()

        # convert NV12 image to RGB image with shape (HxWx3) and add it to the frame stack
        rgb_image = BRIDGE.imgmsg_to_cv2(image, desired_encoding="rgb8")
        rgb_image_np = np.asarray(rgb_image)
        self._frames.append(rgb_image_np)

        if len(self._frames) >= self._frames_per_det:
            activities_detected = self._detector.detect_activities(self._frames)
            log.info(f"activities: {activities_detected}")

            # TODO: publish activities

            # clear out stored frames
            self._frames= []


def main():
    rclpy.init()

    activity_detector = ActivityDetector()

    rclpy.spin(activity_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    activity_detector.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
