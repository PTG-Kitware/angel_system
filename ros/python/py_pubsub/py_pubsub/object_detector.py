import time

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from smqtk_detection.impls.detect_image_objects.resnet_frcnn import ResNetFRCNN

from angel_msgs.msg import ObjectDetection2dSet
from angel_utils.conversion import from_detect_image_objects_result


BRIDGE = CvBridge()


class ObjectDetector(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        self.declare_parameter("image_topic", "PVFrames")
        self.declare_parameter("det_topic", "ObjectDetections")
        self.declare_parameter("use_cuda", False)
        self.declare_parameter("detection_threshold", 0.8)

        self._image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self._det_topic = self.get_parameter("det_topic").get_parameter_value().string_value
        self._use_cuda = self.get_parameter("use_cuda").get_parameter_value().bool_value
        self._detection_threshold = self.get_parameter("detection_threshold").get_parameter_value().double_value

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Use cuda? {self._use_cuda}")
        log.info(f"Detection threshold: {self._detection_threshold}")

        self._subscription = self.create_subscription(
            CompressedImage,
            self._image_topic,
            self.listener_callback,
            1
        )

        self._publisher = self.create_publisher(
            ObjectDetection2dSet,
            self._det_topic,
            1
        )

        self._frames_recvd = 0
        self._prev_time = -1

        # instantiate detector
        self._detector = ResNetFRCNN(img_batch_size=1, use_cuda=self._use_cuda)
        log.info(f"Ready to detect using {self._detector}")

    def listener_callback(self, image):
        log = self.get_logger()
        self._frames_recvd += 1
        if self._prev_time == -1:
            self._prev_time = time.time()
        elif time.time() - self._prev_time > 1:
            log.info(f"Frames rcvd: {self._frames_recvd}")
            self._frames_recvd = 0
            self._prev_time = time.time()

        # convert NV12 image to RGB
        rgb_image = BRIDGE.compressed_imgmsg_to_cv2(image, desired_encoding="rgb8")

        # Send to Detector
        #
        # Since we're passing a single image, we are also just enumerating the
        # returned iterator out to the expected list of size one (containing
        # another iterator)
        #
        detections = list(self._detector.detect_objects([rgb_image]))[0]

        # Construct output message
        #
        # Arbitrarily deciding that output detections, when vectorized, are in
        # descending-score order.
        #
        det_set_msg = from_detect_image_objects_result(
            detections,
            detection_threshold=self._detection_threshold
        )
        # This message time
        det_set_msg.header.stamp = self.get_clock().now().to_msg()
        # Trace to the source
        det_set_msg.header.frame_id = image.header.frame_id
        det_set_msg.source_stamp = image.header.stamp

        self._publisher.publish(det_set_msg)


def main():
    rclpy.init()

    object_detector = ObjectDetector()
    '''
    detection_publisher = DetectionPublisher()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(object_detector)
    executor.add_node(detection_publisher)

    executor.spin()
    '''
    rclpy.spin(object_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    #executor.remove_node(object_detector)
    #executor.remove_node(detection_publisher)
    object_detector.destroy_node()
    #detection_publisher.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
