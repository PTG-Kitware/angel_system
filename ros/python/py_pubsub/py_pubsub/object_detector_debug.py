import struct
import sys
import time
from typing import Optional

from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
import cv2
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from angel_msgs.msg import ObjectDetection2dSet
from angel_utils.conversion import to_confidence_matrix


BRIDGE = CvBridge()


class ObjectDetectorDebug(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        self.declare_parameter("image_topic", "PVFrames")
        self.declare_parameter("det_topic", "ObjectDetections")
        self.declare_parameter("out_image_topic",
                               "ObjectDetectionsDebug")

        self._image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self._det_topic = self.get_parameter("det_topic").get_parameter_value().string_value
        self._out_image_topic = self.get_parameter("out_image_topic").get_parameter_value().string_value

        self._image_subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.listener_callback,
            1
        )

        self._detection_subscription = self.create_subscription(
            ObjectDetection2dSet,
            self._det_topic,
            self.detection_callback,
            1
        )

        # Directly publishing compressed images is the only way for the python
        # node to keep up. Publishing raw images is too much for maintaining
        # higher frame-rates.
        self._pub_debug_detections_image = self.create_publisher(
            CompressedImage,
            self._out_image_topic + "/compressed",
            1
        )

        self._frames_recvd = 0
        self._prev_time = -1

        # The latest detections message received
        self._detection_i = 0
        self._detections: Optional[ObjectDetection2dSet] = None

    def detection_callback(self, msg: ObjectDetection2dSet):
        log = self.get_logger()
        log.info(f"Received detection #{self._detection_i} with "
                 f"{msg.num_detections} detections")

        self._detections = msg
        self._detection_i += 1

    def listener_callback(self, image):
        log = self.get_logger()
        self._frames_recvd += 1
        if self._prev_time == -1:
            self._prev_time = time.time()
        elif time.time() - self._prev_time > 1:
            log.info(f"Frames rcvd {self._frames_recvd}")
            self._frames_recvd = 0
            self._prev_time = time.time()

        # convert NV12 image to RGB
        try:
            yuv_image = np.frombuffer(image.data, np.uint8).reshape(image.height*3//2, image.width)
            rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB_NV12)
        except ValueError:
            rgb_image = BRIDGE.imgmsg_to_cv2(image, desired_encoding="rgb8")
        pil_image = PIL.Image.fromarray(rgb_image)

        latest_dets_msg = self._detections
        # If there are no detections yet, nothing to draw.
        if self._detections is not None:
            # Draw detections onto the image
            draw = PIL.ImageDraw.Draw(pil_image)
            fontpath = "/usr/share/fonts/truetype/ttf-bitstream-vera/Vera.ttf"
            font = PIL.ImageFont.truetype(fontpath, 32)
            det_label_list = latest_dets_msg.label_vec
            det_conf_mat = to_confidence_matrix(latest_dets_msg)
            log.info(f"Drawing {det_conf_mat.shape[0]} detections")
            for i in range(latest_dets_msg.num_detections):
                # Get the maximum confidence label
                label = sorted(zip(det_conf_mat[i], det_label_list))[-1][1]
                # Draw the box and label
                draw.rectangle(
                    [(latest_dets_msg.left[i], latest_dets_msg.top[i]),
                     (latest_dets_msg.right[i], latest_dets_msg.bottom[i])],
                    outline="blue", width=10
                )
                txt_height_px = font.getsize(label)[1]
                draw.text(
                    (latest_dets_msg.left[i], latest_dets_msg.top[i] - txt_height_px),
                    label,
                    font=font
                )

        # Publish out drawn-on image.
        img_mat = np.asarray(pil_image)
        # img_msg = BRIDGE.cv2_to_imgmsg(img_mat, encoding="rgb8")
        img_msg = BRIDGE.cv2_to_compressed_imgmsg(img_mat, dst_format='jpg')
        img_msg.header = image.header
        self._pub_debug_detections_image.publish(img_msg)


def main():
    rclpy.init()

    object_detector = ObjectDetectorDebug()

    rclpy.spin(object_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    object_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
