import struct
import sys
import time

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

from angel_msgs.msg import ObjectDetection


BRIDGE = CvBridge()


class ObjectDetector(Node):

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
            ObjectDetection,
            self._det_topic,
            self.detection_callback,
            1
        )

        self._pub_debug_detections_image = self.create_publisher(
            Image,  # CompressedImage,
            self._out_image_topic,
            1
        )

        self._frames_recvd = 0
        self._prev_time = -1

        self._detections = []

    def detection_callback(self, detection):
        log = self.get_logger()
        log.info(f"Received detection: {detection}")
        if len(detection.label_vec) <= 0:
            log.warning(
                f"Detection received that had no classification content? "
                f":: {detection}"
            )

        object_conf, object_type = sorted(
            zip(detection.label_confidence_vec, detection.label_vec),
            reverse=True
        )[0]
        min_vertex0 = detection.left
        min_vertex1 = detection.top
        max_vertex0 = detection.right
        max_vertex1 = detection.bottom

        # check if we have a similar object already
        add_object = True
        for d in self._detections:
            if d["object_type"] == object_type:
                # already have an object of this time so don't add it
                add_object = False

            # calculate proximity to other boxes
            #if (min_vertex0 * 0.95 < d[1] < min_vertex1)

        if add_object:
            self._detections.append({"object_type": object_type,
                                     "bounding_box": (min_vertex0, min_vertex1,
                                                      max_vertex0, max_vertex1),
                                     "frames_displayed": 0})

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

        draw = PIL.ImageDraw.Draw(pil_image)
        for d in self._detections:
            draw.rectangle(
                            ((d["bounding_box"][0], d["bounding_box"][1]),
                             (d["bounding_box"][2], d["bounding_box"][3])),
                            outline="blue", width=10
                          )

            fontpath = "/usr/share/fonts/truetype/ttf-bitstream-vera/Vera.ttf"
            font = PIL.ImageFont.truetype(fontpath, 32)
            draw.text((d["bounding_box"][0], d["bounding_box"][1] - 30), d["object_type"],
                      font=font)
            d["frames_displayed"] += 1

            if d["frames_displayed"] >= 60:
                self._detections.remove(d)

        img_mat = np.asarray(pil_image)
        img_msg = BRIDGE.cv2_to_imgmsg(img_mat, encoding="rgb8")
        # img_msg = BRIDGE.cv2_to_compressed_imgmsg(img_mat, dst_format='jpg')
        img_msg.header = image.header
        self._pub_debug_detections_image.publish(img_msg)


def convert_class_num_to_label(num):
    label_list = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "hair brush"
    ]

    return label_list[num - 1]


def main():
    rclpy.init()

    object_detector = ObjectDetector()

    rclpy.spin(object_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    object_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
