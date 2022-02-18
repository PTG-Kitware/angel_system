import queue
import socket
import struct
import sys
import time
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8MultiArray
import cv2
import PIL
import PIL.ImageDraw, PIL.ImageFont
import torch
from matplotlib import pyplot as plot

TOPICS = ["PVFrames"]


class ObjectDetector(Node):

    def __init__(self,
        topic: str,
        node_name: str = None,
        detection_threshold: float = 0.8,
    ):
        if node_name is None:
            node_name = self.__class__.__name__
        super().__init__(node_name)

        if topic not in TOPICS:
            print("Error! Invalid topic name")
            sys.exit()

        self._topic = topic

        self._image_subscription = self.create_subscription(
            Image,
            self._topic,
            self.listener_callback,
            100
        )

        self._detection_subscription = self.create_subscription(
            UInt8MultiArray,
            "ObjectDetections",
            self.detection_callback,
            100
        )

        self._frames_recvd = 0
        self._prev_time = -1

        self._detections = []

        self._ax1 = plot.subplot(111)
        self._im1 = self._ax1.imshow(np.zeros(shape=(720, 1280, 3)), vmin=0, vmax=255)

        plot.ion()
        plot.show()


    def detection_callback(self, detection):
        object_type = struct.unpack("I", detection.data[0:4])[0]
        min_vertex0 = struct.unpack("f", detection.data[4:8])[0]
        min_vertex1 = struct.unpack("f", detection.data[8:12])[0]
        max_vertex0 = struct.unpack("f", detection.data[12:16])[0]
        max_vertex1 = struct.unpack("f", detection.data[16:20])[0]

        # check if we have a similar object already
        add_object = True
        for d in self._detections:
            if d["object_type"] == convert_class_num_to_label(object_type):
                # already have an object of this time so don't add it
                add_object = False

            # calculate proximity to other boxes
            #if (min_vertex0 * 0.95 < d[1] < min_vertex1)

        if add_object:
            self._detections.append({"object_type": convert_class_num_to_label(object_type),
                                     "bounding_box": (min_vertex0, min_vertex1,
                                                      max_vertex0, max_vertex1),
                                     "frames_displayed": 0}) # frames displayed


    def listener_callback(self, image):
        self._frames_recvd += 1
        if self._prev_time == -1:
            self._prev_time = time.time()
        elif (time.time() - self._prev_time > 1):
            print("Frames rcvd", self._frames_recvd)
            self._frames_recvd = 0
            self._prev_time = time.time()

        # convert NV12 image to RGB
        yuv_image = np.frombuffer(image.data, np.uint8).reshape(image.height*3//2, image.width)
        rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB_NV12)
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

        self._im1.set_data(pil_image)

        plot.gcf().canvas.flush_events()
        plot.show(block=False)


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
    topic_name = sys.argv[1]
    rclpy.init()

    object_detector = ObjectDetector(topic_name)

    rclpy.spin(object_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    object_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
