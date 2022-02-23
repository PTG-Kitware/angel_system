import queue
import socket
import struct
import sys
import time
import threading

from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8MultiArray
import cv2
import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plot
from smqtk_detection.impls.detect_image_objects.resnet_frcnn import ResNetFRCNN


TOPICS = ["PVFrames"]


BRIDGE = CvBridge()


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
        self._detection_threshold = detection_threshold

        self._subscription = self.create_subscription(
            Image,
            self._topic,
            self.listener_callback,
            100)
        self._subscription  # prevent unused variable warning

        self._publisher = self.create_publisher(
            UInt8MultiArray,
            "ObjectDetections",
            10
        )

        self._frames_recvd = 0
        self._prev_time = -1

        # instantiate detector
        self._detector = ResNetFRCNN(img_batch_size=1)
        print("Ready to detect", self._detector, self._detector.use_cuda)
        self._images = []


    def listener_callback(self, image):
        self._frames_recvd += 1
        if self._prev_time == -1:
            self._prev_time = time.time()
        elif (time.time() - self._prev_time > 1):
            print("Frames rcvd", self._frames_recvd)
            self._frames_recvd = 0
            self._prev_time = time.time()

        # convert NV12 image to RGB
        try:
            yuv_image = np.frombuffer(image.data, np.uint8).reshape(image.height*3//2, image.width)
            rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB_NV12)
        except ValueError:
            rgb_image = BRIDGE.imgmsg_to_cv2(image, desired_encoding="rgb8")
        #print(type(rgb), rgb.size, rgb.shape)
        #print("image_np stuff", rgb.shape, rgb.dtype, image.height, image.width, rgb[0:2])

        # convert np array to tensor
        image_np = (rgb_image / 255.0).astype(np.float32)
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image_tensor = transform(image_tensor)

        self._images.append(image_tensor)
        if len(self._images) >= 1:
            # send to detector
            start = time.time()
            detections = self._detector.detect_objects(self._images)
            end = time.time()
            #print("Time to perform detections", end - start)

            threshold_detections = []
            for detection in detections:
                #print("detection", detection)
                for i in detection:
                    bounding_box = i[0]
                    class_dict = i[1]

                    class_dict_sorted = {k: v for k, v in sorted(class_dict.items(), key=lambda item: item[1], reverse=True)}
                    #print("Results sorted:", list(class_dict_sorted.items())[:5])

                    if list(class_dict_sorted.items())[0][1] > self._detection_threshold:
                        #print("Found something:", list(class_dict_sorted.items())[0], bounding_box)
                        object_type = list(class_dict_sorted.items())[0][0]

                        # publish to the detections topic
                        msg = UInt8MultiArray()
                        object_type = bytearray(struct.pack("I", object_type))
                        min_vertex0 = bytearray(struct.pack("f", bounding_box.min_vertex[0]))
                        min_vertex1 = bytearray(struct.pack("f", bounding_box.min_vertex[1]))
                        max_vertex0 = bytearray(struct.pack("f", bounding_box.max_vertex[0]))
                        max_vertex1 = bytearray(struct.pack("f", bounding_box.max_vertex[1]))
                        msg.data = object_type + min_vertex0 + min_vertex1 + max_vertex0 + max_vertex1
                        self._publisher.publish(msg)
                        #print("Published!", object_type)

            self._images = []


def main():
    topic_name = sys.argv[1]

    rclpy.init()


    object_detector = ObjectDetector(topic_name)
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
