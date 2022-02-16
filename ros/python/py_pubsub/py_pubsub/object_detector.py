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
import cv2
import PIL
import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plot
from smqtk_detection.impls.detect_image_objects.resnet_frcnn import ResNetFRCNN

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
        self._detection_threshold = detection_threshold

        self._subscription = self.create_subscription(
            Image,
            self._topic,
            self.listener_callback,
            100)
        self._subscription  # prevent unused variable warning

        self._frames_recvd = 0
        self._prev_time = -1

        # instantiate detector
        self._detector = ResNetFRCNN(img_batch_size=1)
        print("Ready to detect", self._detector, self._detector.use_cuda)
        self._images = []

        self._ax1 = plot.subplot(111)
        self._im1 = self._ax1.imshow(np.zeros(shape=(720, 1280, 3)), vmin=0, vmax=255)

        plot.ion()
        plot.show()


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
            print("Time to perform detections", end - start)

            threshold_detections = []
            for detection in detections:
                #print("detection", detection)
                for i in detection:
                    bounding_box = i[0]
                    class_dict = i[1]

                    class_dict_sorted = {k: v for k, v in sorted(class_dict.items(), key=lambda item: item[1], reverse=True)}
                    #print("Results sorted:", list(class_dict_sorted.items())[:5])

                    if list(class_dict_sorted.items())[0][1] > self._detection_threshold:
                        print("Found something:", list(class_dict_sorted.items())[0], bounding_box)

                        draw = PIL.ImageDraw.Draw(pil_image)
                        draw.rectangle(((bounding_box.min_vertex[0], bounding_box.min_vertex[1]),
                                        (bounding_box.max_vertex[0], bounding_box.max_vertex[1])),
                                        outline="black", width=10)

                    else:
                        #print("Results sorted:", list(class_dict_sorted.items())[:5])
                        #print(list(class_dict_sorted.items())[0][1])
                        pass
            self._images = []

        self._im1.set_data(pil_image)

        plot.gcf().canvas.flush_events()
        plot.show(block=False)


def main():
    topic_name = sys.argv[1]

    rclpy.init()

    object_detector = ObjectDetector(topic_name)

    rclpy.spin(object_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
