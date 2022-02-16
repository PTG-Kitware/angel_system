import queue
import socket
import struct
import sys
import time
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, ByteMultiArray
from sensor_msgs.msg import Image
import cv2
import PIL
import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plot
from smqtk_detection.impls.detect_image_objects.resnet_frcnn import ResNetFRCNN

TOPICS = ["PVFrames"]


class ObjectDetector(Node):

    def __init__(self, topic):
        super().__init__('object_detector')

        if topic not in TOPICS:
            print("Error! Invalid topic name")
            sys.exit()

        self.topic = topic

        self.subscription = self.create_subscription(
            Image,
            self.topic,
            self.listener_callback,
            100)
        self.subscription  # prevent unused variable warning

        self.frames_recvd = 0
        self.prev_time = -1

        # instantiate detector
        self.detector = ResNetFRCNN(img_batch_size=1)
        print("Ready to detect", self.detector, self.detector.use_cuda)
        self.images = []

        self.ax1 = plot.subplot(111)
        self.im1 = self.ax1.imshow(np.zeros(shape=(720, 1280, 3)), vmin=0, vmax=255)

        plot.ion()
        plot.show()


    def listener_callback(self, msg):
        self.frames_recvd += 1
        if self.prev_time == -1:
            self.prev_time = time.time()
        elif (time.time() - self.prev_time > 1):
            print("Frames rcvd", self.frames_recvd)
            #print(msg.header, msg.height, msg.width, msg.encoding, len(msg.data), msg.data[0:10])
            self.frames_recvd = 0
            self.prev_time = time.time()

        yuv_data = np.frombuffer(msg.data, np.uint8).reshape(msg.height*3//2, msg.width)
        rgb = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2RGB_NV12);
        source_img = PIL.Image.fromarray(rgb)

        #print(type(rgb), rgb.size, rgb.shape)

        # convert to np-array
        #print("image_np stuff", rgb.shape, rgb.dtype, msg.height, msg.width, rgb[0:2])

        image_np = (rgb / 255.0).astype(np.float32)
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1))

        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        image_tensor = transform(image_tensor)

        #print("image!", image_tensor[0][0], image_np.shape)
        self.images.append(image_tensor)

        if len(self.images) >= 1:
            # send to detector
            start = time.time()
            detections = self.detector.detect_objects(self.images)
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

                    if list(class_dict_sorted.items())[0][1] > 0.80:
                        print("Found something:", list(class_dict_sorted.items())[0], bounding_box)

                        draw = PIL.ImageDraw.Draw(source_img)
                        draw.rectangle(((bounding_box.min_vertex[0], bounding_box.min_vertex[1]),
                                        (bounding_box.max_vertex[0], bounding_box.max_vertex[1])),
                                        outline="black", width=10)

                    else:
                        #print("Results sorted:", list(class_dict_sorted.items())[:5])
                        #print(list(class_dict_sorted.items())[0][1])
                        pass
            self.images = []

        self.im1.set_data(source_img)

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
