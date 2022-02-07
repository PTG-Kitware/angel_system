import numpy as np
import queue
import socket
import struct
import sys
import time
import threading

from matplotlib import pyplot as plot
import matplotlib

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, ByteMultiArray
from sensor_msgs.msg import Image

import cv2

TOPICS = ["LFFrames", "RFFrames", "LLFrames", "RRFrames",
          "PVFrames", "DepthFrames", "DepthABFrames"]


class MinimalSubscriber(Node):

    def __init__(self, topic):
        super().__init__('minimal_subscriber')

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

        # create axes
        self.ax1 = plot.subplot(111)
        if self.topic == "PVFrames":
            self.im1 = self.ax1.imshow(np.zeros(shape=(720, 1280, 3)), vmin=0, vmax=255)
        elif self.topic == "DepthFrames" or self.topic == "DepthABFrames":
            self.im1 = self.ax1.imshow(np.zeros(shape=(512, 512, 1)), cmap='gray', vmin=0, vmax=255)
        else:
            self.im1 = self.ax1.imshow(np.zeros(shape=(480, 640, 1)), cmap='gray', vmin=0, vmax=255)
        
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

        if self.topic == "PVFrames":
            yuv_data = np.frombuffer(msg.data, np.uint8).reshape(720*3//2, 1280)
            rgb = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2RGB_NV12);

            self.im1.set_data(rgb)
        elif self.topic == "DepthFrames":
            image_np_orig = np.frombuffer(msg.data, np.uint8)
            #print(image_np_orig[0:10])


            '''
            dt = np.dtype(np.uint16)
            dt = dt.newbyteorder('>')
            image_np_orig = np.frombuffer(msg.data, dtype=dt)
            #print(image_np_orig[0:10])

            for i in range(len(image_np_orig)):
                if image_np_orig[i] > 4090:
                    image_np_orig[i] = 0
                else:
                    image_np_orig[i] = int((image_np_orig[i] / 1000.0 * 255)) & 0xFF
            

            image_np_orig[image_np_orig > 4090] = 0

            image_np_orig = (image_np_orig / 1000.0 * 255)
            
            #print(image_np_orig[0:10])
            '''
            image_np = np.reshape(image_np_orig, (msg.height, msg.width, 1))
            #image_np = image_np.astype(np.uint8)

            self.im1.set_data(image_np)

        elif self.topic == "DepthABFrames":
            image_np_orig = np.frombuffer(msg.data, np.uint8)

            '''
            dt = np.dtype(np.uint16)
            dt = dt.newbyteorder('>')
            image_np_orig = np.frombuffer(msg.data, dtype=dt)

            image_np_orig = (image_np_orig / 1000.0 * 255)
            image_np_orig[image_np_orig > 255] = 0xFF

            #print(image_np_orig[0:10])
            '''

            image_np = np.reshape(image_np_orig, (msg.height, msg.width, 1))
            #image_np = image_np.astype(np.uint8)

            self.im1.set_data(image_np)
        else:
            image_np_orig = np.frombuffer(msg.data, np.uint8)

            image_np = np.reshape(image_np_orig, (msg.height, msg.width, 1))
            image_np = image_np.astype(np.uint8)

            if self.topic == "LFFrames":
                image_np = np.rot90(image_np, k=3)
            if self.topic == "LLFrames":
                image_np = np.rot90(image_np, k=1)
            if self.topic == "RFFrames":
                image_np = np.rot90(image_np, k=1)
            if self.topic == "RRFrames":
                image_np = np.rot90(image_np, k=3)

            self.im1.set_data(image_np)

        plot.gcf().canvas.flush_events()
        plot.show(block=False)


def main():
    topic_name = sys.argv[1]

    rclpy.init()

    minimal_subscriber = MinimalSubscriber(topic_name)

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()