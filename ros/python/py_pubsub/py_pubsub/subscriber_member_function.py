import rclpy
from rclpy.node import Node

from std_msgs.msg import String, ByteMultiArray

import numpy as np
from PIL import Image, ImageDraw
import queue
import socket
import struct
import time
import threading

from matplotlib import pyplot as plot
import matplotlib

# create axes
ax1 = plot.subplot(111)
im1 = ax1.imshow(np.zeros(shape=(480, 640, 1)), cmap='gray', vmin=0, vmax=255)
plot.ion()
plot.show()

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            ByteMultiArray,
            'LFFrames',
            self.listener_callback,
            100)
        self.subscription  # prevent unused variable warning

        self.frames_recvd = 0
        self.prev_time = -1

    def listener_callback(self, msg):
        #self.get_logger().info('Frame rcvd: %d bytes' % (len(msg.data)))

        data = [int.from_bytes(x, "big") for x in msg.data]

        '''
        if data[0:4] != b'\x1a\xcf\xfc\x1d':
            print("Invalid sync pattern", data[0:4])
            return
        '''

        #print(data[0:20])

        self.frames_recvd += 1
        if self.prev_time == -1:
            self.prev_time = time.time()
        elif (time.time() - self.prev_time > 1):
            print("Frames rcvd", self.frames_recvd)
            self.frames_recvd = 0
            self.prev_time = time.time()

        '''
        total_message_length = data[4:8]
        total_message_length = ((total_message_length[0] << 24) |
                                (total_message_length[1] << 16) | 
                                (total_message_length[2] << 8) | 
                                (total_message_length[3] << 0)) 

        #print("message length", total_message_length)
        '''

        image = data

        width = ((image[0] & 0xFF << 24) |
                 (image[1] << 16) | 
                 (image[2] << 8) | 
                 (image[3] << 0)) 
        height = ((image[4] << 24) |
                 (image[5] << 16) | 
                 (image[6] << 8) | 
                 (image[7] << 0)) 
        image = image[8:]

        # convert to np-array
        image_np_orig = np.array(image)

        image_np = np.reshape(image_np_orig, (height, width, 1))
        image_np = image_np.astype(np.uint8)
        image_np = np.rot90(image_np, k=3)

        im1.set_data(image_np)

        plot.gcf().canvas.flush_events()
        plot.show(block=False)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    print("hi josh")
    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()