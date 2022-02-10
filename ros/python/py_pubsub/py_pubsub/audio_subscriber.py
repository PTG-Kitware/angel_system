import numpy as np
import queue
import socket
import struct
import sys
import time
import threading

import simpleaudio as sa
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

SAMPLE_RATE = 48000
BYTES_PER_FRAME = 8192
FRAMES_PER_SECOND = 48
BYTES_PER_SECOND = BYTES_PER_FRAME * FRAMES_PER_SECOND


class AudioSubscriber(Node):

    def __init__(self):
        super().__init__('audio_subscriber')

        self.subscription = self.create_subscription(
            UInt8MultiArray,
            "AudioData",
            self.listener_callback,
            100)
        self.subscription  # prevent unused variable warning

        self.frames_recvd = 0
        self.prev_time = -1

        self.audio_stream = bytearray()

    def listener_callback(self, msg):
        self.frames_recvd += 1
        if self.prev_time == -1:
            self.prev_time = time.time()
        elif (time.time() - self.prev_time > 1):
            print("Frames rcvd", self.frames_recvd)
            self.frames_recvd = 0
            self.prev_time = time.time()

        self.audio_stream.extend(msg.data)
        if len(self.audio_stream) >= (BYTES_PER_SECOND):
            play_obj = sa.play_buffer(self.audio_stream, 2, 4, SAMPLE_RATE)
            self.audio_stream = bytearray()

def main():
    rclpy.init()

    audio_subscriber = AudioSubscriber()
    print ("Created subscriber!")

    rclpy.spin(audio_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()