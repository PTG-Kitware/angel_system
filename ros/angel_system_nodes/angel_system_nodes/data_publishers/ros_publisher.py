import socket
import sys
import time
import threading

from rclpy.node import Node
from std_msgs.msg import ByteMultiArray, Int32MultiArray

from angel_utils import make_default_main


HOST = "169.254.103.120"

LF_IMAGE_PORT = 11000
RF_IMAGE_PORT = 11001
LL_IMAGE_PORT = 11002
RR_IMAGE_PORT = 11003

TOPICS = ["LF_FRAMES", "RF_FRAMES", "LL_FRAMES", "RR_FRAMES"]

PORT_MAP = {
    "LF_FRAMES": 11000,
    "RF_FRAMES": 11001,
    "LL_FRAMES": 11002,
    "RR_FRAMES": 11003,
}


class VLCPublisher(Node):
    def __init__(self, topic):
        super().__init__("vlc_publisher")

        if topic not in TOPICS:
            print("Error! Invalid topic name")
            sys.exit()

        self.topic = topic
        self.port = PORT_MAP[self.topic]

        self.publisher_ = self.create_publisher(Int32MultiArray, self.topic, 10)

        self.server_t = threading.Thread(target=self.server_thread)
        self.server_t.daemon = True
        self.server_t.start()

    def server_thread(self):
        # create TCP socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, self.port))
        s.settimeout(120)

        print("Waiting for connection", self.topic, HOST, self.port)
        s.listen()
        try:
            conn, addr = s.accept()
        except:
            print("Timed out waiting for connection")
            return

        print("Connected!!")

        conn.settimeout(60)

        frames_recvd = 0
        prev_time = -1
        prev_screenshot_data = []
        idx = 0

        ros_msg = Int32MultiArray()

        while True:
            # wait for a message
            try:
                data = conn.recv(8)
            except:
                break

            if data[0:4] != b"\x1a\xcf\xfc\x1d":
                print("Invalid sync pattern", data[0:4])
                print(data[0:4].decode())
                break

            frames_recvd += 1
            if prev_time == -1:
                prev_time = time.time()
            elif time.time() - prev_time > 1:
                print("Frames rcvd", self.topic, frames_recvd)
                frames_recvd = 0
                prev_time = time.time()

            total_message_length = list(bytes(data[4:8]))
            total_message_length = (
                (total_message_length[0] << 24)
                | (total_message_length[1] << 16)
                | (total_message_length[2] << 8)
                | (total_message_length[3] << 0)
            )

            # print("message length", total_message_length)

            # read the rest of the message from the socket using the given length
            # screenshot_data = []
            bytes_read = 0
            default_read_size = 9000  # jumbo packet size
            while bytes_read != total_message_length:
                bytes_remaining = total_message_length - bytes_read

                if default_read_size > bytes_remaining:
                    read_size = bytes_remaining
                elif default_read_size > total_message_length:
                    read_size = total_message_length
                else:
                    read_size = default_read_size

                message = list(conn.recv(read_size))
                # message = conn.recv(read_size).split()
                # print(message)

                bytes_read += len(message)
                # screenshot_data.extend(message)
                ros_msg.data.extend(message)

            # publish the message
            # print(screenshot_data[0:10])
            # print(type(screenshot_data[0]))

            # start = time.time()
            # ros_msg.data = [bytes(x) for x in screenshot_data]
            # ros_msg.data = screenshot_data
            # end = time.time()
            # print("Time to make msg", (end - start))

            # start = time.time()
            self.publisher_.publish(ros_msg)
            # end = time.time()
            # print("Time to publish msg", (end - start))
            ros_msg.data = []


def main():
    # nesting "make" call, so we don't access argv unless actually in the main.
    topic_name = sys.argv[1]
    make_default_main(VLCPublisher, (topic_name,))()


if __name__ == "__main__":
    main()
