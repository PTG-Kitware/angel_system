from threading import (
    Event,
    Thread
)

from cv_bridge import CvBridge
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from angel_system.hl2ss.viewer import hl2ss
from angel_utils.conversion import hl2ss_stamp_to_ros_time


BRIDGE = CvBridge()


class HL2SSVideoPlayer(Node):
    """
    ROS node that demonstrates using the HL2SS client/server to publish
    RGB images from the HoloLens 2.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        # Declare ROS topics
        self._image_topic = (
            self.declare_parameter("image_topic", "")
            .get_parameter_value()
            .string_value
        )
        self.ip_addr = (
            self.declare_parameter("ip_addr", "")
            .get_parameter_value()
            .string_value
        )

        log = self.get_logger()
        if self._image_topic == "":
            raise ValueError("Please provide the image topic with the `image_topic` parameter")
        if self.ip_addr == "":
            raise ValueError("Please provide HL2 IPv4 address with the `ip_addr` parameter")

        self.port = hl2ss.StreamPort.PERSONAL_VIDEO

        log.info(f"Image topic: {self._image_topic}")
        log.info(f"HL2 IP: {self.ip_addr}")

        # Create frame publisher
        self.frame_publisher = self.create_publisher(
            Image,
            self._image_topic,
            1
        )

        log.info("Connecting to HL2SS server...")
        self.connect_hl2ss()
        log.info("Client connected! Starting publishing thread.")

        # Start the frame publishing thread
        self._fp_active = Event()
        self._fp_active.set()
        self._fp_thread = Thread(
            target=self.publish_frames,
            name="publish_frames"
        )
        self._fp_thread.daemon = True
        self._fp_thread.start()
        log.info("Starting publishing thread... Done")


    def connect_hl2ss(self) -> None:
        """
        Creates the HL2SS PV client and connects it to the server on the headset.
        """
        # Operating mode
        # 0: video
        # 1: video + camera pose
        # 2: query calibration (single transfer)
        mode = hl2ss.StreamMode.MODE_1

        # Camera parameters
        width     = 1920
        height    = 1080
        framerate = 30

        # Video encoding profile
        profile = hl2ss.VideoProfile.H265_MAIN

        # Encoded stream average bits per second
        # Must be > 0
        bitrate = 5*1024*1024

        # Decoded format
        decoded_format = 'bgr24'

        #------------------------------------------------------------------------------
        hl2ss.start_subsystem_pv(self.ip_addr, self.port)

        self.hl2ss_pv_client = hl2ss.rx_decoded_pv(
            self.ip_addr,
            self.port,
            hl2ss.ChunkSize.PERSONAL_VIDEO,
            mode,
            width,
            height,
            framerate,
            profile,
            bitrate,
            decoded_format
        )
        self.hl2ss_pv_client.open()


    def shutdown_client(self) -> None:
        """
        Shuts down the frame publishing thread and the HL2SS client.
        """
        # Stop frame publishing thread
        self._fp_active.clear()  # make RT active flag "False"
        self._fp_thread.join()
        self.get_logger().info("Frame publishing thread closed")

        # Close client connection
        self.hl2ss_pv_client.close()
        hl2ss.stop_subsystem_pv(self.ip_addr, self.port)
        self.get_logger().info("HL2SS client disconnected")


    def publish_frames(self) -> None:
        """
        Main thread that gets frames from the HL2SS PV client and publishes
        them to the image topic.
        """
        while self._fp_active.wait(0):  # will quickly return false if cleared.
            data = self.hl2ss_pv_client.get_next_packet()

            try:
                image_msg = BRIDGE.cv2_to_imgmsg(data.payload, encoding="bgr8")
                image_msg.header.stamp = hl2ss_stamp_to_ros_time(data.timestamp)
                image_msg.header.frame_id = "PVFramesRGB"
            except TypeError as e:
                self.get_logger().warning(f"{e}")

            self.frame_publisher.publish(image_msg)


def main():
    rclpy.init()

    hl2ss_video_player = HL2SSVideoPlayer()

    try:
        rclpy.spin(hl2ss_video_player)
    except KeyboardInterrupt:
        hl2ss_video_player.get_logger().info("Keyboard interrupt, shutting down.\n")
        hl2ss_video_player.shutdown_client()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    hl2ss_video_player.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
