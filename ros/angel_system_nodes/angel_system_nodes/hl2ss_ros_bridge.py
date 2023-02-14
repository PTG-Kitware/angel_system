from threading import (
    Event,
    Thread
)

from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import (
    Point,
    Pose,
    Quaternion,
)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from angel_msgs.msg import (
    HandJointPose,
    HandJointPosesUpdate,
)
from angel_system.hl2ss.viewer import hl2ss
from angel_utils.conversion import hl2ss_stamp_to_ros_time


BRIDGE = CvBridge()


class HL2SSROSBridge(Node):
    """
    ROS node that uses HL2SS client/server library to convert HL2SS data to
    ROS messages used throughout the ANGEL system.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        # Declare ROS topics
        self._image_topic = (
            self.declare_parameter("image_topic", "")
            .get_parameter_value()
            .string_value
        )
        self._hand_pose_topic = (
            self.declare_parameter("hand_pose_topic", "")
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
        if self._hand_pose_topic == "":
            raise ValueError("Please provide the hand pose topic with the `hand_pose_topic` parameter")
        if self.ip_addr == "":
            raise ValueError("Please provide HL2 IPv4 address with the `ip_addr` parameter")


        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Hand pose topic: {self._hand_pose_topic}")
        log.info(f"HL2 IP: {self.ip_addr}")

        # Define HL2SS server ports
        self.pv_port = hl2ss.StreamPort.PERSONAL_VIDEO
        self.si_port = hl2ss.StreamPort.SPATIAL_INPUT

        # Create frame publisher
        self.frame_publisher = self.create_publisher(
            Image,
            self._image_topic,
            1
        )
        # Create the hand joint pose publisher
        self.hand_publisher = self.create_publisher(
            HandJointPosesUpdate,
            self._hand_pose_topic,
            1
        )

        log.info("Connecting to HL2SS servers...")
        self.connect_hl2ss_pv()
        log.info("PV client connected!")
        self.connect_hl2ss_si()
        log.info("SI client connected!")

        log.info("Starting publishing threads...")
        # Start the frame publishing thread
        self._pv_active = Event()
        self._pv_active.set()
        self._pv_thread = Thread(
            target=self.pv_publisher,
            name="publish_frames"
        )
        self._pv_thread.daemon = True
        self._pv_thread.start()
        # Start the hand tracking data thread
        self._si_active = Event()
        self._si_active.set()
        self._si_thread = Thread(
            target=self.si_publisher,
            name="publish_frames"
        )
        self._si_thread.daemon = True
        self._si_thread.start()
        log.info("Starting publishing threads... Done")

    def connect_hl2ss_pv(self) -> None:
        """
        Creates the HL2SS PV client and connects it to the server on the headset.
        """
        # Operating mode
        # 0: video
        # 1: video + camera pose
        # 2: query calibration (single transfer)
        mode = hl2ss.StreamMode.MODE_1

        # Camera parameters
        width     = 1280
        height    = 720
        framerate = 30

        # Video encoding profile
        profile = hl2ss.VideoProfile.H265_MAIN

        # Encoded stream average bits per second
        # Must be > 0
        bitrate = 5*1024*1024

        # Decoded format
        decoded_format = 'bgr24'

        #------------------------------------------------------------------------------
        hl2ss.start_subsystem_pv(self.ip_addr, self.pv_port)

        self.hl2ss_pv_client = hl2ss.rx_decoded_pv(
            self.ip_addr,
            self.pv_port,
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

    def connect_hl2ss_si(self) -> None:
        """
        Creates the HL2SS Spatial Input (SI) client and connects it to the
        server on the headset.
        """
        self.hl2ss_si_client = hl2ss.rx_si(
            self.ip_addr, self.si_port, hl2ss.ChunkSize.SPATIAL_INPUT
        )
        self.hl2ss_si_client.open()

    def shutdown_clients(self) -> None:
        """
        Shuts down the frame publishing thread and the HL2SS client.
        """
        # Stop frame publishing thread
        self._pv_active.clear()  # make RT active flag "False"
        self._pv_thread.join()
        self.get_logger().info("PV thread closed")

        # Stop SI publishing thread
        self._si_active.clear()  # make RT active flag "False"
        self._si_thread.join()
        self.get_logger().info("SI thread closed")

        # Close client connections
        self.hl2ss_pv_client.close()
        hl2ss.stop_subsystem_pv(self.ip_addr, self.pv_port)
        self.get_logger().info("PV client disconnected")

        self.hl2ss_si_client.close()
        self.get_logger().info("SI client disconnected")

    def pv_publisher(self) -> None:
        """
        Main thread that gets frames from the HL2SS PV client and publishes
        them to the image topic.
        """
        while self._pv_active.wait(0):  # will quickly return false if cleared.
            data = self.hl2ss_pv_client.get_next_packet()

            try:
                image_msg = BRIDGE.cv2_to_imgmsg(data.payload, encoding="bgr8")
                image_msg.header.stamp = hl2ss_stamp_to_ros_time(data.timestamp)
                image_msg.header.frame_id = "PVFramesRGB"
            except TypeError as e:
                self.get_logger().warning(f"{e}")

            self.frame_publisher.publish(image_msg)

    def si_publisher(self) -> None:
        """
        Thread the gets spatial input packets from the HL2SS SI client, converts
        the SI data to ROS messages, and publishes them.

        Currently only publishes hand tracking data. However, eye gaze data and
        head pose data is also available in the SI data packet.
        """
        while self._si_active.wait(0):  # will quickly return false if cleared.
            data = self.hl2ss_si_client.get_next_packet()
            si_data = hl2ss.unpack_si(data.payload)

            # Publish the hand tracking data if it is valid
            if si_data.is_valid_hand_left():
                hand_msg_left = self.create_hand_pose_msg_from_si_data(
                    si_data, "Left", data.timestamp
                )
                self.hand_publisher.publish(hand_msg_left)
            if si_data.is_valid_hand_right():
                hand_msg_right = self.create_hand_pose_msg_from_si_data(
                    si_data, "Right", data.timestamp
                )
                self.hand_publisher.publish(hand_msg_right)

    def create_hand_pose_msg_from_si_data(
        self,
        si_data: hl2ss.unpack_si,
        hand: str,
        timestamp: int
    ) -> HandJointPosesUpdate:
        """
        Extracts the hand joint poses data from the HL2SS SI structure
        and forms a ROS HandJointPosesUpdate message.
        """
        # List containing joint names that matches the ordering in the HL2SS
        # SI_HandJointKind class
        joint_list = [
            "Palm", "Wrist", "ThumbMetacarpal", "ThumbProximal", "ThumbDistal",
            "ThumbTip", "IndexMetacarpal", "IndexProximal", "IndexIntermediate",
            "IndexDistal", "IndexTip", "MiddleMetacarpal", "MiddleProximal",
            "MiddleIntermediate", "MiddleDistal", "MiddleTip", "RingMetacarpal",
            "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
            "LittleMetacarpal", "LittleProximal", "LittleIntermediate",
            "LittleDistal", "LittleTip",
        ]
        if hand == "Left":
            hand_data = si_data.get_hand_left()
        elif hand == "Right":
            hand_data = si_data.get_hand_right()

        joint_poses = []
        for j in range(0, hl2ss.SI_HandJointKind.TOTAL):
            pose = hand_data.get_joint_pose(j)

            # Extract the position
            position = Point()
            position.x = float(pose.position[0])
            position.y = float(pose.position[1])
            position.z = float(pose.position[2])

            # Extract the orientation
            orientation = Quaternion()
            orientation.x = float(pose.orientation[0])
            orientation.y = float(pose.orientation[1])
            orientation.z = float(pose.orientation[2])
            orientation.w = float(pose.orientation[3])

            # Form the geometry pose message
            pose_msg = Pose()
            pose_msg.position = position
            pose_msg.orientation = orientation

            # Create the hand joint pose message
            joint_pose_msg = HandJointPose()
            joint_pose_msg.joint = joint_list[j]
            joint_pose_msg.pose = pose_msg
            joint_poses.append(joint_pose_msg)

        # Create the top level hand joint poses update message
        hand_msg = HandJointPosesUpdate()
        hand_msg.header.stamp = hl2ss_stamp_to_ros_time(timestamp)
        hand_msg.header.frame_id = "HandJointPosesUpdate"
        hand_msg.hand = hand
        hand_msg.joints = joint_poses

        return hand_msg


def main():
    rclpy.init()

    hl2ss_ros_bridge = HL2SSROSBridge()

    try:
        rclpy.spin(hl2ss_ros_bridge)
    except KeyboardInterrupt:
        hl2ss_ros_bridge.get_logger().info("Keyboard interrupt, shutting down.\n")
        hl2ss_ros_bridge.shutdown_clients()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    hl2ss_ros_bridge.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
