from threading import Event, Thread

from cv_bridge import CvBridge
from geometry_msgs.msg import (
    Point,
    Pose,
    Quaternion,
)
from rclpy.node import Node
from sensor_msgs.msg import Image
import websockets

from angel_msgs.msg import (
    HandJointPose,
    HandJointPosesUpdate,
    HeadsetAudioData,
)
from angel_utils import declare_and_get_parameters, RateTracker
from angel_utils import make_default_main
from angel_utils.conversion import hl2ss_stamp_to_ros_time
from angel_utils.hand import JOINT_LIST
from hl2ss.viewer import hl2ss
from hl2ss.viewer import BBN_redis_frame_load as holoframe


BRIDGE = CvBridge()


PARAM_PV_IMAGES_TOPIC = "image_topic"
PARAM_HAND_POSE_TOPIC = "hand_pose_topic"
PARAM_URL = "url"


class async2sync:
    """Helper to have a method be both sync and async."""

    def __init__(self, func_async):
        import functools

        functools.update_wrapper(self, func_async)
        self.asyncio = func_async

    def __get__(self, inst, own):
        return self.__class__(self.asyncio.__get__(inst, own))

    def __call__(self, *a, **kw):
        import asyncio

        return asyncio.run(self.asyncio(*a, **kw))


class RedisROSBridge(Node):
    """
    ROS node that uses the BBN Redis client/server library to convert data to
    ROS messages used throughout the ANGEL system.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        param_values = declare_and_get_parameters(
            self,
            [
                (PARAM_HAND_POSE_TOPIC,),
                (PARAM_PV_IMAGES_TOPIC,),
                (PARAM_URL,),
            ],
        )

        self._image_topic = param_values[PARAM_PV_IMAGES_TOPIC]
        self._hand_pose_topic = param_values[PARAM_HAND_POSE_TOPIC]
        self._url = param_values[PARAM_URL]

        # Define stream IDs
        self._audio_sid = "mic0"
        self._pv_sid = "main"
        self._si_sid = "si"

        # Create frame publisher
        self.ros_frame_publisher = self.create_publisher(Image, self._image_topic, 1)
        # Create the hand joint pose publisher
        self.ros_hand_publisher = self.create_publisher(
            HandJointPosesUpdate, self._hand_pose_topic, 1
        )

        self.log.info("Starting publishing threads...")
        # Start the frame publishing thread
        self._pv_active = Event()
        self._pv_active.set()
        self._pv_rate_tracker = RateTracker()
        self._pv_thread = Thread(target=self.pv_publisher, name="publish_pv")
        self._pv_thread.daemon = True
        self._pv_thread.start()
        # Start the hand tracking data thread
        self._si_active = Event()
        self._si_active.set()
        self._si_rate_tracker = RateTracker()
        self._si_thread = Thread(target=self.si_publisher, name="publish_si")
        self._si_thread.daemon = True
        self._si_thread.start()
        self.log.info("Starting publishing threads... Done")

    def shutdown_clients(self) -> None:
        """
        Shuts down the publishing threads.
        """
        # Stop frame publishing thread
        self._pv_active.clear()
        self._pv_thread.join()
        self.log.info("PV thread closed")
        # Stop SI publishing thread
        self._si_active.clear()
        self._si_thread.join()
        self.log.info("SI thread closed")

    def pv_publisher(self) -> None:
        """
        Main thread that starts the async image publish method.
        Images are published directly in the `publish_image` method.
        """
        self.publish_images(sid=self._pv_sid)

    def si_publisher(self) -> None:
        """
        Thread the gets spatial input packets from the HL2SS SI client, converts
        the SI data to ROS messages, and publishes them.

        Currently only publishes hand tracking data. However, eye gaze data and
        head pose data is also available in the SI data packet.
        """
        self.publish_si_data(sid=self._si_sid)

    @async2sync
    async def publish_images(
        self,
        sid: str,
    ):
        connection_str = f"ws://{self._url}/data/{sid}/pull?header=0&latest=1"
        self.log.info(f"Connecting websocket to URL: {connection_str}")
        async with websockets.connect(connection_str, max_size=None) as ws:
            while self._pv_active.wait(0):
                # read the data
                data = await ws.recv()
                if not data:
                    self.log.warning("No data yet :(")
                    continue
                d = holoframe.load(data)

                try:
                    # The most recent version of the HL2SS redis streaming
                    # service (git hash 27c223d) is outputting RGB imagery.
                    image_msg = BRIDGE.cv2_to_imgmsg(d["image"], encoding="rgb8")
                    image_msg.header.stamp = hl2ss_stamp_to_ros_time(d["time"])
                    image_msg.header.frame_id = "PVFramesBGR"
                except TypeError as e:
                    self.log.warning(f"{e}")

                self.ros_frame_publisher.publish(image_msg)

                self._pv_rate_tracker.tick()
                self.log.info(
                    f"Published image message (hz: "
                    f"{self._pv_rate_tracker.get_rate_avg()})",
                    throttle_duration_sec=1,
                )

    @async2sync
    async def publish_si_data(
        self,
        sid: str,
    ):
        connection_str = f"ws://{self._url}/data/{sid}/pull?header=0&latest=1"
        self.log.info(f"Connecting websocket to URL: {connection_str}")
        async with websockets.connect(connection_str, max_size=None) as ws:
            while self._si_active.wait(0):
                # read the data
                data = await ws.recv()
                if not data:
                    self.log.warning("No data yet :(")
                    continue
                d = holoframe.load(data)
                si_data = hl2ss.unpack_si(d["data"])

                # Publish the hand tracking data if it is valid
                if si_data.is_valid_hand_left():
                    hand_msg_left = self.create_hand_pose_msg_from_si_data(
                        si_data, "Left", d["time"]
                    )
                    self.ros_hand_publisher.publish(hand_msg_left)
                if si_data.is_valid_hand_right():
                    hand_msg_right = self.create_hand_pose_msg_from_si_data(
                        si_data, "Right", d["time"]
                    )
                    self.ros_hand_publisher.publish(hand_msg_right)

                self._si_rate_tracker.tick()
                self.log.info(
                    f"Published hand pose message (hz: "
                    f"{self._si_rate_tracker.get_rate_avg()})",
                    throttle_duration_sec=1,
                )

    def create_hand_pose_msg_from_si_data(
        self, si_data: hl2ss.unpack_si, hand: str, timestamp: int
    ) -> HandJointPosesUpdate:
        """
        Extracts the hand joint poses data from the HL2SS SI structure
        and forms a ROS HandJointPosesUpdate message.
        """
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
            joint_pose_msg.joint = JOINT_LIST[j]
            joint_pose_msg.pose = pose_msg
            joint_poses.append(joint_pose_msg)

        # Create the top level hand joint poses update message
        hand_msg = HandJointPosesUpdate()
        hand_msg.header.stamp = hl2ss_stamp_to_ros_time(timestamp)
        hand_msg.header.frame_id = "HandJointPosesUpdate"
        hand_msg.hand = hand
        hand_msg.joints = joint_poses

        return hand_msg

    def destroy_node(self):
        """
        Clean up resources.
        """
        self.shutdown_clients()


main = make_default_main(RedisROSBridge)


if __name__ == "__main__":
    main()
