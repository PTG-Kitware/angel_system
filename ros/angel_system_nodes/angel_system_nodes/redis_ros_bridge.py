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
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import websockets

from angel_msgs.msg import (
    HandJointPose,
    HandJointPosesUpdate,
    HeadsetAudioData,
)
import angel_system.hl2ss_viewer.BBN_redis_frame_load as holoframe
from angel_utils import RateTracker
from angel_utils.conversion import hl2ss_stamp_to_ros_time


BRIDGE = CvBridge()


PARAM_PV_IMAGES_TOPIC = "image_topic"
PARAM_URL = "url"


class async2sync:
    '''Helper to have a method be both sync and async.'''
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

        parameter_names = [
            PARAM_PV_IMAGES_TOPIC,
            PARAM_URL,
        ]
        set_parameters = self.declare_parameters(
            namespace="",
            parameters=[(p,) for p in parameter_names],
        )
        # Check for not-set parameters
        some_not_set = False
        for p in set_parameters:
            if p.type_ is rclpy.parameter.Parameter.Type.NOT_SET:
                some_not_set = True
                self.log.error(f"Parameter not set: {p.name}")
        if some_not_set:
            raise ValueError("Some parameters are not set.")

        self._image_topic = self.get_parameter(PARAM_PV_IMAGES_TOPIC).value
        self._url = self.get_parameter(PARAM_URL).value
        self.log.info(f"PV Images topic: "
                      f"({type(self._image_topic).__name__}) "
                      f"{self._image_topic}")
        self.log.info(f"URL: "
                      f"({type(self._url).__name__}) "
                      f"{self._url}")

        # Create frame publisher
        self.ros_frame_publisher = self.create_publisher(
            Image,
            self._image_topic,
            1
        )

        self.log.info("Starting publishing threads...")
        # Start the frame publishing thread
        self._pv_active = Event()
        self._pv_active.set()
        self._pv_rate_tracker = RateTracker()
        self._pv_thread = Thread(
            target=self.pv_publisher,
            name="publish_pv"
        )
        self._pv_thread.daemon = True
        self._pv_thread.start()
        self.log.info("Starting publishing threads... Done")

    def shutdown_clients(self) -> None:
        """
        Shuts down the frame publishing thread and the HL2SS client.
        """
        # Stop frame publishing thread
        self._pv_active.clear()  # make RT active flag "False"
        self._pv_thread.join()
        self.get_logger().info("PV thread closed")

    def pv_publisher(self) -> None:
        """
        Main thread that starts the async image publish method.
        Images are published directly in the `publish_image` method.
        """
        self.publish_images(
            sid="main",
        )

    @async2sync
    async def publish_images(
        self,
        sid: str,
    ):
        async with websockets.connect(
            f'ws://{self._url}/data/{sid}/pull?header=0&latest=1',
            max_size=None
        ) as ws:
            while self._pv_active.wait(0):
                # read the data
                data = await ws.recv()
                if not data:
                    print("No data yet :(")
                    continue
                d = holoframe.load(data)

                try:
                    image_msg = BRIDGE.cv2_to_imgmsg(d["image"], encoding="bgr8")
                    image_msg.header.stamp = hl2ss_stamp_to_ros_time(d["time"])
                    image_msg.header.frame_id = "PVFramesBGR"
                except TypeError as e:
                    self.get_logger().warning(f"{e}")

                self.ros_frame_publisher.publish(image_msg)

                self._pv_rate_tracker.tick()
                self.get_logger().info(f"Published image message (hz: "
                                        f"{self._pv_rate_tracker.get_rate_avg()})",
                                        throttle_duration_sec=1)


def main():
    rclpy.init()

    redis_ros_bridge = RedisROSBridge()

    try:
        rclpy.spin(redis_ros_bridge)
    except KeyboardInterrupt:
        redis_ros_bridge.get_logger().info("Keyboard interrupt, shutting down.\n")
        redis_ros_bridge.shutdown_clients()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    redis_ros_bridge.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
