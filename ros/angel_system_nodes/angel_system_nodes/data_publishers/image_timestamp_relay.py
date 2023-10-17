"""
Simple node to take an image, extract its header's timestamp and forward just
that to the configured topic.
"""

from builtin_interfaces.msg import Time
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image

from angel_utils import declare_and_get_parameters


PARAM_INPUT_TOPIC = "image_topic"
PARAM_OUTPUT_TOPIC = "output_topic"


class ImageTimestampRelay(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)

        param_values = declare_and_get_parameters(
            self,
            [
                (PARAM_INPUT_TOPIC,),
                (PARAM_OUTPUT_TOPIC,),
            ],
        )

        self._sub = self.create_subscription(
            Image,
            param_values[PARAM_INPUT_TOPIC],
            self.input_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._pub = self.create_publisher(
            Time,
            param_values[PARAM_OUTPUT_TOPIC],
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def input_callback(self, msg: Image) -> None:
        self.get_logger().info(f"Forwarding image with TS={msg.header.stamp}")
        self._pub.publish(msg.header.stamp)


def main():
    rclpy.init()

    activity_classifier = ImageTimestampRelay()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(activity_classifier)
    try:
        executor.spin()
    except KeyboardInterrupt:
        activity_classifier.rt_stop()
        activity_classifier.get_logger().info("Keyboard interrupt, shutting down.\n")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    activity_classifier.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
