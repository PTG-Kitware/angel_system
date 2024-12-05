"""
Simple node to take an image, extract its header's timestamp and forward just
that to the configured topic.
"""

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image

from angel_utils import declare_and_get_parameters, RateTracker
from angel_utils import make_default_main
from angel_msgs.msg import ImageMetadata


PARAM_INPUT_TOPIC = "image_topic"
PARAM_OUTPUT_TOPIC = "output_topic"


class ImageMetadataRelay(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)

        param_values = declare_and_get_parameters(
            self,
            [
                (PARAM_INPUT_TOPIC,),
                (PARAM_OUTPUT_TOPIC,),
            ],
        )

        self._rate_tracker = RateTracker()

        self._sub = self.create_subscription(
            Image,
            param_values[PARAM_INPUT_TOPIC],
            self.input_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._pub = self.create_publisher(
            ImageMetadata,
            param_values[PARAM_OUTPUT_TOPIC],
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def input_callback(self, msg: Image) -> None:
        new_msg = ImageMetadata()
        new_msg.image_source_stamp = msg.header.stamp
        new_msg.height = msg.height
        new_msg.width = msg.width
        new_msg.header.stamp = self.get_clock().now().to_msg()

        self._pub.publish(new_msg)
        self._rate_tracker.tick()
        self.get_logger().info(
            f"[input_callback] ImageMetadata message with TS={new_msg.header.stamp}, "
            f"source TS={new_msg.image_source_stamp}, "
            f"image size {msg.width}x{msg.height}, "
            f"rate={self._rate_tracker.get_rate_avg()} Hz",
        )


main = make_default_main(ImageMetadataRelay, multithreaded_executor=2)


if __name__ == "__main__":
    main()
