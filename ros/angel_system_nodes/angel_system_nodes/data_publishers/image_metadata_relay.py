"""
Simple node to take an image, extract its header's timestamp and forward just
that to the configured topic.
"""

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

from angel_utils import declare_and_get_parameters, RateTracker
from angel_utils import make_default_main


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
            CameraInfo,
            param_values[PARAM_OUTPUT_TOPIC],
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def input_callback(self, msg: Image) -> None:
        new_msg = CameraInfo()
        new_msg.header = msg.header
        new_msg.height = msg.height
        new_msg.width = msg.width

        self._pub.publish(new_msg)
        self._rate_tracker.tick()
        self.get_logger().info(
            f"[input_callback] CameraInfo message with TS={msg.header.stamp}, "
            f"image size {msg.width}x{msg.height}, "
            f"rate={self._rate_tracker.get_rate_avg()} Hz",
        )


main = make_default_main(ImageMetadataRelay, multithreaded_executor=2)


if __name__ == "__main__":
    main()
