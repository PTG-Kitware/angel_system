import rclpy
from rclpy.node import Node
import yaml

from angel_msgs.msg import (
    InterpretedAudioUserIntent,
    SystemCommands,
)


class IntentToCommand(Node):
    """
    ROS node that converts user intent messages to system commands.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        self._confirmed_intent_topic = (
            self.declare_parameter("confirmed_intent_topic", "")
            .get_parameter_value()
            .string_value
        )
        self._sys_cmd_topic = (
            self.declare_parameter("sys_cmd_topic", "")
            .get_parameter_value()
            .string_value
        )
        self._intent_to_cmd_map_path = (
            self.declare_parameter("intent_to_cmd_map", "")
            .get_parameter_value()
            .string_value
        )

        if self._confirmed_intent_topic == "":
            raise ValueError(
                "Please provide the confirmed intent topic with the `confirmed_intent_topic` parameter"
            )
        if self._intent_to_cmd_map_path == "":
            raise ValueError(
                "Please provide the intent command map path with the `intent_to_cmd_map` parameter"
            )
        if self._sys_cmd_topic == "":
            raise ValueError(
                "Please provide the system command topic with the `sys_cmd_topic` parameter"
            )

        log = self.get_logger()
        log.info(f"Confirmed intent topic: {self._confirmed_intent_topic}")
        log.info(f"System command topic: {self._sys_cmd_topic}")
        log.info(f"Intent to cmd map: {self._intent_to_cmd_map_path}")

        # Load intent to cmd map (yaml)
        with open(self._intent_to_cmd_map_path, 'r') as f:
            config = yaml.safe_load(f)
        self._intent_to_cmd_map = config["commands"]

        self._subscription = self.create_subscription(
            InterpretedAudioUserIntent,
            self._confirmed_intent_topic,
            self.intent_callback,
            1
        )
        self._sys_cmd_publisher = self.create_publisher(
            SystemCommands,
            self._sys_cmd_topic,
            1
        )

    def intent_callback(self, intent: InterpretedAudioUserIntent) -> None:
        """
        Callback function for the user intent messages. Converts the intent
        to a system command and publishes the command to the system command
        topic.
        """
        log = self.get_logger()

        try:
            sys_cmd = self._intent_to_cmd_map[intent.user_intent]
        except KeyError:
            log.warning(
                f"User intent `{intent.user_intent}` not found in sys cmd map. "
                f"Available intents are {list(self._intent_to_cmd_map.keys())}"
            )
            return

        sys_cmd_msg = SystemCommands()

        # TODO: This only works currently since we only recognize a few boolean
        # based commands (next step, previous step). This will break down if
        # more complex (non-boolean) commands are added.
        try:
            # Set the corresponding sys cmd field to True
            setattr(sys_cmd_msg, sys_cmd, True)
        except AttributeError:
            log.warning(
                f"No command for {sys_cmd} found. Check intent to command mapping."
            )
            return

        # Publish system command message
        log.info(f"Publishing system command with command {sys_cmd}")
        self._sys_cmd_publisher.publish(sys_cmd_msg)


def main():
    rclpy.init()

    node = IntentToCommand()
    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
