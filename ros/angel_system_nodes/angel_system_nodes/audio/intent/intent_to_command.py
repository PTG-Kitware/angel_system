import rclpy
from rclpy.node import Node
import yaml

from angel_msgs.msg import (
    InterpretedAudioUserIntent,
    SystemCommands,
)
from angel_utils import make_default_main


# Parameter name constants
PARAM_CONFIRMED_INTENT_TOPIC = "confirmed_intent_topic"
PARAM_SYS_CMD_TOPIC = "sys_cmd_topic"
PARAM_INTENT_TO_CMD_MAP = "intent_to_cmd_map"


class IntentToCommand(Node):
    """
    ROS node that converts user intent messages to system commands.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        parameter_names = [
            PARAM_CONFIRMED_INTENT_TOPIC,
            PARAM_SYS_CMD_TOPIC,
            PARAM_INTENT_TO_CMD_MAP,
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

        self._confirmed_intent_topic = self.get_parameter(
            PARAM_CONFIRMED_INTENT_TOPIC
        ).value
        self._sys_cmd_topic = self.get_parameter(PARAM_SYS_CMD_TOPIC).value
        self._intent_to_cmd_map_path = self.get_parameter(PARAM_INTENT_TO_CMD_MAP).value

        # log inputs for interpreted type and value
        self.log.info(
            f"Confirmed intent topic: "
            f"({type(self._confirmed_intent_topic).__name__}) "
            f"{self._confirmed_intent_topic}"
        )
        self.log.info(
            f"System command topic: "
            f"({type(self._sys_cmd_topic).__name__}) "
            f"{self._sys_cmd_topic}"
        )
        self.log.info(
            f"Intent to cmd map: "
            f"({type(self._intent_to_cmd_map_path).__name__}) "
            f"{self._intent_to_cmd_map_path}"
        )

        # Load intent to cmd map (yaml)
        with open(self._intent_to_cmd_map_path, "r") as f:
            config = yaml.safe_load(f)
        self._intent_to_cmd_map = config["commands"]

        self._subscription = self.create_subscription(
            InterpretedAudioUserIntent,
            self._confirmed_intent_topic,
            self.intent_callback,
            1,
        )
        self._sys_cmd_publisher = self.create_publisher(
            SystemCommands, self._sys_cmd_topic, 1
        )

    def intent_callback(self, intent: InterpretedAudioUserIntent) -> None:
        """
        Callback function for the user intent messages. Converts the intent
        to a system command and publishes the command to the system command
        topic.
        """
        try:
            sys_cmd = self._intent_to_cmd_map[intent.user_intent]
        except KeyError:
            self.log.warning(
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
            self.log.warning(
                f"No command for {sys_cmd} found. Check intent to command mapping."
            )
            return

        # Publish system command message
        self.log.info(f"Publishing system command with command {sys_cmd}")
        self._sys_cmd_publisher.publish(sys_cmd_msg)


main = make_default_main(IntentToCommand)


if __name__ == "__main__":
    main()
