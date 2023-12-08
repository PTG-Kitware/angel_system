import rclpy
from rclpy.node import Node

from angel_msgs.msg import InterpretedAudioUserIntent, Utterance
from angel_utils import make_default_main


# Please refer to labels defined in
# https://docs.google.com/document/d/1uuvSL5de3LVM9c0tKpRKYazDxckffRHf7IAcabSw9UA .
NEXT_STEP_KEYPHRASES = ["skip", "next", "next step"]
PREV_STEP_KEYPHRASES = ["previous", "previous step", "last step", "go back"]
OVERRIDE_KEYPHRASES = ["angel", "angel system"]

# TODO(derekahmed): Please figure out how to keep this sync-ed with
# config/angel_system_cmds/user_intent_to_sys_cmd_v1.yaml.
LABELS = ["Go to next step", "Go to previous step"]


UTTERANCES_TOPIC = "utterances_topic"
PARAM_EXPECT_USER_INTENT_TOPIC = "expect_user_intent_topic"
PARAM_INTERP_USER_INTENT_TOPIC = "interp_user_intent_topic"


class IntentDetector(Node):
    """
    As of Q12023, intent detection is derived heuristically. This will be shifted
    to a model-based approach in the near-future.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        parameter_names = [
            UTTERANCES_TOPIC,
            PARAM_EXPECT_USER_INTENT_TOPIC,
            PARAM_INTERP_USER_INTENT_TOPIC,
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

        self._utterances_topic = self.get_parameter(UTTERANCES_TOPIC).value
        self._expect_uintent_topic = self.get_parameter(
            PARAM_EXPECT_USER_INTENT_TOPIC
        ).value
        self._interp_uintent_topic = self.get_parameter(
            PARAM_INTERP_USER_INTENT_TOPIC
        ).value
        self.log.info(
            f"Utterances topic: "
            f"({type(self._utterances_topic).__name__}) "
            f"{self._utterances_topic}"
        )
        self.log.info(
            f"Expected User Intent topic: "
            f"({type(self._expect_uintent_topic).__name__}) "
            f"{self._expect_uintent_topic}"
        )
        self.log.info(
            f"Interpreted User Intent topic: "
            f"({type(self._interp_uintent_topic).__name__}) "
            f"{self._interp_uintent_topic}"
        )

        # TODO(derekahmed): Add internal queueing to reduce subscriber queue
        # size to 1.
        self.subscription = self.create_subscription(
            Utterance, self._utterances_topic, self.listener_callback, 10
        )

        self._expected_publisher = self.create_publisher(
            InterpretedAudioUserIntent, self._expect_uintent_topic, 1
        )

        self._interp_publisher = self.create_publisher(
            InterpretedAudioUserIntent, self._interp_uintent_topic, 1
        )

    def listener_callback(self, msg):
        log = self.get_logger()
        intent_msg = InterpretedAudioUserIntent()
        intent_msg.utterance_text = msg.value

        lower_utterance = msg.value.lower()
        if self.contains_phrase(lower_utterance, NEXT_STEP_KEYPHRASES):
            intent_msg.user_intent = LABELS[0]
            intent_msg.confidence = 0.5
        elif self.contains_phrase(lower_utterance, PREV_STEP_KEYPHRASES):
            intent_msg.user_intent = LABELS[1]
            intent_msg.confidence = 0.5
        else:
            log.info(f'Detected no intents for "{msg.value}":')
            return

        if self.contains_phrase(lower_utterance, OVERRIDE_KEYPHRASES):
            intent_msg.confidence = 1.0
            self._expected_publisher.publish(intent_msg)
        else:
            self._interp_publisher.publish(intent_msg)

        log.info(
            f'Detected intents for "{msg.value}":\n'
            + f'"{intent_msg.user_intent}": {intent_msg.confidence}'
        )

    def contains_phrase(self, utterance, phrases):
        for phrase in phrases:
            if phrase in utterance:
                return True
        return False


main = make_default_main(IntentDetector)


if __name__ == "__main__":
    main()
