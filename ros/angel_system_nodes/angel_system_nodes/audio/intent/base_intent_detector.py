import queue
import rclpy
from rclpy.node import Node
from termcolor import colored
import threading

from angel_msgs.msg import InterpretedAudioUserIntent, Utterance
from angel_utils import declare_and_get_parameters

NEXT_STEP_KEYPHRASES = ["skip", "next", "next step"]
PREV_STEP_KEYPHRASES = ["previous", "previous step", "last step", "go back"]
QUESTION_KEYPHRASES = ["question"]
OVERRIDE_KEYPHRASES = ["angel", "angel system"]

# TODO(derekahmed): Please figure out how to keep this sync-ed with
# config/angel_system_cmds/user_intent_to_sys_cmd_v1.yaml.
# Please refer to labels defined in
# https://docs.google.com/document/d/1uuvSL5de3LVM9c0tKpRKYazDxckffRHf7IAcabSw9UA .
INTENT_LABELS = ["next_step", "prev_step", "inquiry", "other"]

UTTERANCES_TOPIC = "utterances_topic"
PARAM_EXPECT_USER_INTENT_TOPIC = "expect_user_intent_topic"
PARAM_INTERP_USER_INTENT_TOPIC = "interp_user_intent_topic"


class BaseIntentDetector(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        # Handle parameterization.
        param_values = declare_and_get_parameters(
            self,
            [
                (UTTERANCES_TOPIC,),
                (PARAM_EXPECT_USER_INTENT_TOPIC,),
                (PARAM_INTERP_USER_INTENT_TOPIC,),
            ],
        )
        self._utterances_topic = param_values[UTTERANCES_TOPIC]
        self._expect_uintent_topic = param_values[PARAM_EXPECT_USER_INTENT_TOPIC]
        self._interp_uintent_topic = param_values[PARAM_INTERP_USER_INTENT_TOPIC]

        # Handle subscription/publication topics.
        self.subscription = self.create_subscription(
            Utterance, self._utterances_topic, self.utterance_callback, 1
        )
        self._expected_publisher = self.create_publisher(
            InterpretedAudioUserIntent, self._expect_uintent_topic, 1
        )
        self._interp_publisher = self.create_publisher(
            InterpretedAudioUserIntent, self._interp_uintent_topic, 1
        )

        self.utterance_message_queue = queue.Queue()
        self.handler_thread = threading.Thread(
            target=self.process_utterance_message_queue
        )
        self.handler_thread.start()

    def utterance_callback(self, msg):
        """
        This is the main ROS node listener callback loop that will process all messages received
        via subscribed topics.
        """
        self.log.debug(f'Received message:\n\n"{msg.value}"')
        self.utterance_message_queue.put(msg)

    def process_utterance_message_queue(self):
        """
        Constant loop to process received messages.
        """
        while True:
            msg = self.utterance_message_queue.get()
            self.log.debug(f'Processing message:\n\n"{msg.value}"')
            intent, score = self.detect_intents(msg)
            if not intent:
                continue
            self.publish_msg(msg.value, intent, score)

    def detect_intents(self, msg):
        """
        Keyphrase search for intent detection. This implementation does simple
        string matching to assign a detected label. When multiple intents are
        detected, the message is classified as the first intent or as an
        'inquiry' if 'inquiry' is one of the classifications.
        """

        def _tiebreak_intents(intents, confidences):
            classification = intents[0]
            score = confidences[0]
            if len(intents) > 1:
                for i, intent in enumerate(intents):
                    if intent == INTENT_LABELS[2]:
                        classification, score = intent, confidences[i]
                self.log.info(
                    f'Detected multiple intents: {intents}. Selected "{classification}".'
                )
            return classification, score

        lower_utterance = msg.value.lower()
        intents = []
        confidences = []
        if self._contains_phrase(lower_utterance, NEXT_STEP_KEYPHRASES):
            intents.append(INTENT_LABELS[0])
            confidences.append(0.5)
        if self._contains_phrase(lower_utterance, PREV_STEP_KEYPHRASES):
            intents.append(INTENT_LABELS[1])
            confidences.append(0.5)
        if self._contains_phrase(lower_utterance, QUESTION_KEYPHRASES):
            intents.append(INTENT_LABELS[2])
            confidences.append(0.5)
        if not intents:
            colored_utterance = colored(msg.value, "light_blue")
            self.log.info(f'No intents detected for:\n>>> "{colored_utterance}":')
            return None, -1.0

        classification, confidence = _tiebreak_intents(intents, confidences)
        classification = colored(classification, "light_green")
        return classification, confidence

    def publish_msg(self, utterance, intent, score):
        """
        Handles message publishing for an utterance with a detected intent.
        """
        intent_msg = InterpretedAudioUserIntent()
        intent_msg.header.frame_id = "Intent Detection"
        intent_msg.header.stamp = self.get_clock().now().to_msg()
        intent_msg.utterance_text = utterance
        intent_msg.user_intent = intent
        intent_msg.confidence = score
        published_topic = None
        if self._contains_phrase(utterance.lower(), OVERRIDE_KEYPHRASES):
            intent_msg.confidence = 1.0
            self._expected_publisher.publish(intent_msg)
            published_topic = PARAM_EXPECT_USER_INTENT_TOPIC
        else:
            self._interp_publisher.publish(intent_msg)
            published_topic = PARAM_INTERP_USER_INTENT_TOPIC

        colored_utterance = colored(utterance, "light_blue")
        colored_intent = colored(intent_msg.user_intent, "light_green")
        self.log.info(
            f'Publishing {{"{colored_intent}": {score}}} to {published_topic} '
            + f'for:\n>>> "{colored_utterance}"'
        )

    def _contains_phrase(self, utterance, phrases):
        for phrase in phrases:
            if phrase in utterance:
                return True
        return False


def main():
    rclpy.init()
    intent_detector = BaseIntentDetector()
    rclpy.spin(intent_detector)
    intent_detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
