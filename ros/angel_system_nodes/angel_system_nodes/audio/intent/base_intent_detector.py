import queue
from rclpy.node import Node
from termcolor import colored
import threading

from angel_msgs.msg import DialogueUtterance
from angel_system_nodes.audio import dialogue
from angel_utils import declare_and_get_parameters
from angel_utils import make_default_main

NEXT_STEP_KEYPHRASES = ["skip", "next", "next step"]
PREV_STEP_KEYPHRASES = ["previous", "previous step", "last step", "go back"]
QUESTION_KEYPHRASES = ["question"]
OVERRIDE_KEYPHRASES = ["angel override", "angel system override"]

# TODO(derekahmed): Please figure out how to keep this sync-ed with
# config/angel_system_cmds/user_intent_to_sys_cmd_v1.yaml.
# Please refer to labels defined in
# https://docs.google.com/document/d/1uuvSL5de3LVM9c0tKpRKYazDxckffRHf7IAcabSw9UA .
INTENT_LABELS = ["next_step", "prev_step", "inquiry", "other"]

INPUT_INTENT_TOPIC = "in_intent_topic"
OUTPUT_INTENT_TOPIC = "out_intent_topic"


class BaseIntentDetector(dialogue.AbstractDialogueNode):
    def __init__(self):
        super().__init__()

        # Handle parameterization.
        param_values = declare_and_get_parameters(
            self,
            [
                (INPUT_INTENT_TOPIC,),
                (OUTPUT_INTENT_TOPIC,),
            ],
        )
        self._in_intent_topic = param_values[INPUT_INTENT_TOPIC]
        self._out_intent_topic = param_values[OUTPUT_INTENT_TOPIC]

        # Handle subscription/publication topics.
        self.subscription = self.create_subscription(
            DialogueUtterance, self._in_intent_topic, self.utterance_callback, 1
        )
        self._publisher = self.create_publisher(
            DialogueUtterance, self._out_intent_topic, 1
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
        self.log.debug(f'Received message:\n\n"{msg.utterance_text}"')
        self.utterance_message_queue.put(msg)

    def process_utterance_message_queue(self):
        """
        Constant loop to process received messages.
        """
        while True:
            msg = self.utterance_message_queue.get()
            self.log.debug(f'Processing message:\n\n"{msg.utterance_text}"')
            intent, score = self.detect_intents(msg)
            if not intent:
                continue
            self.publish_msg(msg, intent, score)

    def detect_intents(self, msg: DialogueUtterance):
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

        lower_utterance = msg.utterance_text.lower()
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
            colored_utterance = colored(msg.utterance_text, "light_blue")
            self.log.info(f'No intents detected for:\n>>> "{colored_utterance}":')
            return None, -1.0

        classification, confidence = _tiebreak_intents(intents, confidences)
        classification = colored(classification, "light_green")
        return classification, confidence

    def publish_msg(
        self, subscribe_msg: DialogueUtterance, intent: str, confidence_score: float
    ):
        """
        Handles message publishing for an utterance with a detected intent.
        """
        publish_msg = self._copy_dialogue_utterance(
            subscribe_msg, "Intent Detection", self.get_clock().now().to_msg()
        )
        publish_msg.intent = intent
        publish_msg.intent_confidence_score = confidence_score
        if self._contains_phrase(
            publish_msg.utterance_text.lower(), OVERRIDE_KEYPHRASES
        ):
            # OVERRIDE_KEYPHRASES will override the intent
            publish_msg.intent_confidence_score = 1.0
        self._publisher.publish(publish_msg)

        colored_utterance = colored(publish_msg.utterance_text, "light_blue")
        colored_intent = colored(publish_msg.intent, "light_green")
        self.log.info(
            f'Classifying "{colored_utterance}" as '
            f'{{"{colored_intent}":{confidence_score}}}. Publishing '
            + f"to {self._out_intent_topic}"
        )

    def _contains_phrase(self, utterance, phrases):
        for phrase in phrases:
            if phrase in utterance:
                return True
        return False


main = make_default_main(BaseIntentDetector)


if __name__ == "__main__":
    main()
