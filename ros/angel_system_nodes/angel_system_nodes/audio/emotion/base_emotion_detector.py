import queue
from termcolor import colored
import threading
from typing import Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from angel_msgs.msg import DialogueUtterance
from angel_system_nodes.audio import dialogue
from angel_utils import declare_and_get_parameters
from angel_utils import make_default_main


INPUT_EMOTION_TOPIC = "in_emotion_topic"
OUTPUT_EMOTION_TOPIC = "out_emotion_topic"

# Currently supported emotions. This is tied with the emotions
# output to VaderSentiment (https://github.com/cjhutto/vaderSentiment) and
# will be subject to change in future iterations.
LABEL_MAPPINGS = {"pos": "positive", "neg": "negative", "neu": "neutral"}

# See https://github.com/cjhutto/vaderSentiment#about-the-scoring for more details.
# The below thresholds are per vaderSentiment recommendation.
VADER_NEGATIVE_COMPOUND_THRESHOLD = -0.05
VADER_POSITIVE_COMPOUND_THRESHOLD = 0.05


class BaseEmotionDetector(dialogue.AbstractDialogueNode):
    """
    Base Emotion Detection node. The default emotion detection algorithm leverages Vader
    Sentiment analysis (https://github.com/cjhutto/vaderSentiment).
    """

    def __init__(self):
        super().__init__()

        # Handle parameterization.
        param_values = declare_and_get_parameters(
            self,
            [
                (INPUT_EMOTION_TOPIC,),
                (OUTPUT_EMOTION_TOPIC,),
            ],
        )

        self._in_emotion_topic = param_values[INPUT_EMOTION_TOPIC]
        self._out_emotion_topic = param_values[OUTPUT_EMOTION_TOPIC]

        # Handle subscription/publication topics.
        self._subscriber = self.create_subscription(
            DialogueUtterance,
            self._in_emotion_topic,
            self.emotion_detection_callback,
            1,
        )
        self._publisher = self.create_publisher(
            DialogueUtterance, self._out_emotion_topic, 1
        )

        self.message_queue = queue.Queue()
        self.handler_thread = threading.Thread(target=self.process_message_queue)
        self.handler_thread.start()

        self.sentiment_analysis_model = SentimentIntensityAnalyzer()

    def _get_vader_sentiment_analysis(self, msg: DialogueUtterance):
        """
        Applies Vader Sentiment Analysis model to assign 'positive,' 'negative,'
        and 'neutral' sentiment labels. Returns with  a 100% confidence.
        """
        polarity_scores = self.sentiment_analysis_model.polarity_scores(
            msg.utterance_text
        )
        if polarity_scores["compound"] >= VADER_POSITIVE_COMPOUND_THRESHOLD:
            classification = LABEL_MAPPINGS["pos"]
        elif polarity_scores["compound"] <= VADER_NEGATIVE_COMPOUND_THRESHOLD:
            classification = LABEL_MAPPINGS["neg"]
        else:
            classification = LABEL_MAPPINGS["neu"]

        confidence = 1.00
        colored_utterance = colored(msg.utterance_text, "light_blue")
        colored_emotion = colored(classification, "light_green")
        self.log.info(
            f'Rated user utterance:\n>>> "{colored_utterance}"'
            + f"\n>>> with emotion scores {polarity_scores}.\n>>> "
            + f'Classifying with emotion="{colored_emotion}" '
            + f"and score={confidence}"
        )
        return (classification, confidence)

    def get_inference(self, msg: DialogueUtterance) -> Tuple[str, int]:
        """
        Abstract away the different model inference calls depending on the
        node's configure model mode.
        """
        return self._get_vader_sentiment_analysis(msg)

    def emotion_detection_callback(self, msg: DialogueUtterance):
        """
        This is the main ROS node listener callback loop that will process
        all messages received via subscribed topics.
        """
        self.log.debug(f'Received message:\n\n"{msg.utterance_text}"')
        if not self._apply_filter(msg):
            return
        self.message_queue.put(msg)

    def process_message_queue(self):
        """
        Constant loop to process received messages.
        """
        while True:
            msg = self.message_queue.get()
            self.log.debug(f'Processing message:\n\n"{msg.utterance_text}"')
            classification, confidence_score = self.get_inference(msg)
            self.publish_detected_emotion(msg, classification, confidence_score)

    def publish_detected_emotion(
        self,
        subscribe_msg: DialogueUtterance,
        classification: str,
        confidence_score: float,
    ):
        """
        Handles message publishing for an utterance with a detected emotion classification.
        """
        publish_msg = self._copy_dialogue_utterance(
            subscribe_msg, "Emotion Detection", self.get_clock().now().to_msg()
        )

        publish_msg.emotion = classification
        publish_msg.emotion_confidence_score = confidence_score
        self._publisher.publish(publish_msg)
        colored_utterance = colored(publish_msg.utterance_text, "light_blue")
        colored_emotion = colored(publish_msg.emotion, "light_green")
        self.log.info(
            f'Classifying "{colored_utterance}" as '
            f'{{"{colored_emotion}":{confidence_score}}}. Publishing '
            + f"to {self._out_emotion_topic}"
        )

    def _apply_filter(self, msg):
        """
        Abstracts away any filtering to apply on received messages. Return
        none if the message should be filtered out. Else, return the incoming
        msg if it can be included.
        """
        return msg


main = make_default_main(BaseEmotionDetector)


if __name__ == "__main__":
    main()
