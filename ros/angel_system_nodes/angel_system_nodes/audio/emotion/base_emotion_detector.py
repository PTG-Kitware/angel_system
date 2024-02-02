import queue
from rclpy.node import Node
from termcolor import colored
import threading
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from angel_msgs.msg import DialogueUtterance
from angel_utils import declare_and_get_parameters
from angel_utils import make_default_main
from angel_system_nodes.audio import dialogue_utterance_processing


IN_TOPIC = "input_topic"
OUT_TOPIC = "user_emotion_topic"

# Currently supported emotions. This is tied with the emotions
# output to VaderSentiment (https://github.com/cjhutto/vaderSentiment) and
# will be subject to change in future iterations.
LABEL_MAPPINGS = {"pos": "positive", "neg": "negative", "neu": "neutral"}

# See https://github.com/cjhutto/vaderSentiment#about-the-scoring for more details.
# The below thresholds are per vaderSentiment recommendation.
VADER_NEGATIVE_COMPOUND_THRESHOLD = -0.05
VADER_POSITIVE_COMPOUND_THRESHOLD = 0.05


class BaseEmotionDetector(Node):
    """
    This is the base emotion detection node that other emotion detection nodes
    should inherit from.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        # Handle parameterization.
        param_values = declare_and_get_parameters(
            self,
            [
                (IN_TOPIC,),
                (OUT_TOPIC,),
            ],
        )

        self._in_topic = param_values[IN_TOPIC]
        self._out_topic = param_values[OUT_TOPIC]

        # Handle subscription/publication topics.
        self._subscription = self.create_subscription(
            DialogueUtterance,
            self._in_expect_uintent_topic,
            self.emotion_detection_callback,
            1,
        )
        self._publication = self.create_publisher(DialogueUtterance, self._out_topic, 1)

        self.message_queue = queue.Queue()
        self.handler_thread = threading.Thread(target=self.process_message_queue)
        self.handler_thread.start()

        self.sentiment_analysis_model = SentimentIntensityAnalyzer()

    def _get_vader_sentiment_analysis(self, utterance: str):
        """
        Applies Vader Sentiment Analysis model to assign 'positive,' 'negative,'
        and 'neutral' sentiment labels. Returns with  a 100% confidence.
        """
        polarity_scores = self.sentiment_analysis_model.polarity_scores(utterance)
        if polarity_scores["compound"] >= VADER_POSITIVE_COMPOUND_THRESHOLD:
            classification = LABEL_MAPPINGS["pos"]
        elif polarity_scores["compound"] <= VADER_NEGATIVE_COMPOUND_THRESHOLD:
            classification = LABEL_MAPPINGS["neg"]
        else:
            classification = LABEL_MAPPINGS["neu"]

        confidence = 1.00
        colored_utterance = colored_utterance = colored(utterance, "light_blue")
        colored_emotion = colored(classification, "light_green")
        self.log.info(
            f'Rated user utterance:\n>>> "{colored_utterance}"'
            + f"\n>>> with emotion scores {polarity_scores}.\n>>> "
            + f'Classifying with emotion="{colored_emotion}" '
            + f"and score={confidence}"
        )
        return (classification, confidence)

    def get_inference(self, msg: DialogueUtterance):
        """
        Abstract away the different model inference calls depending on the
        node's configure model mode.
        """
        return self._get_vader_sentiment_analysis(msg.utterance_text)

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
            self.process_message(msg)

    def process_message(self, msg: DialogueUtterance):
        """
        Handles message publishing for an utterance with a detected emotion classification.
        """
        classification, confidence_score = self.get_inference(msg)
        pub_msg = dialogue_utterance_processing.copy_dialogue_utterance(
            msg,
            node_name="Emotion Detection",
            copy_time=self.get_clock().now().to_msg(),
        )
        # Overwrite the user emotion with the latest classification information.
        pub_msg.emotion = classification
        pub_msg.emotion_confidence_score = confidence_score
        self.emotion_publication.publish(pub_msg)

        # Log emotion detection information.
        colored_utterance = colored(pub_msg.utterance_text, "light_blue")
        colored_emotion = colored(pub_msg.emotion, "light_green")
        self.log.info(
            f'Publishing {{"{colored_emotion}": {confidence_score}}} '
            + f'to {self._out_topic} for:\n>>> "{colored_utterance}"'
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
