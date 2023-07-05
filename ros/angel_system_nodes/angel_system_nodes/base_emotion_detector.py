import queue
import rclpy
from rclpy.node import Node
from termcolor import colored
import threading
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from angel_msgs.msg import InterpretedAudioUserEmotion, InterpretedAudioUserIntent


IN_EXPECT_USER_INTENT_TOPIC = "expect_user_intent_topic"
IN_INTERP_USER_INTENT_TOPIC = "interp_user_intent_topic"
OUT_INTERP_USER_EMOTION_TOPIC = "user_emotion_topic"

# Currently supported emotions. This is tied with the emotions
# output to VaderSentiment (https://github.com/cjhutto/vaderSentiment) and
# will be subject to change in future iterations.
LABEL_MAPPINGS = {
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral"
}

VADER_NEGATIVE_COMPOUND_THRESHOLD = -0.05
VADER_POSITIVE_COMPOUND_THRESHOLD = 0.05

class BaseEmotionDetector(Node):
    '''
    As of Q22023, emotion detection is derived via VaderSentiment
    (https://github.com/cjhutto/vaderSentiment).
    '''

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        parameter_names = [
            IN_EXPECT_USER_INTENT_TOPIC,
            IN_INTERP_USER_INTENT_TOPIC,
            OUT_INTERP_USER_EMOTION_TOPIC
        ]
        set_parameters = self.declare_parameters(
            namespace="",
            parameters=[(p,) for p in parameter_names],
        )
        some_not_set = False
        for p in set_parameters:
            if p.type_ is rclpy.parameter.Parameter.Type.NOT_SET:
                some_not_set = True
                self.log.error(f"Parameter not set: {p.name}")
        if some_not_set:
            raise ValueError("Some parameters are not set.")

        self._in_interp_uintent_topic = self.get_parameter(IN_INTERP_USER_INTENT_TOPIC).value
        self._in_expect_uintent_topic = self.get_parameter(IN_EXPECT_USER_INTENT_TOPIC).value
        self._out_interp_uemotion_topic = self.get_parameter(OUT_INTERP_USER_EMOTION_TOPIC).value
        self.log.info(f"Interpreted User Intents topic: "
                      f"({type(self._in_interp_uintent_topic).__name__}) "
                      f"{self._in_interp_uintent_topic}")
        self.log.info(f"Expected User Intents topic: "
                      f"({type(self._in_expect_uintent_topic).__name__}) "
                      f"{self._in_expect_uintent_topic}")
        self.log.info(f"Interpreted User Emotion topic: "
                      f"({type(self._out_interp_uemotion_topic).__name__}) "
                      f"{self._out_interp_uemotion_topic}")
        
        # Handle subscription/publication topics.
        self.interp_uintent_subscription = self.create_subscription(
            InterpretedAudioUserIntent,
            self._in_interp_uintent_topic,
            self.listener_callback,
            1)
        self.expect_uintent_subscription = self.create_subscription(
            InterpretedAudioUserIntent,
            self._in_expect_uintent_topic,
            self.listener_callback,
            1)
        self._interp_emo_publisher = self.create_publisher(
            InterpretedAudioUserEmotion,
            self._out_interp_uemotion_topic,
            1)
        
        self.message_queue = queue.Queue()
        self.handler_thread = threading.Thread(target=self.process_message_queue)
        self.handler_thread.start()

        self.sentiment_analysis_model = SentimentIntensityAnalyzer()

    def _get_vader_sentiment_analysis(self, utterance: str):
        '''
        Applies Vader Sentiment Analysis model to assign 'positive,' 'negative,'
        and 'neutral' sentiment labels. Returns with  a 100% confidence.
        '''
        polarity_scores = \
            self.sentiment_analysis_model.polarity_scores(utterance)
        if polarity_scores['compound'] >= VADER_POSITIVE_COMPOUND_THRESHOLD:
            classification = LABEL_MAPPINGS['pos']
        elif  polarity_scores['compound'] <= VADER_NEGATIVE_COMPOUND_THRESHOLD:
            classification = LABEL_MAPPINGS['neg']
        else:
            classification = LABEL_MAPPINGS['neu']

        confidence = 1.00
        colored_utterance = colored_utterance = colored(utterance, "light_blue")
        colored_emotion = colored(classification, "light_green")
        self.log.info(f"Rated user utterance:\n>>> \"{colored_utterance}\"" +\
                      f"\n>>> with emotion scores {polarity_scores}.\n>>> " +\
                      f"Classifying with emotion=\"{colored_emotion}\" " +\
                      f"and score={confidence}")     
        return (classification, confidence)

    def get_inference(self, msg):
        '''
        Abstract away the different model inference calls depending on the
        node's configure model mode.
        '''
        return self._get_vader_sentiment_analysis(msg.utterance_text)

    def listener_callback(self, msg):
        '''
        This is the main ROS node listener callback loop that will process
        all messages received via subscribed topics.
        '''
        self.log.debug(f"Received message:\n\n\"{msg.utterance_text}\"")
        if not self._apply_filter(msg):
            return
        self.message_queue.put(msg)

    def process_message_queue(self):
        '''
        Constant loop to process received messages.
        '''
        while True:
            msg = self.message_queue.get()
            self.log.debug(f"Processing message:\n\n\"{msg.utterance_text}\"")
            classification, confidence_score  = self.get_inference(msg)
            self.publish_detected_emotion(msg.utterance_text, classification, confidence_score)

    def publish_detected_emotion(self, utterance: str,  classification: str,
                                  confidence_score: float):
        '''
        Handles message publishing for an utterance with a detected emotion classification.
        '''
        emotion_msg = InterpretedAudioUserEmotion()
        emotion_msg.utterance_text = utterance
        emotion_msg.user_emotion = classification
        emotion_msg.confidence = confidence_score
        self._interp_emo_publisher.publish(emotion_msg)
        colored_utterance = colored(utterance, "light_blue")
        colored_emotion = colored(classification, "light_green")
        self.log.info(f"Publishing {{\"{colored_emotion}\": {confidence_score}}} " +\
                      f"to {self._out_interp_uemotion_topic} for:\n>>> \"{colored_utterance}\"")

    def _apply_filter(self, msg):
        '''
        Abstracts away any filtering to apply on received messages. Return
        none if the message should be filtered out. Else, return the incoming
        msg if it can be included.
        '''
        # if msg.user_intent.lower() == "user inquiry":
        #     return msg
        # else:
        #     return None
        return msg

def main():
    rclpy.init()
    emotion_detector = BaseEmotionDetector()
    rclpy.spin(emotion_detector)
    emotion_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
