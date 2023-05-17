import rclpy
from rclpy.node import Node
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from angel_msgs.msg import InterpretedAudioUserEmotion, InterpretedAudioUserIntent


IN_EXPECT_USER_INTENT_TOPIC = "expect_user_intent_topic"
IN_INTERP_USER_INTENT_TOPIC = "interp_user_intent_topic"
OUT_INTERP_USER_EMOTION_TOPIC = "user_emotion_topic"

# Model implementation configurations. As of 2023Q2, only "vader" mode is
# supported. This is tied with the VaderSentiment
# (https://github.com/cjhutto/vaderSentiment).
EMOTION_DETECTOR_MODE = "model_mode"

# Currently supported emotions. This is tied with the emotions
# output to VaderSentiment (https://github.com/cjhutto/vaderSentiment) and
# will be subject to change in future iterations.
VADER_SENTIMENT_LABELS = ['pos', 'neg', 'neu']
VADER_SENTIMENT_LABEL_MAPPINGS = {
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral"
}

class EmotionDetector(Node):
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
            OUT_INTERP_USER_EMOTION_TOPIC,
            EMOTION_DETECTOR_MODE
        ]
        set_parameters = self.declare_parameters(
            namespace="",
            parameters=[(p,) for p in parameter_names],
        )
        # Check for unset parameters
        some_not_set = False
        for p in set_parameters:
            if p.type_ is rclpy.parameter.Parameter.Type.NOT_SET:
                some_not_set = True
                self.log.error(f"Parameter not set: {p.name}")
        if some_not_set:
            raise ValueError("Some parameters are not set.")

        self._in_interp_uintent_topic = \
            self.get_parameter(IN_INTERP_USER_INTENT_TOPIC).value
        self._in_expect_uintent_topic = \
            self.get_parameter(IN_EXPECT_USER_INTENT_TOPIC).value
        self._out_interp_uemotion_topic = \
            self.get_parameter(OUT_INTERP_USER_EMOTION_TOPIC).value
        self.mode = \
            self.get_parameter(EMOTION_DETECTOR_MODE).value
        self.log.info(f"Interpreted User Intents topic: "
                      f"({type(self._in_interp_uintent_topic).__name__}) "
                      f"{self._in_interp_uintent_topic}")
        self.log.info(f"Expected User Intents topic: "
                      f"({type(self._in_expect_uintent_topic).__name__}) "
                      f"{self._in_expect_uintent_topic}")
        self.log.info(f"Interpreted User Emotion topic: "
                      f"({type(self._out_interp_uemotion_topic).__name__}) "
                      f"{self._out_interp_uemotion_topic}")
        self.log.info(f"Model mode: "
                      f"({type(self.mode).__name__}) "
                      f"{self.mode}")
        
        # TODO(derekahmed): Add internal queueing to reduce subscriber queue
        # size to 1.
        self.interp_uintent_subscription = self.create_subscription(
            InterpretedAudioUserIntent,
            self._in_interp_uintent_topic,
            self.listener_callback,
            10)
        
        self.expect_uintent_subscription = self.create_subscription(
            InterpretedAudioUserIntent,
            self._in_expect_uintent_topic,
            self.listener_callback,
            10)

        self._interp_emo_publisher = self.create_publisher(
            InterpretedAudioUserEmotion,
            self._out_interp_uemotion_topic,
            1
        )
        self.debug_moode = True

        # Add other model modes here in the below conditional cases as they
        # get implemented. As of  2023Q2, only Vader is supported.
        if self.mode == "vader":
            self.sentiment_analysis_model = SentimentIntensityAnalyzer()
        else:
            raise ValueError("Invalid mode provided")

    def _get_vader_sentiment_analysis(self, utterance: str):
        '''
        Applies Vader Sentiment Analysis model to assign 'positive,' 'negative,'
        and 'neutral' sentiment labels.
        '''
        polarity_scores = \
            self.sentiment_analysis_model.polarity_scores(utterance)
        label_scores = {l: polarity_scores[l] for l in VADER_SENTIMENT_LABELS}
        classification = max(label_scores, key=label_scores.get)
        self.log.info(f"Rated user utterance: \"{utterance}\"" +\
                      f" with emotion scores {label_scores}")
        return (VADER_SENTIMENT_LABEL_MAPPINGS[classification],
                label_scores[classification])

    def get_inference(self, utterance: str):
        '''
        Abstract away the different model inference calls depending on the
        node's configure model mode.
        '''
        if self.mode == "vader":
            return self._get_vader_sentiment_analysis(utterance)
        raise ValueError("Invalid Emotion Detector has been configured." +\
                          f"{self.mode} is an invalid mode.")       

    def _apply_filter(self, msg):
        '''
        Abstracts away any filtering to apply on received messages. Return
        none if the message should be filtered out. Else, return the incoming
        msg if it can be included.
        '''
        return msg

    def _publish_detected_emotion(self, utterance: str,  classification: str,
                                  confidence_score: float):
        emotion_msg = InterpretedAudioUserEmotion()
        emotion_msg.utterance_text = utterance
        emotion_msg.user_emotion = classification
        emotion_msg.confidence = confidence_score
        self.log.info(f"Classifying utterance \"{utterance}\" " +\
                      f"with emotion: \"{classification}\" " +\
                      f"and score {emotion_msg.confidence}")        
        self._interp_emo_publisher.publish(emotion_msg)

    def listener_callback(self, msg):
        '''
        This is the main ROS node listener callback loop that will process
        all messages received via subscribed topics.
        '''
        if self.debug_moode:
            self.log.info(f"Received message:\n\n{msg.utterance_text}")
        if not self._apply_filter(msg):
            return
        classification, confidence_score  = self.get_inference(msg.utterance_text)
        self._publish_detected_emotion(msg.utterance_text,
                                       classification, confidence_score)

def main():
    rclpy.init()
    emotion_detector = EmotionDetector()
    rclpy.spin(emotion_detector)
    emotion_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
