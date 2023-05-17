import rclpy
from rclpy.node import Node

from angel_msgs.msg import InterpretedAudioUserEmotion, SystemTextResponse


IN_EMOTION_TOPIC = "user_emotion_topic"
OUT_QA_TOPIC = "system_text_response_topic"

class QuestionAnswerer(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        parameter_names = [
            IN_EMOTION_TOPIC,
            OUT_QA_TOPIC,
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

        self._in_emotion_topic = \
            self.get_parameter(IN_EMOTION_TOPIC).value
        self._out_qa_topic = \
            self.get_parameter(OUT_QA_TOPIC).value
        self.log.info(f"Input Emotion topic: "
                      f"({type(self._in_emotion_topic).__name__}) "
                      f"{self._in_emotion_topic}")
        self.log.info(f"Output Question-Answer topic: "
                      f"({type(self._out_qa_topic).__name__}) "
                      f"{self._out_qa_topic}")

        # TODO(derekahmed): Add internal queueing to reduce subscriber queue
        # size to 1.
        self.subscription = self.create_subscription(
            InterpretedAudioUserEmotion,
            self._in_emotion_topic,
            self.listener_callback,
            10)

        self._qa_publisher = self.create_publisher(
            SystemTextResponse,
            self._out_qa_topic,
            1
        )

    def get_response(self, user_utterance: str, user_emotion: str):
        '''
        Generate a  response to the utterance, enriched with the addition of
        the user's detected emotion. Inference calls can be added and revised
        here.
        '''
        
        utterance_words = user_utterance.split()
        # shortened_utterance = \
        #     ' '.join(utterance_words[:4]) + " ... " + \
        #         ' '.join(utterance_words[-4:]) \
        #         if len(utterance_words) >= 8 else user_utterance
        apology_msg = "I'm sorry. I don't know how to answer your statement."
        return self._red_font(apology_msg) +\
            f" I understand that you feel \"{self._red_font(user_emotion)}\"."

    def _publish_generated_response(self, utterance: str,
                                    response: str):
        msg = SystemTextResponse()
        msg.utterance_text = utterance
        msg.response = response
        self.log.info(f"Responding to utterance \"{utterance}\" " +\
                      f"with:\n{response}")        
        self._qa_publisher.publish(msg)

    def listener_callback(self, msg):
        '''
        This is the main ROS node listener callback loop that will process
        all messages received via subscribed topics.
        '''  
        utterance = msg.utterance_text
        emotion = msg.user_emotion
        response = self.get_response(utterance, emotion)
        self._publish_generated_response(utterance, response)

    def _red_font(self, text):
        return f"\033[91m{text}\033[0m"

def main():
    rclpy.init()
    question_answerer = QuestionAnswerer()
    rclpy.spin(question_answerer)
    question_answerer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
