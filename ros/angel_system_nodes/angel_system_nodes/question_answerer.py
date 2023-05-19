import json
import openai
import os
import queue
import rclpy
from rclpy.node import Node
import requests
import threading
from angel_msgs.msg import InterpretedAudioUserEmotion, SystemTextResponse


openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

IN_EMOTION_TOPIC = "user_emotion_topic"
OUT_QA_TOPIC = "system_text_response_topic"
FEW_SHOT_PROMPT = "few_shot_prompt_file"

class QuestionAnswerer(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        parameter_names = [
            IN_EMOTION_TOPIC,
            OUT_QA_TOPIC,
            FEW_SHOT_PROMPT
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

        self._in_emotion_topic = self.get_parameter(IN_EMOTION_TOPIC).value
        self._out_qa_topic = self.get_parameter(OUT_QA_TOPIC).value
        self.prompt_file = self.get_parameter(FEW_SHOT_PROMPT).value
        self.log.info(f"Input Emotion topic: "
                      f"({type(self._in_emotion_topic).__name__}) "
                      f"{self._in_emotion_topic}")
        self.log.info(f"Output Question-Answer topic: "
                      f"({type(self._out_qa_topic).__name__}) "
                      f"{self._out_qa_topic}")
        self.log.info(f"Few-shot prompt file: "
                      f"({type(self.prompt_file).__name__}) "
                      f"{self.prompt_file}")

        # Handle subscription/publication topics.
        self.subscription = self.create_subscription(
            InterpretedAudioUserEmotion,
            self._in_emotion_topic,
            self.listener_callback,
            1)
        self._qa_publisher = self.create_publisher(
            SystemTextResponse,
            self._out_qa_topic,
            1)

        self.message_queue = queue.Queue()
        self.handler_thread = threading.Thread(target=self.process_message_queue)
        self.handler_thread.start()

        with open(self.prompt_file, 'r') as file:
            self.prompt = file.read()
        self.log.info(f"Initialized few-shot prompt to:\n\n {self.prompt}")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_org_id = os.getenv("OPENAI_ORG_ID")
        # self.log.debug(f"OpenAI API established with API_KEY={self.openai_api_key} and " +\
        #               f"org_id={self.openai_org_id}.")

    def get_response(self, user_utterance: str, user_emotion: str):
        '''
        Generate a  response to the utterance, enriched with the addition of
        the user's detected emotion. Inference calls can be added and revised
        here.
        '''
        try: 
            return self._red_font(self.prompt_gpt(user_utterance))
        except Exception as e:
            self.log.info(e)
            apology_msg = "I'm sorry. I don't know how to answer your statement."
            return self._red_font(apology_msg) +\
                f" I understand that you feel \"{self._red_font(user_emotion)}\"."
        

    def listener_callback(self, msg):
        '''
        This is the main ROS node listener callback loop that will process
        all messages received via subscribed topics.
        '''  
        self.log.debug(f"Received message:\n\n{msg.utterance_text}")
        if not self._apply_filter(msg):
            return
        self.message_queue.put(msg)

    def process_message_queue(self):
        '''
        Constant loop to process received messages.
        '''
        while True:
            msg = self.message_queue.get()
            emotion = msg.user_emotion
            response = self.get_response(msg.utterance_text, emotion)
            self.publish_generated_response(msg.utterance_text, response)

    def publish_generated_response(self, utterance: str, response: str):
        msg = SystemTextResponse()
        msg.utterance_text = utterance
        msg.response = response
        self.log.info(f"Responding to utterance \"{utterance}\" with:\n\"{response}\"")
        self._qa_publisher.publish(msg)

    def prompt_gpt(self, question, model: str = "gpt-3.5-turbo"):
        prompt = self.prompt.format(question)
        self.log.info(f"Prompting OpenAI with\n {prompt}\n")
        payload = {
            "model" : model,
            "messages" : [
                {
                    "role": "user",
                    "content" : prompt
                }
            ]
        }
        req = requests.post("https://api.openai.com/v1/chat/completions", json=payload,
            headers={"Authorization":"Bearer {}".format(self.openai_api_key)})
        return json.loads(req.text)['choices'][0]['message']['content'].split("A:")[-1].lstrip()

    def _apply_filter(self, msg):
        '''
        Abstracts away any filtering to apply on received messages. Return
        none if the message should be filtered out. Else, return the incoming
        msg if it can be included.
        '''
        return msg

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
