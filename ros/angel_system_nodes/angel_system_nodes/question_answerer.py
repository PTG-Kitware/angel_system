import json
import openai
import os
import queue
import rclpy
from rclpy.node import Node
import requests
from termcolor import colored
import threading

from angel_msgs.msg import DialogueUtterance, SystemTextResponse
from angel_system_nodes.base_dialogue_system_node import BaseDialogueSystemNode
from angel_utils import declare_and_get_parameters

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_TOPIC = "input_topic"
OUT_QA_TOPIC = "system_text_response_topic"
FEW_SHOT_PROMPT = "few_shot_prompt_file"
PARAM_TIMEOUT = "timeout"

class QuestionAnswerer(BaseDialogueSystemNode):
    def __init__(self):
        super().__init__()
        self.log = self.get_logger()

        param_values = declare_and_get_parameters(
            self,
            [
                (INPUT_TOPIC,),
                (OUT_QA_TOPIC,),
                (FEW_SHOT_PROMPT,),
                (PARAM_TIMEOUT, 600),
            ],
        )
        self._input_topic = param_values[INPUT_TOPIC]
        self._out_qa_topic = param_values[OUT_QA_TOPIC]
        self.prompt_file = param_values[FEW_SHOT_PROMPT]
        self.timeout = param_values[PARAM_TIMEOUT]

        self.question_queue = queue.Queue()
        self.handler_thread = threading.Thread(target=self.process_question_queue)
        self.handler_thread.start()

        with open(self.prompt_file, "r") as file:
            self.prompt = file.read()
        self.log.info(f"Initialized few-shot prompt to:\n\n {self.prompt}\n\n")

        self.is_openai_ready = True
        if not os.getenv("OPENAI_API_KEY"):
            self.log.info("OPENAI_API_KEY environment variable is unset!")
            self.is_openai_ready = False
        else:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not os.getenv("OPENAI_ORG_ID"):
            self.log.info("OPENAI_ORG_ID environment variable is unset!")
            self.is_openai_ready = False
        else:
            self.openai_org_id = os.getenv("OPENAI_ORG_ID")

        # Handle subscription/publication topics.
        self.subscription = self.create_subscription(
            DialogueUtterance,
            self._input_topic,
            self.question_answer_callback,
            1,
        )
        self._qa_publisher = self.create_publisher(
            SystemTextResponse, self._out_qa_topic, 1
        )

    def get_response(self, sub_msg: DialogueUtterance):
        """
        Generate a  response to the utterance, enriched with the addition of
        the user's detected emotion. Inference calls can be added and revised
        here.
        """
        return_msg = ""
        try:
            if self.is_openai_ready:
                return_msg = colored(
                    self.prompt_gpt(sub_msg.utterance_text) + "\n", "light_green"
                )
        except RuntimeError as err:
            self.log.info(err)
            colored_apology = colored(
                "I'm sorry. I don't know how to answer your statement.", "light_red"
            )
            colored_emotion = colored(sub_msg.emotion, "light_red")
            return_msg = (
                f"{colored_apology} I understand that you feel {colored_emotion}."
            )
        return return_msg

    def question_answer_callback(self, msg):
        """
        This is the main ROS node listener callback loop that will process
        all messages received via subscribed topics.
        """
        self.log.debug(f"Received message:\n\n{msg.utterance_text}")
        if not self._apply_filter(msg):
            return
        self.question_queue.put(msg)

    def process_question_queue(self):
        """
        Constant loop to process received questions.
        """
        while True:
            msg = self.question_queue.get()
            response = self.get_response(msg)
            self.publish_generated_response(msg, response)

    def publish_generated_response(self, sub_msg: DialogueUtterance, response: str):
        pub_msg = SystemTextResponse()
        pub_msg.header.frame_id = "GPT Question Answering"
        pub_msg.header.stamp = self.get_clock().now().to_msg()
        pub_msg.utterance_text = sub_msg.utterance_text
        pub_msg.response = response
        colored_utterance = colored(sub_msg.utterance_text, "light_blue")
        colored_response = colored(response, "light_green")
        self.log.info(
            f'Responding to utterance:\n>>> "{colored_utterance}"\n>>> with:\n'
            + f'>>> "{colored_response}"'
        )
        self._qa_publisher.publish(pub_msg)

    def prompt_gpt(self, question, model: str = "gpt-3.5-turbo"):
        prompt = self.prompt.format(question)
        self.log.debug(f"Prompting OpenAI with\n {prompt}\n")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 64,
        }
        req = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer {}".format(self.openai_api_key)},
            timeout=self.timeout
        )
        return (
            json.loads(req.text)["choices"][0]["message"]["content"]
            .split("A:")[-1]
            .lstrip()
        )

    def _apply_filter(self, msg):
        """
        Abstracts away any filtering to apply on received messages. Return
        none if the message should be filtered out. Else, return the incoming
        msg if it can be included.
        """
        return msg


def main():
    rclpy.init()
    question_answerer = QuestionAnswerer()
    rclpy.spin(question_answerer)
    question_answerer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
