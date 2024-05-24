import json
import openai
import os
import queue

import requests
from termcolor import colored
import threading

from angel_msgs.msg import DialogueUtterance, SystemTextResponse
from angel_system_nodes.audio import dialogue
from angel_utils import declare_and_get_parameters
from angel_utils import make_default_main


openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_QA_TOPIC = "in_qa_topic"
OUT_QA_TOPIC = "out_qa_topic"
FEW_SHOT_PROMPT = "few_shot_prompt_file"


class QuestionAnswerer(dialogue.AbstractDialogueNode):
    def __init__(self):
        super().__init__()

        param_values = declare_and_get_parameters(
            self,
            [
                (INPUT_QA_TOPIC,),
                (OUT_QA_TOPIC,),
                (FEW_SHOT_PROMPT,),
            ],
        )
        self._in_qa_topic = param_values[INPUT_QA_TOPIC]
        self._out_qa_topic = param_values[OUT_QA_TOPIC]
        self.prompt_file = param_values[FEW_SHOT_PROMPT]

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
            self._in_qa_topic,
            self.question_answer_callback,
            1,
        )
        self._qa_publisher = self.create_publisher(
            SystemTextResponse, self._out_qa_topic, 1
        )

    def get_response(self, msg: DialogueUtterance) -> str:
        response_text = ""
        try:
            if self.is_openai_ready:
                response_text = colored(
                    f"{self.prompt_gpt(msg.utterance_text)}\n", "light_green"
                )
        except RuntimeError as err:
            self.log.info(err)
            response_text = colored(
                "I'm sorry. I don't know how to answer your statement.", "light_red"
            )
        return response_text

    def question_answer_callback(self, msg):
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

    def publish_generated_response(
        self, subscribe_msg: DialogueUtterance, response: str
    ):
        publish_msg = SystemTextResponse()
        publish_msg.header.frame_id = "GPT Question Answering"
        publish_msg.header.stamp = self.get_clock().now().to_msg()
        publish_msg.utterance_text = subscribe_msg.utterance_text
        publish_msg.response = response
        colored_utterance = colored(publish_msg.utterance_text, "light_blue")
        colored_response = colored(response, "light_green")
        self.log.info(
            f'Responding to utterance:\n>>> "{colored_utterance}"\n>>> with:\n'
            + f'>>> "{colored_response}"'
        )
        self._qa_publisher.publish(publish_msg)

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


main = make_default_main(QuestionAnswerer)


if __name__ == "__main__":
    main()
