import collections
from dataclasses import dataclass
import json
import openai
import os
import io
import queue
import base64
import PIL.Image
import numpy as np
from cv_bridge import CvBridge

import requests
from termcolor import colored
from sensor_msgs.msg import Image
import threading

from angel_msgs.msg import DialogueUtterance, SystemTextResponse
from angel_system_nodes.audio import dialogue
from angel_utils import declare_and_get_parameters
from angel_utils import make_default_main

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

BRIDGE = CvBridge()

INPUT_QA_TOPIC = "in_qa_topic"
OUT_QA_TOPIC = "out_qa_topic"
FEW_SHOT_PROMPT = "few_shot_prompt_file"
CHAT_HISTORY_LENGTH = "chat_history_length"
IMAGE_TOPIC = "image_topic"

class QuestionAnswerer(dialogue.AbstractDialogueNode):
    def __init__(self):
        super().__init__()

        param_values = declare_and_get_parameters(
            self,
            [
                (INPUT_QA_TOPIC,),
                (OUT_QA_TOPIC,),
                (FEW_SHOT_PROMPT,),
                (CHAT_HISTORY_LENGTH, -1),
                (IMAGE_TOPIC,),
            ],
        )
        self._in_qa_topic = param_values[INPUT_QA_TOPIC]
        self._out_qa_topic = param_values[OUT_QA_TOPIC]
        self._chat_history_length = param_values[CHAT_HISTORY_LENGTH]
        self.prompt_file = param_values[FEW_SHOT_PROMPT]
        self.image_topic = param_values[IMAGE_TOPIC]

        self.question_queue = queue.Queue()
        self.handler_thread = threading.Thread(target=self.process_question_queue)
        self.handler_thread.start()

        with open(self.prompt_file, "r") as file:
            self.prompt = file.read()
        self.log.info(f"Initialized few-shot prompt to:\n\n{self.prompt}\n\n")

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
        self.log.info("Creating subscription to utterance topic")
        self.subscription = self.create_subscription(
            DialogueUtterance,
            self._in_qa_topic,
            self.question_answer_callback,
            1,
        )

        self.log.info("Creating subscription to feedback generator topic")
        self._qa_publisher = self.create_publisher(
            SystemTextResponse, self._out_qa_topic, 1
        )

        # Single slot for latest image message to process detection over.
        self.image_msg: Image = ""

        self.log.info("Creating subscription to image topic")
        # Initialize ROS hooks
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.process_image_callback,
            1,
        )

        self._chat_history = None
        if self._is_using_chat_history():
            self._chat_history = collections.deque([], maxlen=self._chat_history_length)

    def _is_using_chat_history(self):
        return self._chat_history_length > 0

    def get_response(self, msg: DialogueUtterance) -> str:
        response_text = ""
        try:
            if self.is_openai_ready:
                prompt_fn = (
                    self.prompt_gpt
                )
                response_text = prompt_fn(msg.utterance_text)
        except RuntimeError as err:
            self.log.info(err)
            response_text = "I'm sorry. I don't know how to answer your statement."
        return response_text

    def process_image_callback(self, image: Image):
        # image is type sensor_msgs.msg encoding BGR8
        img0 = BRIDGE.imgmsg_to_cv2(image, desired_encoding="bgr8")

        # Convert img0 into RGB and create a PIL image instance.
        img_rgb = PIL.Image.fromarray(img0[:, :, ::-1], mode="RGB")
        img_rgb = img_rgb.resize(np.divide(img_rgb.size, 4).astype(int))
        jpg_container = io.BytesIO()
        img_rgb.save(jpg_container, format="JPEG")
        self.image_msg = base64.encodebytes(jpg_container.getvalue())
        
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

    def prompt_gpt(self, question, model: str = "gpt-4o"):
        prompt = self.prompt.format(question=question)
        self.log.info(f"Prompting OpenAI with\n{prompt}\n")

        if self.image_msg==None or len(self.image_msg)<=1:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 128,
            }
        else:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": [
                    { "type": "text",
                    "text": "Use the image to answer the question."+ prompt},
                    {"type": "image_url",
                    "image_url" : {
                      "url" : "data:image/jpeg;base64,"+self.image_msg
                    }
                    }
                ]}],
            "temperature": 0.0,
            "max_tokens": 128,
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

    def prompt_gpt_with_chat_history(self, question, model: str = "gpt-3.5-turbo"):
        prompt = self.prompt.format(
            chat_history=self._format_chat_history_str(), question=question
        )
        self.log.info(f"Prompting OpenAI with\n {prompt}\n")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 128,
        }
        req = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer {}".format(self.openai_api_key)},
        )
        answer = (
            json.loads(req.text)["choices"][0]["message"]["content"]
            .split("Assistant:")[-1]
            .lstrip()
        )
        self._append_chat_history(role="User", text=question)
        self._append_chat_history(role="Assistant", text=answer)
        return answer

    def _append_chat_history(self, role: str, text: str):
        if self._is_using_chat_history():
            self._chat_history.append(
                QuestionAnswerer.ChatMessage(role=role, text=text)
            )

    def _format_chat_history_str(self):
        result = ""
        for msg in self._chat_history:
            result += f"{msg.role}: {msg.text}\n"
        return result

    def _apply_filter(self, msg):
        """
        Abstracts away any filtering to apply on received messages. Return
        none if the message should be filtered out. Else, return the incoming
        msg if it can be included.
        """
        if "angel" in msg.utterance_text.lower() or "angela" in msg.utterance_text.lower() or "angela," in msg.utterance_text.lower() or "angel," in msg.utterance_text.lower():
            return msg
        return None

    @dataclass(frozen=True)
    class ChatMessage:
        role: str
        text: str


main = make_default_main(QuestionAnswerer)


if __name__ == "__main__":
    main()
