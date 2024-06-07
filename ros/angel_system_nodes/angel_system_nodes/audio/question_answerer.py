import collections
from dataclasses import dataclass
import json
import openai
import os
import queue

import requests
from termcolor import colored
import threading

from angel_msgs.msg import DialogueUtterance, SystemTextResponse, TaskUpdate

from angel_system_nodes.audio import dialogue
from angel_utils import declare_and_get_parameters
from angel_utils import make_default_main

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_QA_TOPIC = "in_qa_topic"
OUT_QA_TOPIC = "out_qa_topic"
FEW_SHOT_PROMPT = "few_shot_prompt_file"
CHAT_HISTORY_LENGTH = "chat_history_length"
TASK_STATE_TOPIC = "task_state_topic"


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
                (TASK_STATE_TOPIC,),
            ],
        )
        self._in_qa_topic = param_values[INPUT_QA_TOPIC]
        self._in_task_state_topic = param_values[TASK_STATE_TOPIC]
        self._out_qa_topic = param_values[OUT_QA_TOPIC]
        self._chat_history_length = param_values[CHAT_HISTORY_LENGTH]
        self.prompt_file = param_values[FEW_SHOT_PROMPT]

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

        publish_msg = SystemTextResponse()
        publish_msg.header.frame_id = "GPT Question Answering"
        publish_msg.header.stamp = self.get_clock().now().to_msg()
        publish_msg.utterance_text = ""
        publish_msg.response = "Hello! Ask me anything. Just start with 'Angel'"
        self._qa_publisher.publish(publish_msg)

        self.log.info("Creating subscription to task topic")
        # Configure the optional task updates subscription.
        self.task_state_subscription = None
        self.current_step = None
        self.completed_steps = None
        if self._in_task_state_topic:
            self.task_state_subscription = self.create_subscription(
                TaskUpdate,
                self._in_task_state_topic,
                self._set_task_topic,
                1,
            )

        self._chat_history = None
        if self._is_using_chat_history():
            self._chat_history = collections.deque([], maxlen=self._chat_history_length)

    def _is_using_chat_history(self):
        return self._chat_history_length > 0

    def _set_task_topic(self, msg: TaskUpdate):
        self.current_step = msg.current_step_id
        self.completed_steps = msg.completed_steps

    def get_response(self, msg: DialogueUtterance, optional_fields: str) -> str:
        response_text = ""
        try:
            if self.is_openai_ready:
                response_text = self.prompt_gpt(msg.utterance_text, optional_fields, msg.pov_frame)
        except RuntimeError as err:
            self.log.info(err)
            response_text = "I'm sorry. I don't know how to answer your statement."
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
            # Get the optional fields.
            optional_fields = self._get_optional_fields_string(
                self.current_step, self.completed_steps
            )

            response = self.get_response(msg, optional_fields)
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

    def prompt_gpt(self, question, optional_fields: str, pov_frame: str, model: str = "gpt-4o"):
        prompt = self.prompt.format(question=question, taskactivity=optional_fields)
        self.log.info(f'Prompting OpenAI with\n{question} with "{optional_fields}"\n')

        if pov_frame == None or len(pov_frame) <= 1:
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 128,
            }
            self.log.info(f'Prompting OpenAI with\n{question} with "{optional_fields}" without pov frame.\n')
        else:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Use the image to answer the question."
                                + prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64," + pov_frame
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 128,
            }
            self.log.info(f'Prompting OpenAI with\n{question} with "{optional_fields}" and pov frame.\n')

        req = requests.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer {}".format(self.openai_api_key)},
        )
        try:
            return (
                json.loads(req.text)["choices"][0]["message"]["content"]
                .split("A:")[-1]
                .lstrip()
            )
        except:
            return "I'm sorry. Please try to repeat your question."

    def _get_optional_fields_string(
        self, current_step: int, completed_steps: list
    ) -> str:
        optional_fields_string = ""

        if current_step == None:
            # non started case
            return "I didn't start the recipe yet."
        else:
            if completed_steps[-1] == True:
                # the last step is finished
                optional_fields_string += f"I am done with all steps."
            elif current_step == 0:
                # user is at step 1
                optional_fields_string += f"I am doing {current_step+1}"
                optional_fields_string += f" and I am about to do {current_step+2}"
            else:
                optional_fields_string += f"I am doing {current_step+1}"
                if current_step <= len(completed_steps) - 2:
                    optional_fields_string += f" and I am about to do {current_step+2}"

        return optional_fields_string.rstrip("\n")

    def prompt_gpt_with_chat_history(self, question, model: str = "gpt-4o"):
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
        if (
            "angela" in msg.utterance_text[:8].lower()
            or "angel" in msg.utterance_text[:8].lower()
            or "angela," in msg.utterance_text[:8].lower()
            or "angel," in msg.utterance_text[:8].lower()
        ):
            return msg
        return None

    @dataclass(frozen=True)
    class ChatMessage:
        role: str
        text: str


main = make_default_main(QuestionAnswerer)


if __name__ == "__main__":
    main()
