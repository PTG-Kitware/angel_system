import itertools
import langchain
from langchain.chains import LLMChain
import json
from langchain.chat_models import ChatOpenAI
import openai
from operator import itemgetter
import os
import queue
import rclpy
from rclpy.node import Node
from termcolor import colored
import threading
from typing import *

from angel_msgs.msg import (
    ActivityDetection,
    InterpretedAudioUserEmotion,
    ObjectDetection2dSet,
    SystemTextResponse,
    TaskUpdate
)
from angel_utils import declare_and_get_parameters

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Below is/are the subscribed topic(s).
IN_EMOTION_TOPIC = "user_emotion_topic"
IN_OBJECT_DETECTION_TOPIC = "object_detections_topic"
IN_ACT_CLFN_TOPIC = "action_classifications_topic"
IN_TASK_STATE_TOPIC = "task_state_topic"

# Below is/are the published topic(s).
OUT_QA_TOPIC = "system_text_response_topic"

# Below are the corresponding model thresholds.
OBJECT_DETECTION_THRESHOLD = "object_detections_threshold"
ACT_CLFN_THRESHOLD = "action_classification_threshold"

# Below is the recipe paths for the intended task.
RECIPE_PATH = "recipe_path"
# Below is the recipe paths for the prompt template.
PROMPT_TEMPLATE_PATH = "prompt_template_path"

CONTEXT_HISTORY_LENGTH = "context_history_length"
DEBUG_MODE = "debug_mode"

# Below is the complete set of prompt instructions.
PROMPT_INSTRUCTIONS = """
You are given a User Scenario. All the objects in front of and observable to the user are included.
Your task is to use the Action Steps to answer the user's Question.

Action Steps: {recipe}

User Scenario:
The User feels {emotion} while doing {action}. The User can see {observables}.

User Question: {question}
Answer: """


class VisualQuestionAnswerer(Node):
    class TimestampedEntity:
        def __init__(self, time, entity: str):
            self.time = time
            self.entity = entity

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        param_values = declare_and_get_parameters(
            self,
            [
                (RECIPE_PATH,),
                (PROMPT_TEMPLATE_PATH,),
                (IN_EMOTION_TOPIC,),
                (IN_TASK_STATE_TOPIC, ""),
                (IN_OBJECT_DETECTION_TOPIC, ""),
                (IN_ACT_CLFN_TOPIC, ""),
                (OBJECT_DETECTION_THRESHOLD, 0.8),
                (ACT_CLFN_THRESHOLD, 0.8),
                (OUT_QA_TOPIC,),
                (CONTEXT_HISTORY_LENGTH, 3),
                (DEBUG_MODE, False),
            ],
        )
        self._in_emotion_topic = param_values[IN_EMOTION_TOPIC]
        self._in_task_state_topic = param_values[IN_TASK_STATE_TOPIC]
        self._in_objects_topic = param_values[IN_OBJECT_DETECTION_TOPIC]
        self._in_actions_topic = param_values[IN_ACT_CLFN_TOPIC]
        self._out_qa_topic = param_values[OUT_QA_TOPIC]
        self.dialogue_history_length = param_values[CONTEXT_HISTORY_LENGTH]
        self.debug_mode = False
        if param_values[DEBUG_MODE]:
            # langchain.debug = True
            self.debug_mode = True

        self._recipe_path = param_values[RECIPE_PATH]
        self.recipe = self._configure_recipe(self._recipe_path)
        self.log.info(f"Configured recipe to be: ~~~~~~~~~~\n{self.recipe}\n~~~~~~~~~~")
        self._prompt_template_path = param_values[PROMPT_TEMPLATE_PATH]
        with open(self._prompt_template_path, "r") as file:
            self.prompt_template = file.read()
            self.log.info(
                f"Prompt Template: ~~~~~~~~~~\n{self.prompt_template}\n~~~~~~~~~~"
            )

        self.object_dtctn_threshold = param_values[OBJECT_DETECTION_THRESHOLD]
        self.action_clfn_threshold = param_values[ACT_CLFN_THRESHOLD]

        self.question_queue = queue.Queue()
        self.current_step = "Unstarted"
        self.action_classification_queue = queue.Queue()
        self.detected_objects_queue = queue.Queue()
        self.handler_thread = threading.Thread(target=self.process_question_queue)
        self.handler_thread.start()

        # Configure the (necessary) emotional detection enriched utterance subscription.
        self.emotion_subscription = self.create_subscription(
            InterpretedAudioUserEmotion,
            self._in_emotion_topic,
            self.question_answer_callback,
            1,
        )
        # Configure the optional task updates subscription.
        self.task_state_subscription = None
        if self._in_task_state_topic:
            self.task_state_subscription = self.create_subscription(
                TaskUpdate,
                self._in_task_state_topic,
                self._set_current_step,
                1,
            )
        # Configure the optional object detection subscription.
        self.objects_subscription = None
        if self._in_emotion_topic:
            self.objects_subscription = self.create_subscription(
                ObjectDetection2dSet,
                self._in_objects_topic,
                self._add_detected_objects,
                1,
            )
        # Configure the optional action classification subscription.
        self.action_subscription = None
        if self.action_subscription:
            self.action_subscription = self.create_subscription(
                ActivityDetection,
                self._in_actions_topic,
                self._add_action_classification,
                1,
            )
        # Configure the sole QA output of this node.
        self._qa_publisher = self.create_publisher(
            SystemTextResponse, self._out_qa_topic, 1
        )

        # Configure OpenAI API.
        self.openai_api_key = self._configure_openai_api_key()
        self.openai_org_id = self._configure_openai_org_id()

        # Configure LangChain.
        self.chain = self._configure_langchain()

        self.dialogue_history = []

    def _configure_openai_org_id(self):
        if not os.getenv("OPENAI_ORG_ID"):
            raise ValueError(
                "OPENAI_ORG_ID environment variable is unset. "
                + f"You should at least set it to garbage output."
            )
        return os.getenv("OPENAI_ORG_ID")

    def _configure_openai_api_key(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is unset. "
                + f"You should at least set it to garbage output."
            )
        return os.getenv("OPENAI_API_KEY")

    def _configure_recipe(self, recipe_path: str):
        """
        Reads a recipe from a JSON file. The top-level keys in this file should correspond
        to each of the steps for a determined task. The next level should contain an "index"
        field to indicate the step number.
        """
        f = open(recipe_path)
        data = json.load(f)
        steps = [None] * len(data.keys())
        for step in data.keys():
            idx = data[step]["index"]
            steps[idx] = f"{idx + 1}. {step}"
        return "\n".join(steps)

    def _configure_langchain(self):
        """
        Handles OpenAI API prompting via LangChain.
        """
        openai_llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key,
            temperature=0.0,
            max_tokens=64,
        )
        # TODO (derekahmed) Figure out how to  include optional  dialogue history
        zero_shot_prompt = langchain.PromptTemplate(
            input_variables=["recipe", "chat_history",
                             "current_step", "emotion", "action", "observables",
                             "question"],
            template=self.prompt_template,
        )
        return LLMChain(llm=openai_llm, prompt=zero_shot_prompt)

    def _get_sec(self, msg) -> int:
        return msg.header.stamp.sec

    def _set_current_step(self, msg: TaskUpdate):
        self.current_step = msg.current_step

    def _get_current_step(self):
        return self.current_step

    def _get_dialogue_history(self):
        last_n = min(len(self.dialogue_history), self.dialogue_history_length)
        last_n_turns = self.dialogue_history[-1 * last_n:]
        return "\n".join(itertools.chain.from_iterable(last_n_turns))

    def _add_action_classification(self, msg: ActivityDetection) -> str:
        """
        Stores the action label with the highest confidence in
        self.action_classification_queue.
        """
        action_classification = max(
            zip(msg.label_vec, msg.conf_vec), key=itemgetter(1)
        )[0]
        te = VisualQuestionAnswerer.TimestampedEntity(
            self._get_sec(msg), action_classification
        )
        self.action_classification_queue.put(te)

    def _add_detected_objects(self, msg: ObjectDetection2dSet) -> str:
        """
        Stores all detected objects with a confidence score above IN_OBJECT_DETECTION_THRESHOLD.
        """
        detected_objects = set()
        for obj, score in zip(msg.label_vec, msg.label_confidences):
            if score < self.object_dtctn_threshold:
                # Optional threshold filtering
                continue
            detected_objects.add(obj)
        if detected_objects:
            te = VisualQuestionAnswerer.TimestampedEntity(
                self._get_sec(msg), detected_objects
            )
            self.detected_objects_queue.put(te)

    def _add_dialogue_history(self, question: str, response: str):
        self.dialogue_history.append((f"Me: {question}",  f"You: {response}"))

    def _get_action_before(self, curr_time: int) -> str:
        """
        Returns the latest action classification in self.action_classification_queue
        that does not occur before a provided time.
        """
        latest_action = "nothing"
        while not self.action_classification_queue.empty():
            next = self.action_classification_queue.queue[0]
            if next.time < curr_time:
                latest_action = next.entity
                self.action_classification_queue.get()
            else:
                break
        return latest_action

    def _get_observables_before(self, curr_time: int) -> str:
        """
        Returns a comma-delimited list of observed objects per all
        entities in self.detected_objects_queue that occurred before a provided time.
        """
        observables = set()
        while not self.detected_objects_queue.empty():
            next = self.detected_objects_queue.queue[0]
            if next.time < curr_time:
                observables.update(next.entity)
                self.detected_objects_queue.get()
            else:
                break
        if not observables:
            return "nothing"
        return ", ".join(list(observables))

    def get_response(self, msg: InterpretedAudioUserEmotion,
                     chat_history: str, current_step: str, action: str, observables: str):
        """
        Generate a  response to the utterance, enriched with the addition of
        the user's detected emotion. Inference calls can be added and revised
        here.
        """
        return_msg = None
        try:
            self.log.info(f"User emotion: {msg.user_emotion}")
            return_msg = self.chain.run(
                recipe=self.recipe,
                chat_history=chat_history,
                current_step=current_step,
                action=action,
                observables=observables,
                emotion=msg.user_emotion,
                question=msg.utterance_text,
            )
            if self.debug_mode:
                sent_prompt = \
                    self.chain.prompt.format_prompt(recipe=self.recipe,
                                                    chat_history=chat_history,
                                                    current_step=current_step,
                                                    action=action,
                                                    observables=observables,
                                                    emotion=msg.user_emotion,
                                                    question=msg.utterance_text,).to_string()
                sent_prompt = colored(sent_prompt, "light_red")
                self.log.info(f"Prompt sent over:~~~~~~~~~~\n{sent_prompt}\n:~~~~~~~~~~")
        except RuntimeError as err:
            self.log.info(err)
            return_msg = "I'm sorry. I don't know how to answer your statement. " +\
                f"I understand that you feel {msg.user_emotion}."
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
            question_msg = self.question_queue.get()
            start_time = self._get_sec(question_msg)

            # Get most recently detected action.
            action = self._get_action_before(start_time)
            self.log.info(f"Latest action: {action}")

            # Get detected objects.
            observables = self._get_observables_before(start_time)
            self.log.info(f"Observed objects: {observables}")

            # Generate response.
            response = self.get_response(question_msg, 
                                         self._get_dialogue_history(),
                                         self._get_current_step(), action, observables)
            self.publish_generated_response(question_msg.utterance_text, response)
            self._add_dialogue_history(question_msg.utterance_text, response)

    def publish_generated_response(self, utterance: str, response: str):
        msg = SystemTextResponse()
        msg.header.frame_id = "GPT Question Answering"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.utterance_text = utterance
        msg.response = response
        colored_utterance = colored(utterance, "light_blue")
        colored_response = colored(response, "light_green")
        self.log.info(
            f'Responding to utterance:\n>>> "{colored_utterance}"\n>>> with:\n'
            + f'>>> "{colored_response}"'
        )
        self._qa_publisher.publish(msg)

    def _apply_filter(self, msg):
        """
        Abstracts away any filtering to apply on received messages. Return
        none if the message should be filtered out. Else, return the incoming
        msg if it can be included.
        """
        return msg


def main():
    rclpy.init()
    question_answerer = VisualQuestionAnswerer()
    rclpy.spin(question_answerer)
    question_answerer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
