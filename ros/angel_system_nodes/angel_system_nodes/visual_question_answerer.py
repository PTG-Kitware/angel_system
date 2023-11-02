from enum import Enum
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
from termcolor import colored
import threading
from typing import *

from angel_msgs.msg import (
    ActivityDetection,
    DialogueUtterance,
    ObjectDetection2dSet,
    SystemTextResponse,
    TaskUpdate,
)
from angel_utils import declare_and_get_parameters
from angel_system.data.common import bounding_boxes
from angel_system_nodes.base_dialogue_system_node import BaseDialogueSystemNode
from angel_system.utils.object_detection_queues import centroid_2d_strategy_queue

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Below is/are the subscribed topic(s).
IN_UTTERANCE_TOPIC = "utterance_topic"
IN_OBJECT_DETECTION_TOPIC = "object_detections_topic"
IN_ACT_CLFN_TOPIC = "action_classifications_topic"
IN_TASK_STATE_TOPIC = "task_state_topic"

# Below is/are the published topic(s).
OUT_QA_TOPIC = "system_text_response_topic"

# Below is used to filter out incoming questions. Toggle this parameter to True if questions
# are only responded to if they contain the TARGET_PHRASE.
PARAM_MUST_CONTAIN_TARGET_PHRASE = "must_contain_target_phrase"
TARGET_PHRASE = "hey angel"

# Below indicates how many of the last n detected objects should be surfaced
# in the LLM prompt. These objects do NOT have to be unique.
PARAM_OBJECT_LAST_N_OBJ_DETECTIONS = "obj_det_last_n"

# Comma-delimited list of objects to ignore.
PARAM_OBJECT_DETECTION_IGNORABLES = "object_det_ignored_objects"

# Below are the corresponding model thresholds.
PARAM_OBJECT_DETECTION_THRESHOLD = "object_det_threshold"
PARAM_ACT_CLFN_THRESHOLD = "action_classification_threshold"

# Below is the recipe paths for the intended task.
PARAM_RECIPE_PATH = "recipe_path"
# Below is the recipe paths for the prompt template.
PARAM_PROMPT_TEMPLATE_PATH = "prompt_template_path"
# Below is how many dialogue turns to keep maintained in the prompt context.
PARAM_CONTEXT_HISTORY_LENGTH = "context_history_length"

# Below configures the width and height of an image. A typical example would be 1280 * 720.
PARAM_IMAGE_WIDTH = "pv_width"
PARAM_IMAGE_HEIGHT = "pv_height"
PARAM_DEBUG_MODE = "debug_mode"

# Below is the complete set of prompt instructions.
PROMPT_INSTRUCTIONS = """
You are given a User Scenario. All the objects in front of and observable to the user are included.
Your task is to use the Action Steps to answer the user's Question.

Action Steps: {recipe}

User Scenario:
The User feels {emotion} while doing {action}. The User can see {observables}.

User Question: {question}
Answer: """

# Below are all the variables. These should correspond to the variables defined in the
# PROMPT_TEMPLATE_PATH and will be indicated by surrounding '{' and '}'.
PROMPT_VARIABLES = [
    "recipe",
    "chat_history",
    "current_step",
    "emotion",
    "action",
    "centered_observables",
    "all_observables",
    "question",
]

PARAM_TIMEOUT = "timeout"

class VisualQuestionAnswerer(BaseDialogueSystemNode):

    class TimestampedEntity:
        """
        This class is used internally as a container for recorded detections and classifications at
        specific instances in time.
        """

        def __init__(self, time, entity):
            self.time = time
            self.entity = entity

    def __init__(self):
        super().__init__()
        self.log = self.get_logger()
        param_values = declare_and_get_parameters(
            self,
            [
                (IN_UTTERANCE_TOPIC,),
                (IN_TASK_STATE_TOPIC, ""),
                (IN_OBJECT_DETECTION_TOPIC, ""),
                (IN_ACT_CLFN_TOPIC, ""),
                (PARAM_RECIPE_PATH,),
                (PARAM_PROMPT_TEMPLATE_PATH,),
                (PARAM_IMAGE_WIDTH,),
                (PARAM_IMAGE_HEIGHT,),
                (PARAM_OBJECT_DETECTION_IGNORABLES,""),
                (PARAM_OBJECT_LAST_N_OBJ_DETECTIONS, 5),
                (PARAM_OBJECT_DETECTION_THRESHOLD, 0.8),
                (PARAM_ACT_CLFN_THRESHOLD, 0.8),
                (OUT_QA_TOPIC,),
                (PARAM_CONTEXT_HISTORY_LENGTH, 3),
                (PARAM_MUST_CONTAIN_TARGET_PHRASE, False),
                (PARAM_TIMEOUT, 600),
                (PARAM_DEBUG_MODE, False),
            ],
        )
        self._in_utterance_topic = param_values[IN_UTTERANCE_TOPIC]
        self._in_task_state_topic = param_values[IN_TASK_STATE_TOPIC]
        self._in_objects_topic = param_values[IN_OBJECT_DETECTION_TOPIC]
        self._in_actions_topic = param_values[IN_ACT_CLFN_TOPIC]
        self._out_qa_topic = param_values[OUT_QA_TOPIC]
        self.dialogue_history_length = param_values[PARAM_CONTEXT_HISTORY_LENGTH]
        self.timeout = param_values[PARAM_TIMEOUT]
        self.debug_mode = False
        if param_values[PARAM_DEBUG_MODE]:
            self.debug_mode = True

        self.param_must_contain_target_phrase = param_values[PARAM_MUST_CONTAIN_TARGET_PHRASE]        

        # Used to obtain the center perspective point and how far detected objects
        # are from it.
        self.pv_width = param_values[PARAM_IMAGE_WIDTH]
        self.pv_height = param_values[PARAM_IMAGE_HEIGHT]
        pv_configured = self.pv_width > 0 and self.pv_height > 0
        self.pv_center_coordinate = (
            [self.pv_width / 2, self.pv_height / 2] if pv_configured else [None, None]
        )

        # Read the configured recipe file.
        self._recipe_path = param_values[PARAM_RECIPE_PATH]
        self.recipe = self._configure_recipe(self._recipe_path)
        self.log.info(f"Configured recipe to be: ~~~~~~~~~~\n{self.recipe}\n~~~~~~~~~~")

        # Read the configured prompt template.
        self._prompt_template_path = param_values[PARAM_PROMPT_TEMPLATE_PATH]
        with open(self._prompt_template_path, "r") as file:
            self.prompt_template = file.read()
            self.log.info(
                f"Prompt Template: ~~~~~~~~~~\n{self.prompt_template}\n~~~~~~~~~~"
            )

        self.object_dtctn_ignorables = set([s.strip() for s in
                                        param_values[PARAM_OBJECT_DETECTION_IGNORABLES].split(",")])
        self.log.info(
            colored(f"Will be ignoring the following objects: {self.object_dtctn_ignorables}",
            "light_red")
        )
        self.object_dtctn_threshold = param_values[PARAM_OBJECT_DETECTION_THRESHOLD]
        self.object_dtctn_last_n_obj_detections = param_values[PARAM_OBJECT_LAST_N_OBJ_DETECTIONS]

        # Configure supplemental input action classification criteria.
        self.action_clfn_threshold = param_values[PARAM_ACT_CLFN_THRESHOLD]

        # Configure supplemental input resources.
        self.question_queue = queue.Queue()
        self.current_step = "Unstarted"
        self.action_classification_queue = queue.Queue()
        self.detected_objects_queue = queue.Queue()
        self.centroid_object_queue = \
            centroid_2d_strategy_queue.Centroid2DStrategyQueue(
                self.object_dtctn_last_n_obj_detections,
                self.pv_center_coordinate[0], self.pv_center_coordinate[1],
                k=1, # the number of top-k objects to obtain from each detection.
                )
        
        self.dialogue_history = []
        self.handler_thread = threading.Thread(target=self.process_question_queue)
        self.handler_thread.start()

        # Configure the (necessary) emotional detection enriched utterance subscription.
        self.subscription = self.create_subscription(
            DialogueUtterance,
            self._in_utterance_topic,
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
        if self._in_objects_topic:
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
            request_timeout=self.timeout
        )
        zero_shot_prompt = langchain.PromptTemplate(
            input_variables=PROMPT_VARIABLES,
            template=self.prompt_template,
        )
        return LLMChain(llm=openai_llm, prompt=zero_shot_prompt)

    def _get_sec(self, msg: DialogueUtterance) -> int:
        return msg.header.stamp.sec

    def _set_current_step(self, msg: TaskUpdate):
        self.current_step = msg.current_step

    def _get_current_step(self):
        return self.current_step

    def _get_dialogue_history(self):
        """
        Gets a string concatenation of the last self.dialogue_history_length turns of conversation.
        """
        last_n = min(len(self.dialogue_history), self.dialogue_history_length)
        last_n_turns = self.dialogue_history[-1 * last_n :]
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
        # We queue timestamped lists of pairs of (detections, confidence scores) for centered
        # objects based on centroid distance from the middle.
        self.centroid_object_queue.add(
            self._get_sec(msg),
            bounding_boxes.BoundingBoxes(
                msg.left, msg.right, msg.top, msg.bottom,
                item=list(zip(msg.label_vec, msg.label_confidences))
                ))
        
        # We queue ALL objects above threshold, regardless if they are centered in the user's
        # perspective.
        self._add_detected_objects_above_threshold(msg)

    def _add_detected_objects_above_threshold(self, msg):
        """
        Queuse all objects above a configured threshold.
        """
        detected_objs = set()
        for obj, score in zip(msg.label_vec, msg.label_confidences):
            if score < self.object_dtctn_threshold:
                continue
            detected_objs.add(obj)
        if detected_objs:
            te = VisualQuestionAnswerer.TimestampedEntity(
                self._get_sec(msg), detected_objs
            )
            self.detected_objects_queue.put(te)

    def _add_dialogue_history(self, question: str, response: str):
        self.dialogue_history.append((f"Me: {question}", f"You: {response}"))

    def _get_latest_action(self, curr_time: int) -> str:
        """
        Returns the latest action classification in self.action_classification_queue
        that does not occur before a provided time.
        """
        latest_action = "not available"
        while not self.action_classification_queue.empty():
            next = self.action_classification_queue.queue[0]
            if next.time < curr_time:
                latest_action = next.entity
                self.action_classification_queue.get()
            else:
                break
        return latest_action

    def _get_latest_centered_observables(self, curr_time: int) -> str:
        """
        Returns a comma-delimited list of "centered" objects per all
        entities in self.detected_objects_queue that occurred before a provided time.
        :param curr_time: The time for which objects must have been detected before.
        :param n: The last n objects.
        :return: returns a string-ified list of the latest observables
        """
        observables = set()
        #  handle 2D centroid distance queueing.
        timestamped_detections = self.centroid_object_queue.get_n_before(timestamp=curr_time)
        if timestamped_detections:
            if self.debug_mode:
                print(f"Timestamped detections based on centroid distance are: " +\
                      f"{timestamped_detections}")
            # Recall that we passed in timestamped lists of pairs of
            # (detection, confidence score).
            centered_obj_detections_lists = [j for _, j in timestamped_detections]
            for centered_obj_detections in centered_obj_detections_lists:
                for centered_obj_detection in centered_obj_detections:
                    centroid, obj_score  = centered_obj_detection
                    obj, score = obj_score
                    observables.add(obj)
            observables = observables - self.object_dtctn_ignorables
            return ", ".join(observables)

    def _get_latest_observables(self, curr_time: int, n: int) -> str:
        """
        Returns a comma-delimited list of all observed objects per all
        entities in self.detected_objects_queue that occurred before a provided time.
        Only refers to the latest n detections.
        :param curr_time: The time for which objects must have been detected before.
        :param n: The last n objects.
        :return: returns a string-ified list of the latest observables
        """
        detections = []
        while not self.detected_objects_queue.empty():
            next_detections = self.detected_objects_queue.queue[0]
            if next_detections.time < curr_time:
                detections.append(self.detected_objects_queue.get())
            else:
                break
        observables = set()
        for detection in detections[-n:]:
            for obj in detection.entity:
                observables.add(obj)
        observables = observables - self.object_dtctn_ignorables               
        return ", ".join(observables)

    def get_response(
        self,
        msg: DialogueUtterance,
        chat_history: str,
        current_step: str,
        action: str,
        centered_observables: str,
        all_observables: str,
    ):
        """
        Generate a response to the utterance, enriched with the addition of
        the user's detected emotion, chat history, current step information, action, and
        detected objects. Inference calls can be added and revised here.
        """
        return_string = None
        try:
            self.log.info(f"User emotion: {msg.emotion}")
            return_string = self.chain.run(
                recipe=self.recipe,
                chat_history=chat_history,
                current_step=current_step,
                action=action,
                centered_observables=centered_observables,
                all_observables=all_observables,
                emotion=self.get_emotion_or(msg),
                question=msg.utterance_text,
            )
            if self.debug_mode:
                sent_prompt = self.chain.prompt.format_prompt(
                    recipe=self.recipe,
                    chat_history=chat_history,
                    current_step=current_step,
                    action=action,
                    centered_observables=centered_observables,
                    all_observables=all_observables,
                    emotion=self.get_emotion_or(msg),
                    question=msg.utterance_text,
                ).to_string()
                sent_prompt = colored(sent_prompt, "light_red")
                self.log.info(
                    f"Prompt sent over:~~~~~~~~~~\n{sent_prompt}\n:~~~~~~~~~~"
                )
        except RuntimeError as err:
            self.log.info(err)
            return_string = (
                "I'm sorry. I don't know how to answer your statement. "
                + f"I understand that you feel {self.get_emotion_or(msg)}."
            )
        return return_string

    def question_answer_callback(self, msg: DialogueUtterance):
        """
        This is the main ROS node listener callback loop that will process
        all messages received via subscribed topics.
        """
        self.log.info(f"Received message:\n\n{msg.utterance_text}")
        if not self._apply_filter(msg):
            return
        self.question_queue.put(msg)

    def process_question_queue(self):
        """
        Constant loop to process received questions.
        """
        self.log.info("Spawning question-processing thread...")
        while True:
            question_msg = self.question_queue.get()
            start_time = self._get_sec(question_msg)
            self.log.info(f"Processing utterance {question_msg.utterance_text}")

            # Get most recently detected action.
            action = self._get_latest_action(start_time)
            self.log.info(f"Latest action: {action}")

            # Get centered detected objects.
            centered_observables = \
                self._get_latest_centered_observables(start_time)
            self.log.info(f"Observed objects: {centered_observables}")
            
            # Get all detected objects.
            all_observables = self._get_latest_observables(start_time,
                                                       self.object_dtctn_last_n_obj_detections)
            self.log.info(f"Observed objects: {all_observables}")

            # Generate response.
            response = self.get_response(
                question_msg,
                self._get_dialogue_history(),
                self._get_current_step(),
                action,
                centered_observables,
                all_observables,
            )
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
        a boolean value indicating if the message passes a filter and should be processed.
        """
        if self.param_must_contain_target_phrase:
            return TARGET_PHRASE in msg.utterance_text.lower()
        else:
            return True


def main():
    rclpy.init()
    question_answerer = VisualQuestionAnswerer()
    rclpy.spin(question_answerer)
    question_answerer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
