from threading import (
    Event,
    RLock,
    Thread,
)
import time
from typing import cast
from typing import Optional
from typing import Set

from builtin_interfaces.msg import Time
import kwcoco
import numpy as np
import numpy.typing as npt
import rclpy
from rclpy.node import Node
import yaml

from angel_msgs.msg import (
    ActivityDetection,
    AruiUserNotification,
    TaskUpdate,
    TaskGraph,
)
from angel_msgs.srv import QueryTaskGraph
from angel_utils import declare_and_get_parameters

from angel_system.global_step_prediction.global_step_predictor import (
    GlobalStepPredictor,
    get_gt_steps_from_gt_activities,
)


PARAM_CONFIG_FILE = "config_file"
PARAM_TASK_STATE_TOPIC = "task_state_topic"
PARAM_TASK_ERROR_TOPIC = "task_error_topic"
PARAM_QUERY_TASK_GRAPH_TOPIC = "query_task_graph_topic"
PARAM_DET_TOPIC = "det_topic"
PARAM_MODEL_FILE = "model_file"


class GlobalStepPredictorNode(Node):
    """
    ROS node that runs the GlobalStepPredictor and publishes TaskUpdate
    messages.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        log = self.get_logger()

        param_values = declare_and_get_parameters(
            self,
            [
                (PARAM_CONFIG_FILE,),
                (PARAM_TASK_STATE_TOPIC,),
                (PARAM_QUERY_TASK_GRAPH_TOPIC,),
                (PARAM_DET_TOPIC,),
                (PARAM_MODEL_FILE,),
            ],
        )
        self._config_file = param_values[PARAM_CONFIG_FILE]
        self._task_state_topic = param_values[PARAM_TASK_STATE_TOPIC]
        self._query_task_graph_topic = param_values[PARAM_QUERY_TASK_GRAPH_TOPIC]
        self._det_topic = param_values[PARAM_DET_TOPIC]
        self._model_file = param_values[PARAM_MODEL_FILE]

        # Instantiate the GlobalStepPredictor module
        self.gsp = GlobalStepPredictor()

        self.gsp.get_average_TP_activations_from_file(self._model_file)
        log.info("Global state predictor loaded")

        # Mapping from recipe to current step. Used to track state changes
        # of the GSP and determine when to publish a TaskUpdate msg.
        self.recipe_steps = {}

        for task in self.gsp.trackers:
            self.recipe_steps[task["recipe"]] = task["current_step"]

        # Initialize ROS hooks
        self._task_update_publisher = self.create_publisher(
            TaskUpdate, self._task_state_topic, 1
        )
        self._task_graph_service = self.create_service(
            QueryTaskGraph, self._query_task_graph_topic, self.query_task_graph_callback
        )
        self._subscription = self.create_subscription(
            ActivityDetection, self._det_topic, self.det_callback, 1
        )

        self.gt_dets_file = "./model_files/test_activity_preds.mscoco.json"
        vid_id = 1
        coco_test = kwcoco.CocoDataset("model_files/test_activity_preds.mscoco.json")
        image_ids = coco_test.index.vidid_to_gids[vid_id]
        video_dset = coco_test.subset(gids=image_ids, copy=True)
        self.step_gts, _ = get_gt_steps_from_gt_activities(video_dset)

    def det_callback(self, activity_msg: ActivityDetection):
        """
        Callback function for the activity detection subscriber topic.
        Adds the activity detection msg to the HMM and then publishes a new
        TaskUpdate message.
        """
        log = self.get_logger()

        # GSP expects confidence array of shape [n_frames, n_acts]
        # In this case, we only ever send 1 frame's dets at a time
        conf_array = np.array(activity_msg.conf_vec)
        conf_array = np.expand_dims(conf_array, 0)

        tracker_dict_list = self.gsp.process_new_confidences(conf_array)

        for task in tracker_dict_list:
            previous_step_id = self.recipe_steps[task["recipe"]]
            current_step_id = task["current_step"]

            # If previous and current are not the same, publish a taskupdate
            if previous_step_id != current_step_id:
                log.info(
                    f"Step change detected: {task['recipe']}. Current step: {current_step_id}"
                    f" Previous step: {previous_step_id}."
                )
                self.publish_task_state_message(task)
                self.recipe_steps[task["recipe"]] = current_step_id

    def publish_task_state_message(
        self,
        task_state,
    ) -> None:
        """
        Forms and sends a `angel_msgs/TaskUpdate` message to the
        TaskUpdates topic.

        :param task_state: TODO
        """
        log = self.get_logger()

        message = TaskUpdate()

        # Populate message header
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "Task message"
        message.task_name = task_state["recipe"]

        # Populate steps and current step
        task_step_str = task_state["step_to_activity_desc"][task_state["current_step"]]
        log.info(f"Publish task update w/ step: {task_step_str}")
        task_steps = task_state["step_to_activity_desc"][1:]  # Exclude background

        task_step = task_steps.index(task_step_str)

        message.current_step_id = task_step
        message.current_step = task_step_str

        # TODO: Lots of missing fields here

        self._task_update_publisher.publish(message)

    def query_task_graph_callback(self, request, response):
        """
        Populate the `QueryTaskGraph` response with the task list
        and return it.
        """
        log = self.get_logger()
        log.info("Received request for the current task graph")

        task_graphs = []  # List of TaskGraphs
        task_titles = []  # List of task titles associated with the graphs
        for task in self.gsp.trackers:
            task_steps = task["step_to_activity_desc"][1:]  # Exclude background
            task_g = TaskGraph()
            task_g.task_steps = task_steps
            task_g.task_levels = [0] * len(task_steps)

            task_graphs.append(task_g)
            task_titles.append(task["recipe"])

        response.task_graphs = task_graphs
        response.task_titles = task_titles
        log.info("Received request for the current task graph -- Done")
        return response

    def destroy_node(self):
        log = self.get_logger()
        log.info("Shutting down runtime threads...")
        log.info("Shutting down runtime threads... Done")
        super()


def main():
    rclpy.init()

    gsp = GlobalStepPredictorNode()

    try:
        rclpy.spin(gsp)
    except KeyboardInterrupt:
        gsp.get_logger().info("Keyboard interrupt, shutting down.\n")
        gsp.gsp.plot_gt_vs_predicted_one_recipe(
            gsp.step_gts, fname_suffix=str("all_activities_20_josh")
        )

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    gsp.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
