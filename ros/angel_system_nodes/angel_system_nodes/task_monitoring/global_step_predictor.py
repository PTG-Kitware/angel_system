from pathlib import Path
from typing import Dict
from typing import Optional

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
)


PARAM_CONFIG_FILE = "config_file"
PARAM_TASK_STATE_TOPIC = "task_state_topic"
PARAM_TASK_ERROR_TOPIC = "task_error_topic"
PARAM_QUERY_TASK_GRAPH_TOPIC = "query_task_graph_topic"
PARAM_DET_TOPIC = "det_topic"
PARAM_MODEL_FILE = "model_file"
# Enable ground-truth plotting mode by specifying the path to an MSCOCO file
# that includes image level `activity_gt` attribute.
# Requires co-specification of the video ID to select out of the COCO file.
PARAM_GT_ACT_COCO = "gt_activity_mscoco"
PARAM_GT_VIDEO_ID = "gt_video_id"
PARAM_GT_OUTPUT_DIR = "gt_output_dir"  # output directory override.


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
                (PARAM_GT_ACT_COCO, ""),
                (PARAM_GT_VIDEO_ID, -1),
                (PARAM_GT_OUTPUT_DIR, "outputs"),
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
        self.recipe_current_step_id = {}

        for task in self.gsp.trackers:
            self.recipe_current_step_id[task["recipe"]] = task["current_granular_step"]

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
        log.info("ROS services initialized.")

        self.gt_video_dset: Optional[kwcoco.CocoDataset] = None
        if param_values[PARAM_GT_ACT_COCO]:
            log.info("GT params specified, initializing data...")
            gt_coco_filepath = Path(param_values[PARAM_GT_ACT_COCO])
            self.gt_output_dir_override: str = param_values[PARAM_GT_OUTPUT_DIR]
            vid_id = param_values[PARAM_GT_VIDEO_ID]
            if not gt_coco_filepath.is_file():
                raise ValueError("Given GT coco filepath did not exist.")
            if vid_id < 0:
                raise ValueError("No GT video ID given or given a negative value.")

            coco_test = kwcoco.CocoDataset(gt_coco_filepath)
            image_ids = coco_test.index.vidid_to_gids[vid_id]
            self.gt_video_dset: Optional[kwcoco.CocoDataset] = coco_test.subset(
                gids=image_ids, copy=True
            )
            log.info("GT params specified, initializing data... Done")

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
            previous_step_id = self.recipe_current_step_id[task["recipe"]]
            current_step_id = task["current_granular_step"]

            # If previous and current are not the same, publish a task-update
            if previous_step_id != current_step_id:
                log.info(
                    f"Step change detected: {task['recipe']}. Current step: {current_step_id}"
                    f" Previous step: {previous_step_id}."
                )
                self.publish_task_state_message(
                    task,
                    previous_step_id,
                    activity_msg.source_stamp_end_frame,
                )
                self.recipe_current_step_id[task["recipe"]] = current_step_id

    def publish_task_state_message(
        self,
        task_state: Dict,
        previous_step_id: int,
        result_ts: Time,
    ) -> None:
        """
        Forms and sends a `angel_msgs/TaskUpdate` message to the
        TaskUpdates topic.

        :param task_state: TODO
        :param previous_step_id: Integer ID of the previous task step.
        :param result_ts: Time of the latest frame input that went into
            estimation of the current task state.
        """
        log = self.get_logger()

        message = TaskUpdate()

        # Populate message header
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "Task message"
        message.task_name = task_state["recipe"]
        message.task_description = message.task_name
        message.latest_sensor_input_time = result_ts

        # Populate steps and current step
        # TODO: This is a temporary implementation until the GSP has its "broad
        #       steps" mapping working.
        task_step_str = task["step_to_full_str"][task["current_broad_step"]]

        log.info(f"Publish task update w/ step: {task_step_str}")
        # Exclude background
        task_step = task_state["current_granular_step"] - 1
        previous_step_str = task["step_to_full_str"][
            max(task["current_broad_step"] - 1, 0)
        ]

        message.current_step_id = task_step
        message.current_step = task_step_str
        message.previous_step = previous_step_str

        # Binary array simply hinged on everything
        completed_steps_arr = np.zeros(
            task_state["total_num_granular_steps"] - 1,
            dtype=bool,
        )
        completed_steps_arr[:task_step] = True
        message.completed_steps = completed_steps_arr.tolist()

        # Task completion confidence is currently binary.
        message.task_complete_confidence = float(np.all(message.completed_steps))

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
            # Retrieve step descriptions in the current task.
            # TODO: This is a temporary implementation until the GSP has its "broad
            #       steps" mapping working.
            # task_steps = task["step_to_activity_desc"][1:]  # Exclude background
            task_steps = task["step_to_full_str"].values()

            task_g = TaskGraph()
            task_g.task_steps = task_steps
            task_g.task_levels = [0] * len(task_steps)

            task_graphs.append(task_g)
            task_titles.append(task["recipe"])

        response.task_graphs = task_graphs
        response.task_titles = task_titles
        log.info("Received request for the current task graph -- Done")
        return response

    def output_gt_plotting(self):
        """
        If enabled, output GT plotting artifacts.
        Assuming this is called at the "end" of a run, i.e. after node has
        exited spinning.
        """
        log = self.get_logger()
        if self.gt_video_dset is None:
            log.info("No GT configured to score against, skipping.")
            return
        # List of per-frame truth activity classification IDs.
        activity_gts = self.gt_video_dset.images().lookup("activity_gt")
        recipe_type = self.gsp.determine_recipe_from_gt_first_activity(activity_gts)
        log.info(f"recipe_type = {recipe_type}")
        if recipe_type == "unknown_recipe_type":
            log.info(f"Skipping plotting due to unknown recipe from activity GT.'")
            return
        config_fn = self.gsp.recipe_configs[recipe_type]
        (
            granular_step_gts,
            granular_step_gts_no_background,
            broad_step_gts,
            broad_step_gts_no_background,
        ) = self.gsp.get_gt_steps_from_gt_activities(self.gt_video_dset, config_fn)

        vid_name = self.gt_video_dset.dataset["videos"][0]["name"]
        vid_id = self.gt_video_dset.dataset["videos"][0]["id"]
        self.gsp.plot_gt_vs_predicted_one_recipe(
            granular_step_gts,
            recipe_type,
            fname_suffix=f"{vid_name}_{str(vid_id)}_granular",
            granular_or_broad="granular",
            output_dir=self.gt_output_dir_override,
        )
        # self.gsp.plot_gt_vs_predicted_one_recipe(
        #     broad_step_gts,
        #     recipe_type,
        #     fname_suffix=f"{vid_name}_{str(vid_id)}_broad",
        #     granular_or_broad="broad",
        # )

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
        gsp.output_gt_plotting()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    gsp.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
