from pathlib import Path
from threading import RLock
from typing import Dict
from typing import Optional

from builtin_interfaces.msg import Time
import kwcoco
import yaml
import numpy as np
import rclpy
from rclpy.node import Node

from angel_msgs.msg import (
    ActivityDetection,
    AruiUserNotification,
    SystemCommands,
    TaskUpdate,
    TaskGraph,
)
from angel_msgs.srv import QueryTaskGraph
from angel_utils import declare_and_get_parameters
from angel_utils import make_default_main

from angel_system.global_step_prediction.global_step_predictor import (
    GlobalStepPredictor,
)


PARAM_CONFIG_FILE = "config_file"
PARAM_ACTIVITY_CONFIG_FILE = "activity_config_file"
PARAM_TASK_STATE_TOPIC = "task_state_topic"
PARAM_TASK_ERROR_TOPIC = "task_error_topic"
PARAM_SYS_CMD_TOPIC = "system_command_topic"
PARAM_QUERY_TASK_GRAPH_TOPIC = "query_task_graph_topic"
PARAM_DET_TOPIC = "det_topic"
PARAM_MODEL_FILE = "model_file"
PARAM_THRESH_FRAME_COUNT = "thresh_frame_count"
PARAM_DEACTIVATE_THRESH_FRAME_COUNT = "deactivate_thresh_frame_count"
PARAM_THRESH_MULTIPLIER_WEAK = "threshold_multiplier_weak"
PARAM_THRESH_FRAME_COUNT_WEAK = "threshold_frame_count_weak"
# The step mode to use for this predictor instance. This must be either "broad"
# or "granular"
PARAM_STEP_MODE = "step_mode"
# Enable ground-truth plotting mode by specifying the path to an MSCOCO file
# that includes image level `activity_gt` attribute.
# Requires co-specification of the video ID to select out of the COCO file.
PARAM_GT_ACT_COCO = "gt_activity_mscoco"
PARAM_GT_VIDEO_ID = "gt_video_id"
PARAM_GT_OUTPUT_DIR = "gt_output_dir"  # output directory override.


VALID_STEP_MODES = {"broad", "granular"}


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
                (PARAM_ACTIVITY_CONFIG_FILE,),
                (PARAM_TASK_STATE_TOPIC,),
                (PARAM_TASK_ERROR_TOPIC,),
                (PARAM_SYS_CMD_TOPIC,),
                (PARAM_QUERY_TASK_GRAPH_TOPIC,),
                (PARAM_DET_TOPIC,),
                (PARAM_MODEL_FILE,),
                (PARAM_THRESH_MULTIPLIER_WEAK,),
                (PARAM_THRESH_FRAME_COUNT,),
                (PARAM_THRESH_FRAME_COUNT_WEAK,),
                (PARAM_DEACTIVATE_THRESH_FRAME_COUNT,),
                (PARAM_STEP_MODE,),
                (PARAM_GT_ACT_COCO, ""),
                (PARAM_GT_VIDEO_ID, -1),
                (PARAM_GT_OUTPUT_DIR, "outputs"),
            ],
        )
        self._config_file = param_values[PARAM_CONFIG_FILE]
        self._activity_config_file = param_values[PARAM_ACTIVITY_CONFIG_FILE]
        self._task_state_topic = param_values[PARAM_TASK_STATE_TOPIC]
        self._task_error_topic = param_values[PARAM_TASK_ERROR_TOPIC]
        self._sys_cmd_topic = param_values[PARAM_SYS_CMD_TOPIC]
        self._query_task_graph_topic = param_values[PARAM_QUERY_TASK_GRAPH_TOPIC]
        self._det_topic = param_values[PARAM_DET_TOPIC]
        self._model_file = param_values[PARAM_MODEL_FILE]
        self._threshold_multiplier_weak = param_values[PARAM_THRESH_MULTIPLIER_WEAK]
        self._thresh_frame_count = param_values[PARAM_THRESH_FRAME_COUNT]
        self._threshold_frame_count_weak = param_values[PARAM_THRESH_FRAME_COUNT_WEAK]
        self._deactivate_thresh_frame_count = param_values[
            PARAM_DEACTIVATE_THRESH_FRAME_COUNT
        ]
        self._step_mode = param_values[PARAM_STEP_MODE]

        if self._step_mode not in VALID_STEP_MODES:
            raise ValueError(
                f"Given step mode '{self._step_mode}' was not valid. Must be "
                f"one of {VALID_STEP_MODES}."
            )

        # Determine what recipes are in the config
        # TODO: make use of angel_system.data.config_structs instead of
        #       manually loading and accessing by string keys.
        with open(self._config_file, "r") as stream:
            config = yaml.safe_load(stream)
        recipe_types = [
            recipe["label"] for recipe in config["tasks"] if recipe["active"]
        ]
        recipe_configs = [
            recipe["config_file"] for recipe in config["tasks"] if recipe["active"]
        ]

        recipe_config_dict = dict(zip(recipe_types, recipe_configs))
        log.info(f"Recipes: {recipe_config_dict}")

        # Instantiate the GlobalStepPredictor module
        self.gsp = GlobalStepPredictor(
            threshold_multiplier_weak=self._threshold_multiplier_weak,
            threshold_frame_count=self._thresh_frame_count,
            threshold_frame_count_weak=self._threshold_frame_count_weak,
            deactivate_thresh_frame_count=self._deactivate_thresh_frame_count,
            recipe_types=recipe_types,
            recipe_config_dict=recipe_config_dict,
            activity_config_fpath=self._activity_config_file,
        )

        self.gsp.get_average_TP_activations_from_file(self._model_file)
        log.info("Global state predictor loaded")

        # Mapping from recipe to current step. Used to track state changes
        # of the GSP and determine when to publish a TaskUpdate msg.
        self.recipe_current_step_id = {}

        # Mapping from recipe to a list of skipped step IDs. Used to ensure
        # that duplicate task error messages are not published for the same
        # skipped step
        self.recipe_skipped_step_ids = {}
        self.recipe_published_last_msg = {}

        # Track the latest activity classification end time sent to the HMM
        # Time is represented as a the ROS Time message
        self._latest_act_classification_end_time = None

        # Control access to GSP
        self._gsp_lock = RLock()

        for task in self.gsp.trackers:
            self.recipe_current_step_id[task["recipe"]] = task[
                f"current_{self._step_mode}_step"
            ]
            self.recipe_skipped_step_ids[task["recipe"]] = []
            self.recipe_published_last_msg[task["recipe"]] = False

        # Initialize ROS hooks
        self._task_update_publisher = self.create_publisher(
            TaskUpdate, self._task_state_topic, 1
        )
        self._task_error_publisher = self.create_publisher(
            AruiUserNotification, self._task_error_topic, 1
        )
        self._task_graph_service = self.create_service(
            QueryTaskGraph, self._query_task_graph_topic, self.query_task_graph_callback
        )
        self._subscription = self.create_subscription(
            ActivityDetection, self._det_topic, self.det_callback, 1
        )
        self._sys_cmd_subscription = self.create_subscription(
            SystemCommands, self._sys_cmd_topic, self.sys_cmd_callback, 1
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

    def sys_cmd_callback(self, sys_cmd_msg: SystemCommands):
        """
        Callback function for the system command subscriber topic.
        Forces an update of the GSP to a new step.
        """
        log = self.get_logger()

        with self._gsp_lock:
            if self._step_mode == "broad" and sys_cmd_msg.next_step:
                update_function = self.gsp.manually_increment_current_broad_step
            elif self._step_mode == "broad" and sys_cmd_msg.previous_step:
                update_function = self.gsp.manually_decrement_current_step
            elif self._step_mode == "granular" and sys_cmd_msg.next_step:
                update_function = self.gsp.increment_granular_step
            elif self._step_mode == "granular" and sys_cmd_msg.previous_step:
                update_function = self.gsp.decrement_granular_step
            else:
                # This should never happen
                return

            try:
                tracker_dict_list = update_function(sys_cmd_msg.task_index)
            except Exception:
                # GSP raises exception if this fails, so just ignore it
                return

            if self._latest_act_classification_end_time is None:
                # No classifications received yet, set time window to now
                start_time = self.get_clock().now().to_msg()
            else:
                start_time = self._latest_act_classification_end_time

            end_time = start_time
            end_time.sec += 1
            self._latest_act_classification_end_time = end_time

            step_mode = self._step_mode
            for task in tracker_dict_list:
                previous_step_id = self.recipe_current_step_id[task["recipe"]]
                current_step_id = task[f"current_{step_mode}_step"]

                # If previous and current are not the same, publish a task-update
                if previous_step_id != current_step_id or (
                    current_step_id == task[f"total_num_{step_mode}_steps"] - 1
                    and task["active"]
                ):
                    log.info(
                        f"Manual step change detected: {task['recipe']}. Current step: {current_step_id}"
                        f" Previous step: {previous_step_id}."
                    )
                    self.publish_task_state_message(
                        task, self._latest_act_classification_end_time
                    )
                    self.recipe_current_step_id[task["recipe"]] = current_step_id

                # If we are on the last step and it is not active, mark it as done
                if (
                    current_step_id == task[f"total_num_{step_mode}_steps"] - 1
                    and not task["active"]
                ):
                    if not self.recipe_published_last_msg[task["recipe"]]:
                        # The last step activity was completed.
                        log.info(
                            f"Final step manually completed: {task['recipe']}. Current step: {current_step_id}"
                        )
                        self.publish_task_state_message(
                            task,
                            self._latest_act_classification_end_time,
                        )

                        self.recipe_published_last_msg[task["recipe"]] = True

                # Undo finishing the recipe
                if (
                    current_step_id == task[f"total_num_{step_mode}_steps"] - 1
                    and task["active"]
                ):
                    log.info(
                        f"Manual step change detected: {task['recipe']}. Current step: {current_step_id}"
                        f" Previous step: {previous_step_id}."
                    )
                    self.publish_task_state_message(
                        task, self._latest_act_classification_end_time
                    )
                    self.recipe_current_step_id[task["recipe"]] = current_step_id

                    self.recipe_published_last_msg[task["recipe"]] = False

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

        with self._gsp_lock:
            tracker_dict_list = self.gsp.process_new_confidences(conf_array)

            step_mode = self._step_mode
            for task in tracker_dict_list:
                previous_step_id = self.recipe_current_step_id[task["recipe"]]
                current_step_id = task[f"current_{step_mode}_step"]

                # If previous and current are not the same, publish a task-update
                if previous_step_id != current_step_id:
                    log.info(
                        f"Step change detected: {task['recipe']}. Current step: {current_step_id}"
                        f" Previous step: {previous_step_id}."
                    )
                    self.publish_task_state_message(
                        task,
                        activity_msg.source_stamp_end_frame,
                    )
                    self.recipe_current_step_id[task["recipe"]] = current_step_id

                # If we are on the last step and it is not active, mark it as done
                if (
                    current_step_id == task[f"total_num_{step_mode}_steps"] - 1
                    and not task["active"]
                ):
                    if not self.recipe_published_last_msg[task["recipe"]]:
                        # The last step activity was completed.
                        log.info(
                            f"Final step completed: {task['recipe']}. Current step: {current_step_id}"
                        )
                        self.publish_task_state_message(
                            task,
                            activity_msg.source_stamp_end_frame,
                        )

                        self.recipe_published_last_msg[task["recipe"]] = True

            # Check for any skipped steps
            skipped_steps_all_trackers = self.gsp.get_skipped_steps_all_trackers()

            for task in skipped_steps_all_trackers:
                for skipped_step in task:
                    recipe = skipped_step["recipe"]
                    skipped_step_id = skipped_step["activity_id"]
                    broad_step_id = skipped_step["part_of_broad"]
                    for idx, tracker_dict in enumerate(tracker_dict_list):
                        if tracker_dict["recipe"] == recipe:
                            broad_step_str = tracker_dict["broad_step_to_full_str"][
                                broad_step_id
                            ]
                            break

                    # "activity_str" is the "full_str" of the activity label.
                    skipped_step_str = (
                        f"Recipe: {recipe}, activity: {skipped_step['activity_str']}, "
                        f"broad step: (id={broad_step_id}) {broad_step_str}"
                    )

                    # New skipped step detected, publish error and add it to the list
                    # of skipped steps
                    if skipped_step_id not in self.recipe_skipped_step_ids[recipe]:
                        self.publish_task_error_message(skipped_step_str)
                        self.recipe_skipped_step_ids[recipe].append(skipped_step_id)

            # Update latest classification timestamp
            self._latest_act_classification_end_time = (
                activity_msg.source_stamp_end_frame
            )

    def publish_task_error_message(self, skipped_step: str):
        """
        Forms and sends a `angel_msgs/AruiUserNotification` message to the
        task errors topic.

        :param skipped_step: Description of the step that was skipped.
        """
        log = self.get_logger()
        log.info(f"Reporting step skipped error: {skipped_step}")

        message = AruiUserNotification()
        message.category = message.N_CAT_NOTICE
        message.context = message.N_CONTEXT_TASK_ERROR

        message.title = "Step skip detected"
        message.description = f"Detected skip: {skipped_step}"

        self._task_error_publisher.publish(message)

    def publish_task_state_message(
        self,
        task_state: Dict,
        result_ts: Time,
    ) -> None:
        """
        Forms and sends a `angel_msgs/TaskUpdate` message to the
        TaskUpdates topic.

        :param task_state: TODO
        :param result_ts: Time of the latest frame input that went into
            estimation of the current task state.
        """
        log = self.get_logger()
        step_mode = self._step_mode

        message = TaskUpdate()

        # Populate message header
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "Task message"
        message.task_name = task_state["recipe"]
        message.task_description = message.task_name
        message.latest_sensor_input_time = result_ts

        # Populate steps and current step
        task_step_str = task_state[f"{step_mode}_step_to_full_str"][
            task_state[f"current_{step_mode}_step"]
        ]

        log.info(f"Publish task {message.task_name} update w/ step: {task_step_str}")
        # Exclude background
        curr_step = task_state[f"current_{step_mode}_step"]
        task_step = curr_step - 1 if curr_step != 0 else 0
        previous_step_str = task_state[f"{step_mode}_step_to_full_str"][
            max(task_state[f"current_{step_mode}_step"] - 1, 0)
        ]

        message.current_step_id = task_step
        message.current_step = task_step_str
        message.previous_step = previous_step_str

        # Binary array simply hinged on everything
        completed_steps_arr = np.zeros(
            task_state[f"total_num_{step_mode}_steps"] - 1,
            dtype=bool,
        )
        completed_steps_arr[:task_step] = True
        if task_step == len(completed_steps_arr) - 1 and not task_state["active"]:
            completed_steps_arr[task_step] = True
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
        with self._gsp_lock:
            for task in self.gsp.trackers:
                # Retrieve step descriptions in the current task.
                task_steps = task[f"{self._step_mode}_step_to_full_str"][
                    1:
                ]  # Exclude background

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
        log.info("Generating plots...")
        out_p = self.gsp.plot_gt_vs_predicted_one_recipe(
            granular_step_gts,
            recipe_type,
            fname_suffix=f"{vid_name}_{str(vid_id)}_granular",
            granular_or_broad="granular",
            output_dir=self.gt_output_dir_override,
        )
        log.info(f"Generated granular plot to: {out_p}")
        # out_p = self.gsp.plot_gt_vs_predicted_one_recipe(
        #     broad_step_gts,
        #     recipe_type,
        #     fname_suffix=f"{vid_name}_{str(vid_id)}_broad",
        #     granular_or_broad="broad",
        # )
        # log.info(f"Generated broad plot to: {out_p}")

    def destroy_node(self):
        self.output_gt_plotting()
        super().destroy_node()


main = make_default_main(GlobalStepPredictorNode)


if __name__ == "__main__":
    main()
