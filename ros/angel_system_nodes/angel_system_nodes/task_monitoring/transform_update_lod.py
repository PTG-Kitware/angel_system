import numpy as np
from rclpy.node import Node

from angel_system.data.common.config_structs import load_multi_task_config
from angel_system.global_step_prediction.global_step_predictor import (
    GlobalStepPredictor,
)

from angel_msgs.msg import TaskUpdate
from angel_utils import declare_and_get_parameters
from angel_utils import make_default_main


###############################################################################
# Parameter labels

PARAM_TOPIC_INPUT = "update_topic_input"
PARAM_TOPIC_OUTPUT = "update_topic_output"

# File for broad/granular step configuration also given to the GSP node.
PARAM_TASK_CONFIG = "task_config_file"
PARAM_TASK_ACTIVITY_CONFIG = "task_activity_config_file"


class TaskUpdateLodTransformerNode(Node):
    """
    Node to transform an input TaskUpdate into another TaskUpdate at a
    different level of detail.

    E.g. w.r.t. GlobalStepTracker verbiage, transform a granular-step update
    into a broad-step update.

    FUTURE: This could take in "user skill" estimations and adapt task update
    levels dynamically. Feedback generator
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        log = self.get_logger()

        params = declare_and_get_parameters(
            self,
            [
                (PARAM_TOPIC_INPUT,),
                (PARAM_TOPIC_OUTPUT,),
                (PARAM_TASK_CONFIG,),
                (PARAM_TASK_ACTIVITY_CONFIG,),
            ],
        )

        # TODO: This is clearly super intrinsic to GSP implementation -- decouple?
        # At a bare minimum, we need to map activity-full-string to broad
        # task-step ID for when we receive a TaskUpdate message. We cannot do
        # this naively because some broad steps are composed of activities
        # shared with other distinct broad steps.
        # We will need to recreate the same thing the GSP class does, which we
        # can do the simplest by just instantiating our own GSP that we only
        # use for structure formation.
        config_multi = load_multi_task_config(params[PARAM_TASK_CONFIG])
        self._gsp = GlobalStepPredictor(
            recipe_types=[t.label for t in config_multi.tasks],
            recipe_config_dict={t.label: t.config_file for t in config_multi.tasks},
            activity_config_fpath=params[PARAM_TASK_ACTIVITY_CONFIG],
        )
        self._task_to_tracker = {t["recipe"]: t for t in self._gsp.trackers}

        # Track previous step ID for different
        self._prev_broad_id = {l: -1 for l in self._task_to_tracker}

        self._pub = self.create_publisher(TaskUpdate, params[PARAM_TOPIC_OUTPUT], 1)
        self._sub = self.create_subscription(
            TaskUpdate,
            params[PARAM_TOPIC_INPUT],
            self.cb_task_update,
            1,
        )
        log.info("Init complete")

    def cb_task_update(self, msg: TaskUpdate) -> None:
        """
        Translate the input TaskUpdate into the target LoD TaskUpdate message.

        :param msg: Message to convert.
        """
        log = self.get_logger()

        log.info(f"Received input message:\n{msg}\n")

        tt = self._task_to_tracker[msg.task_name]

        # If we're in the background step (id=0, step="background"), special
        # case that doesn't transform.
        cur_gran_id = 0 if msg.current_step == "background" else msg.current_step_id + 1
        cur_broad_id = self._gsp.granular_to_broad_step(tt, cur_gran_id)

        # Cannot use GSP provided "prev step" as it is activity class string
        # based, which we know is ambiguous is various locations. GSP node is
        # also not tracking previous correctly... Using locally tracked
        # previous broad ID.
        prev_broad_id = self._prev_broad_id[msg.task_name]

        # If the current and previous step now the same, don't send an update.
        # Except for when the final "completed steps" slot is now true, which
        # means that the final step has completed (final change).
        if cur_broad_id == prev_broad_id and not msg.completed_steps[-1]:
            # No, change, nothing to translate at the broad level
            return

        # Remember to decrement broad_id to "discount" background, but clamp to
        # 0 if we are actually *on* background (and leave "background" as the
        # str).
        msg.current_step_id = max(cur_broad_id - 1, 0)
        msg.current_step = tt["broad_step_to_full_str"][cur_broad_id]
        msg.previous_step = tt["broad_step_to_full_str"][max(prev_broad_id, 0)]

        completed_steps_arr = np.arange(tt["total_num_broad_steps"] - 1) < (
            cur_broad_id - 1
        )
        if msg.completed_steps[-1]:
            # If the final granular step is done, then so is the final broad step.
            completed_steps_arr[-1] = True
        msg.completed_steps = completed_steps_arr.tolist()
        msg.task_complete_confidence = float(np.all(msg.completed_steps))

        log.info(f"Converted into:\n{'v'*79}\n{msg}\n{'^'*79}\n")

        self._prev_broad_id[msg.task_name] = cur_broad_id
        self._pub.publish(msg)


main = make_default_main(TaskUpdateLodTransformerNode)


if __name__ == "__main__":
    main()
