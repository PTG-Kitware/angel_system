import dataclasses
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union
import yaml

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header

from angel_msgs.msg import TaskUpdate
from angel_msgs.srv import QueryTaskGraph
from angel_utils import declare_and_get_parameters
from bbn_integration_msgs.msg import (
    BBNCasualties,
    BBNCasualtyCurrentlyWorkingOn,
    BBNCurrentErrors,
    BBNCurrentSkill,
    BBNCurrentUserActions,
    BBNCurrentUserState,
    BBNHeader,
    BBNNextStepProgress,
    BBNSkillConfidence,
    BBNSkillConfidenceList,
    BBNSkillsDonePerCasualty,
    BBNSkillsOpenPerCasualty,
    BBNStepState,
    BBNUpdate,
)


@dataclasses.dataclass
class BbnStep:
    """
    Metadata about a single BBN-facing task step.
    """

    # Index of the step in the sequence.
    bbn_idx: int
    # Task step body text.
    text: str
    # The KW task step ID that identifies when we are *currently doing*
    # this task step.
    kw_id: int


class TranslateTaskUpdateForBBN(Node):
    """
    Node to maintain task update state and translate to the BBN update format.

    Each update output from the ANGEL system should correlate with one output
    as BBN Update message.

    Certain elements of the BBN Update output are currently hard-coded for demo
    purposes:
        * There is one casualty and its ID is 1. Various elements that refer to
          which casualty is being worked on are coded to refer to ID 1 with a
          confidence of `1.0`.
        * Current skill confidence is coded to be `1.0` as the current ANGEL
          system does not yet support juggling multiple tasks.
        * Next-step progress is coded to be disabled as the current ANGEL
          system does not yet have this measurement functionality.
        * Current-errors is coded to be disabled as it's schema is not yet
          defined by BBN.
        * Current user state is coded to be disabled as it's schema is not yet
          defined by BBN.

    This node will need to build up state as Task Update messages for different
    tasks ("skills" in BBN parlance) are output.
    As tasks are completed, we flip their open/done confidence score from ``0.0``
    to ``1.0``.
    We currently determine task "completion" from an update's
    ``completed_steps`` array being all ``True`` values or not.

    The ``task_name`` from a ``TaskUpdate`` message is currently being
    interpreted as the ``current_skill.number``.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        log = self.get_logger()

        parameter_values = declare_and_get_parameters(
            self,
            [
                ("task_update_topic",),
                ("bbn_update_topic",),
                ("task_graph_srv_topic",),
                ("config",),
            ],
        )
        self._input_task_update_topic = parameter_values["task_update_topic"]
        self._output_bbn_update_topic = parameter_values["bbn_update_topic"]
        self._task_graph_service_topic = parameter_values["task_graph_srv_topic"]
        self._step_mapping_config_path = Path(parameter_values["config"])

        with open(self._step_mapping_config_path) as f:
            self._config = yaml.load(f, yaml.CSafeLoader)

        # Mapping of different task names to a list of (sequential) task steps
        # for BBN consumption steps, each of which also relating which KW task
        # steps relate to the BBN step.
        self._bbn_step_mapping: Dict[str, List[Dict]] = self._config["bbn_to_kw_steps"]

        # Mapping, for different task names, of KW task step index to BBN task
        # step node. Not all KW task step indices may be represented here
        # because there may be intentional gaps in the translation.
        self._kw_step_to_bbn_idx: Dict[str, Dict[int, BbnStep]] = {}
        for task_name, bbn_steps in self._bbn_step_mapping.items():
            local_kw_to_bbn = {}
            self._kw_step_to_bbn_idx[task_name] = local_kw_to_bbn
            for bbn_i, bbn_to_kw in enumerate(bbn_steps):
                bbn_step = BbnStep(bbn_idx=bbn_i, **bbn_to_kw)
                local_kw_to_bbn[bbn_step.kw_id] = bbn_step

        # State capture of task name to the list of task steps.
        # Pre-initialize with
        self._task_to_step_list: Dict[str, List[str]] = {}
        # Track whether a task by name is considered in progress or not with a
        # confidence value in the [0,1] range.
        self._task_in_progress_state: Dict[str, float] = {}
        # Track whether a task by name is considered completed or not with a
        # confidence value in the [0,1] range.
        self._task_completed_state: Dict[str, float] = {}

        self._sub_task_update_cbg = MutuallyExclusiveCallbackGroup()
        self._sub_task_update = self.create_subscription(
            TaskUpdate,
            self._input_task_update_topic,
            self._callback_input_task_update,
            1,
            callback_group=self._sub_task_update_cbg,
        )
        self._pub_bbn_update_cbg = MutuallyExclusiveCallbackGroup()
        self._pub_bbn_update = self.create_publisher(
            BBNUpdate,
            self._output_bbn_update_topic,
            1,
            callback_group=self._pub_bbn_update_cbg,
        )

        self._task_graph_client_cbg = MutuallyExclusiveCallbackGroup()
        self._task_graph_client = self.create_client(
            QueryTaskGraph,
            self._task_graph_service_topic,
            callback_group=self._task_graph_client_cbg,
        )
        # Wait for service to be available. Maybe not required if we assume
        # that the provider of the TaskUpdate messages also supplies the
        # QueryTaskGraph service.
        log.info("Waiting for task graph service...")
        while not self._task_graph_client.wait_for_service(timeout_sec=1.0):
            log.info("Waiting for task graph service...")
        log.info("Task graph service available!")

    def _task_is_in_progress(self, msg: TaskUpdate) -> float:
        """
        Determine if the given task actively in progress.

        Confidence value returned is a floating point in the [0,1] range where
        0 is no confidence and 1 is complete confidence.

        :param msg: TaskUpdate message to base the decision from.
        :return: If the task is judged as actively in progress or not.
        """
        # The `current_step_id` tracks this for us: -1 if no step has been
        # started, and something greater than that if one has.
        current_step_id = msg.current_step_id
        assert current_step_id >= -1, (
            f"Current step ID should have been something >= -1, instead "
            f"was [{current_step_id}]."
        )
        if msg.current_step_id == -1:
            return 0.0
        return 1.0

    def _task_is_complete(self, msg: TaskUpdate) -> float:
        """
        Determine if the given task is complete.

        :param msg: TaskUpdate message to base the decision from.
        :return: If the task is judged as complete or not.
        """
        # Simple logic that checks that the last step is marked completed
        return float(msg.completed_steps[-1])

    def _update_task_graph(self, msg: TaskUpdate) -> bool:
        """
        Query for a task graph from the configured service.

        This asserts that the response is for the same task update we just
        received.

        :returns: True if the update occurred, and false the response returned
            did not match the task update provided.
        """
        log = self.get_logger()
        req = QueryTaskGraph.Request()
        # This works **BECAUSE** we are using multi-threaded executor with
        # separated callback groups.
        log.info("Querying for task-monitor task graph")
        resp = self._task_graph_client.call(req)
        # Use of `self._task_graph_client.call_async(req)` into
        # `rclpy.spin_until_future_complete(self, future)` also works here,
        # but is more or less superfluous.
        # future = self._task_graph_client.call_async(req)
        # rclpy.spin_until_future_complete(self, future)
        # resp: QueryTaskGraph.Response = future.result()
        log.info("Querying for task-monitor task graph -- Done")
        if resp.task_title != msg.task_name:
            self.get_logger().warn(
                f"Received QueryTaskGraph response with mismatching title "
                f"({resp.task_title}) compared to the current task update "
                f"name ({msg.task_name})."
            )
            return False

        # Store the list of step strings for this task
        self._task_to_step_list[msg.task_name] = resp.task_graph.task_steps
        return True

    def _callback_input_task_update(self, msg: TaskUpdate) -> None:
        """
        Handle an input TaskUpdate message.

        Updates the internal state and outputs an appropriate BBNUpdate
        message.
        """
        log = self.get_logger()

        ros_now = self.get_clock().now()
        ros_now_seconds = ros_now.nanoseconds * 1e-9
        latest_sensor_input_time = msg.latest_sensor_input_time
        latest_sensor_input_time_seconds = latest_sensor_input_time.sec + (
            latest_sensor_input_time.nanosec * 1e-9
        )

        # Check if we have the KW Task-monitor reporting steps list, otherwise
        # attempt to query the task-graph service (probably provided by the
        # task-monitor) for it.
        task_name = msg.task_name
        if task_name not in self._task_to_step_list:
            if not self._update_task_graph(msg):
                # Could not get the task graph for the current task update message.
                # Cannot proceed with populating the BBN Update message.
                # The above method outputs a warning to the logger, so just
                # returning early.
                return

        self._task_in_progress_state[task_name] = self._task_is_in_progress(msg)
        self._task_completed_state[task_name] = self._task_is_complete(msg)

        current_kw_task_steps = self._task_to_step_list[task_name]
        assert len(current_kw_task_steps) == len(msg.completed_steps), (
            "Misalignment between state of task steps, steps completed "
            "bools, and step "
        )

        # Determine the current user task step state list
        #
        # A step is DONE if it is marked as complete in msg.completed_steps
        # A step is IMPLIED if it is NOT marked as complete and the current step is beyond this step
        # A step is CURRENT for the msg.current_step_id if it is not marked completed in msg.completed_steps
        # A step is UNOBSERVED if it is not marked as complete and after the current step ID
        #
        current_step_id = msg.current_step_id
        task_step_state_map = {}
        # Equivalent to check `in` against `_bbn_step_mapping` or  `_kw_step_to_bbn_idx`.
        mapping_in_effect = task_name in self._kw_step_to_bbn_idx
        for step_i, (step_name, step_completed) in enumerate(
            zip(current_kw_task_steps, msg.completed_steps)
        ):
            # Getting the state first while `step_i` and `current_step_id` are
            # still relative to each other (below may translate `step_i` into
            # something else).
            if step_completed:
                if step_i == current_step_id:
                    state = BBNStepState.STATE_CURRENT
                elif step_i < current_step_id:
                    state = BBNStepState.STATE_DONE
                else:  # current_step_id < step_i
                    log.warn(
                        f"Step {step_i} ('{step_name}') is marked as "
                        f"completed but is beyond the current step "
                        f"{current_step_id} "
                        f"('{current_kw_task_steps[current_step_id]}'). "
                        f"Calling step {step_i} DONE but we are in "
                        f"possibly a wonky state / regressed the task."
                    )
                    state = BBNStepState.STATE_DONE
            elif step_i < current_step_id:
                state = BBNStepState.STATE_IMPLIED
            elif step_i == current_step_id:
                log.warn(
                    f"Step {step_i} ('{step_name}') marked current but it "
                    f"is also marked incomplete. This is an unexpected "
                    f"condition coming from TaskMonitor v2. Calling it "
                    f"UNOBSERVED for now."
                )
                state = BBNStepState.STATE_UNOBSERVED
            else:  # current_step_id < step_i
                state = BBNStepState.STATE_UNOBSERVED

            # If the current task has configured translations, adjust the
            # `step_i` and `step_name`, or even skip this KW task step if it
            # does not map to something.
            if mapping_in_effect:
                mapping = self._kw_step_to_bbn_idx[task_name]
                if step_i in mapping:
                    bbn_step = mapping[step_i]
                    log.info(
                        f'Mapping KW step ({step_i}) "{step_name}" '
                        f'into BBN step ({bbn_step.bbn_idx}) "{bbn_step.text}".'
                    )
                    # Translate into new index and naming.
                    step_i = bbn_step.bbn_idx
                    step_name = bbn_step.text
                else:
                    # Current KW step has no mapping into BBN steps, skipping.
                    continue

            task_step_state_map[step_i] = BBNStepState(
                number=step_i, name=step_name, state=state, confidence=1.0
            )

        # Check that the index-to-step mapping is contiguous across step indices.
        # If this is not true, there was a translation mapping and the mapping
        # was not properly configured against the input KW steps for the task.
        # Cannot proceed.
        if set(task_step_state_map.keys()) != set(range(len(task_step_state_map))):
            log.error(
                f"Translated steps is not composed of contiguous "
                f"indices. There is an error in translation "
                f"configuration for task {task_name}."
            )
            return
        task_step_state_list = [
            task_step_state_map[_i] for _i in range(len(task_step_state_map))
        ]

        # Construct outgoing message
        out_msg = BBNUpdate(
            header=Header(
                stamp=ros_now.to_msg(), frame_id="Kitware"
            ),  # TODO: not sure what to put here in this case
            bbn_header=BBNHeader(
                sender="Kitware",
                sender_software_version="1.0",
                transmit_timestamp=ros_now_seconds,
                # TaskUpdate that has not seen any input yet may have this set to
                # zero. TODO: Maybe skip sending a BBN update if this is the case?
                closest_hololens_dataframe_timestamp=latest_sensor_input_time_seconds,
            ),
            # Hard-coded requirement for current BBN system state
            casualties=BBNCasualties(),
            # Build skills open list based on observation state
            # NOTE: Just one casualty subject supported at the moment.
            skills_open_per_casualty=BBNSkillsOpenPerCasualty(
                # Only populated if there is anything in our progress state.
                populated=True if self._task_in_progress_state else False,
                casualty_ids=[
                    1,
                ],
                skill_confidences=[
                    BBNSkillConfidenceList(
                        list=[
                            BBNSkillConfidence(
                                label=lbl,
                                confidence=c,
                            )
                            for lbl, c in self._task_in_progress_state.items()
                        ]
                    ),
                ],
            ),
            # Build skills open list based on observation state
            # NOTE: Just one casualty subject supported at the moment.
            skills_done_per_casualty=BBNSkillsDonePerCasualty(
                # Only populated if there is anything in our completed state
                populated=True if self._task_completed_state else False,
                casualty_ids=[
                    1,
                ],
                skill_confidences=[
                    BBNSkillConfidenceList(
                        list=[
                            BBNSkillConfidence(
                                label=lbl,
                                confidence=c,
                            )
                            for lbl, c in self._task_completed_state.items()
                        ]
                    )
                ],
            ),
            current_user_actions=BBNCurrentUserActions(
                populated=True if task_step_state_list else False,
                # Hard coded for Demo 1
                casualty_currently_working_on=BBNCasualtyCurrentlyWorkingOn(),
                current_skill=BBNCurrentSkill(
                    number=task_name,
                    # No support in ANGEL for multitask juggling, so only one
                    # concurrent task is possible --> current task is
                    # definitely the current task.
                    confidence=1.0,
                ),
                steps=task_step_state_list,
            ),
            # We currently do not have a measure of progress towards the next
            # task step
            next_step_progress=BBNNextStepProgress(),
            # Currently not filling in any error propagation.
            current_errors=BBNCurrentErrors(),
            # Nothing defined yet here from BBN's side.
            current_user_state=BBNCurrentUserState(),
        )

        log.info("Publishing BBN Update translation")
        self._pub_bbn_update.publish(out_msg)


def main():
    rclpy.init()

    node = TranslateTaskUpdateForBBN()

    executor = MultiThreadedExecutor(num_threads=5)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.\n")
        executor.shutdown()

    node.get_logger().info("Destroying node")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
