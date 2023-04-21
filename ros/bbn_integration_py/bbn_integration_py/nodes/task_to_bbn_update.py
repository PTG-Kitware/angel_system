from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from angel_msgs.msg import TaskUpdate
from angel_msgs.srv import QueryTaskGraph
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


class TranslateTaskUpdateForBBN(Node):
    """
    Node to maintain task update state and translate to the BBN update format.

    Each update output from the ANGEL system should correlate with one output
    as BBN Update message.

    Certain elements of the BBN Update output are currently hard-coded for demo
    purposes:
        * There is one casualty and its ID is 1
        *

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

        parameter_values = self._declare_get_parameters([
            ("task_update_topic",),
            ("bbn_update_topic",),
            ("task_graph_srv_topic",)
        ])
        self._input_task_update_topic = parameter_values["task_update_topic"]
        self._output_bbn_update_topic = parameter_values["bbn_update_topic"]
        self._task_graph_service_topic = parameter_values["task_graph_srv_topic"]

        # TODO: Setup state tracking variables.
        # State capture of task name to the list of task steps.
        self._task_to_step_list: Dict[str, List[str]] = {}
        # Track whether a task by name is considered in progress or not with a
        # confidence value in the [0,1] range.
        self._task_in_progress_state: Dict[str, float] = {}
        # Track whether a task by name is considered completed or not with a
        # confidence value in the [0,1] range.
        self._task_completed_state: Dict[str, float] = {}

        self._sub_task_update = self.create_subscription(
            TaskUpdate,
            self._input_task_update_topic,
            self._callback_input_task_update,
            1
        )
        self._pub_bbn_update = self.create_publisher(
            BBNUpdate,
            self._output_bbn_update_topic,
            1
        )

        self._task_graph_client = self.create_client(
            QueryTaskGraph,
            self._task_graph_service_topic,
        )
        # Wait for service to be available. Maybe not required if we assume
        # that the provider of the TaskUpdate messages also supplies the
        # QueryTaskGraph service.
        log.info("Waiting for task graph service...")
        while not self._task_graph_client.wait_for_service(timeout_sec=1.0):
            log.info("Waiting for task graph service...")
        log.info("Task graph service available!")

    def _declare_get_parameters(
        self,
        name_default_tuples: Sequence[Union[Tuple[str], Tuple[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Declare note parameters, get their values and return them in a tuple.

        Logs parameters declared and parsed to the info channel.

        TODO: Port this more generalized form to the utilities package,
              replacing `self` with an input node instance.
        """
        log = self.get_logger()
        parameters = self.declare_parameters(
            namespace="",
            parameters=name_default_tuples,
        )
        # Check for not-set parameters
        some_not_set = False
        for p in parameters:
            if p.type_ is rclpy.parameter.Parameter.Type.NOT_SET:
                some_not_set = True
                log.error(f"Parameter not set: {p.name}")
        if some_not_set:
            raise ValueError("Some input parameters are not set.")

        # Log parameters
        log = self.get_logger()
        log.info("Input parameters:")
        for p in parameters:
            log.info(f"- {p.name} = ({p.type_}) {p.value}")

        return {p.name: p.value for p in parameters}

    def _task_is_in_progress(self, msg: TaskUpdate) -> bool:
        """
        Determine if the given task actively in progress.

        :param msg: TaskUpdate message to base the decision from.
        :return: If the task is judged as actively in progress or not.
        """
        # The `current_step_id` tracks this for us: -1 if no step has been
        # started, and something greater than that if one has.
        current_step_id = msg.current_step_id
        assert current_step_id >= -1, \
               f"Current step ID should have been something >= -1, instead " \
               f"was [{current_step_id}]."
        if msg.current_step_id == -1:
            return False
        return True

    def _task_is_complete(self, msg: TaskUpdate) -> bool:
        """
        Determine if the given task is complete.

        :param msg: TaskUpdate message to base the decision from.
        :return: If the task is judged as complete or not.
        """
        # Simple logic that checks that the last step is marked completed
        return msg.completed_steps[-1]

    def _update_task_graph(self, msg: TaskUpdate) -> bool:
        """
        Query for a task graph from the configured service.

        This asserts that the response is for the same task update we just
        received.

        :returns: True if the update occurred, and false the response returned
            did not match the task update provided.
        """
        req = QueryTaskGraph.Request()
        future = self._task_graph_client.call_async(req)
        rclpy.spin_until_future_complete(future)
        resp: QueryTaskGraph.Response = future.result()
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
        ros_now = self.get_clock().now()
        ros_now_seconds = ros_now.nanoseconds * 1e-9
        latest_sensor_input_time = msg.latest_sensor_input_time
        latest_sensor_input_time_seconds = (
            latest_sensor_input_time.sec + (latest_sensor_input_time.nanosec * 1e-9)
        )

        # Update internal state if needed
        if msg.task_name not in self._task_to_step_list:
            self._update_task_graph(msg)
        self._task_in_progress_state[msg.task_name] = self._task_is_in_progress(msg)
        self._task_completed_state[msg.task_name] = current_task_complete = self._task_is_complete(msg)

        current_task_steps = self._task_to_step_list[msg.task_name]
        assert len(current_task_steps) == len(msg.completed_steps), \
               "Misalignment between state of task steps, steps completed " \
               "bools, and step "

        # Determine the current user task step state list
        #
        # A step is DONE if it is marked as complete in msg.completed_steps
        # A step is IMPLIED if it is NOT marked as complete and the current step is beyond the step
        # A step is CURRENT for the msg.current_step_id if it is not marked completed in msg.completed_steps
        # A step is UNOBSERVED if it is after the current step ID
        current_step_id = msg.current_step_id
        task_step_state_list = []
        for step_i, (step_name, step_completed) in enumerate(zip(current_task_steps, msg.completed_steps)):
            if step_completed:
                state = BBNStepState.STATE_DONE
            # The following imply step not completed condition
            elif step_i < current_step_id:
                state = BBNStepState.STATE_IMPLIED
            elif step_i == current_step_id:
                state = BBNStepState.STATE_CURRENT
            else:  # current_step_id < step_i
                state = BBNStepState.STATE_UNOBSERVED

            task_step_state_list.append(BBNStepState(
                number=step_i,
                name=step_name,
                state=state,
                confidence=1.0
            ))

        # Construct outgoing message
        out_msg = BBNUpdate(
            header=Header(
                stamp=ros_now.to_msg(),
                frame_id="Kitware"  # TODO: not sure what to put here in this case
            ),
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
                    BBNSkillConfidenceList(list=[
                        BBNSkillConfidence(
                            label=lbl,
                            confidence=c,
                        )
                        for lbl, c in self._task_in_progress_state.items()
                    ]),
                ]
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
                    BBNSkillConfidenceList(list=[
                        BBNSkillConfidence(
                            label=lbl,
                            confidence=c,
                        )
                        for lbl, c in self._task_completed_state.items()
                    ])
                ]
            ),
            current_user_actions=BBNCurrentUserActions(
                populated=True if msg.task_name in self._task_to_step_list else False,
                # Hard coded for Demo 1
                casualty_currently_working_on=BBNCasualtyCurrentlyWorkingOn(),
                current_skill=BBNCurrentSkill(
                    number=msg.task_name,
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

        self._pub_bbn_update.publish(out_msg)


def main():
    rclpy.init()

    node = TranslateTaskUpdateForBBN()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
