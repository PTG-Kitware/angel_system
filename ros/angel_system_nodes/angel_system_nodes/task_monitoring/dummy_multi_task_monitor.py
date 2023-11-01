from dataclasses import dataclass, field
from threading import (
    Thread,
)
from typing import Dict
from typing import List
from typing import Set
import yaml

from builtin_interfaces.msg import Time
import numpy as np
from pynput import keyboard
import rclpy
from rclpy.node import Node

from angel_msgs.msg import (
    TaskGraph,
    TaskUpdate,
)
from angel_msgs.srv import QueryTaskGraph
from angel_utils import declare_and_get_parameters
from angel_utils.conversion import (
    nano_to_ros_time,
    SEC_TO_NANO,
    time_to_int,
)


PARAM_CONFIG_FILE = "config_file"
PARAM_TASK_STATE_TOPIC = "task_state_topic"
PARAM_TASK_ERROR_TOPIC = "task_error_topic"
PARAM_QUERY_TASK_GRAPH_TOPIC = "query_task_graph_topic"


@dataclass
class TaskStateInformation:
    """
    Stores relevant task state information for a single task.
    """

    task_title: str
    n_steps: int

    # Tracks the current/previous steps
    # Current step is the most recently completed step the user has finished
    # as predicted by the HMM.
    # Previous step is the step before the current step.
    # Values will be either None or the step string label.
    current_step = None
    current_step_id = 0  # HMM ID
    previous_step = None
    previous_step_id = 0  # HMM ID
    # Track the step skips we previously notified the user about to avoid
    # sending duplicate notifications for the same error.
    previous_step_skip = None
    current_step_skip = None
    # Track the indices of steps we have emitted skip errors for in order
    # to not repeatedly emit many errors for the same steps.
    # This may be removed from if step progress regresses.
    steps_skipped_cache: Set[int] = field(default_factory=set)
    # Track the latest activity classification end time sent to the HMM
    # Time is in integer nanoseconds.
    latest_act_classification_end_time = None
    # List of task steps
    steps: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Boolean vector the length of task steps
        # that indicates which steps to report as having been completed.
        self.steps_complete = np.zeros(self.n_steps, dtype=bool)
        # Skipped steps are those steps at or before the current
        # skipped (haven't happened yet).
        self.steps_skipped = np.zeros(self.n_steps, dtype=bool)


class DummyMultiTaskMonitor(Node):
    """
    ROS node that simulates a node that could monitor multiple tasks
    simultaneously. The keyboard is used to advance between task steps.
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
            ],
        )
        self._config_file = param_values[PARAM_CONFIG_FILE]
        self._task_state_topic = param_values[PARAM_TASK_STATE_TOPIC]
        self._query_task_graph_topic = param_values[PARAM_QUERY_TASK_GRAPH_TOPIC]

        # Load the task configurations
        with open(self._config_file, "r") as f:
            config = yaml.safe_load(f)

        self._task_state_dict: Dict[int, TaskStateInformation] = {}
        for t in config["tasks"]:
            # Load the config for this task
            with open(t["config_file"], "r") as f:
                task_config = yaml.safe_load(f)

            n_steps = len(task_config["labels"][1:])

            task_state_info = TaskStateInformation(
                f"{t['id']}_{task_config['title']}",
                n_steps,
            )

            # Load the list of task steps
            task_state_info.steps = [
                step["full_str"] for step in task_config["labels"][1:]
            ]

            self._task_state_dict[t["id"]] = task_state_info

        log.info("Loaded the following tasks:")
        for key, value in self._task_state_dict.items():
            log.info(f"    Task {key}: {value.task_title}")

        # Initialize ROS hooks
        self._task_update_publisher = self.create_publisher(
            TaskUpdate, self._task_state_topic, 1
        )
        self._task_graph_service = self.create_service(
            QueryTaskGraph, self._query_task_graph_topic, self.query_task_graph_callback
        )

        # Start the keyboard monitoring thread
        log.info("Starting keyboard threads")
        self._keyboard_t = Thread(target=self.monitor_keypress)
        self._keyboard_t.daemon = True
        self._keyboard_t.start()
        log.info("Starting keyboard threads... done")

    def publish_task_state_message(
        self,
        task_id: int,
    ) -> None:
        """
        Forms and sends a `angel_msgs/TaskUpdate` message to the
        TaskUpdates topic.

        :param task_id: Task ID of the task state that is being published.
        """
        log = self.get_logger()

        task_state = self._task_state_dict[task_id]

        message = TaskUpdate()

        # Populate message header
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "Task message"
        message.task_name = task_state.task_title
        if task_state.latest_act_classification_end_time is not None:
            message.latest_sensor_input_time = nano_to_ros_time(
                task_state.latest_act_classification_end_time
            )
        else:
            # Fill in a default time of 0.0 seconds.
            message.latest_sensor_input_time = Time(0, 0)

        # Populate steps and current step
        if task_state.current_step is None:
            message.current_step_id = -1
            message.current_step = "None"
        else:
            # Getting the index of the step in the list-of-strings
            message.current_step_id = task_state.current_step_id
            message.current_step = task_state.current_step

        if task_state.previous_step is None:
            message.previous_step = "N/A"
        else:
            message.previous_step = task_state.previous_step

        message.completed_steps = task_state.steps_complete.tolist()
        log.info(f"Steps complete: {message.completed_steps}")

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
        for task_id, task in self._task_state_dict.items():
            task_g = TaskGraph()
            task_g.task_steps = task.steps
            task_g.task_levels = [0] * len(task.steps)

            task_graphs.append(task_g)
            task_titles.append(task.task_title)

        response.task_graphs = task_graphs
        response.task_titles = task_titles
        log.info("Received request for the current task graph -- Done")
        return response

    def monitor_keypress(self):
        log = self.get_logger()
        log.info(
            f"Starting keyboard monitor. Use the 0-9 number keys to advance"
            f" between tasks."
        )
        # Collect events until released
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def advance_step(self, task_id: int, forward: bool) -> None:
        """
        Handles advancing a task between its steps.
        """
        log = self.get_logger()

        try:
            task_state = self._task_state_dict[task_id]
        except KeyError:
            log.warning(f"Task {task_id} not found")
            return

        if task_state.latest_act_classification_end_time is None:
            # No classifications received yet
            # Set time window to now + 1 second
            start_time = time_to_int(
                self.get_clock().now().to_msg()
            )  # time_to_int returns ns
            end_time = start_time + SEC_TO_NANO  # 1 second later
        else:
            # Assuming ~30Hz frame rate, so set start one frame later
            start_time = task_state.latest_act_classification_end_time + int(
                (1 / 30.0) * SEC_TO_NANO
            )
            end_time = start_time + SEC_TO_NANO  # 1 second later

        task_state.latest_act_classification_end_time = end_time

        curr_step_id = task_state.current_step_id
        if forward:
            new_step_id = curr_step_id + 1

            if new_step_id > len(task_state.steps):
                log.debug("Attempting to advance past end of list... ignoring")
                return
            if new_step_id == len(task_state.steps):
                # Do not change the step id
                new_step_id = curr_step_id

            # Mark this step as done
            task_state.steps_complete[curr_step_id] = True
        else:
            # Check if we are at the start of the list
            if task_state.current_step is None or curr_step_id <= 0:
                log.debug("Attempting to advance before start of list... ignoring")
                return

            new_step_id = curr_step_id - 1

            # Unmark this step as done
            task_state.steps_complete[curr_step_id - 1] = False

        task_state.previous_step_id = task_state.current_step_id
        task_state.previous_step = task_state.steps[task_state.previous_step_id]

        task_state.current_step_id = new_step_id
        task_state.current_step = task_state.steps[new_step_id]

        log.info(
            f"Advanced task {task_id} to step {new_step_id}: {task_state.current_step}"
        )
        self.publish_task_state_message(task_id)

    def on_press(self, key):
        """
        Callback function for keypress events. Uses the number keys to advance
        between tasks.
        """
        if key == keyboard.KeyCode.from_char("1"):
            task_id = 0
            forward = True
        elif key == keyboard.KeyCode.from_char("2"):
            task_id = 0
            forward = False
        elif key == keyboard.KeyCode.from_char("3"):
            task_id = 1
            forward = True
        elif key == keyboard.KeyCode.from_char("4"):
            task_id = 1
            forward = False
        elif key == keyboard.KeyCode.from_char("5"):
            task_id = 2
            forward = True
        elif key == keyboard.KeyCode.from_char("6"):
            task_id = 2
            forward = False
        elif key == keyboard.KeyCode.from_char("7"):
            task_id = 3
            forward = True
        elif key == keyboard.KeyCode.from_char("8"):
            task_id = 3
            forward = False
        elif key == keyboard.KeyCode.from_char("9"):
            task_id = 4
            forward = True
        elif key == keyboard.KeyCode.from_char("0"):
            task_id = 4
            forward = False
        else:
            return  # ignore

        self.advance_step(task_id, forward)


def main():
    rclpy.init()

    multi_task_monitor = DummyMultiTaskMonitor()

    try:
        rclpy.spin(multi_task_monitor)
    except KeyboardInterrupt:
        multi_task_monitor.get_logger().info("Keyboard interrupt, shutting down.\n")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    multi_task_monitor.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
