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
import numpy as np
import numpy.typing as npt
import rclpy
from rclpy.node import Node
import yaml

from angel_msgs.msg import (
    ActivityDetection,
    AruiUserNotification,
    SystemCommands,
    TaskUpdate,
    TaskGraph,
)
from angel_msgs.srv import QueryTaskGraph
from angel_system.activity_hmm.core import ActivityHMMRos
from angel_utils.conversion import time_to_int, nano_to_ros_time


class HMMNode(Node):
    """
    ROS node that runs the HMM and publishes TaskUpdate messages. The HMM is
    called on a separate thread.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        log = self.get_logger()

        self._det_topic = (
            self.declare_parameter("det_topic", "ActivityDetections")
            .get_parameter_value()
            .string_value
        )
        self._config_file = (
            self.declare_parameter(
                "config_file", "config/tasks/task_steps_config-recipe_coffee.yaml"
            )
            .get_parameter_value()
            .string_value
        )
        self._task_state_topic = (
            self.declare_parameter("task_state_topic", "TaskUpdates")
            .get_parameter_value()
            .string_value
        )
        self._task_error_topic = (
            self.declare_parameter("task_error_topic", "TaskErrors")
            .get_parameter_value()
            .string_value
        )
        self._query_task_graph_topic = (
            self.declare_parameter("query_task_graph_topic", "query_task_graph")
            .get_parameter_value()
            .string_value
        )
        self._step_complete_threshold = (
            self.declare_parameter("step_complete_threshold", 0.5)
            .get_parameter_value()
            .double_value
        )
        self._enable_manual_progression = (
            self.declare_parameter("enable_manual_progression", True)
            .get_parameter_value()
            .bool_value
        )
        self._allow_rollback = (
            self.declare_parameter("allow_rollback", True)
            .get_parameter_value()
            .bool_value
        )
        self._sys_cmd_topic = (
            self.declare_parameter("sys_cmd_topic", "")
            .get_parameter_value()
            .string_value
        )
        if self._sys_cmd_topic == "":
            raise ValueError(
                "Please provide the system command topic with the `sys_cmd_topic` parameter"
            )

        log.info(f"Detection topic: {self._det_topic}")
        log.info(f"Task updates topic: {self._task_state_topic}")
        log.info(f"Task error topic: {self._task_error_topic}")
        log.info(f"Query task graph topic: {self._query_task_graph_topic}")
        log.info(f"System command topic: {self._sys_cmd_topic}")

        log.info(f"Step complete threshold: {self._step_complete_threshold}")
        log.info(f"Enable manual progression: {self._enable_manual_progression}")
        log.info(f"Allow progress rollback: {self._allow_rollback}")

        # Instantiate the HMM module
        self._hmm = ActivityHMMRos(self._config_file)
        log.info(f"HMM node initialized with {self._config_file}")

        # Cache the quantity of "real" task steps configured
        # (discounting HMM background "step").
        self._n_steps = self._hmm.num_steps - 1

        # Get the task title from the config
        with open(self._config_file, "r") as f:
            config = yaml.safe_load(f)

        self._task_title = config["title"]
        log.info(f"Task: {self._task_title}")

        # Tracks the current/previous steps
        # Current step is the most recently completed step the user has finished
        # as predicted by the HMM.
        # Previous step is the step before the current step.
        # Values will be either None or the step string label.
        self._current_step = None
        self._current_step_id = 0  # HMM ID
        self._previous_step = None
        self._previous_step_id = 0  # HMM ID
        # Track the step skips we previously notified the user about to avoid
        # sending duplicate notifications for the same error.
        self._previous_step_skip = None
        self._current_step_skip = None
        # Track the indices of steps we have emitted skip errors for in order
        # to not repeatedly emit many errors for the same steps.
        # This may be removed from if step progress regresses.
        self._steps_skipped_cache: Set[int] = set()
        # Track the latest activity classification end time sent to the HMM
        # Time is in floating-point seconds up to nanosecond precision.
        self._latest_act_classification_end_time = None

        # Initialize ROS hooks
        self._subscription = self.create_subscription(
            ActivityDetection, self._det_topic, self.det_callback, 1
        )
        self._sys_cmd_subscription = self.create_subscription(
            SystemCommands, self._sys_cmd_topic, self.sys_cmd_callback, 1
        )
        self._task_update_publisher = self.create_publisher(
            TaskUpdate, self._task_state_topic, 1
        )
        self._task_error_publisher = self.create_publisher(
            AruiUserNotification, self._task_error_topic, 1
        )
        self._task_graph_service = self.create_service(
            QueryTaskGraph, self._query_task_graph_topic, self.query_task_graph_callback
        )

        # Control access to HMM
        self._hmm_lock = RLock()

        # Start the HMM threads
        log.info(f"Starting HMM threads")
        self._hmm_active = Event()
        self._hmm_active.set()
        self._hmm_active_heartbeat = 0.1

        self._hmm_awake_evt = Event()

        self._hmm_thread = Thread(target=self.thread_run_hmm, name="hmm_runtime")
        self._hmm_thread.daemon = True
        self._hmm_thread.start()

        log.info(f"Starting HMM threads... done")

        if self._enable_manual_progression:
            # Intentional delayed import to enable use without a running X-server.
            from pynput import keyboard

            # Store module for local use.
            self._keyboard = keyboard
            # Start the keyboard monitoring thread
            log.info(f"Starting keyboard threads")
            self._keyboard_t = Thread(target=self.monitor_keypress)
            self._keyboard_t.daemon = True
            self._keyboard_t.start()
            log.info(f"Starting keyboard threads... done")

    def _clean_skipped_cache(self, current_hmm_step_id: int) -> None:
        """Clear the error cache any IDs greater than the given."""
        s = np.asarray(list(self._steps_skipped_cache))
        to_remove_ids = s[s > current_hmm_step_id]
        if to_remove_ids.size:
            self.get_logger().info(
                f"Clearing step IDs from skipped cache: " f"{to_remove_ids.tolist()}"
            )
        [self._steps_skipped_cache.remove(_id) for _id in to_remove_ids]

    def det_callback(self, activity_msg: ActivityDetection):
        """
        Callback function for the activity detection subscriber topic.
        Adds the activity detection msg to the HMM and then publishes a new
        TaskUpdate message.
        """
        if self.hmm_alive():
            # time_to_int returns ns, converting to float seconds
            source_stamp_start_frame_sec = (
                time_to_int(activity_msg.source_stamp_start_frame) * 1e-9
            )
            source_stamp_end_frame_sec = (
                time_to_int(activity_msg.source_stamp_end_frame) * 1e-9
            )

            # Add activity classification to the HMM
            with self._hmm_lock:
                if (
                    self._latest_act_classification_end_time is not None
                    and self._latest_act_classification_end_time
                    >= source_stamp_start_frame_sec
                ):
                    # We already sent an activity classification to the HMM
                    # that was after this frame window's start time
                    return

                self._latest_act_classification_end_time = source_stamp_end_frame_sec

                act_id_vec = np.arange(len(activity_msg.conf_vec))
                self._hmm.add_activity_classification(
                    act_id_vec,
                    activity_msg.conf_vec,
                    source_stamp_start_frame_sec,
                    source_stamp_end_frame_sec,
                )

            # Tell the HMM thread to wake up
            self._hmm_awake_evt.set()

    def publish_task_state_message(
        self,
        steps_complete_vec: npt.NDArray[bool],
        hmm_step_conf: npt.NDArray[float],
    ) -> None:
        """
        Forms and sends a `angel_msgs/TaskUpdate` message to the
        TaskUpdates topic.

        :param steps_complete_vec: Boolean vector the length of task steps (as
            reported by `query_task_graph_callback`) that indicates which steps
            to report as having been completed.
        :param hmm_step_conf: Confidence vector of the HMM that we are in any
            particular step.
        """
        log = self.get_logger()

        message = TaskUpdate()

        # Populate message header
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "Task message"
        message.task_name = self._task_title
        if self._latest_act_classification_end_time is not None:
            message.latest_sensor_input_time = nano_to_ros_time(
                int(self._latest_act_classification_end_time * 1e9)
            )
        else:
            # Fill in a default time of 0.0 seconds.
            message.latest_sensor_input_time = Time(0, 0)

        # Populate steps and current step
        with self._hmm_lock:
            # TODO: DEPRECATE steps field here. Use query service instead.
            message.steps = self._hmm.model.class_str[1:]  # exclude background
            last_step_id = len(message.steps) - 1

            if self._current_step is None:
                message.current_step_id = -1
                message.current_step = "None"
            else:
                # Getting the index of the step in the list-of-strings
                steps_list_id = message.steps.index(self._current_step)
                message.current_step_id = steps_list_id
                message.current_step = message.steps[message.current_step_id]

            if self._previous_step is None:
                message.previous_step = "N/A"
            else:
                message.previous_step = self._previous_step

        if message.current_step_id == last_step_id:
            message.task_complete_confidence = 1.0
        else:
            message.task_complete_confidence = 0.0

        message.completed_steps = steps_complete_vec.tolist()
        log.debug(f"Steps complete: {message.completed_steps}")
        message.hmm_step_confidence = hmm_step_conf.tolist()
        log.debug(f"HMM step confidence: {message.hmm_step_confidence}")

        # TODO: Do we need to fill in the other fields

        self._task_update_publisher.publish(message)

    def publish_task_error_message(self, skipped_step: str, complete_confidence: float):
        """
        Forms and sends a `angel_msgs/AruiUserNotification` message to the
        task errors topic.

        :param skipped_step: Description of the step that was skipped.
        :param complete_confidence: Float completion confidence of the step
            that was determined skipped.
        """
        log = self.get_logger()
        log.info(
            f"Reporting step skipped error: "
            f"(complete_conf={complete_confidence}) {skipped_step}"
        )

        # TODO: Using AruiUserNotification for this error is a temporary
        # placeholder. There should be a new message created for this task
        # error.
        message = AruiUserNotification()
        message.category = message.N_CAT_NOTICE
        message.context = message.N_CONTEXT_TASK_ERROR

        message.title = "Step skip detected"
        message.description = (
            f"Detected skip with confidence {1.0 - complete_confidence}: "
            f"{skipped_step}."
        )

        self._task_error_publisher.publish(message)

    def query_task_graph_callback(self, request, response):
        """
        Populate the `QueryTaskGraph` response with the task list
        and return it.
        """
        log = self.get_logger()
        log.debug("Received request for the current task graph")
        task_g = TaskGraph()

        with self._hmm_lock:
            task_g.task_steps = self._hmm.model.class_str[1:]  # exclude background
            # TODO: support different task levels?
            task_g.task_levels = [0] * len(self._hmm.model.class_str)

        response.task_graph = task_g

        response.task_title = self._task_title
        log.debug("Received request for the current task graph -- Done")
        return response

    def thread_run_hmm(self):
        """
        HMM runtime thread that gets the current state and skip score.
        Thread is awakened each time a new ActivityDetection message is
        received.

        It is expected that the HMM's computation will finish
        well before a new ActivityDetection message arrives, but we put it in a
        separate thread just to be safe.
        """
        log = self.get_logger()
        log.info("HMM thread started")

        while self._hmm_active.wait(0):  # will quickly return false if cleared
            if self._hmm_awake_evt.wait(self._hmm_active_heartbeat):
                log.debug("HMM loop awakened")
                self._hmm_awake_evt.clear()

                # Get the HMM prediction
                start_time = time.time()
                with self._hmm_lock:
                    # REMINDER: state_seq indices refer to HMM step index
                    #   perspective, i.e. 0 == background step.
                    # REMINDER: `step_finished_conf` DOES NOT include the
                    #   background class at index 0. Index 0 of this vector is
                    #   the true first user step.
                    # unfilt_step_conf: ndarray of [0,1] range raw confidences
                    #   from the HMM. Includes the background class (HMM ID
                    #   perspective).
                    (
                        times,
                        state_seq,
                        step_finished_conf,
                        unfilt_step_conf,
                    ) = self._hmm.analyze_current_state()
                    log.debug(f"HMM computation time: {time.time() - start_time}")
                    log.debug(f"HMM State Sequence: {state_seq}")
                    log.debug(f"HMM Steps Finished: {step_finished_conf}")

                    # Get the latest non-zero (non-background) step-ID in the
                    # state sequence to indicate the current state.
                    ss_nonzero = np.nonzero(state_seq)[0]
                    if ss_nonzero.size == 0:
                        # No non-zero entries yet, nothing to do.
                        continue

                    # There are non-zero entries. hmm_step_id should never be
                    # zero.
                    hmm_step_id = state_seq[ss_nonzero[-1]]
                    assert hmm_step_id != 0, (
                        "Should not be able to be set to background ID at " "this point"
                    )
                    step_str = self._hmm.model.class_str[hmm_step_id]

                    # Only change steps if we have a new step, and it is not
                    # background (ID=0).
                    if self._current_step != step_str:
                        if (
                            not self._allow_rollback
                            and hmm_step_id < self._current_step_id
                        ):
                            # Ignore backwards progress
                            log.debug(
                                f"Predicted step {hmm_step_id} but we are not allowing rollback from step {self._current_step_id}"
                            )
                        else:
                            self._previous_step = self._current_step
                            self._previous_step_id = self._current_step_id
                            self._current_step = step_str
                            self._current_step_id = hmm_step_id

                            user_step_id = (
                                self._current_step_id - 1
                            )  # no user "background" step

                            steps_complete = cast(
                                npt.NDArray[bool],
                                step_finished_conf >= self._step_complete_threshold,
                            )

                            # Force steps beyond the current to not be considered
                            # finished (haven't happened yet). hmm_step_id includes
                            # addressing background, so `hmm_step_id == user_step_id+1`
                            steps_complete[user_step_id + 1 :] = False

                            # Skipped steps are those steps at or before the current
                            # step that are strictly below the step complete threshold.
                            steps_skipped = cast(
                                npt.NDArray[bool],
                                step_finished_conf < self._step_complete_threshold,
                            )
                            # Force steps beyond the current to not be considered
                            # skipped (haven't happened yet).
                            steps_skipped[user_step_id + 1 :] = False

                        # Handle regression in steps actions
                        if hmm_step_id < self._previous_step_id:
                            # Now "later" steps should be removed from the
                            # skipped cache.
                            self._clean_skipped_cache(self._current_step_id)

                    log.debug(
                        f"Most recently completed step {self._current_step_id}: {self._current_step}"
                    )

                    if steps_skipped.max():
                        skipped_step_ids = np.nonzero(steps_skipped)[0]
                        for s_id in skipped_step_ids:
                            if s_id not in self._steps_skipped_cache:
                                s_str = self._hmm.model.class_str[s_id + 1]
                                self.publish_task_error_message(
                                    s_str, step_finished_conf[s_id]
                                )
                                self._steps_skipped_cache.add(s_id)

                    # Publish a new TaskUpdate message
                    self.publish_task_state_message(
                        steps_complete,
                        unfilt_step_conf,
                    )

    def hmm_alive(self) -> bool:
        """
        Check that the HMM runtime is still alive and raise an exception
        if it is not.
        """
        alive = self._hmm_thread.is_alive()
        if not alive:
            self.get_logger().warn("HMM thread no longer alive.")
            self._hmm_active.clear()  # make HMM active flag "False"
            self._hmm_awake_evt.clear()
            self._hmm_thread.join()
        return alive

    def destroy_node(self):
        log = self.get_logger()
        log.info("Shutting down runtime threads...")
        self._hmm_active.clear()  # make HMM active flag "False"
        self._hmm_awake_evt.clear()
        self._hmm_thread.join()
        log.info("Shutting down runtime threads... Done")
        super()

    def monitor_keypress(self):
        log = self.get_logger()
        log.info(
            f"Starting keyboard monitor. Press the right-bracket key, `]`,"
            f"to proceed to the next step. Press the left-bracket key, `[`, "
            f"to go back to the previous step."
        )
        # Collect events until released
        with self._keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def advance_step(self, forward: bool) -> None:
        """
        Handles advancing the HMM forward and backwards.
        """
        log = self.get_logger()

        with self._hmm_lock:
            if self._latest_act_classification_end_time is None:
                # No classifications received yet
                # Set time window to now + 1 second
                start_time = (
                    time_to_int(self.get_clock().now().to_msg()) * 1e-9
                )  # time_to_int returns ns
                end_time = start_time + 1  # 1 second later
            else:
                # Assuming ~30Hz frame rate, so set start one frame later
                start_time = self._latest_act_classification_end_time + (1 / 30.0)
                end_time = start_time + 1  # 1 second later

            self._latest_act_classification_end_time = end_time

            # Get the current HMM state
            steps = self._hmm.model.class_str
            if self._current_step is None:
                curr_step_id = 0  # technically the "background" step, see +1 below
            else:
                curr_step_id = steps.index(self._current_step)

            if forward:
                # Create confidence vector as pulled from the HMM means
                # ("ideal" input confidence vector for a step).
                # Check if we are at the end of the list (current == "done")
                if curr_step_id == (len(steps) - 1):
                    log.debug("Attempting to advance past end of list... ignoring")
                    return
                log.debug(
                    f"Manually progressing forward pass step: " f"{self._current_step}"
                )
                # Getting the mean vector for the step *after* the current one.
                conf_vec = self._hmm.get_hmm_mean_and_std()[0][curr_step_id + 1]
                label_vec = np.arange(conf_vec.size)

                # Add activity classification to the HMM
                self._hmm.add_activity_classification(
                    label_vec, conf_vec, start_time, end_time
                )

                # Tell the HMM thread to wake up
                self._hmm_awake_evt.set()
            else:
                # Check if we are at the start of the list
                if self._current_step is None:
                    log.debug("Attempting to advance before start of list... ignoring")
                    return

                new_step_id = curr_step_id - 1
                if new_step_id == 0:
                    # Attempting to go back to before the first step, so just
                    # clear out the stored HMM models steps and times
                    self._hmm.X = None
                    self._hmm.times = None
                    self._current_step = None
                    self._previous_step = None
                    self._steps_skipped_cache.clear()
                    steps_complete = np.zeros(self._n_steps, dtype=bool)
                    zero_step_conf = np.zeros(self._hmm.num_steps, dtype=float)

                    self.publish_task_state_message(steps_complete, zero_step_conf)
                    log.debug("HMM reset to beginning")
                else:
                    self._hmm.revert_to_step(new_step_id)
                    self._clean_skipped_cache(new_step_id)

                    # Tell the HMM thread to wake up
                    self._hmm_awake_evt.set()

    def on_press(self, key):
        """
        Callback function for keypress events. If the right arrow is pressed,
        the task monitor advances to the next step. If the left arrow is
        pressed, the task monitor advances to the previous step.
        """
        if key == self._keyboard.KeyCode.from_char("]"):
            forward = True
        elif key == self._keyboard.KeyCode.from_char("["):
            forward = False
        else:
            return  # ignore

        self.advance_step(forward)

    def sys_cmd_callback(self, cmd: SystemCommands):
        """
        Callback function for SystemCommands messages. Executes the provided
        system command.

        Currently only examines the next_step and previous_step system commands.
        """
        if cmd.next_step:
            forward = True
        elif cmd.previous_step:
            forward = False
        else:
            return  # ignore

        self.advance_step(forward)


def main():
    rclpy.init()

    hmm = HMMNode()

    try:
        rclpy.spin(hmm)
    except KeyboardInterrupt:
        hmm.get_logger().info("Keyboard interrupt, shutting down.\n")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    hmm.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
