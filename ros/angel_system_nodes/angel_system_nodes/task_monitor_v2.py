from threading import (
    Event,
    RLock,
    Thread,
)
import time
import yaml

from pynput import keyboard
import rclpy
from rclpy.node import Node

from angel_msgs.msg import (
    ActivityDetection,
    AruiUserNotification,
    TaskUpdate,
    TaskGraph,
)
from angel_msgs.srv import QueryTaskGraph
from angel_system.activity_hmm.core import ActivityHMMRos
from angel_utils.conversion import time_to_int


KEY_LEFT_SQBRACKET = keyboard.KeyCode.from_char("[")
KEY_RIGHT_SQBRACKET = keyboard.KeyCode.from_char("]")


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
            self.declare_parameter("config_file",
                                   "config/tasks/task_steps_config-recipe_coffee.yaml")
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
        self._skip_score_threshold = (
            self.declare_parameter("skip_score_threshold", 300.0)
            .get_parameter_value()
            .double_value
        )

        # Instantiate the HMM module
        self._hmm = ActivityHMMRos(self._config_file)
        log.info(f"HMM node initialized with {self._config_file}")

        # Get the task title from the config
        with open(self._config_file, 'r') as f:
            config = yaml.safe_load(f)

        self._task_title = config["title"]
        log.info(f"Task: {self._task_title}")

        # Tracks the current/previous steps
        # Current step is the most recently completed step the user has finished
        # as predicted by the HMM.
        # Previous step is the step before the current step.
        # Values will be either None or the step string label.
        self._previous_step = None
        self._current_step = None
        # Track the step skips we previously notified the user about to avoid
        # sending duplicate notifications for the same error.
        self._previous_step_skip = None
        self._current_step_skip = None
        # Track the latest activity classification end time sent to the HMM
        self._latest_act_classification_end_time = None

        # HMM's confidence that a step was skipped
        self._skip_score = 0.0

        # Initialize ROS hooks
        self._subscription = self.create_subscription(
            ActivityDetection,
            self._det_topic,
            self.det_callback,
            1
        )
        self._task_update_publisher = self.create_publisher(
            TaskUpdate,
            self._task_state_topic,
            1
        )
        self._task_error_publisher = self.create_publisher(
            AruiUserNotification,
            self._task_error_topic,
            1
        )
        self._task_graph_service = self.create_service(
            QueryTaskGraph,
            self._query_task_graph_topic,
            self.query_task_graph_callback
        )

        # Control access to HMM
        self._hmm_lock = RLock()

        # Start the HMM threads
        log.info(f"Starting HMM threads")
        self._hmm_active = Event()
        self._hmm_active.set()
        self._hmm_active_heartbeat = 0.1

        self._hmm_awake_evt = Event()

        self._hmm_thread = Thread(
            target=self.thread_run_hmm,
            name="hmm_runtime"
        )
        self._hmm_thread.daemon = True
        self._hmm_thread.start()

        log.info(f"Starting HMM threads... done")

        # Start the keyboard monitoring thread
        log.info(f"Starting keyboard threads")
        self._keyboard_t = Thread(target=self.monitor_keypress)
        self._keyboard_t.daemon = True
        self._keyboard_t.start()
        log.info(f"Starting keyboard threads... done")

    def det_callback(self, activity_msg: ActivityDetection):
        """
        Callback function for the activity detection subscriber topic.
        Adds the activity detection msg to the HMM and then publishes a new
        TaskUpdate message.
        """
        if self.hmm_alive():
            source_stamp_start_frame_sec = time_to_int(
                activity_msg.source_stamp_start_frame
            ) * 1e-9  # time_to_int returns ns
            source_stamp_end_frame_sec = time_to_int(
                activity_msg.source_stamp_end_frame
            ) * 1e-9  # time_to_int returns ns

            # Add activity classification to the HMM
            with self._hmm_lock:
                if (
                    self._latest_act_classification_end_time is not None and
                    self._latest_act_classification_end_time >= source_stamp_start_frame_sec
                ):
                    # We already sent an activity classification to the HMM
                    # that was after this frame window's start time
                    return

                self._latest_act_classification_end_time = source_stamp_end_frame_sec

                self._hmm.add_activity_classification(
                    activity_msg.label_vec,
                    activity_msg.conf_vec,
                    source_stamp_start_frame_sec,
                    source_stamp_end_frame_sec,
                )

            # Tell the HMM thread to wake up
            self._hmm_awake_evt.set()

    def publish_task_state_message(self):
        """
        Forms and sends a `angel_msgs/TaskUpdate` message to the
        TaskUpdates topic.
        """
        log = self.get_logger()

        message = TaskUpdate()

        # Populate message header
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "Task message"
        message.task_name = self._task_title

        # Populate steps and current step
        with self._hmm_lock:
            message.steps = self._hmm.model.class_str[1:]  # exclude background
            last_step_id = len(message.steps) - 1

            if self._current_step is None:
                message.current_step_id = -1
                message.current_step = "None"
            else:
                steps_list_id = message.steps.index(
                    self._current_step
                )
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

        # TODO: Do we need to fill in the other fields

        self._task_update_publisher.publish(message)

    def publish_task_error_message(self):
        """
        Forms and sends a `angel_msgs/AruiUserNotification` message to the
        task errors topic.
        """
        log = self.get_logger()

        # TODO: Using AruiUserNotification for this error is a temporary
        # placeholder. There should be a new message created for this task
        # error.
        message = AruiUserNotification()
        message.category = message.N_CAT_NOTICE
        message.context = message.N_CONTEXT_TASK_ERROR

        message.title = "Step skip detected"
        with self._hmm_lock:
            message.description = (
                f"Detected skip with confidence {self._skip_score}. "
                f"Current step: {self._current_step}, "
                f"previous step: {self._previous_step}."
            )

        self._task_error_publisher.publish(message)

    def query_task_graph_callback(self, request, response):
        """
        Populate the `QueryTaskGraph` response with the task list
        and return it.
        """
        log = self.get_logger()
        task_g = TaskGraph()

        with self._hmm_lock:
            task_g.task_steps = self._hmm.model.class_str[1:] # exclude background
            # TODO: support different task levels?
            task_g.task_levels = [0] * len(self._hmm.model.class_str)

        response.task_graph = task_g

        response.task_title = self._task_title
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
                log.info("HMM loop awakened")
                self._hmm_awake_evt.clear()

                # Get the HMM prediction
                start_time = time.time()
                with self._hmm_lock:
                    skip_idx, no_skip_idx, self._skip_score = (
                        self._hmm.analyze_current_state()
                    )
                    log.info(f"HMM computation time: {time.time() - start_time}")

                    step_id = skip_idx[-1]
                    step = self._hmm.model.class_str[step_id]

                    # Only change steps if we have a new step, and it is not background
                    if self._current_step != step and step_id != 0:
                        self._previous_step = self._current_step
                        self._current_step = step

                    log.info(f"Last completed step: {self._current_step}")
                    log.info(f"Skip score: {self._skip_score}")

                    if self._skip_score > self._skip_score_threshold:
                        # Check if we've already sent a notification for this step
                        # skip to avoid sending lots of notifications for the same
                        # error.
                        if (
                            self._current_step != self._current_step_skip or
                            self._previous_step != self._previous_step_skip
                        ):
                            self.publish_task_error_message()
                            self._current_step_skip = self._current_step
                            self._previous_step_skip = self._previous_step

                    # Publish a new TaskUpdate message
                    self.publish_task_state_message()

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
        log.info(f"Starting keyboard monitor. Press the right-bracket key, `]`,"
                 f"to proceed to the next step. Press the left-bracket key, `[`, "
                 f"to go back to the previous step.")
        # Collect events until released
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def on_press(self, key):
        """
        Callback function for keypress events. If the right arrow is pressed,
        the task monitor advances to the next step. If the left arrow is
        pressed, the task monitor advances to the previous step.
        """
        log = self.get_logger()
        if key == KEY_RIGHT_SQBRACKET:
            forward = True
        elif key == KEY_LEFT_SQBRACKET:
            forward = False
        else:
            return  # ignore

        with self._hmm_lock:
            if self._latest_act_classification_end_time is None:
                # No classifications received yet
                # Set time window to now + 1 second
                start_time = time_to_int(
                    self.get_clock().now().to_msg()
                ) * 1e-9  # time_to_int returns ns
                end_time = start_time + 1  # 1 second later
            else:
                # Assuming ~30Hz frame rate, so set start one frame later
                start_time = (
                    self._latest_act_classification_end_time + (1 / 30.0)
                )
                end_time = start_time + 1  # 1 second later

            self._latest_act_classification_end_time = end_time


            # Get the current HMM state
            steps = self._hmm.model.class_str
            if self._current_step is None:
                curr_step_id = 0
            else:
                curr_step_id = steps.index(self._current_step)

            # Create confidence vector where all classes are
            # zero except the new desired step
            conf_vec = [0.0] * len(steps)
            if forward:
                # Check if we are at the end of the list
                if curr_step_id == (len(steps) - 1):
                    log.info("Attempting to advance past end of list... ignoring")
                    return
                conf_vec[curr_step_id + 1] = 1.0

                # Add activity classification to the HMM
                self._hmm.add_activity_classification(
                    steps,
                    conf_vec,
                    start_time,
                    end_time
                )

                # Tell the HMM thread to wake up
                self._hmm_awake_evt.set()
            else:
                # Check if we are at the start of the list
                if self._current_step is None:
                    log.info("Attempting to advance before start of list... ignoring")
                    return

                new_step_id = curr_step_id - 1
                if new_step_id == 0:
                    # Attempting to go back to before the first step, so just
                    # clear out the stored HMM models steps and times
                    self._hmm.X = None
                    self._hmm.times = None
                    self._current_step = None
                    self._previous_step = None

                    self.publish_task_state_message()
                    log.info("HMM reset to beginning")
                else:
                    self._hmm.revert_to_step(new_step_id)

                    # Tell the HMM thread to wake up
                    self._hmm_awake_evt.set()


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


if __name__ == '__main__':
    main()
