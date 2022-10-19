from threading import (
    Event,
    RLock,
    Thread,
)
import time

import rclpy
from rclpy.node import Node

from angel_msgs.msg import (
    ActivityDetection,
    TaskUpdate,
    TaskGraph,
)
from angel_msgs.srv import QueryTaskGraph
from angel_system.activity_hmm.core import ActivityHMMRos
from angel_utils.conversion import time_to_int


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

        # Instantiate the HMM module
        self._hmm = ActivityHMMRos(self._config_file)
        log.info(f"HMM node initialized with {self._config_file}")

        # Tracks the current/previous steps
        self._previous_step = None
        self._current_step = None

        # HMM's confidence that task is done
        self._task_complete_confidence = 0.0

        # Initialize ROS hooks
        self._subscription = self.create_subscription(
            ActivityDetection,
            self._det_topic,
            self.det_callback,
            1
        )
        self._publisher = self.create_publisher(
            TaskUpdate,
            self._task_state_topic,
            1
        )
        self._task_graph_service = self.create_service(
            QueryTaskGraph,
            "query_task_graph",
            self.query_task_graph_callback
        )

        # Control access to HMM
        self._hmm_lock = RLock()

        # Start the HMM state threads
        log.info(f"Starting HMM threads")
        self._hmm_active = Event()
        self._hmm_active.set()
        self._hmm_active_heartbeat = 0.1

        self._hmm_state_awake_evt = Event()
        self._hmm_skip_awake_evt = Event()

        self._hmm_state_thread = Thread(
            target=self.thread_run_hmm_state,
            name="hmm_state_runtime"
        )
        self._hmm_state_thread.daemon = True
        self._hmm_state_thread.start()

        self._hmm_skip_thread = Thread(
            target=self.thread_run_hmm_skip,
            name="hmm_skip_runtime"
        )
        self._hmm_skip_thread.daemon = True
        self._hmm_skip_thread.start()
        log.info(f"Starting HMM threads... done")

    def det_callback(self, activity_msg: ActivityDetection):
        """
        Callback function for the activity detection subscriber topic.
        Adds the activity detection msg to the HMM and then publishes a new
        TaskUpdate message.
        """
        if self.hmm_alive():
            source_stamp_start_frame_sec = time_to_int(
                activity_msg.source_stamp_start_frame
            ) * 1e-9 # time_to_int returns ns
            source_stamp_end_frame_sec = time_to_int(
                activity_msg.source_stamp_end_frame
            ) * 1e-9 # time_to_int returns ns

            # Add activity classification to the HMM
            with self._hmm_lock:
                self._hmm.add_activity_classification(
                    activity_msg.label_vec,
                    activity_msg.conf_vec,
                    source_stamp_start_frame_sec,
                    source_stamp_end_frame_sec,
                )

            # Tell the HMM threads to wake up
            self._hmm_state_awake_evt.set()
            self._hmm_skip_awake_evt.set()

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

        # Populate steps and current step
        message.steps = self._hmm.model.class_str
        message.current_step = self._current_step
        message.current_step_id = self._hmm.model.class_str.index(self._current_step)

        if self._previous_step is None:
            message.previous_step = "N/A"
        else:
            message.previous_step = self._previous_step

        message.task_complete_confidence = self._task_complete_confidence

        # TODO: Populate task name and description
        # TODO: Do we need to fill in current/next activity?
        # TODO: Fill in time remaining?

        self._publisher.publish(message)

    def query_task_graph_callback(self, request, response):
        """
        Populate the `QueryTaskGraph` response with the task list
        and return it.
        """
        log = self.get_logger()
        task_g = TaskGraph()

        task_g.task_steps = self._hmm.model.class_str

        # TODO: support different task levels?
        task_g.task_levels = [0] * len(self._hmm.model.class_str)

        response.task_graph = task_g
        return response

    def thread_run_hmm_state(self):
        """
        HMM runtime thread that gets the current state.
        Thread is awakened each time a new ActivityDetection message is
        received.

        It is expected that the HMM's current state computation will finish
        well before a new ActivityDetection message arrives, but we put it in a
        separate thread just to be safe.
        """
        log = self.get_logger()
        log.info("HMM state thread started")

        while self._hmm_active.wait(0): # will quickly return false if cleared
            if self._hmm_state_awake_evt.wait(self._hmm_active_heartbeat):
                log.info("HMM state loop awakened")
                self._hmm_state_awake_evt.clear()

                # Get the HMM prediction
                start_time = time.time()
                with self._hmm_lock:
                    step = self._hmm.get_current_state()
                log.info(f"HMM get_current_state time: {time.time() - start_time}")

                if self._current_step != step:
                    self._previous_step = self._current_step
                    self._current_step = step

                log.info(f"Current step {self._current_step}")

                # Publish a new TaskUpdate message
                self.publish_task_state_message()

    def thread_run_hmm_skip(self):
        """
        HMM runtime thread that gets the current likelihood of a step skip.
        Thread is awakened each time a new ActivityDetection message is
        received.

        It is expected that the skip score computation will finish well before
        a new ActivityDetection message arrives, but we put it in a separate
        thread just to be safe.
        """
        log = self.get_logger()
        log.info("HMM skip thread started")

        while self._hmm_active.wait(0): # will quickly return false if cleared
            if self._hmm_skip_awake_evt.wait(self._hmm_active_heartbeat):
                log.info("HMM skip loop awakened")
                self._hmm_skip_awake_evt.clear()

                # Get the HMM prediction
                start_time = time.time()
                with self._hmm_lock:
                    skip_score = self._hmm.get_skip_score()
                log.info(f"HMM get_skip_score time: {time.time() - start_time}")
                log.info(f"Skip score: {skip_score}")

                # TODO: how does skip score relate to task completion conf.
                self._task_complete_confidence = skip_score

    def hmm_alive(self) -> bool:
        """
        Check that the HMM runtime is still alive and raise an exception
        if it is not.
        """
        alive = (
            self._hmm_state_thread.is_alive() and self._hmm_skip_thread.is_alive()
        )
        if not alive:
            self.get_logger().warn("HMM thread no longer alive.")
            self._hmm_active.clear()  # make HMM active flag "False"
            self._hmm_state_thread.join()
            self._hmm_skip_thread.join()
        return alive

    def destroy_node(self):
        log = self.get_logger()
        log.info("Shutting down runtime threads...")
        self._hmm_active.clear()  # make HMM active flag "False"
        self._hmm_state_thread.join()
        self._hmm_skip_thread.join()
        log.info("Shutting down runtime threads... Done")
        super()


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
