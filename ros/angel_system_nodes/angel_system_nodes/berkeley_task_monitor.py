import os
import rclpy
import yaml
import threading
import numpy as np

from pynput import keyboard
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from angel_msgs.msg import TaskGraph, TaskUpdate
from angel_msgs.srv import QueryTaskGraph

from angel_system.berkeley.demo import predictor, model


BRIDGE = CvBridge()

KEY_LEFT_SQBRACKET = keyboard.KeyCode.from_char("[")
KEY_RIGHT_SQBRACKET = keyboard.KeyCode.from_char("]")


class TaskMonitor(Node):
    """
    ROS node responsible for keeping track of the current task being performed.
    The task is represented as a state machine with the `transitions` python
    library.

    Uses `angel_msgs/ActivityDetections` to determine the current activity and then
    publishes `angel_msgs/TaskUpdate` messages representing the current state of the
    task.

    The `task_trigger_thresholds` wants to be given the path to file that is of
    JSON format and defines a mapping of activity names to a confidence
    threshold. If we observe a detected activity whose confidence is above this
    threshold we consider that activity to have occurred. The keys of this
    mapping must match the labels output by the activity detector that is
    feeding this node, otherwise input activities that do not match an entry in
    this mapping will be ignored. Thresholds are triggered if values meet or
    exceed the values provided.
    """
    def __init__(self):
        super().__init__(self.__class__.__name__)

        # Inputs
        self._image_topic = (
            self.declare_parameter("image_topic", "debug/PVFrames")
            .get_parameter_value()
            .string_value
        )
        self._config_file = (
            self.declare_parameter("config_file",
                                   "config/tasks/task_steps_berkeley_config-recipe_coffee.yaml")
            .get_parameter_value()
            .string_value
        )
        self._draw_output = True
        #(
        #    self.declare_parameter("draw_output", 1)
        #    .get_parameter_value()
        #    .string_value
        #)
        self._task_state_topic = (
            self.declare_parameter("task_state_topic", "TaskUpdates")
            .get_parameter_value()
            .string_value
        )
        self._topic_output_images_topic = (
            self.declare_parameter("topic_output_images", "BerkeleyFrames")
            .get_parameter_value()
            .string_value
        )
        self._query_task_graph_topic = (
            self.declare_parameter("query_task_graph_topic", "query_task_graph")
            .get_parameter_value()
            .string_value
        )

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Config file: {self._config_file}")
        log.info(f"draw_output: {self._draw_output}")
        log.info(f"Task state topic: {self._task_state_topic}")
        log.info(f"Output image topic: {self._topic_output_images_topic}")
        log.info(f"Query task graph topic: {self._query_task_graph_topic}")

        # Initialize ROS hooks
        self._subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.listener_callback,
            1
        )
        self._task_update_publisher = self.create_publisher(
            TaskUpdate,
            self._task_state_topic,
            1
        )
        self._generated_image_publisher = self.create_publisher(
            Image,
            self._topic_output_images_topic,
            10  # TODO: Learn QoS meanings
        )
        self._task_graph_service = self.create_service(
            QueryTaskGraph,
            self._query_task_graph_topic,
            self.query_task_graph_callback
        )

        # Get the task info from the config
        with open(self._config_file, 'r') as f:
            config = yaml.safe_load(f)

        self._task_title = config["title"]
        log.info(f"Task: {self._task_title}")

        self._steps = [i["description"] for i in config["steps"]]

        self._completed_steps = np.array(np.zeros(len(self._steps)), dtype=bool)

        # Step classifier
        berkeley_root = "angel_system/berkeley"
        parser = model.get_parser()

        config_file = f"{berkeley_root}/configs/MC50-InstanceSegmentation/mask_rcnn_R_101_FPN_1x_demo.yaml"
        conf_thr = 0.7
        args = parser.parse_args(f"--config-file {config_file} --confidence-threshold {conf_thr}".split())

        log.info("Arguments: " + str(args))

        cfg = model.setup_cfg(args)

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.demo = predictor.VisualizationDemo_add_smoothing(
            cfg,
            last_time=2,
            draw_output=self._draw_output
        )

        self.idx = 0

        # Represents the current state of the task
        self._current_step = None
        self._current_step_id = 0
        self._previous_step = None
        self._previous_step_id = 0

        # Control thread access to advancing the task step
        self._task_lock = threading.RLock()

        # Start the keyboard monitoring thread
        log.info(f"Starting keyboard threads")
        self._keyboard_t = threading.Thread(target=self.monitor_keypress)
        self._keyboard_t.daemon = True
        self._keyboard_t.start()
        log.info(f"Starting keyboard threads... done")

        # Publish task update to indicate the initial state
        self.publish_task_state_message()

    def query_task_graph_callback(self, request, response):
        """
        Populate the `QueryTaskGraph` response with the task list
        and return it.
        """
        log = self.get_logger()
        task_g = TaskGraph()
        
        # TODO: support different task levels?
        task_g.task_steps = self._steps
        task_g.task_levels = [0] * len(self._steps)

        response.task_graph = task_g

        response.task_title = self._task_title
        return response

    def listener_callback(self, image):
        """
        """
        log = self.get_logger()
        self.idx += 1

        # Convert ROS img msg to CV2 image
        bgr_image = BRIDGE.imgmsg_to_cv2(image, desired_encoding="rgb8")

        predictions, step_infos, visualized_output = self.demo.run_on_image_smoothing_v2(
            bgr_image, current_idx=self.idx)

        # publish visualized output
        if self._draw_output:
            visualized_image = visualized_output.get_image()
            image_message = BRIDGE.cv2_to_imgmsg(visualized_image, encoding="passthrough")
            self._generated_image_publisher.publish(image_message)

        # Update current step
        finished_sub_step = False
        if not step_infos[0] == 'Need more test !':
            step, sub_step, next_sub_step, gt = step_infos
            if sub_step[-1] == -1:
                _current_step = 'background'
                _current_sub_step = 'background'
            else:
                if 'end_frame' in list(step[-1].keys()):
                    _current_step =  step[-1]['step'] + ' (finished)'
                else:
                    _current_step = step[-1]['step']
                if 'end_frame' in list(sub_step[-1].keys()):
                    # Sub-step is finished
                    _current_sub_step = sub_step[-1]['sub-step']
                    self._completed_steps[self._steps.index(_current_sub_step)] = True
                    finished_sub_step = True
                    log.info(f"{_current_sub_step} (finished)")
                   # _current_sub_step = _current_sub_step + ' (finished)'
                else:
                    _current_sub_step = sub_step[-1]['sub-step']
                #if '(finished)' in _current_sub_step or '(finished)' in _current_step:
                #if self._completed_steps[self._steps.index(_current_sub_step)]:
                #    _current_step = 'background'
                #    _current_sub_step = 'background'
                #else:
                #    _next_sub_step = next_sub_step[-1]
        else:
            _current_sub_step = None

        log.info(f"current sub-step: {_current_sub_step}")

        # Only change steps if we have a new step or a step finished, and it is not
        # background (ID=0)
        if _current_sub_step != 'background':
            if finished_sub_step or (_current_sub_step != self._current_step):
                log.info("Publishing message")
                # Moving forward
                self._previous_step = self._current_step
                self._previous_step_id = self._current_step_id
                self._current_step = _current_sub_step
                self._current_step_id = self._steps.index(self._current_step)

                self.publish_task_state_message()

    def publish_task_state_message(self):
        """
        Forms and sends a `angel_msgs/TaskUpdate` message to the TaskUpdates topic.
        """
        log = self.get_logger()

        message = TaskUpdate()

        # Populate message header
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "Task message"

        # Populate task name and description
        message.task_name = self._task_title
        
        # Populate task items list

        # Populate step list
        if self._current_step is None:
                message.current_step_id = -1
                message.current_step = "None"
        else:
            message.current_step = self._current_step

        if self._previous_step is None:
            message.previous_step = "N/A"
        else:
            message.previous_step = self._previous_step

        message.completed_steps = self._completed_steps.tolist()

        # Publish
        self._task_update_publisher.publish(message)

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
        the task monitor advances to the next step.
        """
        log = self.get_logger()
        if key == KEY_RIGHT_SQBRACKET:
            forward = True
        elif key == KEY_LEFT_SQBRACKET:
            forward = False
        else:
            return  # ignore

        with self._task_lock:
            if forward:
                if (self._current_step_id + 1) >= len(self._steps):
                    log.warn(f"Tried to trigger on invalid state")
                    return

                log.info(f"Proceeding to next step.")

                self._previous_step = self._current_step
                self._previous_step_id = self._current_step_id

                self._current_step_id += 1
                self._current_step = self._steps[self._current_step_id]

                # TODO: how to move demo to next step?
            else:
                log.info(f"Proceeding to previous step")
                if (self._current_step_id - 1) < 0:
                    log.warn(f"Tried to set machine to invalid state")
                    return

                if self._current_step_id == 0:
                    self._current_step = None
                    self._previous_step = None
                else:
                    self._current_step = self._previous_step
                    self._current_step_id = self._previous_step_id

                    self._previous_step_id = self._current_step_id - 1
                    if self._previous_step_id < 0:
                        self._previous_step = None
                    else:
                        self._previous_step = self._steps[self._previous_step_id]
                
                # TODO: how to move demo to previous step?

            log.info(f"Current step is now: {self._current_step}")
            log.info(f"Previous step is now: {self._previous_step}")

            self.publish_task_state_message()

def main():
    rclpy.init()

    task_monitor = TaskMonitor()

    try:
        rclpy.spin(task_monitor)
    except KeyboardInterrupt:
        task_monitor.get_logger().info("Keyboard interrupt, shutting down.\n")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    task_monitor.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()