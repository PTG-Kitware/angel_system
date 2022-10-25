import json
import threading
import time
import uuid
from typing import Optional

import numpy as np
from pynput import keyboard
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from transitions import Machine
import transitions

from angel_msgs.msg import ActivityDetection, TaskUpdate, TaskItem, TaskGraph, TaskNode
from angel_msgs.srv import QueryTaskGraph


class TeaTask():
    """
    Representation of the simple weak tea task defined
    by its steps and transitions between them.
    """
    def __init__(self):
        self.name = 'Making Tea'

        self.items = {'water bottle': 1, 'tea bag': 1, 'cup': 1}

        self.description = ('Open the water bottle and pour the water into a tea cup.' +
                            ' Place the tea bag in the cup.' +
                            ' Wait 20 seconds while the tea bag steeps, then drink and enjoy!')

        self.steps = [{'name': 'open_bottle_and_pour_water_into_cup'},
                      {'name': 'place_tea_bag_into_cup'},
                      {'name': 'steep_for_20_seconds'},
                      {'name': 'enjoy'},
                     ]

        self.transitions = [
            { 'trigger': 'open_bottle', 'source': 'open_bottle_and_pour_water_into_cup', 'dest': 'place_tea_bag_into_cup' },
            { 'trigger': 'make_tea', 'source': 'place_tea_bag_into_cup', 'dest': 'steep_for_20_seconds' },
        ]

        self.machine = Machine(model=self, states=self.steps,
                               transitions=self.transitions, initial='open_bottle_and_pour_water_into_cup')

        self.machine.states['steep_for_20_seconds'].timer_length = 20.0

        # Mapping from state name to the to_state function, which provides a way to get
        # to the state from anywhere.
        # The to_* functions are created automatically when the Machine is initialized.
        self.to_state_dict = {
            'open_bottle_and_pour_water_into_cup': self.to_open_bottle_and_pour_water_into_cup,
            'place_tea_bag_into_cup': self.to_place_tea_bag_into_cup,
            'steep_for_20_seconds': self.to_steep_for_20_seconds,
            'enjoy': self.to_enjoy,
        }


class CoffeeDemoTask():
    """
    Representation of the coffee demo recipe defined
    by its steps and transitions between them.

    For details on using the pytransitions package, see:
    https://github.com/pytransitions/transitions#quickstart

    NOTE: This currently only represents first 13 steps of the coffee demo.
    Steps are listed under v1.2 here:
    https://docs.google.com/document/d/1MfbZdRS6tOGzqNSN-_22Xwmq-huk5_WdKeDSEulFOL0/edit#heading=h.l369fku95vnn

    :param task_steps_file: Path to the file that contains the steps to use
        for this task. This file should be a json file containing a dictionary
        mapping activity names to step indices. The activity names should match
        the output of the activity detector node.
    """
    def __init__(
        self,
        task_steps_file: str
    ):
        self.name = 'Pour-over coffee'

        self.items = {'scale': 1, '25g coffee beans': 1, 'mug': 1, '12 oz water': 1,
                      'kettle': 1, 'grinder': 1, 'measuring cup': 1, 'bowl': 1}

        self.description = ('Boil water, grind coffee beans, and' +
                            ' place the coffee filter into the dripper,'
                            ' and then place the dripper on top of the mug')

        # Load the task steps from the provided steps file
        with open(task_steps_file, "r") as f:
            self._task_steps = json.load(f)

        # Task graph information
        self.task_graph_steps = []
        self.task_graph_step_levels = []

        # State machine steps
        self.steps = []
        self.uids = {}

        # Create the list of steps for the state machine and fill out
        # the task graph information
        for name, step in self._task_steps.items():
            self.task_graph_steps.append(name)
            self.task_graph_step_levels.append(step['level'])

            # NOTE: Only extracting the sub steps for use in the state machine
            # for now
            for sub_step_name, sub_step in step['sub-steps'].items():
                self.task_graph_steps.append(sub_step_name)
                self.task_graph_step_levels.append(sub_step['level'])

                task_name = sub_step['activity']
                self.steps.append({'name': task_name})
                self.uids[task_name] = str(uuid.uuid4())

        # Manually add the finish step
        self.task_graph_steps.append("Done")
        self.task_graph_step_levels.append(0)
        self.steps.append({'name': 'Done'})
        self.uids['Done'] = str(uuid.uuid4())

        # Create the transitions between steps, assuming linear steps
        # TODO: use the step index in the task steps file to decide
        # the destination step
        self.transitions = []
        for idx, step in enumerate(self.steps):
            # No transition needed for the last step
            if idx < (len(self.steps) - 1):
                self.transitions.append({'trigger': step['name'],
                                         'source': step['name'],
                                         'dest': self.steps[idx + 1]['name']})

        self.machine = Machine(model=self, states=self.steps,
                               transitions=self.transitions, initial=self.steps[0]['name'])


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

        self._det_topic = self.declare_parameter("det_topic", "ActivityDetections").get_parameter_value().string_value
        self._task_state_topic = self.declare_parameter("task_state_topic", "TaskUpdates").get_parameter_value().string_value
        self._task_steps = self.declare_parameter("task_steps", "default_task_label_config.json").get_parameter_value().string_value
        # Path to the JSON file that maps steps
        self._task_trigger_thresholds_fp = self.declare_parameter(
            "task_trigger_thresholds",
            "default_task_trigger_thresholds.json"
        ).get_parameter_value().string_value

        log = self.get_logger()

        # Load step thresholds structure
        with open(self._task_trigger_thresholds_fp, 'r') as infile:
            self._task_trigger_thresholds = json.load(infile)

        self._task = CoffeeDemoTask(self._task_steps)

        # Represents the current state of the task
        self._current_step = self._task.state
        self._previous_step = None

        # Represents the current action being performed
        self._current_activity = None
        self._next_activity = None

        # Tracks whether a timer is currently active
        self._timer_active = False
        self._timer_lock = threading.RLock()

        # Control thread access to advancing the task step
        self._task_lock = threading.RLock()

        # Initialize ROS hooks
        self._subscription = self.create_subscription(
            ActivityDetection,
            self._det_topic,
            self.listener_callback,
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

        # Publish task update to indicate the initial state
        self.publish_task_state_message()

    def listener_callback(self, activity_msg: ActivityDetection):
        """
        Callback function for the activity detection subscriber topic.

        Upon receiving a new activity message, this function checks if that
        activity matches any of the defined activities for the current task,
        and attempts to advance the task state machine.

        A new task update message is published if the task's state changed.
        """
        log = self.get_logger()

        # If we are currently in a timer state, exit early since we need to wait for
        # the timer to finish
        with self._timer_lock:
            if self._timer_active:
                log.info(f"Waiting for timer to finish for {self._task.state}")
                return

        # We are expecting a "next" step activity. We observe that this
        # activity has been performed if the confidence of that activity has
        # met or exceeded the associated confidence threshold.
        lbl = self._task.state
        try:
            # Index of the current state label in the activity detection output
            lbl_idx = activity_msg.label_vec.index(lbl)
        except ValueError:
            log.warn(f"Current state ({lbl}) not represented in activity "
                     f"detection results. Received: {activity_msg.label_vec}")
            return

        conf = activity_msg.conf_vec[lbl_idx]
        current_activity: Optional[str] = None
        log.info(f"Awaiting sufficiently high confidence for activity '{lbl}'. "
                 f"Currently: {conf}. Need {self._task_trigger_thresholds[lbl]}.")
        if conf >= self._task_trigger_thresholds[lbl]:
            log.info("Threshold exceeded, setting as current activity.")
            current_activity = lbl
        self._current_activity = current_activity

        if current_activity is None:
            # No activity matching current task.
            return

        # Attempt to advance to the next step
        try:
            with self._task_lock:
                # Attempt to advance to the next state
                self._task.trigger(current_activity)
            log.info(f"Proceeding to next step. Current step: {self._task.state}")

            # Update state tracking vars
            self._previous_step = self._current_step
            self._current_step = self._task.state

        except transitions.core.MachineError as e:
            pass

        self.publish_task_state_message()

        # Check to see if this new state has a timer associated with it
        try:
            if self._task.machine.states[self._task.state].timer_length > 0:
                # Spawn a thread to track the timer and publish state messages
                with self._timer_lock:
                    self._timer_active = True
                    t = threading.Thread(target=self.task_timer_thread)
                    t.start()
        except AttributeError as e:
            # No timer associated with this state
            pass

    def query_task_graph_callback(self, request, response):
        """
        Populate the `QueryTaskGraph` response with the task list
        and return it.
        """
        log = self.get_logger()
        task_g = TaskGraph()

        task_g.task_steps = self._task.task_graph_steps
        task_g.task_levels = self._task.task_graph_step_levels

        response.task_title = self._task.name
        response.task_graph = task_g
        return response

    def publish_task_state_message(self, activity=None):
        """
        Forms and sends a `angel_msgs/TaskUpdate` message to the TaskUpdates topic.
        """
        log = self.get_logger()

        message = TaskUpdate()

        # Populate message header
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "Task message"

        # Populate task name and description
        message.task_name = self._task.name
        message.task_description = self._task.description

        # Populate task items list
        for i, q in self._task.items.items():
            item = TaskItem()
            item.item_name = i
            item.quantity = q
            message.task_items.append(item)

        # Populate step list
        for idx, step in enumerate(self._task.task_graph_steps):
            # TODO: This field is not needed anymore with the
            # now implemented query_task_graph service
            message.steps.append(step)

            # Set the current step index
            if self._current_step == step:
                message.current_step_id = idx

        message.current_step = self._current_step

        if self._previous_step is None:
            message.previous_step = "N/A"
        else:
            message.previous_step = self._previous_step

        if self._current_activity is None:
            message.current_activity = "N/A"
        else:
            message.current_activity = self._current_activity

        for t in self._task.transitions:
            if t['source'] == self._task.state:
                self._next_activity = t['trigger']
                break

        message.next_activity = self._next_activity

        try:
            message.time_remaining_until_next_task = int(self._task.machine.states[self._task.state].timer_length)
        except AttributeError as e:
            message.time_remaining_until_next_task = -1

        self._publisher.publish(message)

    def task_timer_thread(self):
        """
        Thread to track the time left on a current time-based task.
        Publishes a task update message once per second with the time remaining
        until the next task.
        At the end of the timer, it moves the task monitor to the next task.
        """
        log = self.get_logger()
        loops = self._task.machine.states[self._task.state].timer_length

        # Record the current state
        curr_state = self._task.state

        # Publish a task update message once per second
        for i in range(int(loops)):
            self.publish_task_state_message()

            try:
                self._task.machine.states[curr_state].timer_length -= 1
            except AttributeError:
                # The current state was probably changed elsewhere
                break

            time.sleep(1)

        with self._timer_lock:
            self._timer_active = False

        # Make sure that the state has not changed during the timer delay
        if curr_state != self._task.state:
            log.warn("State change detecting during timer loop."
                     + " Next state will NOT be triggered via timer.")
            return

        # Advance to the next state
        with self._task_lock:
            self._task.trigger(self._task.state)

        # Update state tracking vars
        self._previous_step = self._current_step
        self._current_step = self._task.state

        self.publish_task_state_message()

    def monitor_keypress(self):
        log = self.get_logger()
        log.info(f"Starting keyboard monitor. Press the right arrow key to"
                 + " proceed to the next step. Press the left arrow key to"
                 + " go back to the previous step.")
        # Collect events until released
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def on_press(self, key):
        """
        Callback function for keypress events. If the right arrow is pressed,
        the task monitor advances to the next step.
        """
        log = self.get_logger()
        if key == keyboard.Key.right:
            with self._task_lock:
                try:
                    self._task.trigger(self._task.state)
                except AttributeError:
                    log.warn(f"Tried to trigger on invalid state: {self._task.state}")
                    return

                log.info(f"Proceeding to next step. Current step: {self._task.state}")

                # Update state tracking vars
                self._previous_step = self._current_step
                self._current_step = self._task.state

                self.publish_task_state_message()
        elif key == keyboard.Key.left:
            with self._task_lock:
                log.info(f"Proceeding to previous step")
                try:
                    self._task.machine.set_state(self._previous_step)
                except ValueError:
                    log.warn(f"Tried to set machine to invalid state: {self._previous_step}")
                    return

                # Update current step
                self._current_step = self._task.state

                # Find the index of the current step
                curr_step_index = self._task.steps.index({'name': self._current_step,
                                                          'ignore_invalid_triggers': None})

                # Lookup the new previous step
                prev_step_index = curr_step_index - 1
                if prev_step_index >= 0:
                    self._previous_step = self._task.steps[prev_step_index]['name']
                else:
                    self._previous_step = None

                log.info(f"Current step is now: {self._task.state}")
                log.info(f"Previous step is now: {self._previous_step}")

                self.publish_task_state_message()


def main():
    rclpy.init()

    task_monitor = TaskMonitor()

    keyboard_t = threading.Thread(target=task_monitor.monitor_keypress)
    keyboard_t.daemon = True
    keyboard_t.start()

    rclpy.spin(task_monitor)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    task_monitor.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
