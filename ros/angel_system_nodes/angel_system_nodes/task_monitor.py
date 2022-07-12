import json
import threading
import time
import uuid
from typing import Dict

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from angel_msgs.msg import ActivityDetection, TaskUpdate, TaskItem, TaskGraph, TaskNode
from transitions import Machine
import transitions

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
            self._task_steps= json.load(f)

        # Create the list of steps
        self.steps = []
        self.uids = {}
        for key, value in self._task_steps.items():
            task_name = key.replace(' ', '_')
            self.steps.append({'name': task_name})
            self.uids[task_name] = str(uuid.uuid4())

        # Manually add the finish step
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
    """
    def __init__(self):
        super().__init__(self.__class__.__name__)

        self._det_topic = self.declare_parameter("det_topic", "ActivityDetections").get_parameter_value().string_value
        self._task_state_topic = self.declare_parameter("task_state_topic", "TaskUpdates").get_parameter_value().string_value
        self._task_steps = self.declare_parameter("task_steps", "default_task_label_config.json").get_parameter_value().string_value

        log = self.get_logger()

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

        self._task = CoffeeDemoTask(self._task_steps)

        self._task_graph_service = self.create_service(
            QueryTaskGraph,
            "query_task_graph",
            self.query_task_graph_callback
        )

        # Represents the current state of the task
        self._current_step = self._task.state
        self._previous_step = None

        # Represents the current action being performed
        self._current_activity = None
        self._next_activity = None

        # Tracks whether or not a timer is currently active
        self._timer_active = False
        self._timer_lock = threading.RLock()

        self.publish_task_state_message()


    def listener_callback(self, activity_msg):
        """
        Callback function for the activity detection subscriber topic.

        Upon receiving a new activity message, this function checks if that
        activity matches any of the defined activities for the current task,
        and attempts to advance the task state machine.

        A new task update message is published if the task's state changed.
        """
        log = self.get_logger()

        # See if any of the predicted activities are in the current task's
        # defined activities
        current_activity = None
        for a in activity_msg.label_vec:
            if a in self._task._task_steps.keys():
                # Label vector is sorted by probability so the first activity we find
                # that pertains to the current task is the most likely
                current_activity = a
                break

        if current_activity is None:
            # No activity matching current task... update the current activity and exit
            self._current_activity = activity_msg.label_vec[0]
            self.publish_task_state_message()
            return

        self._current_activity = current_activity

        # If we are currently in a timer state, exit early since we need to wait for
        # the timer to finish
        with self._timer_lock:
            if self._timer_active:
                return

        # Attempt to advance to the next step
        try:
            # Attempt to advance to the next state
            self._task.trigger(current_activity.replace(' ', '_'))
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

        task_g.task_nodes = []
        for t in self._task.steps:
            t_node = TaskNode()

            t_node.name = t['name']
            t_node.uid = self._task.uids[t_node.name]
            # TODO: Add other parameters

            task_g.task_nodes.append(t_node)
        log.info(f"Tasks: {task_g.task_nodes}")

        task_g.node_edges = []
        for tr in self._task.transitions:
            task_g.node_edges.append([i for i, x in enumerate(task_g.task_nodes) if x.name == tr['source']][0])
            task_g.node_edges.append([i for i, x in enumerate(task_g.task_nodes) if x.name == tr['dest']][0])
        log.info(f"Edges: {task_g.node_edges}")

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
        for idx, step in enumerate(self._task.steps):
            try:
                message.steps.append(step['name'].replace('_', ' '))
            except:
                message.steps.append(step.replace('_', ' '))

            # Set the current step index
            if self._current_step == step['name']:
                message.current_step_id = idx

        message.current_step = self._current_step.replace('_', ' ')

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

        # Publish a task update message once per second
        for i in range(int(loops)):
            self.publish_task_state_message()
            self._task.machine.states[self._task.state].timer_length -= 1

            time.sleep(1)

        with self._timer_lock:
            self._timer_active = False

        # Advance to the next state
        self._task.trigger(self._task.state)

        # Update state tracking vars
        self._previous_step = self._current_step
        self._current_step = self._task.state

        self.publish_task_state_message()


def main():
    rclpy.init()

    task_monitor = TaskMonitor()

    rclpy.spin(task_monitor)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    task_monitor.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()

