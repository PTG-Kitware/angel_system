import time
from typing import Dict

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from angel_msgs.msg import ActivityDetection, TaskUpdate, TaskItem
from transitions import Machine, State
import transitions


class Task():
    """
    Representation of a task defined by its steps and transistions between them.
    """
    def __init__(self):
        self.name = 'Making Tea'

        self.items = {'water bottle': 1, 'tea bag': 1, 'cup': 1}

        self.description = ('Open the water bottle and pour the water into a tea cup.' +
                            ' Place the tea bag in the cup.' +
                            ' Wait 20 seconds while the tea bag steeps, then drink and enjoy!')

        self.steps = [{'name': 'pour_water_into_cup'},
                      {'name': 'place_tea_bag_into_cup'},
                      {'name': 'steep_for_20_seconds'},
                      {'name': 'drink_tea'},
                      {'name': 'enjoy'},
                     ]

        self.transitions = [
            { 'trigger': 'open_bottle', 'source': 'pour_water_into_cup', 'dest': 'place_tea_bag_into_cup' },
            { 'trigger': 'make_tea', 'source': 'place_tea_bag_into_cup', 'dest': 'steep_for_20_seconds' },
            { 'trigger': 'drink_tea', 'source': 'drink_tea', 'dest': 'enjoy!' },
        ]

        self.time_remaining_until_next_step = -1

        self.machine = Machine(model=self, states=self.steps,
                               transitions=self.transitions, initial='pour_water_into_cup')

        self.machine.states['pour_water_into_cup'].timer = 20.0


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

        self._task = Task()

        # Represents the current state of the task
        self._current_step = self._task.state
        self._previous_step = None

        # Represents the current action being performed
        self._current_activity = None
        self._next_activity = None

        self._activity_action_dict = {
            'opening bottle': self._task.open_bottle,
            'making tea': self._task.make_tea,
            'drinking': self._task.drink_tea,
        }

        #print("dir:", dir(self._task))
        print("task state", self._task.machine.states[self._task.state].timer)
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
            if a in self._activity_action_dict.keys():
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

        # Attempt to advance to the next step
        try:
            self._activity_action_dict[current_activity]()
            log.info(f"Proceeding to next step. Current step: {self._task.state}")

            # Update state tracking vars
            self._previous_step = self._current_step
            self._current_step = self._task.state

        except transitions.core.MachineError as e:
            print("error!", e)
            pass

        self.publish_task_state_message()

        # Check to see if this new state has a timer associated with it
        # TODO
        try:
            print("task state", self._task.machine.states[self._task.state].timer)
            if self._task.machine.states[self._task.state].timer:
                # Spawn a thread to track the timer and publish state messages
        except transitions.core.MachineError as e:

    def publish_task_state_message(self, activity=None):
        """
        Forms and sends a `angel_msgs/TaskUpdate` message to the TaskUpdates topic.
        """
        log = self.get_logger()

        message = TaskUpdate()

        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "Task message"

        message.task_name = self._task.name
        message.task_description = self._task.description

        # Populate task items list
        for i, q in self._task.items.items():
            item = TaskItem()
            item.item_name = i
            item.quantity = q
            message.task_items.append(item)

        for step in self._task.steps:
            try:
                message.steps.append(step['name'])
            except:
                message.steps.append(step)
        #message.steps = self._task.steps
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

        self._publisher.publish(message)


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

