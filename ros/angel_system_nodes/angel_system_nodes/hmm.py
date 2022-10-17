import json
import time
import uuid
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from angel_msgs.msg import (
    ActivityDetection,
    TaskUpdate,
    TaskItem,
    TaskGraph,
    TaskNode
)
from angel_msgs.srv import QueryTaskGraph
from angel_system.activity_hmm.core import ActivityHMMRos
from angel_utils.conversion import time_to_int


class HMMNode(Node):
    """
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

        # Tracks the previous step
        self._previous_step = None

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

    def det_callback(self, activity_msg: ActivityDetection):
        """
        Callback function for the activity detection subscriber topic.
        Adds the activity detection msg to the HMM and then publishes a new
        TaskUpdate message.
        """
        source_stamp_start_frame_sec = time_to_int(
            activity_msg.source_stamp_start_frame
        ) * 1e-9 # time_to_int returns ns
        source_stamp_end_frame_sec = time_to_int(
            activity_msg.source_stamp_end_frame
        ) * 1e-9 # time_to_int returns ns

        # Add activity classification to the HMM
        self._hmm.add_activity_classification(
            activity_msg.label_vec,
            activity_msg.conf_vec,
            source_stamp_start_frame_sec,
            source_stamp_end_frame_sec,
        )

        self.publish_task_state_message()

    def publish_task_state_message(self, activity=None):
        """
        Forms and sends a `angel_msgs/TaskUpdate` message to the
        TaskUpdates topic.
        """
        log = self.get_logger()

        message = TaskUpdate()

        current_step = self._hmm.get_current_state()
        log.info(f"Current step: {current_step}")

        # Populate message header
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "Task message"

        # Populate steps and current step
        message.steps = self._hmm.model.class_str
        message.current_step = current_step
        message.current_step_id = self._hmm.model.class_str.index(current_step)

        if self._previous_step is None:
            message.previous_step = "N/A"
        else:
            message.previous_step = self._previous_step

        # TODO: Populate task name and description
        # TODO: Do we need to fill in current/next activity?
        # TODO: Fill in time remaining?

        self._publisher.publish(message)

        # Update state tracking vars
        self._previous_step = current_step

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


def main():
    rclpy.init()

    hmm = HMMNode()

    rclpy.spin(hmm)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    hmm.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
