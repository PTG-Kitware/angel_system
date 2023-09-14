from collections import deque
from threading import Event, RLock, Thread
from typing import Callable
from typing import Deque
from typing import List
from typing import Tuple

from builtin_interfaces.msg import Time
import hydra
import numpy as np
from omegaconf import DictConfig
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import torch

from angel_system.data.recipe_task_graph import load_recipe
from angel_system.impls.detect_activities.detections_to_activities.utils import (
    obj_det2d_set_to_feature,
)

from angel_msgs.msg import (
    ObjectDetection2dSet,
    ActivityDetection,
)
from angel_utils import declare_and_get_parameters
from angel_utils.activity_classification import InputWindow, InputBuffer
from angel_utils.conversion import time_to_int


# Input ROS topic for object detections.
PARAM_DET_TOPIC = "det_topic"
# Output ROS topic for activity classifications.
PARAM_ACT_TOPIC = "act_topic"
# Filesystem path to the TCN model weights
PARAM_MODEL_WEIGHTS = "model_weights"
# Key string for the recipe activity label set used for this model.
PARAM_MODEL_LABELS_KEY = "model_labels_key"
# Version of the detections-to-descriptors algorithm the model is compatible
# with
PARAM_MODEL_DETS_CONV_VERSION = "model_dets_conv_version"
# Number of
# Runtime thread checkin heartbeat interval in seconds.
PARAM_RT_HEARTBEAT = "rt_thread_heartbeat"


def get_hydra_config(dir_path: str, cfg_name: str) -> DictConfig:
    @hydra.main(dir_path, cfg_name)
    def inner(_cfg):
        nonlocal cfg
        cfg = _cfg

    cfg: DictConfig
    inner()
    return cfg


class ActivityClassifierTCN(Node):
    """
    ROS node that publishes `ActivityDetection` messages using a classifier and
    `ObjectDetection2dSet` messages.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        log = self.get_logger()

        param_values = declare_and_get_parameters(
            self,
            [
                (PARAM_DET_TOPIC,),
                (PARAM_ACT_TOPIC,),
                (PARAM_MODEL_WEIGHTS,),
                (PARAM_MODEL_LABELS_KEY,),
                (PARAM_MODEL_DETS_CONV_VERSION, 2),
                (PARAM_RT_HEARTBEAT, 0.1),
            ],
        )
        self._det_topic = param_values[PARAM_DET_TOPIC]
        self._act_topic = param_values[PARAM_ACT_TOPIC]

        # TODO: Load in TCN classification model and weights
        # TODO: https://github.com/PTG-Kitware/TCN_HPL/blob/main/src/models/ptg_module.py
        torch.load(param_values[PARAM_MODEL_WEIGHTS], map_location="cpu")
        # Load labels list from configured activity_labels YAML file.
        _, activity_labels_cfg = load_recipe(param_values[PARAM_MODEL_LABELS_KEY])
        self._activity_label_to_id = {
            d['label']: d['id']
            for d in activity_labels_cfg
            # NOTE: current models are not known to encode the done activity label
            if d['label'].lower() != "done"
        }
        breakpoint()
        # Sequence of activity labels in ID order
        self._activity_label_vec = []
        # Feature version aligned with model current architecture
        self._feat_version = param_values[PARAM_MODEL_DETS_CONV_VERSION]

        # Input data buffer for temporal windowing.
        # Data should be tuple pairing a timestamp (ROS Time) of the source
        # image frame with the object detections descriptor vector.
        self._buffer: Deque[Tuple[Time, np.ndarray]] = deque()
        # Access lock for the buffer
        self._buffer_lock = RLock()

        # Create ROS subscribers and publishers.
        # These are being purposefully being allocated before the
        # runtime-thread allocation.
        self._det_subscriber = self.create_subscription(
            ObjectDetection2dSet, self._det_topic, self.det_callback, 1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._activity_publisher = self.create_publisher(
            ActivityDetection, self._act_topic, 1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Start windowed prediction runtime thread.
        log.info("Starting runtime thread...")
        # On/Off Switch for runtime loop
        self._rt_active = Event()
        self._rt_active.set()
        # seconds to occasionally time out of the wait condition for the loop
        # to check if it is supposed to still be alive.
        self._rt_active_heartbeat = param_values[PARAM_RT_HEARTBEAT]
        # Event to notify runtime it should try processing now.
        self._rt_awake_evt = Event()
        self._rt_thread = Thread(
            target=self.rt_loop, name="prediction_runtime"
        )
        self._rt_thread.daemon = True
        self._rt_thread.start()

    def det_callback(self, msg: ObjectDetection2dSet) -> None:
        """
        Callback function for `ObjectDetection2dSet` messages. Runs the classifier,
        creates an `ActivityDetection` message from the results the classifier,
        and publish the `ActivityDetection` message.
        """
        log = self.get_logger()

        feature_vec = obj_det2d_set_to_feature(
            msg.label_vec,
            msg.left,
            msg.right,
            msg.top,
            msg.bottom,
            msg.label_confidences,
            msg.descriptors,
            msg.obj_obj_contact_state,
            msg.obj_obj_contact_conf,
            msg.obj_hand_contact_state,
            msg.obj_hand_contact_conf,
            self._activity_label_to_id,
            version=self._feat_version,
        )
        if self.rt_alive():
            with self._buffer_lock:
                log.info(f"Queueing detections descriptor (ts={msg.header.stamp})")
                self._buffer.append((msg.source_stamp, feature_vec))

        # # Call activity classifier function. It is expected that this
        # # function will return a dictionary mapping activity labels to confidences.
        # conf = self.clf.predict_proba(feature_vec.reshape(1, -1)).ravel()
        # label_conf_dict = {self.act_str_list[i]: conf[i] for i in range(len(conf))}
        #
        # activity_det_msg = ActivityDetection()
        # activity_det_msg.header.stamp = self.get_clock().now().to_msg()
        # activity_det_msg.header.frame_id = "ActivityDetection"
        #
        # # Set the activity det start/end frame time to the previously recieved
        # # frame (start) and the current frame (end).
        # activity_det_msg.source_stamp_start_frame = self._source_stamp_start_frame
        # activity_det_msg.source_stamp_end_frame = msg.source_stamp
        #
        # activity_det_msg.label_vec = list(label_conf_dict.keys())
        # activity_det_msg.conf_vec = list(label_conf_dict.values())
        #
        # self._activity_publisher.publish(activity_det_msg)
        # log.debug("Publish activity detection msg")
        # log.debug(f"highest conf: {max(activity_det_msg.conf_vec)}")
        # log.debug(
        #     f"- msg start time: {time_to_int(activity_det_msg.source_stamp_start_frame)}"
        # )
        # log.debug(
        #     f"- msg end time  : {time_to_int(activity_det_msg.source_stamp_end_frame)}"
        # )

    def rt_alive(self) -> bool:
        """
        Check that the prediction runtime is still alive and raise an exception
        if it is not.
        """
        alive = self._rt_thread.is_alive()
        if not alive:
            self.get_logger().warn("Runtime thread no longer alive.")
            self._rt_thread.join()
        return alive

    def rt_stop(self) -> None:
        """
        Indicate that the runtime loop should cease.
        """
        self._rt_active.clear()

    def rt_loop(self):
        """
        Activity classification prediction runtime function.
        """
        log = self.get_logger()
        log.info("Runtime loop starting")

        while self._rt_active.wait(0):  # will quickly return false if cleared.
            # Get most recent window of inputs
            input_window: List[Tuple[Time, np.ndarray]] = []
            with self._buffer_lock:
                # TODO: Do we want just the last-X detection outputs, or
                #       relative to RGB frames?
                #       Remember: object detections are spase relative to the
                #       frame stream.
                #       The last-X image frames or seconds of object detections
                #       may be variable in size. Last-X detections results will
                #       be variable in time-window.
                #       Maybe reuse InputBuffer() if base on last-X frames
                ...

            earliest_time: Time
            # TODO: Some continue logic if input_window is not ripe to work on.

            # TODO: Remove content of buffer before the earliest edge of
            #       this window (left-side).

            act_msg: ActivityDetection = self._process_window(input_window)

            self._activity_publisher.publish(act_msg)

        log.info("Runtime function end.")

    def _process_window(self, input_window: List[Tuple[Time, np.ndarray]]) -> ActivityDetection:
        """
        Process an input window and output an activity classification message.
        """
        # TODO: Invoke model with descriptor inputs
        # TODO: What are the model outputs?

        activity_msg = ActivityDetection()
        activity_msg.header.frame_id = "Activity Classification"
        activity_msg.header.stamp = self.get_clock().now().to_msg()
        activity_msg.source_stamp_start_frame = input_window[0][0]
        activity_msg.source_stamp_end_frame = input_window[-1][0]
        activity_msg.label_vec
        return activity_msg

    def destroy_node(self):
        print("Shutting down runtime thread...")
        self._rt_active.clear()  # make RT active flag "False"
        self._rt_thread.join()
        print("Shutting down runtime thread... Done")
        super().destroy_node()


def main():
    rclpy.init()

    activity_classifier = ActivityClassifierTCN()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(activity_classifier)
    try:
        executor.spin()
    except KeyboardInterrupt:
        activity_classifier.rt_stop()
        activity_classifier.get_logger().info("Keyboard interrupt, shutting down.\n")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    activity_classifier.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
