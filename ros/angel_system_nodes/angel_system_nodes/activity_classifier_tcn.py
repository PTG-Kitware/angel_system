"""
TCN config: https://github.com/PTG-Kitware/TCN_HPL/blob/c987b3d4f65ff7d4f9696333443ee138310893e0/configs/experiment/feat_v2.yaml
Use get_hydra_config to get cfg dict, use eval.py content as how-to-call example using
trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)
"""

import json
import os
from threading import Event, Thread
from typing import Callable
from typing import List
from typing import Optional

from builtin_interfaces.msg import Time
import numpy as np
import numpy.typing as npt
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import torch

from tcn_hpl.models.ptg_module import PTGLitModule

from angel_system.impls.detect_activities.detections_to_activities.utils import (
    obj_det2d_set_to_feature,
)
from angel_system.utils.simple_timer import SimpleTimer

from angel_msgs.msg import (
    ObjectDetection2dSet,
    ActivityDetection,
)
from angel_utils import declare_and_get_parameters
from angel_utils.activity_classification import InputWindow, InputBuffer
from angel_utils.conversion import time_to_int


# Input ROS topic for RGB Image Timestamps
PARAM_IMG_TS_TOPIC = "image_ts_topic"
# Input ROS topic for object detections.
PARAM_DET_TOPIC = "det_topic"
# Output ROS topic for activity classifications.
PARAM_ACT_TOPIC = "act_topic"
# Filesystem path to the TCN model weights
PARAM_MODEL_WEIGHTS = "model_weights"
# Filesystem path to the class mapping file.
PARAM_MODEL_MAPPING = "model_mapping"
# Filesystem path to the input object detection label mapping.
# This is expected to be a JSON file containing a list of strings.
PARAM_MODEL_OD_MAPPING = "model_det_label_mapping"
# Device the model should be loaded onto. "cuda" and "cpu" are
PARAM_MODEL_DEVICE = "model_device"
# Version of the detections-to-descriptors algorithm the model is expecting as
# input.
PARAM_MODEL_DETS_CONV_VERSION = "model_dets_conv_version"
# Number of (image) frames to consider as the "window" when collating
# correlated data.
PARAM_WINDOW_FRAME_SIZE = "window_size"
# Maximum amount of data we will buffer in seconds.
PARAM_BUFFER_MAX_SIZE_SECONDS = "buffer_max_size_seconds"
# Runtime thread checkin heartbeat interval in seconds.
PARAM_RT_HEARTBEAT = "rt_thread_heartbeat"


class NoActivityClassification(Exception):
    """
    Raised when the window processing function is unable to generate an
    activity classification for an input window.
    """


class Event2(Event):
    """
    Simple subclass that adds a wait-and-clear method to simultaneously wait
    for the lock and clear the set flag upon successful waiting.
    """

    def wait_and_clear(self, timeout=None):
        """Block until the internal flag is true, then clear the flag if it was
        set.

        If the internal flag is true on entry, return immediately. Otherwise,
        block until another thread calls set() to set the flag to true, or until
        the optional timeout occurs.

        When the timeout argument is present and not None, it should be a
        floating point number specifying a timeout for the operation in seconds
        (or fractions thereof).

        This method returns the internal flag on exit, so it will always return
        True except if a timeout is given and the operation times out.

        """
        with self._cond:
            signaled = self._flag
            if not signaled:
                signaled = self._cond.wait(timeout)
            self._flag = False
            return signaled


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
                (PARAM_IMG_TS_TOPIC,),
                (PARAM_DET_TOPIC,),
                (PARAM_ACT_TOPIC,),
                (PARAM_MODEL_WEIGHTS,),
                (PARAM_MODEL_MAPPING,),
                (PARAM_MODEL_OD_MAPPING,),
                (PARAM_MODEL_DEVICE, "cuda"),
                (PARAM_MODEL_DETS_CONV_VERSION, 2),
                (PARAM_WINDOW_FRAME_SIZE,),
                (PARAM_BUFFER_MAX_SIZE_SECONDS,),
                (PARAM_RT_HEARTBEAT, 0.1),
            ],
        )
        self._img_ts_topic = param_values[PARAM_IMG_TS_TOPIC]
        self._det_topic = param_values[PARAM_DET_TOPIC]
        self._act_topic = param_values[PARAM_ACT_TOPIC]

        # Load in TCN classification model and weights
        with SimpleTimer("Loading inference module", log.info):
            mapping_file_dir = os.path.abspath(os.path.dirname(param_values[PARAM_MODEL_MAPPING]))
            mapping_file_name = os.path.basename(param_values[PARAM_MODEL_MAPPING])
            self._model_device = torch.device(param_values[PARAM_MODEL_DEVICE])
            self._model = PTGLitModule.load_from_checkpoint(
                param_values[PARAM_MODEL_WEIGHTS],
                map_location=self._model_device,
                # HParam overrides
                data_dir=mapping_file_dir,
                mapping_file_name=mapping_file_name,
            )
            self._model = self._model.eval()

        # Load labels list from configured activity_labels YAML file.
        with open(param_values[PARAM_MODEL_OD_MAPPING]) as infile:
            det_label_list = json.load(infile)
        self._det_label_to_id = {
            c: i
            for i, c in enumerate(det_label_list)
        }
        # Feature version aligned with model current architecture
        self._feat_version = param_values[PARAM_MODEL_DETS_CONV_VERSION]

        # Input data buffer for temporal windowing.
        # Data should be tuple pairing a timestamp (ROS Time) of the source
        # image frame with the object detections descriptor vector.
        self._window_size = param_values[PARAM_WINDOW_FRAME_SIZE]
        self._buffer = InputBuffer(
            0,  # Not using msgs with tolerance.
            self.get_logger,
        )
        self._buffer_max_size_nsec = int(param_values[PARAM_BUFFER_MAX_SIZE_SECONDS] * 1e9)

        # Used by a _window_criterion_new_leading_frame to track previous
        # window's leading frame time.
        self._prev_leading_time_ns = None

        # Create ROS subscribers and publishers.
        # These are being purposefully being allocated before the
        # runtime-thread allocation.
        self._img_ts_subscriber = self.create_subscription(
            Time, self._img_ts_topic, self.img_ts_callback, 1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
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
        # Condition that the runtime should perform processing
        self._rt_awake_evt = Event2()
        self._rt_thread = Thread(
            target=self.rt_loop, name="prediction_runtime"
        )
        self._rt_thread.daemon = True
        self._rt_thread.start()

    def img_ts_callback(self, msg: Time) -> None:
        """
        Capture a detection source image timestamp message.
        """
        if self.rt_alive() and self._buffer.queue_image(None, msg):
            self.get_logger().debug(f"Queueing image TS {msg}")
            # Inform the runtime that it should process a cycle.
            self._rt_awake_evt.set()

    def det_callback(self, msg: ObjectDetection2dSet) -> None:
        """
        Callback function for `ObjectDetection2dSet` messages. Runs the classifier,
        creates an `ActivityDetection` message from the results the classifier,
        and publish the `ActivityDetection` message.
        """
        if self.rt_alive() and self._buffer.queue_object_detections(msg):
            self.get_logger().debug(f"Queueing object detections (ts={msg.header.stamp})")

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

    def _window_criterion_correct_size(self, window: InputBuffer) -> bool:
        window_ok = (len(window) == self._window_size)
        if not window_ok:
            self.get_logger().warn(
                f"Window is not the appropriate size "
                f"(actual:{len(window)} != {self._window_size}:expected)"
            )
        return window_ok

    def _window_criterion_new_leading_frame(self, window: InputWindow) -> bool:
        """
        The new window's leading frame should be beyond a previous window's
        leading frame.
        """
        if len(window) == 0:
            self.get_logger().warn("Window has no content, no leading frame "
                                   "to check.")
            return False
        cur_leading_time_ns = time_to_int(window.frames[-1][0])
        prev_leading_time_ns = self._prev_leading_time_ns
        if prev_leading_time_ns is not None:
            window_ok = prev_leading_time_ns < cur_leading_time_ns
            if not window_ok:
                # current window is earlier/same lead as before, so not a good
                # window
                self.get_logger().warn("Input window has duplicate leading "
                                       "frame time.")
                return False
            # Window is OK, save new latest leading frame time below.
        # Else:This is the first window with non-zero frames. The first history
        # will be recorded below.
        self._prev_leading_time_ns = cur_leading_time_ns
        return True

    def rt_loop(self):
        """
        Activity classification prediction runtime function.
        """
        log = self.get_logger()
        log.info("Runtime loop starting")

        # These criterion predicates must all return true for us to proceed
        # with processing activity classification for a window.
        # Function order should consider short-circuiting rules.
        window_processing_criterion_fn_list: List[Callable[[InputWindow], bool]] = [
            self._window_criterion_correct_size,
            self._window_criterion_new_leading_frame,
        ]

        while self._rt_active.wait(0):  # will quickly return false if cleared.
            if self._rt_awake_evt.wait_and_clear(self._rt_active_heartbeat):
                log.info("Runtime loop awakened")

                # We want to fire off a prediction if the current window of
                # data is "valid" based on our registered criterion.
                window = self._buffer.get_window(self._window_size)
                if all(fn(window) for fn in window_processing_criterion_fn_list):
                    log.info("Runtime loop starting processing block")
                    # After validating a window, and before processing it, clear
                    # out older data from the buffer based on the timestamp of
                    # the first item in the current window.
                    clear_before_time = window.frames[1][0]
                    clear_before_ns = time_to_int(clear_before_time)
                    self._buffer.clear_before(clear_before_ns)

                    try:
                        act_msg = self._process_window(window)
                        self._activity_publisher.publish(act_msg)
                    except NoActivityClassification:
                        # No ramifications, but don't publish activity message.
                        log.warn("Runtime loop window processing function did "
                                 "not yield an activity classification for "
                                 "publishing.")
                else:
                    log.info("Runtime loop window criterion check(s) failed.")
                    with self._buffer:
                        # Clear to at least our maximum buffer size even if we
                        # didn't process anything (if there is anything *in*
                        # our buffer).
                        try:
                            clear_before_ns = (
                                time_to_int(self._buffer.latest_time())
                                - self._buffer_max_size_nsec
                            )
                            self._buffer.clear_before(clear_before_ns)
                        except RuntimeError:
                            # Nothing in the buffer, nothing to clear.
                            pass
            else:
                log.info("Runtime loop heartbeat timeout: checking alive "
                         "status.",
                         throttle_duration_sec=1)

        log.info("Runtime function end.")

    def _process_window(self, window: InputWindow) -> ActivityDetection:
        """
        Process an input window and output an activity classification message.

        :raises NoActivityClassification: No activity classification could be
            determined for this input window.
        """
        log = self.get_logger()

        # Construct feature vector for each object detection aligned with a
        # frame. Where there is a frame and *no* object detections, create a
        # 0-vector of equivalent dimensionality.
        obj_det_vec_list: List[Optional[npt.NDArray]] = [None] * len(window.obj_dets)
        obj_det_vec_ndim = None
        obj_det_vec_dtype = None
        with SimpleTimer("[_process_window] Convert detections to vectors", log.debug):
            for i, obj_det_msg in enumerate(window.obj_dets):
                if obj_det_msg is not None:
                    obj_det_vec_list[i] = obj_det2d_set_to_feature(
                        obj_det_msg.label_vec,
                        obj_det_msg.left,
                        obj_det_msg.right,
                        obj_det_msg.top,
                        obj_det_msg.bottom,
                        obj_det_msg.label_confidences,
                        obj_det_msg.descriptors,
                        obj_det_msg.obj_obj_contact_state,
                        obj_det_msg.obj_obj_contact_conf,
                        obj_det_msg.obj_hand_contact_state,
                        obj_det_msg.obj_hand_contact_conf,
                        self._det_label_to_id,
                        version=self._feat_version,
                    ).astype(np.float32)
                    obj_det_vec_ndim = obj_det_vec_list[i].shape
                    obj_det_vec_dtype = obj_det_vec_list[i].dtype
            # Second pass, create zero-vectors
            if obj_det_vec_ndim is None:
                log.warn("[_process_window] No object detection messages in "
                         "input window. Continuing.")
                raise NoActivityClassification()
            z_vec = np.zeros(obj_det_vec_ndim, obj_det_vec_dtype)
            for i in range(len(obj_det_vec_list)):
                if obj_det_vec_list[i] is None:
                    obj_det_vec_list[i] = z_vec
            obj_det_vec_t = torch.tensor(obj_det_vec_list).T.to(self._model_device)

        # Hannah said to look at
        #   https://github.com/PTG-Kitware/TCN_HPL/blob/main/tcn_hpl/data/components/PTG_dataset.py#L46
        # ¯\_(ツ)_/¯
        mask_t = torch.ones(obj_det_vec_t.shape[1]).to(self._model_device)

        with SimpleTimer("[_process_window] Predicting activity proba", log.debug):
            # Invoke model with descriptor inputs
            with torch.no_grad():
                logits = self._model(obj_det_vec_t, mask_t[None, :])
            # Logits access mirrors model step function argmax access here:
            #   tcn_hpl.models.ptg_module --> PTGLitModule.model_step
            # ¯\_(ツ)_/¯
            pred = torch.argmax(logits[-1, :, :, -1], dim=1)[0].cpu()
            proba: torch.Tensor = torch.softmax(logits[-1, :, :, -1], dim=1)[0].cpu()

        with SimpleTimer("[_process_window] Constructing output msg", log.debug):
            activity_msg = ActivityDetection()
            activity_msg.header.frame_id = "Activity Classification"
            activity_msg.header.stamp = self.get_clock().now().to_msg()
            activity_msg.source_stamp_start_frame = window.frames[0][0]
            activity_msg.source_stamp_end_frame = window.frames[-1][0]
            activity_msg.label_vec = self._model.classes
            activity_msg.conf_vec = proba.tolist()

        log.info(f"Activity classification -- "
                 f"{activity_msg.label_vec[pred]} @ {activity_msg.conf_vec[pred]} "
                 f"(time: {activity_msg.source_stamp_end_frame} - "
                 f"{activity_msg.source_stamp_end_frame})")

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
