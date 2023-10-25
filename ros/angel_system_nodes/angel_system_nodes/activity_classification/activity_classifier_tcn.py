"""
TCN config: https://github.com/PTG-Kitware/TCN_HPL/blob/c987b3d4f65ff7d4f9696333443ee138310893e0/configs/experiment/feat_v2.yaml
Use get_hydra_config to get cfg dict, use eval.py content as how-to-call example using
trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)
"""
import json
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import kwcoco
from builtin_interfaces.msg import Time
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import rclpy.logging
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import torch

from angel_system.tcn_hpl.predict import (
    load_module,
    ObjectDetectionsLTRB,
    objects_to_feats,
    predict,
    ResultsCollector,
)
from angel_system.utils.event import WaitAndClearEvent
from angel_system.utils.simple_timer import SimpleTimer

from angel_msgs.msg import (
    ObjectDetection2dSet,
    ActivityDetection,
)
from angel_utils import declare_and_get_parameters, RateTracker
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
# Width in pixels of the imagery that object detections were predicted from.
PARAM_IMAGE_PIX_WIDTH = "image_pix_width"
# Height in pixels of the imagery that object detections were predicted from.
PARAM_IMAGE_PIX_HEIGHT = "image_pix_height"
# Runtime thread checkin heartbeat interval in seconds.
PARAM_RT_HEARTBEAT = "rt_thread_heartbeat"
# Where we should output an MS-COCO file with our activity predictions in it
# per frame. NOTE: activity format is very custom, pending common utilities.
# If no value or an empty string is provided, we will not accumulate
# predictions. If a path is provided, we will accumulate and output at node
# closure.
PARAM_OUTPUT_COCO_FILEPATH = "output_predictions_kwcoco"
# Optional input COCO file of video frame object detections to be used as input
# for activity classification. This should not be used simultaneously when
# interfacing with ROS-based object detection input - behavior is undefined.
PARAM_INPUT_COCO_FILEPATH = "input_obj_det_kwcoco"


class NoActivityClassification(Exception):
    """
    Raised when the window processing function is unable to generate an
    activity classification for an input window.
    """


def obj_det_msg_to_max_lbl_conf(
    msg: ObjectDetection2dSet,
) -> Tuple[List[str], List[float]]:
    """
    Get out a tuple of the maximally confident class label and
    confidence value for each detection as a tuple of two lists for
    expansion into the `ObjectDetectionsLTRB` constructor
    """
    mat_shape = (msg.num_detections, len(msg.label_vec))
    conf_mat = np.asarray(msg.label_confidences).reshape(mat_shape)
    max_conf_idxs = conf_mat.argmax(axis=1)
    max_confs = conf_mat[np.arange(conf_mat.shape[0]), max_conf_idxs]
    max_labels = np.asarray(msg.label_vec)[max_conf_idxs]
    return max_labels, max_confs


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
                (PARAM_IMAGE_PIX_WIDTH,),
                (PARAM_IMAGE_PIX_HEIGHT,),
                (PARAM_RT_HEARTBEAT, 0.1),
                (PARAM_OUTPUT_COCO_FILEPATH, ""),
                (PARAM_INPUT_COCO_FILEPATH, ""),
            ],
        )
        self._img_ts_topic = param_values[PARAM_IMG_TS_TOPIC]
        self._det_topic = param_values[PARAM_DET_TOPIC]
        self._act_topic = param_values[PARAM_ACT_TOPIC]
        self._img_pix_width = param_values[PARAM_IMAGE_PIX_WIDTH]
        self._img_pix_height = param_values[PARAM_IMAGE_PIX_HEIGHT]

        # Load in TCN classification model and weights
        with SimpleTimer("Loading inference module", log.info):
            self._model_device = torch.device(param_values[PARAM_MODEL_DEVICE])
            self._model = load_module(
                param_values[PARAM_MODEL_WEIGHTS],
                param_values[PARAM_MODEL_MAPPING],
                self._model_device,
            ).eval()

        # Load labels list from configured activity_labels YAML file.
        with open(param_values[PARAM_MODEL_OD_MAPPING]) as infile:
            det_label_list = json.load(infile)
        self._det_label_to_id = {c: i for i, c in enumerate(det_label_list)}
        # Feature version aligned with model current architecture
        self._feat_version = param_values[PARAM_MODEL_DETS_CONV_VERSION]

        # Setup optional results output to a COCO file at end of runtime.
        tmp_str: str = param_values[PARAM_OUTPUT_COCO_FILEPATH]
        self._output_kwcoco_path: Optional[Path] = Path(tmp_str) if tmp_str else None
        self._results_collector: Optional[ResultsCollector] = None
        if self._output_kwcoco_path:
            log.info(
                f"Collecting predictions and outputting to: "
                f"{self._output_kwcoco_path}"
            )
            self._results_collector = ResultsCollector(
                self._output_kwcoco_path,
                {i: c for i, c in enumerate(self._model.classes)},
            )
            self._results_collector.set_video("ROS2 Stream")

        # Input data buffer for temporal windowing.
        # Data should be tuple pairing a timestamp (ROS Time) of the source
        # image frame with the object detections descriptor vector.
        # Buffer initialization must be before ROS callback and runtime-loop
        # initialization.
        self._window_size = param_values[PARAM_WINDOW_FRAME_SIZE]
        self._buffer = InputBuffer(
            0,  # Not using msgs with tolerance.
            self.get_logger,
        )
        self._buffer_max_size_nsec = int(
            param_values[PARAM_BUFFER_MAX_SIZE_SECONDS] * 1e9
        )

        # Time of the most recent window extracted from the buffer in the
        # runtime loop.
        # This is a little different
        self._window_extracted_time_ns: Optional[int] = None
        # Protected access across threads.
        self._window_extracted_time_ns_cond = Condition()

        # Track the time of the most recently processed window's leading frame
        # time. Assuming only used on the same thread (runtime loop thread).
        # Used by a `_window_criterion_new_leading_frame`.
        # Intentionally before runtime-loop initialization.
        self._window_processed_time_ns: Optional[int] = None

        # Create ROS subscribers and publishers.
        # These are being purposefully being allocated before the
        # runtime-thread allocation.
        # This is intentionally before runtime-loop initialization.
        self._img_ts_subscriber = self.create_subscription(
            Time,
            self._img_ts_topic,
            self.img_ts_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._det_subscriber = self.create_subscription(
            ObjectDetection2dSet,
            self._det_topic,
            self.det_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._activity_publisher = self.create_publisher(
            ActivityDetection,
            self._act_topic,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Rate tracker used in the window processing function.
        # This needs to be initialized before starting the runtime-loop which
        # calls that method.
        self._rate_tracker = RateTracker()

        # Start windowed prediction runtime thread.
        # On/Off Switch for runtime loop, initializing to "on" position.
        # Clear this event to deactivate the runtime loop.
        self._rt_active = Event()
        self._rt_active.set()
        # seconds to occasionally time out of the wait condition for the loop
        # to check if it is supposed to still be alive.
        self._rt_active_heartbeat = param_values[PARAM_RT_HEARTBEAT]
        # Condition that the runtime should perform processing
        self._rt_awake_evt = WaitAndClearEvent()
        self._rt_thread = Thread(target=self.rt_loop, name="prediction_runtime")
        self._rt_thread.daemon = True

        # Optionally initialize buffer-feeding from input COCO-file of object
        # detections.
        # This is intentionally after the runtime thread initialization.
        tmp_str = param_values[PARAM_INPUT_COCO_FILEPATH]
        input_coco_path: Optional[Path] = Path(tmp_str) if tmp_str else None
        # TODO: Variable to signal that processing of all file-loaded
        #       detections has completed.
        self._coco_complete_lock = Lock()
        self._coco_load_thread = None
        if input_coco_path is not None:
            self._coco_load_thread = Thread(
                target=self._thread_populate_from_coco,
                name="coco_loader",
                args=(input_coco_path,),
            )
            self._coco_load_thread.daemon = True

        # Start threads
        # Should be the last part of the constructor.
        log.info("Starting runtime thread...")
        self._rt_thread.start()
        if self._coco_load_thread:
            log.info("Starting COCO loading thread...")
            self._coco_load_thread.start()

    def _thread_populate_from_coco(self, input_coco_path: Path) -> None:
        """
        Function to populate the buffer from a loaded COCO dataset of object
        detections.
        """
        log = self.get_logger()
        with SimpleTimer("Loading COCO object detections...", log.info):
            with open(input_coco_path, "r") as infile:
                dset = kwcoco.CocoDataset(data=json.load(infile))

        # Only supporting processing of a single video's worth of detections.
        # We will be buffering into a window-based buffer, so we need to only
        # buffer one video's worth of detections at a time. Supporting only one
        # video's worth of inputs is the simplest to support initially.
        if len(dset.videos()) != 1:
            log.error(
                f"Input object detections COCO file did not have, or had more "
                f"than, one video's worth of detections. "
                f"Had: {len(dset.videos())}"
            )
            self._rt_active.clear()
            return

        # Store detection annotation category labels as a vector once.
        # * categories() will using ascending ID order when not given any
        #   explicit IDs to retrieve.
        obj_labels = dset.categories().name

        # Scan images by frame_index attribute
        # Type annotations for `dset.images().get` is not accurate.
        image_id_to_frame_index = dset.images().get("frame_index", keepid=True)
        for image_id, frame_index in sorted(
            image_id_to_frame_index.items(), key=lambda v: v[1]
        ):
            # Arbitrary time for alignment in windowing
            image_ts = Time(sec=0, nanosec=frame_index)
            image_ts_ns = time_to_int(image_ts)

            # Detection set message
            det_msg = ObjectDetection2dSet()
            det_msg.header.stamp = image_ts
            det_msg.source_stamp = image_ts
            det_msg.label_vec = obj_labels

            image_annots = dset.annots(dset.index.gid_to_aids[image_id])  # type: ignore
            det_msg.num_detections = n_dets = len(image_annots)

            if n_dets > 0:
                det_bbox_tlbr = image_annots.boxes.to_tlbr().data.T
                det_msg.top.extend(det_bbox_tlbr[0])
                det_msg.left.extend(det_bbox_tlbr[1])
                det_msg.bottom.extend(det_bbox_tlbr[2])
                det_msg.right.extend(det_bbox_tlbr[3])

                # Creates [n_det, n_label] matrix, which we assign to and then
                # ravel into the message slot.
                conf_mat = np.zeros((n_dets, len(obj_labels)), dtype=np.float64)
                conf_mat[
                    np.arange(n_dets), image_annots.get("category_id")
                ] = image_annots.get("confidence")
                det_msg.label_confidences.extend(conf_mat.ravel())

            # Calling the image callback last since image frames define the
            # window bounds, creating a new window for processing.
            log.info(f"Queuing from COCO: n_dets={n_dets}, ts_index={image_ts}")
            self.det_callback(det_msg)
            self.img_ts_callback(image_ts)

            # Wait until `image_ts` was considered in the runtime loop before
            # proceeding into the next iteration.
            with self._window_extracted_time_ns_cond:
                self._window_extracted_time_ns_cond.wait_for(
                    lambda: (
                        self._window_extracted_time_ns is not None
                        and self._window_extracted_time_ns >= image_ts_ns
                    ),
                )

        log.info("Completed COCO file object yielding")
        rclpy.shutdown()

    def img_ts_callback(self, msg: Time) -> None:
        """
        Capture a detection source image timestamp message.
        """
        if self.rt_alive() and self._buffer.queue_image(None, msg):
            self.get_logger().info(f"Queueing image TS {msg}")
            # Let the runtime know we've queued something.
            self._rt_awake_evt.set()

    def det_callback(self, msg: ObjectDetection2dSet) -> None:
        """
        Callback function for `ObjectDetection2dSet` messages. Runs the classifier,
        creates an `ActivityDetection` message from the results the classifier,
        and publish the `ActivityDetection` message.
        """
        if self.rt_alive() and self._buffer.queue_object_detections(msg):
            self.get_logger().info(
                f"Queueing object detections (ts={msg.header.stamp})"
            )
            # Let the runtime know we've queued something.
            self._rt_awake_evt.set()

    def rt_alive(self) -> bool:
        """
        Check that the prediction runtime is still alive and return false if it
        is not.
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
        self._rt_awake_evt.set()  # intentionally second

    def _rt_keep_looping(self) -> bool:
        """
        Indicator that the runtime-loop should keep looping.

        The runtime should still be active when:
        * The `_rt_active` event is still set (on/off switch)
        * The input-file-mode EOF has not been reached (if in that mode).
        """
        # This will quickly return False if it has been `.clear()`ed
        rt_active = self._rt_active.wait(0)
        # TODO: add has-finished-processing-file-input check.
        return rt_active

    def _window_criterion_correct_size(self, window: InputBuffer) -> bool:
        window_ok = len(window) == self._window_size
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
            self.get_logger().warn("Window has no content, no leading frame to check.")
            return False
        cur_leading_time_ns = time_to_int(window.frames[-1][0])
        prev_leading_time_ns = self._window_processed_time_ns
        if prev_leading_time_ns is not None:
            window_ok = prev_leading_time_ns < cur_leading_time_ns
            if not window_ok:
                # current window is earlier/same lead as before, so not a good
                # window
                self.get_logger().warn("Input window has duplicate leading frame time.")
                return False
            # Window is OK, save new latest leading frame time below.
        # Else: This is the first window with non-zero frames.
        return True

    def _window_criterion_coco_input_mode(self, window: InputWindow) -> bool:
        """
        When input is coming from COCO file, we expect that all input window
        slots are filled and there are no None values.
        Basically only need to check the latest time column as inputs are
        lock-step buffered.
        """
        if len(window) == 0:
            return True  # nothing to check, defer to other checks
        if None in window.frames or None in window.obj_dets:
            self.get_logger().warn("Inputs not jointly filled yet.")
            return False
        return True

    def rt_loop(self):
        """
        Activity classification prediction runtime function.
        """
        log = self.get_logger()
        log.debug("Runtime loop starting")

        # These criterion predicates must all return true for us to proceed
        # with processing activity classification for a window.
        # Function order should consider short-circuiting rules.
        window_processing_criterion_fn_list: List[Callable[[InputWindow], bool]] = [
            self._window_criterion_correct_size,
            self._window_criterion_new_leading_frame,
        ]

        # If we're in COCO input mode, add the associated criterion
        if self._coco_load_thread is not None:
            window_processing_criterion_fn_list.append(
                self._window_criterion_coco_input_mode
            )

        while self._rt_keep_looping():
            if self._rt_awake_evt.wait_and_clear(self._rt_active_heartbeat):
                # We want to fire off a prediction if the current window of
                # data is "valid" based on our registered criterion.
                window = self._buffer.get_window(self._window_size)

                # Time of the leading frame of the extracted window.
                window_time_ns: Optional[int] = None
                if window.frames:  # maybe there are no frames yet in there.
                    window_time_ns = time_to_int(window.frames[-1][0])

                with self._window_extracted_time_ns_cond:
                    self._window_extracted_time_ns = window_time_ns
                    self._window_extracted_time_ns_cond.notify_all()

                if all(fn(window) for fn in window_processing_criterion_fn_list):
                    # After validating a window, and before processing it, clear
                    # out older data at and before the first item in the window.
                    self._buffer.clear_before(time_to_int(window.frames[1][0]))

                    try:
                        log.info(
                            f"Processing window with leading image TS: "
                            f"{window.frames[-1][0]}"
                        )
                        act_msg = self._process_window(window)
                        self._collect_results(act_msg)
                        self._activity_publisher.publish(act_msg)
                    except NoActivityClassification:
                        # No ramifications, but don't publish activity message.
                        log.warn(
                            "Runtime loop window processing function did "
                            "not yield an activity classification for "
                            "publishing."
                        )

                    # This window has completed processing - record its leading
                    # timestamp now.
                    self._window_processed_time_ns = window_time_ns
                else:
                    log.debug("Runtime loop window criterion check(s) failed.")
                    with self._buffer:
                        # Clear to at least our maximum buffer size even if we
                        # didn't process anything (if there is anything *in*
                        # our buffer). It's OK if the buffer's latest time has
                        # progress since the start of this loop: that's just
                        # the state it's in now.
                        try:
                            self._buffer.clear_before(
                                time_to_int(self._buffer.latest_time())
                                - self._buffer_max_size_nsec
                            )
                        except RuntimeError:
                            # Nothing in the buffer, nothing to clear.
                            pass
            else:
                log.debug(
                    "Runtime loop heartbeat timeout: checking alive status.",
                    throttle_duration_sec=1,
                )

        log.info("Runtime function end.")

    def _process_window(self, window: InputWindow) -> ActivityDetection:
        """
        Process an input window and output an activity classification message.

        :raises NoActivityClassification: No activity classification could be
            determined for this input window.
        """
        log = self.get_logger()

        # TCN wants to know the label and confidence for the maximally
        # confident class only. Input object detection messages
        frame_object_detections = [
            ObjectDetectionsLTRB(
                det_msg.left,
                det_msg.top,
                det_msg.right,
                det_msg.bottom,
                *obj_det_msg_to_max_lbl_conf(det_msg),
            )
            if det_msg is not None
            else None
            for det_msg in window.obj_dets
        ]
        log.debug(
            f"Window vector presence: "
            f"{[(v is not None) for v in frame_object_detections]}"
        )

        try:
            feats, mask = objects_to_feats(
                frame_object_detections,
                self._det_label_to_id,
                self._feat_version,
                self._img_pix_width,
                self._img_pix_height,
            )
        except ValueError:
            # feature detections were all None
            raise NoActivityClassification()
        feats = feats.to(self._model_device)
        mask = mask.to(self._model_device)

        proba = predict(self._model, feats, mask).cpu()
        pred = torch.argmax(proba)

        # Prepare output message
        activity_msg = ActivityDetection()
        activity_msg.header.frame_id = "Activity Classification"
        activity_msg.header.stamp = self.get_clock().now().to_msg()
        activity_msg.source_stamp_start_frame = window.frames[0][0]
        activity_msg.source_stamp_end_frame = window.frames[-1][0]
        activity_msg.label_vec = self._model.classes
        activity_msg.conf_vec = proba.tolist()

        log.info(
            f"Activity classification -- "
            f"{activity_msg.label_vec[pred]} @ {activity_msg.conf_vec[pred]} "
            f"(time: {time_to_int(activity_msg.source_stamp_start_frame)} - "
            f"{time_to_int(activity_msg.source_stamp_end_frame)})"
        )

        self._rate_tracker.tick()
        log.info(
            f"Activity classification rate (hz: "
            f"{self._rate_tracker.get_rate_avg()})",
        )

        return activity_msg

    def _collect_results(self, msg: ActivityDetection):
        """
        Collect into our ResultsCollector instance from the produced activity
        classification message if we were initialized to do that.

        This method does nothing if this node has not been initialized to
        collect results.

        :param msg: ROS2 activity classification message that would be output.
        """
        rc = self._results_collector
        if rc is not None:
            # Assigning result to the "id" if the last frame in the window.
            frame_index = time_to_int(msg.source_stamp_end_frame)
            pred_cls_idx = int(np.argmax(msg.conf_vec))
            rc.collect(
                frame_index=frame_index,
                name=f"ros-frame-nsec-{frame_index}",
                activity_pred=pred_cls_idx,
                activity_conf_vec=list(msg.conf_vec),
            )

    def _save_results(self):
        """
        Save results if we have been initialized to do that.

        This method does nothing if this node has not been initialized to
        collect results.
        """
        rc = self._results_collector
        if rc is not None:
            self.get_logger().info(
                f"Writing classification results to: {self._output_kwcoco_path}"
            )
            self._results_collector.write_file()

    def destroy_node(self):
        print("Shutting down runtime thread...")
        self._rt_active.clear()  # make RT active flag "False"
        self._rt_thread.join()
        print("Shutting down runtime thread... Done")
        self._save_results()
        super().destroy_node()


def main():
    rclpy.init()
    log = rclpy.logging.get_logger("main")

    activity_classifier = ActivityClassifierTCN()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(activity_classifier)
    try:
        executor.spin()
    except KeyboardInterrupt:
        log.info("Keyboard interrupt, shutting down.\n")
        rclpy.shutdown()
    finally:
        log.info("Stopping node runtime")
        activity_classifier.rt_stop()

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        activity_classifier.destroy_node()


if __name__ == "__main__":
    main()
