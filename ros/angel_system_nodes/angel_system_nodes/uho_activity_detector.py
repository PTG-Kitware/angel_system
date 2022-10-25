from builtin_interfaces.msg import Time
from collections import deque
from cv_bridge import CvBridge
import cv2
from dataclasses import dataclass, field
import itertools
import numpy as np
import numpy.typing as npt
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from threading import Event, RLock, Thread
from typing import Any
from typing import Callable
from typing import Deque
from typing import List
from typing import Optional
from typing import Tuple

from angel_msgs.msg import (
    ActivityDetection,
    HandJointPosesUpdate,
    ObjectDetection2dSet
)

from angel_system.utils.activity_classification import gt_predict
from angel_system.uho.predict import get_uho_classifier, predict
from angel_system.uho.deprecated_src.data_helper import create_batch
from angel_system.utils.matching import descending_match_with_tolerance
from angel_system.utils.simple_timer import SimpleTimer
from angel_utils.conversion import get_hand_pose_from_msg
from angel_utils.conversion import time_to_int


BRIDGE = CvBridge()


@dataclass
class InputWindow:
    """
    Structure encapsulating a window of aligned data.

    It should be the case that all lists contained in this window are of the
    same length. The `frames` field should always be densely packed, but the
    other fields may have None values, which indicates that no message of that
    field's associated type was matched to the index's corresponding frame.
    """
    # Buffer of RGB image matrices and the associated timestamp.
    # Set at construction time with the known window of frames.
    frames: List[Tuple[Time, npt.NDArray]]
    # Buffer of left-hand pose messages
    hand_pose_left: List[Optional[HandJointPosesUpdate]]
    # Buffer of right-hand pose messages
    hand_pose_right: List[Optional[HandJointPosesUpdate]]
    # Buffer of object detection predictions
    obj_dets: List[Optional[ObjectDetection2dSet]]

    def __len__(self):
        return len(self.frames)


@dataclass
class InputBuffer:
    """
    Protected container for buffering input data.

    Frames are the primary index of this buffer to which everything else needs
    to associate to.

    Data queueing methods enforce that the new message is temporally *after*
    the latest message of that type. If this is not the case, i.e. we received
    an out-of-order message, we do not queue the message.

    Hand pose sensor outputs are known to be output asynchronously and at a
    different rate than the images. Thus, we cannot presume there will be any
    hand messages that directly align with an image. To associate a message
    with the nearest image, we require a tolerance such that messages closer
    than this value to an image may be considered "associated" with the image.

    Object detection outputs are known to correlate strictly with an image
    frame via the timestamp value.

    Contained deques are in ascending time order (later indices are farther
    ahead in time).

    NOTE: `__post_init__` is a thing if we need it.
    """
    # Tolerance in nanoseconds for associating hand-pose messages to a frame.
    hand_msg_tolerance_nsec: int

    # Function to get the logger from the parent ROS2 node.
    get_logger_fn: Callable[[], Any]  # don't know where to get the type for this...

    # Buffer of RGB image matrices and the associated timestamp
    frames: Deque[Tuple[Time, npt.NDArray]] = field(default_factory=deque,
                                                    init=False, repr=False)
    # Buffer of left-hand pose messages
    hand_pose_left: Deque[HandJointPosesUpdate] = field(default_factory=deque,
                                                        init=False, repr=False)
    # Buffer of right-hand pose messages
    hand_pose_right: Deque[HandJointPosesUpdate] = field(default_factory=deque,
                                                         init=False, repr=False)
    # Buffer of object detection predictions
    obj_dets: Deque[ObjectDetection2dSet] = field(default_factory=deque,
                                                  init=False, repr=False)

    __state_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __enter__(self):
        """
        For when you want to call multiple things on the buffer in the same
        locking context.
        """
        # Same as RLock.__enter__
        self.__state_lock.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Same as RLock.__exit__
        self.__state_lock.release()

    def queue_image(self, msg: Image) -> bool:
        """
        Queue up a new image frame.
        :returns: True if the image was queued, otherwise false because it was
            not newer than the current latest frame.
        """
        # Convert ROS img msg to CV2 image and add it to the frame stack
        rgb_image = BRIDGE.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        rgb_image_np = np.asarray(rgb_image)
        with self.__state_lock:
            # before the current lead frame?
            if (
                self.frames and
                time_to_int(msg.header.stamp) <= time_to_int(self.frames[-1][0])
            ):
                self.get_logger_fn().warn(
                    f"Input image frame was NOT after the previous latest: "
                    f"(prev) {self.frames[-1][0]} !< {msg.header.stamp} (new)"
                )
                return False
            self.frames.append((msg.header.stamp, rgb_image_np))
            return True

    def queue_hand_pose(self, msg: HandJointPosesUpdate) -> bool:
        """
        Input hand pose may be of the left or right hand, as indicated by
        `msg.hand`.
        :returns: True if the message was queued, otherwise false because it was
            not newer than the current latest message.
        """
        hand_list: Deque[HandJointPosesUpdate]
        if msg.hand == 'Right':
            hand_list = self.hand_pose_right
        elif msg.hand == 'Left':
            hand_list = self.hand_pose_left
        else:
            raise ValueError(f"Input hand pose for hand '{msg.hand}'? What?")
        with self.__state_lock:
            # before the current lead pose?
            if (
                hand_list and
                time_to_int(msg.header.stamp) <= time_to_int(hand_list[-1].header.stamp)
            ):
                self.get_logger_fn().warn(
                    f"Input hand pose was NOT after the previous latest: "
                    f"(prev) {hand_list[-1].header.stamp} !< {msg.header.stamp} (new)"
                )
                return False
            hand_list.append(msg)
            return True

    def queue_object_detections(self, msg: ObjectDetection2dSet) -> bool:
        """
        Queue up an object detection set for the
        """
        with self.__state_lock:
            # before the current lead pose?
            if (
                self.obj_dets and
                time_to_int(msg.header.stamp) <= time_to_int(self.obj_dets[-1].header.stamp)
            ):
                self.get_logger_fn().warn(
                    f"Input object detection result was NOT after the previous latest: "
                    f"(prev) {self.obj_dets[-1].header.stamp} !< {msg.header.stamp} (new)"
                )
                return False
            self.obj_dets.append(msg)
            return True

    @staticmethod
    def _hand_msg_to_time_ns(msg: HandJointPosesUpdate):
        return time_to_int(msg.header.stamp)

    @staticmethod
    def _objdet_msg_to_time_ns(msg: ObjectDetection2dSet):
        # Using stamp that should associate to the source image
        return time_to_int(msg.source_stamp)

    def get_window(self, window_size: int) -> InputWindow:
        """
        Get a window buffered data as it is associated to frame data.

        Data other than the image frames may not have direct association to a
        particular frame, e.g. "missing" for that frame. In those cases there
        will be a None in the applicable slot.

        :param window_size: number of frames from the head of the buffer to
            consider "the window."

        :return: Mapping of associated data, each of window_size items.
        """
        # Knowns:
        # - Object detections occur on a specific frame as associated by
        #   timestamp *exactly*.

        with self.__state_lock:
            # Cache self accesses
            hand_nsec_tol = self.hand_msg_tolerance_nsec

            # This window's frame in ascending time order
            # deques don't support slicing, so thus the following madness
            window_frames = list(itertools.islice(reversed(self.frames), window_size))[::-1]
            window_frame_times: List[Time] = [wf[0] for wf in window_frames]
            window_frame_times_ns: List[int] = [time_to_int(wft) for wft
                                                in window_frame_times]

            # tolerance associate hand messages, left and right
            # - For each frame backwards, reverse-iterate through hand messages
            #   until encountering one that is more time-distance or out of
            #   tolerance, which triggers moving on to the next frame.
            # - carry variable for when the item being checked in previous
            #   iteration did not match the current frame.
            window_lhand = descending_match_with_tolerance(
                window_frame_times_ns,
                self.hand_pose_left,
                hand_nsec_tol,
                time_from_value_fn=self._hand_msg_to_time_ns,
            )
            window_rhand = descending_match_with_tolerance(
                window_frame_times_ns,
                self.hand_pose_right,
                hand_nsec_tol,
                time_from_value_fn=self._hand_msg_to_time_ns,
            )

            # Direct associate object detections within window time. For
            # detections known to be in the window, creating a mapping (key=ts)
            # to access the detection for a specific time.
            window_dets = descending_match_with_tolerance(
                window_frame_times_ns,
                self.obj_dets,
                0,  # we expect exact matches for object detections.
                time_from_value_fn=self._objdet_msg_to_time_ns,
            )

            output = InputWindow(
                frames=window_frames,
                hand_pose_left=window_lhand,
                hand_pose_right=window_rhand,
                obj_dets=window_dets,
            )
            return output

    def clear_before(self, time_nsec: int) -> None:
        """
        Clear content in the buffer that is strictly older than the given
        timestamp.
        """
        # for each deque, traverse from the left (the earliest time) and pop if
        # the ts is < the given.
        with self.__state_lock:
            while (self.frames and
                   time_to_int(self.frames[0][0]) < time_nsec):
                self.frames.popleft()
            while (self.hand_pose_left and
                   time_to_int(self.hand_pose_left[0].header.stamp) < time_nsec):
                self.hand_pose_left.popleft()
            while (self.hand_pose_right and
                   time_to_int(self.hand_pose_right[0].header.stamp) < time_nsec):
                self.hand_pose_right.popleft()
            while (self.obj_dets and
                   time_to_int(self.obj_dets[0].source_stamp) < time_nsec):
                self.obj_dets.popleft()


class UHOActivityDetector(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        # Declare ROS topics
        self._image_topic = (
            self.declare_parameter("image_topic", "PVFramesRGB")
            .get_parameter_value()
            .string_value
        )
        self._hand_topic = (
            self.declare_parameter("hand_pose_topic", "HandJointPoseData")
            .get_parameter_value()
            .string_value
        )
        self._obj_det_topic = (
            self.declare_parameter("obj_det_topic", "ObjectDetections")
            .get_parameter_value()
            .string_value
        )
        self._torch_device = (
            self.declare_parameter("torch_device", "cuda")
            .get_parameter_value()
            .string_value
        )
        self._det_topic = (
            self.declare_parameter("det_topic", "ActivityDetections")
            .get_parameter_value()
            .string_value
        )
        self._min_time_topic = (
            self.declare_parameter("min_time_topic", "ObjDetMinTime")
            .get_parameter_value()
            .string_value
        )
        self._frames_per_det = (
            self.declare_parameter("frames_per_det", 32)
            .get_parameter_value()
            .integer_value
        )
        # The number of object detections we require to be in the input buffer
        # window before considering it value for processing.
        self._obj_dets_per_window = (
            self.declare_parameter("object_dets_per_window", 2)
            .get_parameter_value()
            .integer_value
        )
        # Maximum size for our data buffer in terms of seconds.
        # We will use this in our prediction runtime to make sure that we don't
        # build up too much data if we happen to not predict for a while.
        # Default value assumes frame-rate of ~30Hz and a processing window of
        # about 1 seconds-worth of data (`frames_per_det` above).
        self._buffer_max_size_seconds = (
            self.declare_parameter("buffer_max_size_seconds", 2.0)
            .get_parameter_value()
            .double_value
        )
        self._model_checkpoint = (
            self.declare_parameter("model_checkpoint",
                                   "/angel_workspace/model_files/uho_epoch_090.ckpt")
            .get_parameter_value()
            .string_value
        )
        self._labels_file = (
            self.declare_parameter("labels_file",
                                   "/angel_workspace/model_files/uho_epoch_090_labels.txt")
            .get_parameter_value()
            .string_value
        )
        # Model specific top-K parameter.
        self._topk = (
            self.declare_parameter("top_k", 5)
            .get_parameter_value()
            .integer_value
        )
        # Ground truth file for classifications (optional)
        # This is expecting a feather file format. Activity prediction vector
        # size will be dictated by the number of unique activities labeled
        # (+ background), and the confidence vector order will be determined
        # by the order labels appear in this file.
        self._gt_file = (
            self.declare_parameter("gt_file", "")
            .get_parameter_value()
            .string_value
        )
        # Whether overlapping windows are passed to the classifier
        self._overlapping_mode = (
            self.declare_parameter("overlapping_mode", False)
            .get_parameter_value()
            .bool_value
        )

        self._buffer_max_size_nanosec = int(self._buffer_max_size_seconds * 1e9)
        self._slop_ns = (5 / 60.0) * 1e9  # slop (hand msgs have rate of ~60hz per hand)

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Hand topic: {self._hand_topic}")
        log.info(f"Object detections topic: {self._obj_det_topic}")
        log.info(f"Device? {self._torch_device}")
        log.info(f"Frames per detection: {self._frames_per_det}")
        log.info(f"Checkpoint: {self._model_checkpoint}")
        log.info(f"Labels: {self._labels_file}")
        log.info(f"GT file: {self._gt_file}")

        # Subscribers for input data channels.
        # These will collect on their own threads, adding to buffers from
        # activity classification will draw from.
        # - Image data
        self._image_subscription_cb_group = MutuallyExclusiveCallbackGroup()
        self._image_subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.image_callback,
            1,
            callback_group=self._image_subscription_cb_group,
        )
        # - Hand pose
        self._hand_subscription_cb_group = MutuallyExclusiveCallbackGroup()
        self._hand_subscription = self.create_subscription(
            HandJointPosesUpdate,
            self._hand_topic,
            self.hand_callback,
            1,
            callback_group=self._hand_subscription_cb_group,
        )
        # - Object detections
        self._obj_det_subscriber_cb_group = MutuallyExclusiveCallbackGroup()
        self._obj_det_subscriber = self.create_subscription(
            ObjectDetection2dSet,
            self._obj_det_topic,
            self.obj_det_callback,
            1,
            callback_group=self._obj_det_subscriber_cb_group,
        )

        # Channel over which we communicate to the object detector a timestamp
        # before which to skip processing/publication.
        self._min_time_publisher_cb_group = MutuallyExclusiveCallbackGroup()
        self._min_time_publisher = self.create_publisher(
            Time,
            self._min_time_topic,
            1,
            callback_group=self._min_time_publisher_cb_group,
        )
        self._activity_publisher_cb_group = MutuallyExclusiveCallbackGroup()
        self._activity_publisher = self.create_publisher(
            ActivityDetection,
            self._det_topic,
            1,
            callback_group=self._activity_publisher_cb_group,
        )

        # Create the runtime thread to trigger processing and buffer cleanup
        # appropriately.
        self._input_buffer = InputBuffer(int(self._slop_ns), self.get_logger)

        # Instantiate the activity detector models
        self._detector = get_uho_classifier(
            self._model_checkpoint,
            self._labels_file,
            self._torch_device,
        )
        log.info(f"UHO Detector initialized")
        # TODO: Warmup detector?

        # Variables used by window criterion methods
        self._prev_leading_time_ns = None
        self._prev_last_time_ns = None

        # Start the runtime thread
        log.info("Starting runtime thread...")
        # switch for runtime loop
        self._rt_active = Event()
        self._rt_active.set()
        # seconds to occasionally time out of the wait condition for the loop
        # to check if it is supposed to still be alive.
        self._rt_active_heartbeat = 0.1  # TODO: Parameterize?
        # Event to notify runtime it should try processing now.
        self._rt_awake_evt = Event()
        self._rt_thread = Thread(
            target=self.thread_predict_runtime,
            name="prediction_runtime"
        )
        self._rt_thread.daemon = True
        self._rt_thread.start()
        log.info("Starting runtime thread... Done")

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

    def stop_runtime(self) -> None:
        """
        Indicate that the runtime loop should cease.
        """
        self._rt_active.clear()

    def image_callback(self, image: Image) -> None:
        """
        Callback function for images. Messages are saved in the images list.
        """
        # Check first, so we don't accumulate anything when we aren't using it.
        if self.rt_alive() and self._input_buffer.queue_image(image):
            self.get_logger().info(f"Queueing image (ts={image.header.stamp})")
            # On new imagery, tell the RT loop to wake up and try to form a window
            # for processing.
            self._rt_awake_evt.set()

    def hand_callback(self, hand_pose: HandJointPosesUpdate) -> None:
        """
        Callback function for hand poses. Messages are saved in the hand_poses
        list.
        """
        # Check first, so we don't accumulate anything when we aren't using it.
        if self.rt_alive() and self._input_buffer.queue_hand_pose(hand_pose):
            self.get_logger().info(f"Queueing hand pose (hand={hand_pose.hand}) "
                                   f"(ts={hand_pose.header.stamp})")

    def obj_det_callback(self, msg):
        """
        Callback for the object detection set message. If there are enough frames
        accumulated for the activity detector and there is an object detection
        message received for the last frame in the frame set or after it,
        the activity detector model is called and a new activity detection message
        is published with the current activity predictions.
        """
        log = self.get_logger()

        if msg.num_detections < self._topk:
            log.warn(f"Received msg with less than {self._topk} detections. "
                     f"Skipping.")
            return

        # Check first, so we don't accumulate anything when we aren't using it.
        if self.rt_alive() and self._input_buffer.queue_object_detections(msg):
            log.info(f"Queueing object detections (ts={msg.header.stamp})")

    def _window_criterion_correct_size(self, window: InputWindow) -> bool:
        window_ok = len(window) == self._frames_per_det
        if not window_ok:
            self.get_logger().warn(f"Window is not the appropriate size "
                                   f"(actual:{len(window)} != "
                                   f"{self._frames_per_det}:expected)")
        return window_ok

    def _window_criterion_enough_dets(self, window: InputWindow) -> bool:
        num_dets = len(list(filter(None, window.obj_dets)))
        self.get_logger().info(f"Window num dets: {num_dets}")
        window_ok = num_dets >= self._obj_dets_per_window
        if not window_ok:
            self.get_logger().warn(f"Window num dets ({num_dets}) not at "
                                   f"least {self._obj_dets_per_window}")
        return window_ok

    def _window_criterion_new_leading_frame(self, window: InputWindow) -> bool:
        """
        The new window's leading frame should be beyond a previous window's
        leading frame.
        """
        # Assuming _window_criterion_correct_size already passed.
        cur_leading_time_ns = time_to_int(window.frames[-1][0])
        prev_leading_time_ns = self._prev_leading_time_ns
        if prev_leading_time_ns is not None:
            window_ok = prev_leading_time_ns < cur_leading_time_ns
            if not window_ok:
                # current window is earlier/same lead as before, so not a good
                # window
                self.get_logger().warn("RT duplicate window detected, skipping.")
                return False
            # Window is OK, save new latest leading frame time below.
        # Else:This is the first window. The first history will be recorded
        # below.
        self._prev_leading_time_ns = cur_leading_time_ns
        return True

    def _window_criterion_ensure_non_overlapping(self, window: InputWindow) -> bool:
        """
        The new window's leading frame should be beyond a previous window's
        last frame.
        """
        # Assuming _window_criterion_correct_size already passed.
        cur_first_time_ns = time_to_int(window.frames[0][0])
        prev_last_time_ns = self._prev_last_time_ns
        if prev_last_time_ns is not None:
            window_ok = prev_last_time_ns < cur_first_time_ns
            if not window_ok:
                # current window is earlier/same lead as before, so not a good
                # window
                self.get_logger().warn("RT overlapping window detected, skipping.")
                return False
            # Window is OK, save new latest prev last time below.
        # Else: This is the first window, save the last frame's time below.
        self._prev_last_time_ns = time_to_int(window.frames[-1][0])
        return True

    def thread_predict_runtime(self):
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
            self._window_criterion_enough_dets,
            self._window_criterion_new_leading_frame,
        ]

        if self._overlapping_mode is False:
            window_processing_criterion_fn_list.append(
                self._window_criterion_ensure_non_overlapping
            )

        # If the ground truth file is provided, use the gt processing function
        if self._gt_file == "":
            process_window_fn = self._process_window
        else:
            process_window_fn = self._process_window_gt

        while self._rt_active.wait(0):  # will quickly return false if cleared.
            if self._rt_awake_evt.wait(self._rt_active_heartbeat):
                log.info("RT loop awakened")
                # reset the flag for the next go-around
                self._rt_awake_evt.clear()

                # We want to fire off a prediction run when:
                # - The latest window is at least self._frames_per_det in size
                # - The latest window contains at least self._obj_dets_per_window
                #   object detection result associations.
                # TODO: Include forced latency?
                #       request window X seconds back from latest.
                #       Paul's laptop: catching ~4 dets per 32-frames
                #           ~1 every 8 frames, so (1/30)*8 ~=0.267 latency
                window = self._input_buffer.get_window(self._frames_per_det)
                if all(fn(window) for fn in window_processing_criterion_fn_list):
                    log.info("RT Starting processing block")

                    # After getting validating a window, before processing,
                    # clear out older data from the buffer using the timestamp
                    # of window first frame + a nanosecond.
                    # TODO: Change this to first + forced latency time?
                    old_time = window.frames[1][0]
                    old_time_ns = time_to_int(old_time)
                    self._input_buffer.clear_before(old_time_ns)

                    # Before processing publish the latest frame time to the
                    # self._min_time_publisher.
                    # Also inform any listeners that we no
                    self._min_time_publisher.publish(old_time)

                    act_msg = process_window_fn(window)
                    log.info("RT publishing activity classification results")
                    self._activity_publisher.publish(act_msg)
                else:
                    # Clear at least to our max buffer size even if we didn't
                    # process anything.
                    old_time_ns = (
                        time_to_int(window.frames[-1][0])
                        - self._buffer_max_size_nanosec
                    )
                    self._input_buffer.clear_before(old_time_ns)

            else:
                # wait timeout triggered, just loop.
                log.debug("RT heartbeat timeout: checking alive status")

        log.info("Runtime function end.")

    def _process_window(self, window: InputWindow) -> ActivityDetection:
        """
        Invoke activity classifier for this window of data, performing all
        appropriate conversions to and from algorithm specific needs.

        Assuming window has passed appropriate criterion checks.
        """
        frame_set = [frm[1] for frm in window.frames]

        # Convert hand messages that we have into appropriate vector form.
        known_hand_dim = 21 * 3  # model-specific, based on joints used
        hand_zero_vec = np.zeros(known_hand_dim, dtype=np.float)
        lhand_arrays: List[Optional[npt.NDArray]] = [None] * len(frame_set)
        for i, msg in enumerate(window.hand_pose_left):
            if msg is not None:
                lhand_arrays[i] = get_hand_pose_from_msg(msg)
            else:
                lhand_arrays[i] = hand_zero_vec.copy()
        rhand_arrays: List[Optional[npt.NDArray]] = [None] * len(frame_set)
        for i, msg in enumerate(window.hand_pose_right):
            if msg is not None:
                rhand_arrays[i] = get_hand_pose_from_msg(msg)
            else:
                rhand_arrays[i] = hand_zero_vec.copy()
        # TODO: zero out both vector pairs if any one is a zero vector.

        frame_set, aux_data = create_batch(
            frame_set, lhand_arrays, rhand_arrays, window.obj_dets,
            self._topk
        )

        # Model input format notes/questions
        # - Hand input
        #   - should include both hands in a flattened vector,p
        #     (J*3*2), where Dawei is expecting J=21.
        #     shape [32 x (J*3*2)] (J=21 -> 126)
        #   - If a one hand is missing from the frame, zero the whole
        #     vector out.
        #   - Found a Left --> Right order detail in the transformer code
        #   - What joints order was the model trained on? Can we check that?
        # - Detection descriptors and bboxes should not have any zero
        #   vectors. There is some expectation of "regularly" spaced
        #   detections. A sample-step was described that is set to 5,
        #   meaning that the input detections were rigidly assigned to
        #   input frames.
        #   - new discussion, provided packed matrix, K=<top-k size>
        #       [32*K x 2048]
        #       [32*K x 4]
        with SimpleTimer("Activity classification prediction", self.get_logger().info):
            pred_conf, pred_labels = predict(self._detector, frame_set, aux_data)

        # Create activity message from results
        activity_msg = ActivityDetection()
        activity_msg.header.frame_id = "Activity Classification"
        activity_msg.header.stamp = self.get_clock().now().to_msg()
        activity_msg.source_stamp_start_frame = window.frames[0][0]
        activity_msg.source_stamp_end_frame = window.frames[-1][0]
        activity_msg.label_vec = pred_labels
        activity_msg.conf_vec = pred_conf[0].squeeze().tolist()

        return activity_msg

    def _process_window_gt(self, window: InputWindow) -> ActivityDetection:
        """
        Invoke the ground truth activity classifier stub for this window of
        data.

        Assuming window has passed appropriate criterion checks.
        """
        with SimpleTimer("Activity classification prediction", self.get_logger().info):
            pred_conf, pred_labels = gt_predict(
                self._gt_file,
                time_to_int(window.frames[0][0]) * 1e-9,
                time_to_int(window.frames[-1][0]) * 1e-9,
            )

        # Create activity message from results
        activity_msg = ActivityDetection()
        activity_msg.header.frame_id = "Activity Classification"
        activity_msg.header.stamp = self.get_clock().now().to_msg()
        activity_msg.source_stamp_start_frame = window.frames[0][0]
        activity_msg.source_stamp_end_frame = window.frames[-1][0]
        activity_msg.label_vec = pred_labels
        activity_msg.conf_vec = pred_conf[0].squeeze().tolist()

        return activity_msg

    def destroy_node(self):
        print("Shutting down runtime thread...")
        self._rt_active.clear()  # make RT active flag "False"
        self._rt_thread.join()
        print("Shutting down runtime thread... Done")
        super()


def main():
    rclpy.init()

    detector = UHOActivityDetector()

    # If things are going wrong, set this False to debug in a serialized setup.
    do_multithreading = True

    if do_multithreading:
        # Don't really want to use *all* available threads...
        # 5 threads because:
        # - 3 known subscribers which have their own groups
        # - 1 for default group
        # - 1 for publishers
        executor = MultiThreadedExecutor(num_threads=5)
        executor.add_node(detector)
        try:
            executor.spin()
        except KeyboardInterrupt:
            detector.stop_runtime()
            detector.get_logger().info("Keyboard interrupt, shutting down.\n")
    else:
        try:
            rclpy.spin(detector)
        except KeyboardInterrupt:
            detector.stop_runtime()
            detector.get_logger().info("Keyboard interrupt, shutting down.\n")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    #  when the garbage collector destroys the node object... if it gets to it)
    detector.destroy_node()

    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
