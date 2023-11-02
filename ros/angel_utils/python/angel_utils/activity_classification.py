from collections import deque
from dataclasses import dataclass, field
import itertools
from threading import RLock
from typing import Any
from typing import Callable
from typing import Deque
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import numpy.typing as npt

# ROS Message types
from builtin_interfaces.msg import Time

from angel_msgs.msg import ActivityDetection, HandJointPosesUpdate, ObjectDetection2dSet

from angel_system.utils.matching import descending_match_with_tolerance
from angel_utils.conversion import time_to_int


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


# TODO: A more generic version of InputBuffer
#       Want to be able to input a description of the data to be buffered:
#           - key name of that data
#           - how to to get the timestamp for data of that type
#           - if the type is the "key-frame" type (there can only be one)


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

    :param hand_msg_tolerance_nsec: Integer nanoseconds tolerance around a
        key-frame to consider other data to be matched/associated against that
        key-frame.
    """

    # Tolerance in nanoseconds for associating hand-pose messages to a frame.
    hand_msg_tolerance_nsec: int

    # Function to get the logger from the parent ROS2 node.
    get_logger_fn: Callable[[], Any]  # don't know where to get the type for this...

    # Buffer of RGB image matrices and the associated timestamp
    frames: Deque[Tuple[Time, npt.NDArray]] = field(
        default_factory=deque, init=False, repr=False
    )
    # Buffer of left-hand pose messages
    hand_pose_left: Deque[HandJointPosesUpdate] = field(
        default_factory=deque, init=False, repr=False
    )
    # Buffer of right-hand pose messages
    hand_pose_right: Deque[HandJointPosesUpdate] = field(
        default_factory=deque, init=False, repr=False
    )
    # Buffer of object detection predictions
    obj_dets: Deque[ObjectDetection2dSet] = field(
        default_factory=deque, init=False, repr=False
    )

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

    def __len__(self):
        with self.__state_lock:
            return len(self.frames)

    def latest_time(self) -> Time:
        """
        Get timestamp of the most recent image queued in this buffer.

        :raises RuntimeError: No data has yet been buffered.

        :returns: Time message of the latest data queued in this buffer.
        """
        # NOTE: Only considering `frames` for timestamps.
        with self.__state_lock:
            if not self.frames:
                raise RuntimeError("No data buffered for there to be a latest time.")
            return self.frames[-1][0]

    def queue_image(
        self, img_mat: npt.NDArray[np.uint8], img_header_stamp: Time
    ) -> bool:
        """
        Queue up a new image frame.
        :returns: True if the image was queued, otherwise false because it was
            not newer than the current latest frame.
        """
        with self.__state_lock:
            # before the current lead frame?
            if self.frames and time_to_int(img_header_stamp) <= time_to_int(
                self.frames[-1][0]
            ):
                self.get_logger_fn().warn(
                    f"Input image frame was NOT after the previous latest: "
                    f"(prev) {time_to_int(self.frames[-1][0])} "
                    f"!< {time_to_int(img_header_stamp)} (new)"
                )
                return False
            self.frames.append((img_header_stamp, img_mat))
            return True

    def queue_hand_pose(self, msg: HandJointPosesUpdate) -> bool:
        """
        Input hand pose may be of the left or right hand, as indicated by
        `msg.hand`.
        :returns: True if the message was queued, otherwise false because it was
            not newer than the current latest message.
        """
        hand_list: Deque[HandJointPosesUpdate]
        if msg.hand == "Right":
            hand_list = self.hand_pose_right
        elif msg.hand == "Left":
            hand_list = self.hand_pose_left
        else:
            raise ValueError(f"Input hand pose for hand '{msg.hand}'? What?")
        with self.__state_lock:
            # before the current lead pose?
            if hand_list and time_to_int(msg.header.stamp) <= time_to_int(
                hand_list[-1].header.stamp
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
            if self.obj_dets and time_to_int(msg.header.stamp) <= time_to_int(
                self.obj_dets[-1].header.stamp
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
        Get a window of buffered data as it is associated to frame data.

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
            window_frames = list(itertools.islice(reversed(self.frames), window_size))[
                ::-1
            ]
            window_frame_times: List[Time] = [wf[0] for wf in window_frames]
            window_frame_times_ns: List[int] = [
                time_to_int(wft) for wft in window_frame_times
            ]

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
            while self.frames and time_to_int(self.frames[0][0]) < time_nsec:
                self.frames.popleft()
            while (
                self.hand_pose_left
                and time_to_int(self.hand_pose_left[0].header.stamp) < time_nsec
            ):
                self.hand_pose_left.popleft()
            while (
                self.hand_pose_right
                and time_to_int(self.hand_pose_right[0].header.stamp) < time_nsec
            ):
                self.hand_pose_right.popleft()
            while (
                self.obj_dets and time_to_int(self.obj_dets[0].source_stamp) < time_nsec
            ):
                self.obj_dets.popleft()
