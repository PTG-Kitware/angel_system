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
import pandas as pd

# ROS Message types
from builtin_interfaces.msg import Time

from angel_msgs.msg import (
    ActivityDetection,
    HandJointPosesUpdate,
    ObjectDetection2dSet,
    HandJointPose,
)

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
    frames: List[Tuple[Time, npt.NDArray, int]]
    obj_dets: List[Optional[ObjectDetection2dSet]]
    # Buffer for patient poses
    patient_joint_kps: List[Optional[HandJointPosesUpdate]]

    def __len__(self):
        return len(self.frames)

    def __repr__(self):
        """
        Construct a tabular representation of the window state.

        This table will show, for some frame timestamp (nanoseconds):
            * number of object detections
            * number of pose key-points

        Either of the above may show as "NaN" if there were no object
        detections or pose key-points received for that particular frame.

        The order of the table representation will show the most recent
        timestamp frame and correlated data **lower** in the table (higher
        index). This order is arbitrary.
        """
        return repr(
            pd.DataFrame(
                data={
                    "frame_number": [f[2] for f in self.frames],
                    "frame_nsec": [time_to_int(f[0]) for f in self.frames],
                    "detections": [
                        (d.num_detections if d else None) for d in self.obj_dets
                    ],
                    "poses": [
                        (len(p.joints) if p else None) for p in self.patient_joint_kps
                    ],
                },
                dtype=pd.Int64Dtype,
            )
        )


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
    frames: Deque[Tuple[Time, npt.NDArray, int]] = field(
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

    patient_joint_kps: Deque[HandJointPosesUpdate] = field(
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
        self, img_mat: npt.NDArray[np.uint8], img_header_stamp: Time, image_frame_number: int
    ) -> bool:
        """
        Queue up a new image frame.
        :returns: True if the image was queued, otherwise false because it was
            not newer than the current latest frame.
        """
        # self.get_logger_fn().info(f"image header stamp: {img_header_stamp}")
        # self.get_logger_fn().info(f"self.frames[-1][0] header stamp: {self.frames[-1][0]}")
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
            self.frames.append((img_header_stamp, img_mat, image_frame_number))
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

    def queue_joint_keypoints(self, msg: HandJointPosesUpdate) -> bool:
        """
        Queue up an object detection set for the
        """

        with self.__state_lock:
            # before the current lead pose?
            if self.patient_joint_kps and time_to_int(msg.header.stamp) <= time_to_int(
                self.patient_joint_kps[-1].header.stamp
            ):
                self.get_logger_fn().warn(
                    f"Input pose estimation results was NOT after the previous latest: "
                    f"(prev) {self.patient_joint_kps[-1].header.stamp} !< {msg.header.stamp} (new)"
                )
                return False
            self.patient_joint_kps.append(msg)
            return True

    @staticmethod
    def _hand_msg_to_time_ns(msg: HandJointPosesUpdate):
        return time_to_int(msg.header.stamp)

    @staticmethod
    def _objdet_msg_to_time_ns(msg: ObjectDetection2dSet):
        # Using stamp that should associate to the source image
        return time_to_int(msg.source_stamp)

    @staticmethod
    def _joints_msg_to_time_ns(msg: HandJointPosesUpdate):
        # Using stamp that should associate to the source image
        return time_to_int(msg.source_stamp)

    def get_window(
        self,
        window_size: int,
        have_leading_object: bool = False,
    ) -> InputWindow:
        """
        Get a window of buffered data as it is associated to frame data.

        Data other than the image frames may not have direct association to a
        particular frame, e.g. "missing" for that frame. In those cases there
        will be a None in the applicable slot.

        :param window_size: number of frames from the head of the buffer to
            consider "the window."
        :param have_leading_object: Indicate that we want the leading frame of
            the returned window to be the most recent frame with object
            detections associated with it, as opposed to the most recent frame
            period.

        :return: Mapping of associated data, each of window_size items.
        """
        # Knowns:
        # - Object detections occur on a specific frame as associated by
        #   timestamp *exactly*.

        with self.__state_lock:
            # Normally, we consider frames for the window starting with the
            # most recent frame. When the frames deque is reverse iterated,
            # this would be indexed zero.
            window_frame_start_idx = 0

            if have_leading_object:
                # Determine the slice of `window_frames` to use such that the
                # most recent frame of the window is the same frame for which
                # our most recent object detections are for.

                # If `obj_dets` is empty, return empty window.
                if not self.obj_dets:
                    return InputWindow(frames=[], obj_dets=[], patient_joint_kps=[])

                # Use the last object detection to get a timestamp. Find that
                # timestamp in the `frames` deque, traversing backwards.
                last_det_ts = self._objdet_msg_to_time_ns(self.obj_dets[-1])
                last_det_frame_idx = None
                for i, frame in enumerate(reversed(self.frames)):
                    if time_to_int(frame[0]) == last_det_ts:
                        last_det_frame_idx = i
                        break
                if last_det_frame_idx is None:
                    # Failed to find a queued frame for the object detection.
                    # This is not expected, but can technically happen.
                    # Return an empty window.
                    return InputWindow(frames=[], obj_dets=[], patient_joint_kps=[])
                # Update the window frame start index to the
                window_frame_start_idx = last_det_frame_idx

            # This window's frame in ascending time order.
            # `deque` instances don't support slicing, so we use
            # `itertools.islice` of the reverse iterator of the frames deque to
            # slice backwards from the most recent frame (or specified starting
            # index).
            # The `window_size` is increased by the starting index so that we
            # actually get a `window_size` list back regardless of where we are
            # starting.
            # Finally, the extracted slice is reversed in order again so that
            # the final list is in temporally ascending order.
            window_frames = list(
                itertools.islice(
                    reversed(self.frames),
                    window_frame_start_idx,
                    window_size + window_frame_start_idx,
                )
            )[::-1]
            window_frame_times: List[Time] = [wf[0] for wf in window_frames]
            window_frame_times_ns: List[int] = [
                time_to_int(wft) for wft in window_frame_times
            ]

            window_dets = descending_match_with_tolerance(
                window_frame_times_ns,
                self.obj_dets,
                0,  # we expect exact matches for object detections.
                time_from_value_fn=self._objdet_msg_to_time_ns,
            )

            # print(f"Window window detections: {window_dets}")

            window_joint_kps = descending_match_with_tolerance(
                window_frame_times_ns,
                self.patient_joint_kps,
                0,  # we expect exact matches for object detections.
                time_from_value_fn=self._joints_msg_to_time_ns,
            )

            output = InputWindow(
                frames=window_frames,
                obj_dets=window_dets,
                patient_joint_kps=window_joint_kps,
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

            while (
                self.patient_joint_kps
                and time_to_int(self.patient_joint_kps[0].source_stamp) < time_nsec
            ):
                self.patient_joint_kps.popleft()
