"""
Various conversion functions into and out of angel_msg types.
"""
import array
import itertools
import math
from typing import Dict
from typing import Hashable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from builtin_interfaces.msg import Time
import cv2
import numpy as np
import numpy.typing as npt

from angel_msgs.msg import HandJointPosesUpdate, ObjectDetection2dSet
from smqtk_detection.utils.bbox import AxisAlignedBoundingBox


SEC_TO_NANO = int(1e9)
NANO_TO_SEC = 1e-9
HECTO_NS = 1e7


def time_to_int(time_msg: Time) -> int:
    """
    Convert the given time message into an integer representing nanoseconds,
    which is easily comparable and index-able.
    :param time_msg:
    :return: Integer nanoseconds
    """
    return (time_msg.sec * SEC_TO_NANO) + time_msg.nanosec


def time_to_float(time_msg: Time) -> float:
    """
    Convert the given time message into a floating point value representing
    seconds.
    :param time_msg:
    :return: Floating point seconds.
    """
    return time_msg.sec + (time_msg.nanosec * NANO_TO_SEC)


def nano_to_ros_time(timestamp: int) -> Time:
    """
    Convert an integer representing time in nanoseconds to ROS2 Time message
    instance.

    :param timestamp: Input time in nanoseconds (ns).
    :return: ROS2 Time message representing the input time.
    """
    sec = timestamp // SEC_TO_NANO
    nanosec = timestamp % SEC_TO_NANO
    return Time(sec=sec, nanosec=nanosec)


def hl2ss_stamp_to_ros_time(timestamp: int) -> Time:
    """
    Convert the HL2SS timestamp which is an integer in hundreds of ns (1e7) to
    a ROS Time message.
    :param timestamp: Integer timestamp of the HL2SS packet
    :return: Time message
    """
    sec = int(timestamp // HECTO_NS)
    nanosec = int(timestamp % HECTO_NS)
    return Time(sec=sec, nanosec=nanosec)


def from_detect_image_objects_result(
    detections: Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]],
    detection_threshold: float = 0,
) -> ObjectDetection2dSet:
    """
    Convert an iterable of detection results from a smqtk-detection
    DetectImageObjects algorithm instance into a new ObjectDetection2dSet
    message.

    This function does *not* touch the `header` or `source_stamp` fields of the
    ObjectDetection2dSet message output. The user of the instance should
    populate those fields appropriately according to the context.

    :param detections: Iterable of detection prediction results.
    :param detection_threshold: Do not include detections whose maximum
        confidence score is less than this threshold value.

    :returns: New ObjectDetection2dSet instance containing the detections.
    """
    # We'll be taking multiple passes over detections, so make sure it is
    # expanded.
    detections = tuple(detections)

    # Aggregate all detections, create "master set" of labels, ordered
    # alphabetically for determinism.
    label_union = set()
    for _, det_preds in detections:
        label_union.update(det_preds.keys())
    label_list = sorted(label_union)

    # List of (left, top, right, bottom) quadruples per detection.
    # Should end up with shape [n_detections x 4] after threshold filtering.
    det_ltrb = []

    # Matrix of detection confidences
    # Should end up with shape [n_detections x n_labels] after threshold
    # filtering.
    det_confidence = []

    for i, (det_bbox, det_preds) in enumerate(detections):
        # Get predicted confidences in the order determined above, filling in
        # 0's for labels not present in this particular prediction.
        confidence_vec = [det_preds.get(label, 0.0) for label in label_list]

        # Skip detection if the maximally confident class is less than
        # our confidence threshold. If "background" or equivalent classes
        # are included in the chosen detector, then every class may be
        # output...
        max_conf = np.max(confidence_vec)
        if max_conf < detection_threshold:
            continue

        det_confidence.append(confidence_vec)
        det_ltrb.append((*det_bbox.min_vertex, *det_bbox.max_vertex))

    assert len(det_ltrb) == len(det_confidence)
    n_detections = len(det_confidence)

    # If there are no detections post-filtering, empty out the label vec since
    # there will be nothing in this message to refer to it.
    if n_detections == 0:
        label_list = []

    msg = ObjectDetection2dSet()
    msg.label_vec = label_list
    msg.num_detections = n_detections
    if n_detections > 0:
        bounds_mat = np.asarray(det_ltrb, dtype=np.float32).T
        msg.left = bounds_mat[0].tolist()
        msg.top = bounds_mat[1].tolist()
        msg.right = bounds_mat[2].tolist()
        msg.bottom = bounds_mat[3].tolist()
        msg.label_confidences = (
            np.asarray(det_confidence, dtype=np.float64).ravel().tolist()
        )

    return msg


def to_confidence_matrix(msg: ObjectDetection2dSet) -> np.ndarray:
    """
    Get the detection predicted confidences as a 2D matrix.
    :param msg: Message to get the matrix confidences from.
    :return: New numpy ndarray of 2 dimensions with shape [nDets x nClasses].
    """
    return np.asarray(msg.label_confidences).reshape(
        (msg.num_detections, len(msg.label_vec))
    )


def convert_nv12_to_rgb(nv12_image: array.array, height: int, width: int) -> np.ndarray:
    """
    Converts an image in NV12 format to RGB.

    :param nv12_image: Buffer containing the image data in NV12 format
    :param height: image pixel height
    :param width: image pixel width

    :returns: RGB image
    """
    yuv_image = np.frombuffer(nv12_image, np.uint8).reshape(height * 3 // 2, width)
    rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return rgb_image


def hand_joint_poses_to_struct(
    msg: HandJointPosesUpdate,
) -> Tuple[List[str], npt.NDArray[np.float64]]:
    """
    Convert a hand joint pose message into a list of joint labels and the
    matrix of 3D joint positions ([x, y, z] format).

    The output matrix will be of shape `[nJoints x 3]`, where `nJoints` is the
    size of the `joints` vector in the input message.

    :param msg:

    :return:
    """
    n_joints = len(msg.joints)
    joint_labels = [""] * n_joints
    joint_poses = np.empty((n_joints, 3))
    for i, j in enumerate(msg.joints):
        joint_labels[i] = j.joint
        joint_poses[i] = [
            j.pose.position.x,
            j.pose.position.y,
            j.pose.position.z,
        ]
    return joint_labels, joint_poses


def sparse_hand_joint_poses_to_structs(
    msgs: Sequence[Optional[HandJointPosesUpdate]],
) -> Tuple[List[str], List[Optional[npt.NDArray[np.float64]]]]:
    """
    Convert a sparse sequence of hand joint labels and sparse pose matrices
    whose coordinates correspond to the order of the label sequence.

    If no messages in the input sequence, the returned label sequence will be
    empty and the position mats list will be a list of `None` values equivalent
    to the input sequence length.

    If subsequent messages do not have the same joint labels order as the first
    non-None message, we raise a ValueError.

    :param msgs: Sparse sequence of HandJointPosesUpdate messages.
    :return: List of hand joint labels, and an equivalently sparse list of
        hand joint position matrices.
    """
    ret_labels = None
    position_mats: List[Optional[npt.NDArray]] = [None] * len(msgs)
    # Progress to the first not-None value, storing label list order and
    # the first position matrix. Then progress through the remainder of the
    # value messages, asserting the label order is consistent.
    msg_it = enumerate(msgs)
    for i, msg in msg_it:
        if msg is not None:
            # First non-None item in the given sequence
            ret_labels, poses = hand_joint_poses_to_struct(msg)
            position_mats[i] = poses
            break
    for i, msg in msg_it:
        if msg is not None:
            labels, poses = hand_joint_poses_to_struct(msg)
            if labels != ret_labels:
                raise ValueError(
                    "Subsequent message does not have the same "
                    "joints labels order as the first."
                )
            position_mats[i] = poses
    return ret_labels, position_mats
