"""
Various conversion functions into and out of angel_msg types.
"""
import array
from typing import Dict
from typing import Hashable
from typing import Iterable
from typing import List
from typing import Tuple

import cv2
import numpy as np

from angel_msgs.msg import ObjectDetection2dSet
from smqtk_detection.utils.bbox import AxisAlignedBoundingBox


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
        msg.label_confidences = np.asarray(det_confidence, dtype=np.float64).ravel().tolist()

    return msg


def to_confidence_matrix(msg: ObjectDetection2dSet) -> np.ndarray:
    """
    Get the detection predicted confidences as a 2D matrix.
    :param msg: Message to get the matrix confidences from.
    :return: New numpy ndarray of 2 dimensions with shape [nDets x nClasses].
    """
    return (
        np.asarray(msg.label_confidences)
          .reshape((msg.num_detections, len(msg.label_vec)))
    )


def convert_nv12_to_rgb(nv12_image: array.array,
                        height: int, width: int) -> np.ndarray:
    """
    Converts an image in NV12 format to RGB.

    :param nv12_image: Buffer containing the image data in NV12 format
    :param height: image pixel height
    :param width: image pixel width

    :returns: RGB image
    """
    yuv_image = np.frombuffer(nv12_image, np.uint8).reshape(height*3//2, width)
    rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return rgb_image
