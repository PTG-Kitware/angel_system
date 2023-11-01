from typing import List
from typing import Tuple

import numpy as np
import numpy.typing as npt

from angel_msgs.msg import ObjectDetection2dSet


def max_labels_and_confs(
    msg: ObjectDetection2dSet,
) -> Tuple[npt.NDArray[str], npt.NDArray[float]]:
    """
    Get out a tuple of the maximally confident class label and
    confidence value for each detection as a tuple of two lists for
    expansion into the `ObjectDetectionsLTRB` constructor

    :param msg: Input 2D object detection set message.

    :returns: Two 1S arrays of the labels and confidence values associated with
        the maximally confident prediction of each detection as numpy arrays.
        Each array's size should be equal to the number of detections present
        in the message (i.e. `msg.num_detections`).
    """
    mat_shape = (msg.num_detections, len(msg.label_vec))
    conf_mat = np.asarray(msg.label_confidences).reshape(mat_shape)
    max_conf_idxs = conf_mat.argmax(axis=1)
    max_confs = conf_mat[np.arange(conf_mat.shape[0]), max_conf_idxs]
    max_labels = np.asarray(msg.label_vec)[max_conf_idxs]
    return max_labels, max_confs
