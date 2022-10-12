"""
Helper functions for forming the input data to the UHO model.
"""
from dataclasses import dataclass
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

from angel_msgs.msg import ObjectDetection2dSet
import numpy as np
import torch


@dataclass
class AuxData:
    """
    Class for representing the aux data dictionary required by the UHO module.

    It is expected that all list attributes are of the same length.

    Attributes:
        lhand: List of np.ndarrays of size (63,) representing the 63 joint poses
            of the left hand.
        rhand: List of np.ndarrays of size (63,) representing the 63 joint poses
            of the right hand.
        dets: List of torch.Tensors of size [N x D], where N is the number of
            detections and D is the size of detection's descriptor vector.
        bbox: List of torch.Tensors of size [N x 4], where N is the number of
            detections.

    """
    lhand: List[np.ndarray]
    rhand: List[np.ndarray]
    dets: List[torch.Tensor]
    bbox: List[torch.Tensor]


def create_batch(
    frame_set: List[np.array],
    lhand_pose_set: List[np.array],
    rhand_pose_set: List[np.array],
    detection_set: List[ObjectDetection2dSet],
    topk: int = 5,
) -> Tuple[List[np.array], AuxData]:
    """
    Processes the input data to create a batch suitable for the UHO model input.
    """
    assert len(frame_set) == len(lhand_pose_set) == len(rhand_pose_set)
    aux_data = AuxData(
        lhand=lhand_pose_set,
        rhand=rhand_pose_set,
        dets=[],
        bbox=[],
    )

    # Format the object detections into descriptors and bboxes
    for det in detection_set:
        # Get the topk detection confidences
        det_confidences = (
            torch.Tensor(det.label_confidences)
            .reshape((det.num_detections, len(det.label_vec)))
        )

        det_max_confidences = det_confidences.max(axis=1).values
        _, top_det_idx = torch.topk(det_max_confidences, topk)

        det_descriptors = (
            torch.Tensor(det.descriptors).reshape((det.num_detections, det.descriptor_dim))
        )

        # Grab the descriptors corresponding to the top predictions
        det_descriptors = det_descriptors[top_det_idx]
        aux_data.dets.append(det_descriptors)

        # Grab the bboxes corresponding to the top predictions
        bboxes = [
            torch.Tensor((det.left[i], det.top[i], det.right[i], det.bottom[i])) for i in top_det_idx
        ]
        bboxes = torch.stack(bboxes)

        aux_data.bbox.append(bboxes)

    # Check if we didn't get any detections in the time range of the frame set
    if len(aux_data.dets) == 0 or len(aux_data.bbox) == 0:
        aux_data.dets = [torch.zeros((topk, msg.descriptor_dim))]
        aux_data.bbox = [torch.zeros((topk, 4))]

    return frame_set, aux_data
