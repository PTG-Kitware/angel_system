"""
Helper functions for forming the input data to the UHO model.
"""
from dataclasses import dataclass
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import torch

from angel_system.uho.aux_data import AuxData

from angel_msgs.msg import ObjectDetection2dSet


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
    # The current logic only converts real detection messages (skips Nones)
    # NOTE: WE KNOW THIS IS NOT CORRECT, BUT MAINTAINING CURRENT LOGIC FOR NOW
    for frame_dets in filter(None, detection_set):
        # Get the topk detection confidences
        det_confidences = (
            torch.Tensor(frame_dets.label_confidences)
            .reshape((frame_dets.num_detections, len(frame_dets.label_vec)))
        )

        det_max_confidences = det_confidences.max(axis=1).values
        _, top_det_idx = torch.topk(det_max_confidences, topk)

        det_descriptors = (
            torch.Tensor(frame_dets.descriptors).reshape((frame_dets.num_detections, frame_dets.descriptor_dim))
        )

        # Grab the descriptors corresponding to the top predictions
        det_descriptors = det_descriptors[top_det_idx]
        aux_data.dets.append(det_descriptors)

        # Grab the bboxes corresponding to the top predictions
        bboxes = [
            torch.Tensor((frame_dets.left[i], frame_dets.top[i], frame_dets.right[i], frame_dets.bottom[i])) for i in top_det_idx
        ]
        bboxes = torch.stack(bboxes)

        aux_data.bbox.append(bboxes)

    # Check if we didn't get any detections in the time range of the frame set
    if len(aux_data.dets) == 0 or len(aux_data.bbox) == 0:
        aux_data.dets = [torch.zeros((topk, 2048))]
        aux_data.bbox = [torch.zeros((topk, 4))]

    return frame_set, aux_data
