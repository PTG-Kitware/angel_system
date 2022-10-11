"""
Helper functions for forming the input data to the UHO model.
"""
from typing import Callable
from typing import Dict
from typing import List

from angel_msgs.msg import ObjectDetection2dSet
import numpy as np
import torch

def create_batch(
    frame_set: List[np.array],
    lhand_pose_set: List[np.array],
    rhand_pose_set: List[np.array],
    detection_set: List[ObjectDetection2dSet],
    device: str,
    fcn_transform: Callable,
    topk: int = 5,
):
    """
    Processes the input data to create a batch suitable for the UHO model input.
    """
    assert len(frame_set) == len(lhand_pose_set) == len(rhand_pose_set)
    aux_data = dict(
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
        aux_data['dets'].append(det_descriptors)

        # Grab the bboxes corresponding to the top predictions
        bboxes = [
            torch.Tensor((det.left[i], det.top[i], det.right[i], det.bottom[i])) for i in top_det_idx
        ]
        bboxes = torch.stack(bboxes)

        aux_data['bbox'].append(bboxes)

    # Check if we didn't get any detections in the time range of the frame set
    if len(aux_data["dets"]) == 0 or len(aux_data["bbox"]) == 0:
        aux_data["dets"] = [torch.zeros((topk, msg.descriptor_dim))]
        aux_data["bbox"] = [torch.zeros((topk, 4))]

    # Preprocess aux data
    data_dict = {}

    labels = {"l_hand": [], "r_hand": []}
    labels["l_hand"] = torch.stack([torch.from_numpy(k) for k in aux_data["lhand"]]).to(device)
    labels["l_hand"] = labels["l_hand"].unsqueeze(0)
    labels["r_hand"] = torch.stack([torch.from_numpy(k) for k in aux_data["rhand"]]).to(device)
    labels["r_hand"] = labels["r_hand"].unsqueeze(0)

    data_dict["labels"] = labels

    if len(aux_data["dets"]) == 0:
        data_dict["dets"] = torch.empty((0, 2048)).to(device)
    else:
        data_dict["dets"] = torch.cat(aux_data["dets"]).to(device)
        data_dict["dets"] = data_dict["dets"].reshape(
            [1,
             data_dict["dets"].shape[0],
             data_dict["dets"].shape[1]]
        )
    if len(aux_data["bbox"]) == 0:
        data_dict["bbox"] = torch.empty((0, 2048)).to(device)
    else:
        data_dict["bbox"] = torch.cat(aux_data["bbox"]).to(device)
        data_dict["bbox"] = data_dict["bbox"].reshape(
            [1,
             data_dict["bbox"].shape[0],
             data_dict["bbox"].shape[1]]
        )

    # Preprocess frames
    frames = [fcn_transform(f) for f in frame_set]
    frame_tensor = torch.stack(frames)
    frame_tensor = frame_tensor.to(device=torch.device(device))

    return frame_tensor, data_dict
