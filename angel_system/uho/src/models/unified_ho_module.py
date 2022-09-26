import os
from typing import Any, Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms

from .components.unified_fcn import UnifiedFCNModule

class UnifiedHOModule(torch.nn.Module):
    """This class implements the spatio-temporal model used for unified
    representation of hands and interacting objects in the scene.

    This model also performs the activity recognition for the given frame
    sequence.
    """

    def __init__(
        self,
        fcn: torch.nn.Module,
        temporal: torch.nn.Module,
        checkpoint: str
    ):
        super().__init__()
        self.fcn = fcn
        self.temporal = temporal

        m = torch.load(checkpoint)
        self.load_state_dict(m["state_dict"])

        # data transformations (Normalization values recommended by
        # torchvision model zoo)
        self.fcn_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.temporal_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, frame_data, aux_data):
        # Preprocess frames
        frames = [self.fcn_transform(f) for f in frame_data]
        frame_tensor = torch.stack(frames)

        frame_tensor = frame_tensor.to(device=torch.device("cuda"))

        # Extract video features
        frame_feats = self.fcn(frame_tensor)

        '''
        Original:
            Batch data format = (
                {
                    'feats' = (1, n_frames, 2048) # RGB features, used
                    'act' = tensor([n_frames]) # Action labels, not used for inference
                    'idx' = tensor([n_frames]) # Video clip index, not used for inference
                    'labels' = { # Hand poses, used
                            'l_hand' = (1, n_frames, 63),
                            'r_hand' = (1, n_frames, 63),
                        }
                    'dets' = (1, n_detects, 2048) # Detection descriptor, used
                    'bbox' = (1, n_detects, 4) # Detection bounding boxes, used
                    'dcls' = (1, n_detects) # Detection classes, not used for inference
                },
                tensor([n_frames]) # frame lengths
            )
        Inference only:
            Batch data format = {
                    'feats' = (1, n_frames, 2048) # RGB features, used
                    'labels' = { # Hand poses, used
                            'l_hand' = (1, n_frames, 63),
                            'r_hand' = (1, n_frames, 63),
                        }
                    'dets' = (1, n_detects, 2048) # Detection descriptor, used
                    'bbox' = (1, n_detects, 4) # Detection bounding boxes, used
                }
        '''
        device = "cuda"

        # TODO- preprocess aux data
        data_dict = {}
        data_dict["feats"] = frame_feats.unsqueeze(0)

        labels = {"l_hand": [], "r_hand": []}

        labels["l_hand"] = torch.stack([torch.from_numpy(k) for k in aux_data["lhand"]]).to(device)
        labels["l_hand"] = labels["l_hand"].unsqueeze(0)
        labels["r_hand"] = torch.stack([torch.from_numpy(k) for k in aux_data["rhand"]]).to(device)
        labels["r_hand"] = labels["r_hand"].unsqueeze(0)

        data_dict["labels"] = labels

        if len(aux_data["dets"]) == 0:
            data_dict["dets"] = torch.empty((0, 2048)).to(device)
        else:
            data_dict["dets"] = torch.stack(aux_data["dets"]).to(device)
            data_dict["dets"] = data_dict["dets"].reshape(
                [1,
                 data_dict["dets"].shape[0]*data_dict["dets"].shape[1],
                 data_dict["dets"].shape[2]]
            )
        if len(aux_data["bbox"]) == 0:
            data_dict["bbox"] = torch.empty((0, 2048)).to(device)
        else:
            data_dict["bbox"] = torch.stack(aux_data["bbox"]).to(device)
            data_dict["bbox"] = data_dict["bbox"].reshape(
                [1,
                 data_dict["bbox"].shape[0]*data_dict["bbox"].shape[1],
                 data_dict["bbox"].shape[2]]
            )

        out = self.temporal(data_dict)

        return out
