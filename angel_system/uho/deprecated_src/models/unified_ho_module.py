import json
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
        checkpoint: str,
        labels_file: str,
        device: str = "cuda"
    ):
        super().__init__()
        self.fcn = fcn
        self.temporal = temporal
        self.labels_file = labels_file
        self.device = device

        m = torch.load(checkpoint)
        self.load_state_dict(m)

        # data transformations (Normalization values recommended by
        # torchvision model zoo)
        self.fcn_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.labels = []
        with open(self.labels_file, "r") as f:
            for line in f:
                self.labels.append(line.rstrip())

    def forward(self, frame_data, aux_data):
        """
        Data format:
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
        """
        # Preprocess frames
        frames = [self.fcn_transform(f) for f in frame_data]
        frame_tensor = torch.stack(frames)
        frame_tensor = frame_tensor.to(device=torch.device(self.device))

        # Extract frame features
        frame_feats = self.fcn(frame_tensor)

        '''
        test_tensor = torch.arange(8 * 3 * 720 * 1280).reshape((8, 3, 720, 1280)).type(torch.FloatTensor).to(device=torch.device("cuda"))
        print(test_tensor)
        results = self.fcn(test_tensor)
        print(f"forward {results} {results[0].shape}")
        '''

        # Preprocess aux data
        data_dict = {}
        data_dict["feats"] = frame_feats.unsqueeze(0)

        labels = {"l_hand": [], "r_hand": []}

        labels["l_hand"] = torch.stack([torch.from_numpy(k) for k in aux_data["lhand"]]).to(self.device)
        labels["l_hand"] = labels["l_hand"].unsqueeze(0)
        labels["r_hand"] = torch.stack([torch.from_numpy(k) for k in aux_data["rhand"]]).to(self.device)
        labels["r_hand"] = labels["r_hand"].unsqueeze(0)

        data_dict["labels"] = labels

        if len(aux_data["dets"]) == 0:
            data_dict["dets"] = torch.empty((0, 2048)).to(self.device)
        else:
            data_dict["dets"] = torch.cat(aux_data["dets"]).to(self.device)
            data_dict["dets"] = data_dict["dets"].reshape(
                [1,
                 data_dict["dets"].shape[0],
                 data_dict["dets"].shape[1]]
            )
        if len(aux_data["bbox"]) == 0:
            data_dict["bbox"] = torch.empty((0, 2048)).to(self.device)
        else:
            data_dict["bbox"] = torch.cat(aux_data["bbox"]).to(self.device)
            data_dict["bbox"] = data_dict["bbox"].reshape(
                [1,
                 data_dict["bbox"].shape[0],
                 data_dict["bbox"].shape[1]]
            )

        '''
        hand_tensor = torch.arange(1 * 8 * 63).reshape((1, 8, 63)).type(torch.FloatTensor).to(device=torch.device("cuda"))

        dets_tensor = torch.arange(40 * 2048).reshape((1, 40, 2048)).type(torch.FloatTensor).to(device=torch.device("cuda"))
        bbox_tensor = torch.arange(40 * 4).reshape((1, 40, 4)).type(torch.FloatTensor).to(device=torch.device("cuda"))

        temp_data = {}
        temp_data["feats"] = results.unsqueeze(0)
        temp_data["act"] = torch.tensor(0)
        temp_data["idx"] = torch.tensor(0)
        temp_data["labels"] = dict(l_hand=hand_tensor, r_hand=hand_tensor)
        temp_data["dets"] = dets_tensor
        temp_data["bbox"] = bbox_tensor

        out = self.temporal(temp_data)
        print(out)

        data_dict_json = {}
        for key, item in data_dict.items():
            if key != "labels":
                print(key)
                data_dict_json[key] = item.cpu().numpy().tolist()
            else:
                data_dict_json[key] = dict(
                    l_hand=item["l_hand"].cpu().numpy().tolist(),
                    r_hand=item["r_hand"].cpu().numpy().tolist()
                )
        print(data_dict_json)

        with open("batch.json", "w") as f:
            json.dump(data_dict_json, f)
        '''
        '''
        with open("batch_hydra.json", "r") as f:
            batch_data = json.load(f)

        new_batch_data = {}
        for key, value in batch_data.items():
            print(key)
            if key == "labels":
                new_batch_data[key] = dict(
                    l_hand=torch.Tensor(np.array(value["l_hand"])).to(device=torch.device("cuda"))[0].unsqueeze(0),
                    r_hand=torch.Tensor(np.array(value["r_hand"])).to(device=torch.device("cuda"))[0].unsqueeze(0)
                )
            else:
                new_batch_data[key] = torch.Tensor(np.array(value)).to(device=torch.device("cuda"))[0].unsqueeze(0)
        '''

        out = self.temporal(data_dict)

        return out, self.labels
