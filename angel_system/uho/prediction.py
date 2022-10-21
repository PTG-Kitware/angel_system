import os
import random
from typing import List

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import tqdm

from .src.datamodules.ros_datamodule import ROSVideoDataset
from .src.models.components.fcn import UnifiedFCNModule
from .src.models.components.transformer import TemTRANSModule
from .aux_data import AuxData


# prepare input data
def collate_fn_pad(batch):
    """Padds batch of variable length.

    :params batch (Dict): Batch data with labels + auxiliary data from
        dataloader

    :returns batch (Dict): Processed batch with labels + auxiliary data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dic = {}

    labels = {"l_hand": [], "r_hand": []}
    for t in batch:
        labels["l_hand"].append(torch.cat([k["l_hand"] for k in t[0]["labels"]]).to(device))
        labels["r_hand"].append(torch.cat([k["r_hand"] for k in t[0]["labels"]]).to(device))
    labels["l_hand"] = torch.stack(labels["l_hand"])
    labels["r_hand"] = torch.stack(labels["r_hand"])
    data_dic["labels"] = labels

    dets = []
    bbox = []
    frms = []
    topK = 10
    for t in batch:
        # collect detections
        det1 = [torch.from_numpy(tmp[:topK]) for tmp in t[0]["dets"]]
        det1 = torch.stack(det1)
        det1 = det1.reshape([det1.shape[0]*det1.shape[1],1,det1.shape[2]])
        dets.append(det1.to(device))
        # collect detection bboxes
        bbox1 = [torch.from_numpy(tmp[:topK]) for tmp in t[0]["bbox"]]
        bbox1 = torch.stack(bbox1)
        bbox1 = bbox1.reshape([bbox1.shape[0]*bbox1.shape[1],1,bbox1.shape[2]])
        bbox.append(bbox1.to(device))
        # collect frames
        frms1 = [tmp for tmp in t[0]["frm"]]
        frms1 = torch.stack(frms1)
        frms.append(frms1.to(device))

    data_dic["dets"] = torch.stack(dets).squeeze()
    data_dic["bbox"] = torch.stack(bbox).squeeze()
    data_dic["frm"] = torch.stack(frms).squeeze()

    batch = data_dic

    return batch


def load_model(model: TemTRANSModule, checkpoint_path: str) -> TemTRANSModule:
    """
    Load a checkpoint file's state into the given TemTRANSModule instance.
    :param model:
    :param checkpoint_path:
    :return:
    """
    print("loading checkpoint at "+checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    own_state = model.state_dict()
    for k, v in checkpoint['state_dict'].items():
        key = k.replace("temporal.","")
        if key in own_state.keys():
            own_state[key].copy_(v)

    model.load_state_dict(own_state)

    return model


def get_uho_classifier(checkpoint_path: str,
                       device: str) -> ...:
    """
    Instantiate and return the UHO activity classification model to be given to
    the prediction function.

    :param checkpoint_path: Filepath to the temporal model weights.
    :param device: The device identifier to load the model onto.

    :return: New UHO classifier model instance.
    """
    # TODO: Implement such that we return the instantiated ALL models
    #       required to perform activity classification, and any tightly
    #       associated structures/metadata (e.g. image normalization
    #       transform).


def get_uho_classifier_labels() -> List[str]:
    """
    Get the ordered sequence of labels that are index-associative to the class
    indices that are returned from the UHO classifier.
    """
    # TODO: Return the semantic labels that the classifier model was trained
    #       on.


def predict(model: ...,
            frames: List[npt.NDArray],
            aux_data: AuxData):
    """
    Predict an activity classification for a single window of image frames and
    auxiliary data.

    `frames` must be of a window length that the given model supports.

    :param model:
    :param frames:
    :param aux_data:
    :return: Two tensors, the first of which is the vector of class
        confidences, and the second of which is ...
        TODO: fill in what this second component is.
    """
    # TODO: Implement everything required to transform images and aux-data into
    #       activity classification confidences, including use of the fcn,
    #       image normalization, etc.


if __name__ == "__main__":
    # paths and hyper-parameters
    root_path = "/data/dawei.du/datasets/ROS/Data" # dataset path
    action_set = "all_activities_action_val.txt" # path of video clips
    model_path = "checkpoints/PTG_transformer.ckpt" # path of checkpoint, which can be found in Kitware data
    test_list = os.path.join(root_path, "label_split", action_set)
    batch_size = 1 # we deal with 1 video clip in inference
    num_workers = 0 # use all workers to deal with 32 frames per video clip
    topK = 10 # we extract top 10 detections for each frame
    torch.manual_seed(25)
    random.seed(25)
    # dataloader
    trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data_test = ROSVideoDataset(root_path, test_list, frames_per_segment=32, transform=trans, test_mode=True)
    dataloader = DataLoader(dataset=data_test, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
                shuffle=False, collate_fn=collate_fn_pad)

    # models
    fcn = UnifiedFCNModule("resnext").cuda() # feature extractor
    temporal = TemTRANSModule(27, 256, dropout=0.1, depth=6).cuda() # transformer for temporal modeling
    temporal = load_model(temporal, model_path) # loading checkpoint
    fcn.eval()
    temporal.eval()

    # inference
    for data in tqdm.tqdm(dataloader):
        # calculate resnet features
        feats = fcn(data)

        # simulation: select detections at arbitrary 4~8 frames
        max_sample_fr = 8
        min_sample_fr = 4
        num_det_fr = np.random.randint(max_sample_fr-min_sample_fr+1)+min_sample_fr
        all_fr = np.random.permutation(feats.shape[0])
        valid_fr = np.sort(all_fr[:num_det_fr])
        bbox = data["bbox"]
        # zero out bbox at invalid frames
        for k in all_fr[num_det_fr:]:
            bbox[k*topK:(k+1)*topK,:] = 0

        # predict actions
        data["feats"] = feats.unsqueeze(0)
        data["dets"] = data["dets"].unsqueeze(0)
        data["bbox"] = bbox.unsqueeze(0)
        action_prob, action_pred = temporal.predict(data)
        #print(action_prob, action_pred)
