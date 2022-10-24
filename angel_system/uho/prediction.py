import tqdm
import sys
import os
import random
import torch
import numpy as np
from typing import List
import numpy.typing as npt
from torch.utils.data import DataLoader
from src.datamodules.ros_datamodule import AngelDataset
from src.models.components.fcn import UnifiedFCNModule
from src.models.components.transformer import TemTRANSModule
from aux_data import AuxData
import pdb

# prepare input data
def collate_fn_pad(batch):
    """Padds batch of variable length.

    :params batch (Dict): Batch data with labels + auxiliary data from
        dataloader

    :returns batch (Dict): Processed batch with labels + auxiliary data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aux_data = {}

    lhand, rhand = [], []
    dets, bbox, frms = [], [], []
    topK = 10
    for t in batch:
        lhand = [torch.from_numpy(tmp).to(device) for tmp in t[0]["l_hand"]]
        rhand = [torch.from_numpy(tmp).to(device) for tmp in t[0]["l_hand"]]

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

    aux_data["lhand"] = torch.stack(lhand)
    aux_data["rhand"] = torch.stack(rhand)
    aux_data["dets"] = torch.stack(dets).squeeze()
    aux_data["bbox"] = torch.stack(bbox).squeeze()

    frames = torch.stack(frms).squeeze()

    return frames, aux_data

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
    UHO_classifier = TemTRANSModule(27, 256, dropout=0.1, depth=6).to(device) # transformer for temporal modeling
    UHO_classifier = load_model(UHO_classifier, model_path) # loading checkpoint

    fcn = UnifiedFCNModule("resnext").cuda() # feature extractor
    fcn.eval()
    UHO_classifier.eval()

    return fcn, UHO_classifier

def get_uho_classifier_labels(index: int) -> List[str]:
    """
    Get the ordered sequence of labels that are index-associative to the class
    indices that are returned from the UHO classifier.
    """
    # TODO: Return the semantic labels that the classifier model was trained
    #       on.

    uho_labels = ["Background",
                   "Measure 12 ounces of water in the liquid measuring cup",
                   "Pour the water from the liquid measuring cup into the electric kettle",
                   "Turn on the Kettle",
                   "Place the dripper on top of the mug",
                   "Take the coffee filter and fold it in half to create a semi-circle",
                   "Fold the filter in half again to create a quarter-circle",
                   "Place the folded filter into the dripper such that the the point of the quarter-circle rests in the center of the dripper",
                   "Spread the filter open to create a cone inside the dripper",
                   "Turn on the kitchen scale",
                   "Place a bowl on the scale",
                   "Zero the scale",
                   "Add coffee beans to the bowl until the scale reads 25 grams",
                   "Pour the measured coffee beans into the coffee grinder",
                   "Set timer for 20 seconds",
                   "Turn on the timer",
                   "Grind the coffee beans by pressing and holding down on the black part of the lid",
                   "Pour the grounded coffee beans into the filter cone prepared in step 2",
                   "Turn on the thermometer",
                   "Place the end of the thermometer into the water",
                   "Set timer to 30 seconds",
                   "Pour a small amount of water over the grounds in order to wet the grounds",
                   "Slowly pour the water over the grounds in a circular motion. Do not overfill beyond the top of the paper filter",
                   "Allow the rest of the water in the dripper to drain",
                   "Remove the dripper from the cup",
                   "Remove the coffee grounds and paper filter from the dripper", 
                   "Discard the coffee grounds and paper filter"] 

    return uho_labels[index]


def predict(fcn: ...,
            temporal: ...,
            frames: List[npt.NDArray],
            aux_data: AuxData):
    """
    Predict an activity classification for a single window of image frames and
    auxiliary data.

    `frames` must be of a window length that the given model supports.

    :param fcn:
    :param temporal:
    :param frames:
    :param aux_data:
    :return: Two tensors, the first of which is the vector of class
        confidences, and the second of which is the index of predicted class
    """
    # TODO: Implement everything required to transform images and aux-data into
    #       activity classification confidences, including use of the fcn,
    #       image normalization, etc.

    # calculate resnet features
    feats = fcn(frames)
 
    # simulation: select detections at arbitrary 4~8 frames
    max_sample_fr = 8
    min_sample_fr = 4
    num_det_fr = np.random.randint(max_sample_fr-min_sample_fr+1)+min_sample_fr
    all_fr = np.random.permutation(feats.shape[0])
    valid_fr = np.sort(all_fr[:num_det_fr])
    bbox = aux_data["bbox"]
    # zero out bbox at invalid frames
    for k in all_fr[num_det_fr:]:
        bbox[k*topK:(k+1)*topK,:] = 0

    # predict actions
    data = {}
    data["feats"] = feats.unsqueeze(0)
    data["lhand"], data["rhand"] = aux_data["lhand"].unsqueeze(0), aux_data["rhand"].unsqueeze(0)
    data["dets"] = aux_data["dets"].unsqueeze(0)
    data["bbox"] = bbox.unsqueeze(0)
    action_prob, action_index = temporal.predict(data)

    return action_prob, action_index

if __name__ == "__main__":
    # paths and hyper-parameters
    root_path = "/data/dawei.du/datasets/ROS/Data" # dataset path
    action_set = "all_activities_action_val.txt" # path of video clips
    model_path = "checkpoints/epoch_018.ckpt" # path of checkpoint, which can be found in Kitware data
    test_list = os.path.join(root_path, "label_split", action_set)

    batch_size = 1 # we deal with 1 video clip in inference
    num_workers = 0 # use all workers to deal with 32 frames per video clip
    topK = 10 # we extract top 10 detections for each frame
    torch.manual_seed(25)
    random.seed(25)

    # dataloader
    data_test = AngelDataset(root_path, test_list)
    dataloader = DataLoader(dataset=data_test, batch_size=batch_size, num_workers=num_workers, pin_memory=False,
                shuffle=False, collate_fn=collate_fn_pad)

    # models
    fcn, uho_classifier = get_uho_classifier(model_path, "cuda")

    # inference
    for frames, aux_data in tqdm.tqdm(dataloader):
        action_prob, action_index = predict(fcn, uho_classifier, frames, aux_data)
        action_step = get_uho_classifier_labels(action_index)
        print(action_step)
        pdb.set_trace()
