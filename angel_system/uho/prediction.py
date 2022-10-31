import os
import random

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader
import tqdm
from typing import List
from typing import Optional
from typing import Tuple

from angel_system.utils.simple_timer import SimpleTimer
from .src.datamodules.ros_datamodule import get_common_transform, AngelDataset
from .src.models.components.fcn import UnifiedFCNModule
from .src.models.components.transformer import TemTRANSModule
from .aux_data import AuxData

import pdb


# TODO: Pull in from a resource file instead of defining via hard-coding.
UHO_LABELS = ["Background",
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
        rhand = [torch.from_numpy(tmp).to(device) for tmp in t[0]["r_hand"]]

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
                       device: str) -> Tuple[UnifiedFCNModule, TemTRANSModule]:
    """
    Instantiate and return the UHO activity classification model to be given to
    the prediction function.

    :param checkpoint_path: Filepath to the temporal model weights.
    :param device: The device identifier to load the model onto.

    :return: New UHO classifier model instances for (FCN model, temporal
        model).
    """
    # transformer for temporal modeling
    UHO_classifier = TemTRANSModule(27, 256, dropout=0.1, depth=6).to(device)
    # loading checkpoint
    UHO_classifier = load_model(UHO_classifier, checkpoint_path)

    # feature extractor
    fcn = UnifiedFCNModule("resnext").to(device)
    fcn.eval()
    UHO_classifier.eval()

    return fcn, UHO_classifier


def get_uho_classifier_labels() -> List[str]:
    """
    Get the ordered sequence of labels that are index-associative to the class
    indices that are returned from the UHO classifier.

    :returns: List of string labels in index-associative order to the temporal
        model prediction output.
    """
    return UHO_LABELS


def predict(fcn: UnifiedFCNModule,
            temporal: TemTRANSModule,
            frames: List[npt.NDArray],
            aux_data: AuxData,
            fcn_batch_size: Optional[int] = None):
    """
    Predict an activity classification for a single window of image frames and
    auxiliary data.

    `frames` must be of a window length that the given model supports.

    Auxiliary multi-modal data may be sparse, but the lists contained must be
    of the same length as the input frames sequence, as we make index
    association assumptions within.

    :param fcn: Torch FCN model as returned by `get_uho_classifier`.
    :param temporal: Torch temporal model as returned by `get_uho_classifier`.
    :param frames: List of temporally sequential RGB image matrices that
        compose the temporal prediction window. This is expected to be dense,
        i.e. we there is an image for every "slot".
    :param aux_data: Auxiliary multi-modal data for this window.
    :param fcn_batch_size: If an integer value, compute frame descriptors in
        batches of the given size.

    :return: Two tensors, the first of which is the vector of class
        confidences, and the second of which is the index of predicted class
    """
    topk = temporal.det_topk
    n_frames = len(frames)

    # Assuming that all the fcn's parameters are on the same device.
    # Generally a safe assumption...
    device = next(fcn.parameters()).device

    # Input sanity checking.
    n_frames = len(frames)
    assert len(aux_data.lhand) == n_frames, (
        f"Auxiliary left-hand sequence must be the same size as the number of "
        f"frames: (frames) {n_frames} != {len(aux_data.lhand)}"
    )
    assert len(aux_data.rhand) == n_frames, (
        f"Auxiliary right-hand sequence must be the same size as the number of "
        f"frames: (frames) {n_frames} != {len(aux_data.lhand)}"
    )
    assert len(aux_data.scores) == n_frames, (
        f"Auxiliary detection scores sequence must be the same size as the "
        f"number of frames: (frames) {n_frames} != {len(aux_data.lhand)}"
    )
    assert len(aux_data.dets) == n_frames, (
        f"Auxiliary detection descriptors sequence must be the same size as "
        f"the number of frames: (frames) {n_frames} != {len(aux_data.lhand)}"
    )
    assert len(aux_data.bbox) == n_frames, (
        f"Auxiliary detection boxes sequence must be the same size as the "
        f"number of frames: (frames) {n_frames} != {len(aux_data.lhand)}"
    )

    transform = get_common_transform()
    with SimpleTimer("Transform images for FCN feature extraction"):
        frames_tformed = torch.stack([transform(f) for f in frames]).to(device)
    with SimpleTimer("FCN feature extraction"):
        if fcn_batch_size is None or fcn_batch_size >= len(frames):
            feats = fcn(frames_tformed)
        else:
            feats_batched = []
            batch_sections = np.arange(fcn_batch_size, n_frames, fcn_batch_size)
            for frame_batch in np.split(frames_tformed, batch_sections):
                # calculate resnet features
                feats_batched.append(
                    fcn(frame_batch.to(device))
                )
            feats = torch.cat(feats_batched)

    # Collate left/right hand positions into vectors for the network.
    # Known quantity that network needs a 21*3=63 length tensor.
    # Missing hands for a frame should be filled with zero-vectors.
    #
    # Will reject hand pose joints not in OpenPose hand skeleton format.
    # TODO: Invert and set the order of expected joint names? network likely
    #  requires a specific order that is currently just implicitly followed
    #  through from training.
    reject_joint_list = {'ThumbMetacarpalJoint',
                         'IndexMetacarpal',
                         'MiddleMetacarpal',
                         'RingMetacarpal',
                         'PinkyMetacarpal'}
    assert temporal.fc_h.in_features % 2 == 0
    per_hand_dim = temporal.fc_h.in_features // 2
    with SimpleTimer("Composing sparse hand pos tensors"):
        lhand_tensor = torch.zeros(n_frames, per_hand_dim)
        rhand_tensor = torch.zeros(n_frames, per_hand_dim)
        if aux_data.hand_joint_names:
            # Collect indices of non-rejected labels, use that to index into
            # position matrices
            keep_indices = [i
                            for i, joint_label
                            in enumerate(aux_data.hand_joint_names)
                            if joint_label not in reject_joint_list]
            for i, j_pos in enumerate(aux_data.lhand):
                if j_pos is not None:
                    lhand_tensor[i] = torch.from_numpy(j_pos[keep_indices].flatten()).float()
            for i, j_pos in enumerate(aux_data.rhand):
                if j_pos is not None:
                    rhand_tensor[i] = torch.from_numpy(j_pos[keep_indices].flatten()).float()
    with SimpleTimer("Composing data package for temporal network - lhand"):
        data["lhand"] = lhand_tensor.unsqueeze(0).to(device)
    with SimpleTimer("Composing data package for temporal network - rhand"):
        data["rhand"] = rhand_tensor.unsqueeze(0).to(device)

    # Collate detection descriptors and boxes based on top-k detections by max
    # label confidence score.
    # Missing detections for a frame should be filled with zero-vectors.
    #       dets should have shape: [ n_frames*top_k x 2048 ]
    #       bbox should have shape: [ n_frames*top_k x 4 ]
    n_dets_feats = temporal.fc_d.in_features
    n_bbox_feats = temporal.fc_b.in_features
    with SimpleTimer("Composing sparse dets/bbox tensors"):
        dets = torch.zeros(n_frames, topk, n_dets_feats)
        bbox = torch.zeros(n_frames, topk, n_bbox_feats)
        if aux_data.labels:
            # There are labels, so there must be at least one detection
            # Assuming scores/dets/bbox are all the same length for a frame.
            for i, (s, d, b) in enumerate(zip(aux_data.scores, aux_data.dets, aux_data.bbox)):
                if None not in (s, d, b):
                    assert s.shape[0] == d.shape[0] == b.shape[0]
                    # determine indices of top-k determine by det-wise max score
                    s_max = s.max(axis=1).values
                    topk_idx = torch.topk(s_max, topk).indices
                    d_topk = d[topk_idx]
                    b_topk = b[topk_idx]
                    dets[i] = d_topk
                    bbox[i] = b_topk
        # Bring into network expected shapes:
        dets = dets.reshape(n_frames * topk, n_dets_feats)
        bbox = bbox.reshape(n_frames * topk, n_bbox_feats)
    with SimpleTimer("Composing data package for temporal network - dets"):
        data["dets"] = dets.unsqueeze(0).to(device)
    with SimpleTimer("Composing data package for temporal network - bbox"):
        data["bbox"] = bbox.unsqueeze(0).to(device)

    # Compute FCN features for input frames
    transform = get_common_transform()
    with SimpleTimer("Transform images for FCN feature extraction"):
        frames_tformed = torch.stack([transform(f) for f in frames]).to(device)
    with SimpleTimer("FCN feature extraction"):
        if fcn_batch_size is None or fcn_batch_size >= len(frames):
            feats = fcn(frames_tformed)
        else:
            feats_batched = []
            batch_sections = np.arange(fcn_batch_size, n_frames, fcn_batch_size)
            for frame_batch in np.split(frames_tformed, batch_sections):
                # calculate resnet features
                feats_batched.append(
                    fcn(frame_batch.to(device))
                )
            feats = torch.cat(feats_batched)
    with SimpleTimer("Composing data package for temporal network - feats"):
        data["feats"] = feats.unsqueeze(0)  # already on device

    with SimpleTimer("Predicting against temporal network"):
        action_prob, _ = temporal.predict(data)

    return action_prob.squeeze()


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
    dataloader = DataLoader(dataset=data_test, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=False,
                shuffle=False, collate_fn=collate_fn_pad)

    # models
    fcn, uho_classifier = get_uho_classifier(model_path, "cuda")

    # inference
    for frames, aux_data in tqdm.tqdm(dataloader):
        aux_data_struct = AuxData(
            lhand=aux_data["lhand"],
            rhand=aux_data["rhand"],
            dets=aux_data["dets"],
            bbox=aux_data["bbox"],
        )
        action_prob, action_index = predict(fcn, uho_classifier, frames, aux_data_struct)
        action_step = get_uho_classifier_labels(action_index)
        print(action_step)
        pdb.set_trace()
