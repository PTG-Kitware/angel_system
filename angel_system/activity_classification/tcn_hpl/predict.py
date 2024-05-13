from dataclasses import dataclass
from pathlib import Path
import os
from threading import RLock
from typing import List
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

import kwcoco
import numpy as np
import numpy.typing as npt
import torch

from tcn_hpl.data.components.augmentations import NormalizePixelPts, NormalizeFromCenter
from tcn_hpl.models.ptg_module import PTGLitModule

from angel_system.activity_classification.utils import (
    tlbr_to_xywh,
    obj_det2d_set_to_feature,
)


def load_module(
    checkpoint_file, label_mapping_file, torch_device, topic
) -> PTGLitModule:
    """

    :param checkpoint_file:
    :param label_mapping_file:
    :param torch_device:
    :param topic:
    :return:
    """
    # # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # lightning.fabric.utilities.seed.seed_everything(12345)

    mapping_file_dir = os.path.abspath(os.path.dirname(label_mapping_file))
    mapping_file_name = os.path.basename(label_mapping_file)
    model_device = torch.device(torch_device)
    model = PTGLitModule.load_from_checkpoint(
        checkpoint_file,
        map_location=model_device,
        # HParam overrides
        data_dir=mapping_file_dir,
        mapping_file_name=mapping_file_name,
        topic=topic,
    )

    # print(f"CLASSES IN MODEL: {model.classes}")
    # print(f"class_ids IN MODEL: {model.class_ids}")

    return model


@dataclass
class ObjectDetectionsLTRB:
    """
    Expected object detections format for a single frame from the ROS2
    ecosystem.
    """

    # Identifier for this set of detections.
    id: int
    # Vectorized detection bbox left pixel bounds
    left: Tuple[float]
    # Vectorized detection bbox top pixel bounds
    top: Tuple[float]
    # Vectorized detection bbox right pixel bounds
    right: Tuple[float]
    # Vectorized detection bbox bottom pixel bounds
    bottom: Tuple[float]
    # Vectorized detection label of the most confident class.
    labels: Tuple[str]
    # Vectorized detection confidence value of the most confidence class.
    confidences: Tuple[float]


@dataclass
class PatientPose:
    # Identifier for this set of detections.
    id: int
    # Vectorized keypoints
    positions: list
    # Vectorized orientations
    # orientations: list
    # Vectorized keypoint label
    labels: str


def normalize_detection_features(
    det_feats: npt.ArrayLike,
    feat_version: int,
    top_k_objects: int,
    img_width: int,
    img_height: int,
    num_det_classes: int,
    normalize_pixel_pts: bool,
    normalize_center_pts: bool,
) -> None:
    """
    Normalize input object detection descriptor vectors, outputting new vectors
    of the same shape.

    Expecting input `det_feats` to be in the shape `[window_size, num_feats]'.

    NOTE: This method normalizes in-place, so be sure to clone the input array
    if that is not desired.

    :param det_feats: Object Detection features to be normalized.

    :return: Normalized object detection features.
    """
    if normalize_pixel_pts:
        # This method is known to normalize in-place.
        # Shape [window_size, n_feats]
        NormalizePixelPts(
            img_width, img_height, num_det_classes, feat_version, top_k_objects
        )(det_feats)
    if normalize_center_pts:
        NormalizeFromCenter(
            img_width, img_height, num_det_classes, feat_version, top_k_objects
        )(det_feats)


def objects_to_feats(
    frame_object_detections: Sequence[Optional[ObjectDetectionsLTRB]],
    frame_patient_poses: Sequence[Optional[PatientPose]],
    det_label_to_idx: Dict[str, int],
    feat_version: int,
    image_width: int,
    image_height: int,
    feature_memo: Optional[Dict[int, npt.NDArray]] = None,
    top_k_objects: int = 1,
    normalize_pixel_pts=False,
    normalize_center_pts=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert some object detections for some window of frames into a feature
    vector of version requested.

    :param frame_object_detections: Sequence of object detections for some
        window of frames. The window size is dictated by this length of this
        sequence. Some frame "slots" may be None to indicate there were no
        object detections for that frame.
    :param det_label_to_idx: Mapping of object detector classes to the
        activity-classifier input index expectation.
    :param feat_version: Integer version of the feature vector to generate.
        See the `obj_det2d_set_to_feature` function for details.
    :param image_width: Integer pixel width of the image that object detections
        were generated on.
    :param image_height: Integer pixel height of the image that object
        detections were generated on.
    :param feature_memo: Optional memoization cache to given us that we will
        access and insert into based on the IDs given to `ObjectDetectionsLTRB`
        instances encountered.

    :raises ValueError: No object detections nor patient poses passed in.
    :raises ValueError: No non-None object detections in the given input
        window.
    :raises ValueError: No non-None patient poses in the given input
        window.

    :return: Window of normalized feature vectors for the given object
        detections (shape=[window_size, n_feats]), and an appropriate mask
        vector for use with activity classification (shape=[window_size]).
    """
    if 0 in [len(frame_patient_poses), len(frame_object_detections)]:
        raise ValueError(
            "Need at least one patient pose or object det in input sequences"
        )
    if all([d is None for d in frame_object_detections]):
        raise ValueError("No frames with detections in input.")
    if all([p is None for p in frame_patient_poses]):
        raise ValueError("No frames with patient poses in input.")

    feat_memo = {} if feature_memo is None else feature_memo

    window_size = len(frame_object_detections)

    # Shape [window_size, None|n_feats]
    feature_list: List[Optional[npt.NDArray]] = [None] * window_size
    feature_ndim = None
    feature_dtype = None

    # hands-joints offset vectors
    zero_joint_offset = [0 for i in range(22)]

    # for pose in frame_patient_poses:
    for i, (pose, detections) in enumerate(
        zip(frame_patient_poses, frame_object_detections)
    ):
        pose_keypoints = []
        print(pose)
        if detections is None:
            continue

        detection_id = detections.id
        confidences = detections.confidences
        if detection_id in feat_memo.keys():
            # We've already processed this set
            feat = feat_memo[detection_id]
        else:
            labels = detections.labels
            xs, ys, ws, hs = tlbr_to_xywh(
                detections.top,
                detections.left,
                detections.bottom,
                detections.right,
            )

            if pose is not None:
                for joint in pose:
                    kwcoco_format_joint = {
                        "xy": [joint.positions.x, joint.positions.y],
                        "keypoint_category_id": -1,  # TODO: not in message
                        "keypoint_category": joint.labels,
                    }
                    pose_keypoints.append(kwcoco_format_joint)

            feat = (
                obj_det2d_set_to_feature(
                    labels,
                    xs,
                    ys,
                    ws,
                    hs,
                    confidences,
                    pose_keypoints=(
                        pose_keypoints if pose_keypoints else zero_joint_offset
                    ),
                    obj_label_to_ind=det_label_to_idx,
                    version=feat_version,
                    top_k_objects=top_k_objects,
                )
                .ravel()
                .astype(np.float32)
            )

            feat_memo[detection_id] = feat

        feature_ndim = feat.shape
        feature_dtype = feat.dtype
        feature_list[i] = feat
    # Already checked that we should have non-zero frames with detections above
    # so feature_ndim/_dtype should not be None at this stage
    assert feature_ndim is not None
    assert feature_dtype is not None

    # Create mask vector, which should indicate which window indices should not
    # be considered.
    # NOTE: The expected network is not yet trained to do this, so the mask is
    #       always 1's right now.
    # Shape [window_size]
    mask = torch.ones(window_size)

    # Fill in the canonical "empty" feature vector for those frames that had no
    # detections.
    empty_vec = np.zeros(shape=feature_ndim, dtype=feature_dtype)
    for i in range(window_size):
        if feature_list[i] is None:
            feature_list[i] = empty_vec

    # Shape [window_size, n_feats]
    feature_vec = torch.tensor(feature_list)

    # Normalize features
    # Shape [window_size, n_feats]
    if normalize_pixel_pts or normalize_center_pts:
        normalize_detection_features(
            feature_vec,
            feat_version,
            top_k_objects,
            image_width,
            image_height,
            len(det_label_to_idx),
            normalize_pixel_pts,
            normalize_center_pts,
        )

    return feature_vec, mask


def predict(
    model: PTGLitModule,
    window_feats: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Compute model activity classifications, returning a tensor of softmax
    probabilities.

    We assume the input model and tensors are already on the appropriate
    device.

    We assume that input features normalized before being provided to this
    function. See :ref:`normalize_detection_features`.

    The "prediction" of this result can be determined via the `argmax`
    function::

        proba = predict(model, window_feats, mask)
        pred = torch.argmax(proba)

    :param model: PTGLitModule instance to use.
    :param window_feats: Window (sequence) of *normalized* object detection
        features. Shape: [window_size, feat_dim].
    :param mask: Boolean array indicating which frames of the input window for
        the network to consider. Shape: [window_size].

    :return: Probabilities (softmax) of the activity classes.
    """
    x = window_feats.T.unsqueeze(0).float()
    m = mask[None, :]
    # print(f"window_feats: {x.shape}")
    # print(f"mask: {m.shape}")
    with torch.no_grad():
        logits = model(x, m)
    # Logits access mirrors model step function argmax access here:
    #   tcn_hpl.models.ptg_module --> PTGLitModule.model_step
    # ¯\_(ツ)_/¯
    return torch.softmax(logits[-1, :, :, -1], dim=1)[0]


@dataclass
class ResultElement:
    """
    Single activity classification result element for later inclusion in an
    output COCO file.
    """

    # Class index of the single most confident activity class.
    pred_index: int
    # Vector of confidence values for all activity classes.
    conf_vec: List[float]


class ResultsCollector:
    """
    Utility class to collect TCN-HPL prediction results, encoding the ability
    to output the results to a COCO file.

    This is a beginning to generalizing results coco file aggregation and
    output, based on output from `python-tpl/TCN_HPL/tcn_hpl/models/ptg_module.py`
    in `on_validation_epoch_end` and `on_test_epoch_end` methods.

    TODO: Promote into TCN_HPL repo?
    """

    def __init__(self, output_filepath: Path, id_to_action: Mapping[int, str]):
        self._lock = RLock()  # for thread safety
        self._collection: Dict[int, ResultElement] = {}
        self._dset = dset = kwcoco.CocoDataset()
        dset.fpath = output_filepath.as_posix()
        dset.dataset["info"].append({"activity_labels": id_to_action})
        self._vid: Optional[int] = None

    def set_video(self, video_name: str) -> None:
        """
        Set the video for which we are currently collecting results for.

        :param video_name: Semantic name of the video   .
        """
        with self._lock:
            video_lookup = self._dset.index.name_to_video
            if video_name in video_lookup:
                self._vid = video_lookup[video_name]["id"]
            else:
                self._vid = self._dset.add_video(name=video_name)

    def collect(
        self,
        frame_index: int,
        activity_pred: int,
        activity_conf_vec: Sequence[float],
        name: Optional[str] = None,
        file_name: Optional[str] = None,
        activity_gt: Optional[int] = None,
    ) -> None:
        """
        See `CocoDataset.add_image` for more details.
        """
        with self._lock:
            if self._vid is None:
                raise RuntimeError(
                    "No video set before results collection. See `set_video` method."
                )
            packet = dict(
                video_id=self._vid,
                frame_index=frame_index,
                activity_pred=activity_pred,
                activity_conf=list(activity_conf_vec),
            )
            if name is not None:
                packet["name"] = name
            if file_name is not None:
                packet["file_name"] = file_name
            if activity_gt is not None:
                packet["activity_gt"] = activity_gt
            self._dset.add_image(**packet)

    def write_file(self):
        """
        Write COCO file to the set output path.
        """
        with self._lock:
            dset = self._dset
            dset.dump(dset.fpath, newlines=True)


###############################################################################
# Functions for debugging things in an interpreter
#
def windows_from_all_feature(
    all_features: npt.ArrayLike, window_size: int
) -> npt.ArrayLike:
    """
    Iterate over overlapping windows in the frame detections features given.

    :param all_features: All object detection feature vectors for all frames to
        consider. Shape: [n_frames, n_feats]
    :param window_size: Size of the window to slide.

    :return: Generator yielding different windows of feature vectors.
    """
    i = 0
    stride = 1
    while (i + window_size) < np.shape(all_features)[0]:
        yield all_features[i : (i + window_size), :]
        i += stride


def debug_from_array_file() -> None:
    import functools
    import re
    import numpy as np
    import torch
    from tqdm import tqdm
    from tcn_hpl.data.components.augmentations import NormalizePixelPts
    from angel_system.tcn_hpl.predict import (
        load_module,
        predict,
        windows_from_all_feature,
    )

    # Pre-computed, un-normalized features per-frame extracted from the
    # training harness, in temporally ascending order.
    # Shape = [n_frames, n_feats]
    all_features = torch.tensor(
        np.load("./model_files/all_activities_20.npy").astype(np.float32).T
    ).to("cuda")

    model = load_module(
        "./model_files/activity_tcn-coffee-checkpoint.ckpt",
        "./model_files/activity_tcn-coffee-mapping.txt",
        "cuda",
    ).eval()

    # Above model window size = 30
    mask = torch.ones(30).to("cuda")

    # Normalize features
    # The `objects_to_feats` above includes normalization along with the
    # bounding box conversion, so this needs to be applied explicitly outside
    # using `objects_to_feats` (though, using the same normalize func).
    norm_func = functools.partial(
        normalize_detection_features,
        feat_version=5,
        img_width=1280,
        img_height=720,
        num_det_classes=42,
    )

    # Shape [n_windows, window_size, n_feats]
    all_windows = list(windows_from_all_feature(all_features, 30))

    all_proba = list(
        tqdm(
            (predict(model, norm_func(w.clone()), mask) for w in all_windows),
            total=len(all_windows),
        )
    )

    all_preds_idx = np.asarray([int(torch.argmax(p)) for p in all_proba])
    all_preds_lbl = [model.classes[p] for p in all_preds_idx]

    # Load Hannah preds
    comparison_preds_file = "./model_files/all_activities_20_preds.txt"
    re_window_pred = re.compile(r"^gt: (\d+), pred: (\d+)$")
    comparison_gt = []
    comparison_preds_idx = []
    with open(comparison_preds_file) as infile:
        for l in infile.readlines():
            m = re_window_pred.match(l.strip())
            comparison_gt.append(int(m.groups()[0]))
            comparison_preds_idx.append(int(m.groups()[1]))
    comparison_preds_idx = np.asarray(comparison_preds_idx)

    ne_mask = all_preds_idx != comparison_preds_idx
    all_preds_idx[ne_mask], comparison_preds_idx[ne_mask]
