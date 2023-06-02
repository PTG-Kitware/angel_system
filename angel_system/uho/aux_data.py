from dataclasses import dataclass
from typing import Optional
from typing import Sequence

import numpy.typing as npt


@dataclass
class AuxData:
    """
    Class for representing the aux data dictionary required by the UHO module.

    It is expected that all sequence attributes are of the same length.

    Attributes:
        hand_joint_names: Sequence of string labels of joints. This applies to
            both `lhand` and `rhand` position matrices when not None. This may
            be empty of there are no hand joints in this structure.
        lhand: Sequence of arrays of shape [nJoints x 3] representing the joint
            poses of the left hand. The combined shape is expected to be
            [nImages x nJoints x 3].
        rhand: Sequence of arrays of shape [nJoints x 3] representing the joint
            poses of the right hand. The combined shape is expected to be
            [nImages x nJoints x 63].
        labels: Sequence of detect class score labels. This should be non-empty
            if there are any detections in this structure.
        scores: Sequence of detection scores of shape [N x L], where N is the
            number of detections and L is the number of classes. The combined
            shape is expected to be [nImages x N x L]
        dets: Sequence of arrays of shape [N x D], where N is the number
            of detections and D is the size of detection's descriptor vector.
            The combined shape is expected to be [nImages x N x D].
        bbox: Sequence of arrays of shape [N x 4], where N is the number
            of detections. The combined shape is expected to be
            [nImages x N x 4].
    """

    # Hand stuff
    hand_joint_names: Sequence[str]
    lhand: Sequence[Optional[npt.ArrayLike]]
    rhand: Sequence[Optional[npt.ArrayLike]]
    # Detection stuff
    labels: Sequence[str]
    scores: Sequence[Optional[npt.NDArray]]
    dets: Sequence[Optional[npt.ArrayLike]]
    bbox: Sequence[Optional[npt.ArrayLike]]
