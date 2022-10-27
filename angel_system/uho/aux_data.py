from dataclasses import dataclass
from typing import Sequence

import numpy.typing as npt


@dataclass
class AuxData:
    """
    Class for representing the aux data dictionary required by the UHO module.

    It is expected that all sequence attributes are of the same length.

    Attributes:
        lhand: Sequence of arrays of shape [63] representing the 63 joint
            poses of the left hand. The combined shape is expected to be
            [nImages x 63].
        rhand: Sequence of arrays of shape [63] representing the 63 joint
            poses of the right hand. The combined shape is expected to be
            [nImages x 63].
        dets: Sequence of arrays of shape [N x D], where N is the number
            of detections and D is the size of detection's descriptor vector.
            The combined shape is expected to be [nImages x N x D].
        bbox: Sequence of arrays of shape [N x 4], where N is the number
            of detections. The combined shape is expected to be
            [nImages x N x 4].
    """
    lhand: Sequence[npt.ArrayLike]
    rhand: Sequence[npt.ArrayLike]
    dets: Sequence[npt.ArrayLike]
    bbox: Sequence[npt.ArrayLike]
