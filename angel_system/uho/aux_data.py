from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class AuxData:
    """
    Class for representing the aux data dictionary required by the UHO module.

    It is expected that all list attributes are of the same length.

    Attributes:
        lhand: List of np.ndarrays of size (63,) representing the 63 joint poses
            of the left hand.
        rhand: List of np.ndarrays of size (63,) representing the 63 joint poses
            of the right hand.
        dets: List of torch.Tensors of size [N x D], where N is the number of
            detections and D is the size of detection's descriptor vector.
        bbox: List of torch.Tensors of size [N x 4], where N is the number of
            detections.

    """
    lhand: List[np.ndarray]
    rhand: List[np.ndarray]
    dets: List[torch.Tensor]
    bbox: List[torch.Tensor]
