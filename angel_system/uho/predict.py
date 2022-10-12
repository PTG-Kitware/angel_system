from dataclasses import asdict
from dataclasses import dataclass
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import torch

from angel_system.uho.src.data_helper import AuxData
from angel_system.uho.src.models.components.transformer import TemTRANSModule
from angel_system.uho.src.models.components.unified_fcn import UnifiedFCNModule
from angel_system.uho.src.models.unified_ho_module import UnifiedHOModule


def get_uho_detector(
    checkpoint_path: str,
    labels_path: str,
    device: str,
    net: str = "resnext", # Defaults from rulstm config
    num_cpts: int = 21,
    obj_classes: int = 9,
    verb_classes: int = 12,
    act_classes: int = 27,
    hidden: int = 256,
    dropout: float = 0.1,
    depth: int = 6,
) -> UnifiedHOModule:
    """
    Instantiates and returns a UHO module with a two-stage UnifiedFCNModule
    and a TemTRANSModule. The returned module should be used for evalation
    only.
    """
    fcn = UnifiedFCNModule(
        net=net,
        num_cpts=num_cpts,
        obj_classes=obj_classes,
        verb_classes=verb_classes
    )
    temporal = TemTRANSModule(
        act_classes=act_classes,
        hidden=hidden,
        dropout=dropout,
        depth=depth
    )

    detector: UnifiedHOModule = UnifiedHOModule(
        fcn=fcn,
        temporal=temporal,
        checkpoint=checkpoint_path,
        device=device,
        labels_file=labels_path
    )
    detector.eval()
    return detector.to(device=device)


def predict(
    model: UnifiedHOModule,
    frames: List[np.ndarray],
    aux_data: AuxData
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], List[str]]:
    """
    Returns the result of the forward call for the UHO model.

    Returns:
        A tuple consisting of two elements:
            0: Detection results: A tuple consisting of a [1 x n_classes] tensor
               representing the detector's confidence for each class and a tensor
               containing the index of the max confidence in the confidence tensor.
            1: A list of length n_classes for mapping the detector confidence
               indices to class strings.
    """
    # Convert aux_data class to dict as that is currently required
    # by the UHO module transformer
    return model.forward(frames, asdict(aux_data))
