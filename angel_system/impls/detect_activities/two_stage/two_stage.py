import pdb

import torch
from torch import nn

from .spatial.fcn import SpatialFCNModule
from .temporal.rulstm import RULSTM


class TwoStageModule(nn.Module):
    """This class implements the spatio-temporal model used for unified
    representation of multi-modal inputs in the scene. This model also
    performs the activity recognition for the given frame sequence.

    :params checkpoint: Path to the checkpoint file to be loaded
    :params num_classes: Number of activity classes in the data
    """

    def __init__(self, checkpoint: str, num_classes: int):
        super().__init__()

        # FCN: input image (3, 1280, 720) -> Output feature from ResNext (2048)
        self.fcn = SpatialFCNModule("resnext")
        # Temporal: (Dict) Input feature from ResNext (2048) + Aux_data ->
        # Output (1, num_classes)
        self.temporal = RULSTM(num_classes, hidden=128, dropout=0, depth=3)
        self.fcn.eval()

        # Load checkpoint
        # Expecting a state dict including only the temporal layer weights
        self.temporal.load_state_dict(torch.load(checkpoint))
        self.temporal.eval()

    def forward(self, data, aux):
        frame_feats = self.fcn(data)
        # pdb.set_trace()
        out = self.temporal(frame_feats.unsqueeze(1), aux)

        return out
