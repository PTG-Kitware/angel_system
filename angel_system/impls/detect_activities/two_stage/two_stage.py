import pdb

import torch
from torch import nn
# from torchmetrics import MaxMetric
# from torchmetrics.classification.accuracy import Accuracy
import sys
sys.path.insert(0, '/angel_workspace/angel_system/impls/detect_activities/two_stage')

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
        self.fcn = SpatialFCNModule('resnext')
        # Temporal: (Dict) Input feature from ResNext (2048) + Aux_data ->
        # Output (1, num_classes)
        self.temporal = RULSTM(num_classes, hidden=128, dropout=0, depth=3)
        check = torch.load(checkpoint)
        
        self.fcn.eval()
        
        # Load checkpoint
        temp_check = dict()

        for key in check['state_dict'].keys():
            #extract just the temporal weights from the checkpoint
            #fcn doesn't change, so no need to load in weights for it
            if 'temporal' in key:
                new_key = key.split('temporal.')[1]
                temp_check[new_key] = check['state_dict'][key]
        self.temporal.load_state_dict(temp_check)
        self.temporal.eval()

    def forward(self, data, aux):
        frame_feats = self.fcn(data)
        # pdb.set_trace()
        out = self.temporal(frame_feats.unsqueeze(1), aux)

        return out
