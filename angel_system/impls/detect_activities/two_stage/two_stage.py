"""

TODO:
* Update documentation

"""

import pdb

import torch
from torch import nn
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from .spatial.fcn import SpatialFCNModule
from .temporal.rulstm import RULSTM


class TwoStageModule(nn.Module):
    """This class implements the spatio-temporal model used for unified
    representation of multi-modal inputs in the scene. This model also
    performs the activity recognition for the given frame sequence.

    Args: TBD
    """

    def __init__(self, checkpoint: str, num_classes: int):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.fcn = SpatialFCNModule('resnext')
        self.temporal = RULSTM(num_classes, hidden=128, dropout=0, depth=3)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, data):
        results = self.fcn(data)
        results = self.temporal(results)

        return results

