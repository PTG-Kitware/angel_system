from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
import os
import numpy as np
import pdb

class UnifiedHOModule(LightningModule):
    """This class implements the spatio-temporal model used for unified
    representation of hands and interacting objects in the scene.

    This model also performs the activity recognition for the given frame
    sequence.
    """

    def __init__(
        self,
        temporal: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.temporal = temporal

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy(ignore_index=-1)
        self.val_acc = Accuracy(ignore_index=-1)
        self.test_acc = Accuracy(ignore_index=-1)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, data, mode):

        return self.temporal(data, mode)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any, mode: bool):
        feats, preds, loss, idx = self.forward(batch, mode)

        return loss, preds, feats, idx

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, _, _ = self.step(batch, True)

        acc = self.train_acc(preds, batch[0]["act"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": batch[0]["act"]}


    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, _, _ = self.step(batch, False)

        acc = self.val_acc(preds, batch[0]["act"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": batch[0]["act"]}


    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log(
            "val/acc_best",
            self.val_acc_best.compute(),
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, feats, idx = self.step(batch, False)

        acc = self.test_acc(preds, batch[0]["act"])
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": batch[0]["act"]}


    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization."""
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
