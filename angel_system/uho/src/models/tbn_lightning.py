"""###### Work in Progress ###### Implementation of Temporal Binding Networks.

(official paper - "EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition")
(official implentation - http://github.com/ekazakos/temporal-binding-network)
"""

import pdb
from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy


class TBNLightning(LightningModule):
    """This class implements the mulit-modal TBN model in Pytorch Lightning
    Framework.

    Args: TBD
    """

    def __init__(
        self,
        tbn: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        data_type: str = "frame",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.train_mode = data_type

        self.tbn = tbn

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, data):
        if self.train_mode == "frame":
            results = self.fcn(data)
        elif self.train_mode == "video":
            results = self.temporal(data)
        return results

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        feats, preds, loss = self.forward(batch)

        return loss, preds

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds = self.step(batch)

        if self.train_mode == "video":
            acc = self.train_acc(preds, batch[0]["act"])
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

            return {"loss": loss, "preds": preds, "targets": batch[0]["act"]}

        # log train metrics
        obj_acc = self.train_acc(preds["obj"], batch["obj_label"])
        verb_acc = self.train_acc(preds["verb"], batch["verb"])
        self.log("train/obj_loss", loss["obj_loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/obj_acc", obj_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/verb_loss", loss["verb_loss"], on_step=False, on_epoch=True, prog_bar=False
        )
        self.log("train/verb_acc", verb_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {
            "loss": sum([loss[l] for l in loss]),
            "obj_loss": loss["obj_loss"],
            "verb_loss": loss["verb_loss"],
            "obj_pred": preds["obj"],
            "verb_pred": preds["verb"],
            # "obj_targets": batch["obj_label"],
            # "verb_targets": batch["verb"],
        }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # pdb.set_trace()
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds = self.step(batch)

        if self.train_mode == "video":
            acc = self.val_acc(preds, batch[0]["act"])
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

            return {"loss": loss, "preds": preds, "targets": batch[0]["act"]}

        # log val metrics
        obj_acc = self.val_acc(preds["obj"], batch["obj_label"])
        verb_acc = self.val_acc(preds["verb"], batch["verb"])
        self.log("val/obj_loss", loss["obj_loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/obj_acc", obj_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/verb_loss", loss["verb_loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/verb_acc", verb_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", (obj_acc + verb_acc) / 2, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "loss": sum([loss[l] for l in loss]),
            "obj_loss": loss["obj_loss"],
            "verb_loss": loss["verb_loss"],
            "obj_pred": preds["obj"],
            "verb_pred": preds["verb"],
            # "obj_targets": batch["obj_label"],
            # "verb_targets": batch["verb"],
        }

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
        loss, preds = self.step(batch)

        if self.train_mode == "video":
            acc = self.test_acc(preds, batch[0]["act"])
            self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

            return {"loss": loss, "preds": preds, "targets": batch[0]["act"]}

        # log val metrics
        obj_acc = self.test_acc(preds["obj"], batch["obj_label"])
        verb_acc = self.test_acc(preds["verb"], batch["verb"])
        self.log("test/obj_loss", loss["obj_loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/obj_acc", obj_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/verb_loss", loss["verb_loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/verb_acc", verb_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "loss": sum([loss[l] for l in loss]),
            "obj_loss": loss["obj_loss"],
            "verb_loss": loss["verb_loss"],
            "obj_pred": preds["obj"],
            "verb_pred": preds["verb"],
            # "obj_targets": batch["obj_label"],
            # "verb_targets": batch["verb"],
        }

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
