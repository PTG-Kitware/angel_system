from typing import Dict, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from .components.frame_dataset import ROSFrameDataset
from .components.video_dataset import H2OVideoDataset


def collate_fn_pad(batch):
    """Padds batch of variable length.

    :params batch (Dict): Batch data with labels + auxiliary data from
        dataloader

    :returns batch (Dict): Processed batch with labels + auxiliary data
    :returns lengths (torch.Tensor): Lengths of sequences of each
        sample in batch
    """
    ## get sequence lengths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lengths = torch.tensor([len(t[0]["feats"]) for t in batch]).to(device)
    ## padd
    data_dic = {}
    feats = [torch.cat(t[0]["feats"]).to(device) for t in batch]
    data_dic["feats"] = torch.nn.utils.rnn.pad_sequence(feats)
    data_dic["act"] = torch.Tensor([t[1] for t in batch]).to(device).int()
    labels = {"l_hand": [], "r_hand": []}
    for t in batch:
        labels["l_hand"].append(torch.cat([k["l_hand"] for k in t[0]["labels"]]).to(device))
        labels["r_hand"].append(torch.cat([k["r_hand"] for k in t[0]["labels"]]).to(device))
    labels["l_hand"] = torch.nn.utils.rnn.pad_sequence(labels["l_hand"])
    labels["r_hand"] = torch.nn.utils.rnn.pad_sequence(labels["r_hand"])
    data_dic["labels"] = labels
    batch = data_dic

    return batch, lengths


class ROSDataModule(LightningDataModule):
    """``LightningDataModule`` for Hololens2 data collected from a ROS node.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        pose_files: Dict,
        action_files: Dict,
        data_dir: str = "data/h2o",
        data_type: str = "video",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        frames_per_segment: int = 1,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.data_type = data_type
        self.frames_per_segment = frames_per_segment

        # data transformations (Normalization values recommended by 
        # torchvision model zoo)
        if self.data_type == "frame":
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif self.data_type == "video":
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 6  # Action (interaction) classes

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`,
        `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and
        `trainer.test()`, so be careful not to execute the random split twice!
        The `stage` can be used to differentiate whether it's called before
        trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            if self.data_type == "frame":
                self.data_train = ROSFrameDataset(
                    self.hparams.data_dir,
                    self.hparams.pose_files["train_list"],
                    transform=self.transforms,
                )
                self.data_val = ROSFrameDataset(
                    self.hparams.data_dir,
                    self.hparams.pose_files["val_list"],
                    transform=self.transforms,
                )
                self.data_test = ROSFrameDataset(
                    self.hparams.data_dir,
                    self.hparams.pose_files["test_list"],
                    transform=self.transforms,
                )
            elif self.data_type == "video":
                self.data_train = H2OVideoDataset(
                    self.hparams.data_dir,
                    self.hparams.action_files["train_list"],
                    frames_per_segment=self.frames_per_segment,
                    transform=self.transforms,
                )
                self.data_val = H2OVideoDataset(
                    self.hparams.data_dir,
                    self.hparams.action_files["val_list"],
                    frames_per_segment=self.frames_per_segment,
                    transform=self.transforms,
                    test_mode=True,
                )
                self.data_test = H2OVideoDataset(
                    self.hparams.data_dir,
                    self.hparams.action_files["test_list"],
                    frames_per_segment=self.frames_per_segment,
                    transform=self.transforms,
                    test_mode=True,
                )

    def train_dataloader(self):
        if self.data_type == "video":
            dataloader = DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
                collate_fn=collate_fn_pad,
            )
        else:
            dataloader = DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
            )

        return dataloader

    def val_dataloader(self):
        if self.data_type == "video":
            dataloader = DataLoader(
                dataset=self.data_val,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                collate_fn=collate_fn_pad,
            )
        else:
            dataloader = DataLoader(
                dataset=self.data_val,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
            )

        return dataloader

    def test_dataloader(self):
        if self.data_type == "video":
            dataloader = DataLoader(
                dataset=self.data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                collate_fn=collate_fn_pad,
            )
        else:
            dataloader = DataLoader(
                dataset=self.data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
            )

        return dataloader
