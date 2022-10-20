from typing import Any, Dict, List, Tuple, Union, Optional

import os
import os.path
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image

from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

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
    data_dic["feats"] = torch.stack(feats)
    data_dic["act"] = torch.Tensor([t[1] for t in batch]).to(device).int()
    data_dic["idx"] = torch.Tensor([t[2] for t in batch]).to(device).int()
    labels = {"l_hand": [], "r_hand": []}
    for t in batch:
        labels["l_hand"].append(torch.cat([k["l_hand"] for k in t[0]["labels"]]).to(device))
        labels["r_hand"].append(torch.cat([k["r_hand"] for k in t[0]["labels"]]).to(device))
    labels["l_hand"] = torch.stack(labels["l_hand"])
    labels["r_hand"] = torch.stack(labels["r_hand"])
    data_dic["labels"] = labels

    dets = []
    dcls = []
    bbox = []
    frms = []
    topK = 10
    for t in batch:
        # collect detections
        det1 = [torch.from_numpy(tmp[:topK]) for tmp in t[0]["dets"]]
        det1 = torch.stack(det1)
        det1 = det1.reshape([det1.shape[0]*det1.shape[1],1,det1.shape[2]])
        dets.append(det1.to(device))
        # collect detection outputs
        dcls1 = [torch.from_numpy(tmp[:topK]) for tmp in t[0]["dcls"]]
        dcls1 = torch.stack(dcls1)
        dcls1 = dcls1.reshape([dcls1.shape[0]*dcls1.shape[1],1])
        dcls.append(dcls1.to(device))
        # collect detection outputs
        bbox1 = [torch.from_numpy(tmp[:topK]) for tmp in t[0]["bbox"]]
        bbox1 = torch.stack(bbox1)
        bbox1 = bbox1.reshape([bbox1.shape[0]*bbox1.shape[1],1,bbox1.shape[2]])
        bbox.append(bbox1.to(device))

    data_dic["dets"] = torch.stack(dets).squeeze()
    data_dic["dcls"] = torch.stack(dcls).squeeze()
    data_dic["bbox"] = torch.stack(bbox).squeeze()

    batch = data_dic

    return batch, lengths

class VideoRecord(object):
    """Helper class for class H2OVideoDataset. This class represents a video
    sample's metadata.

    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where
             1) The first element is the path to the video sample's frames excluding
                the root_datapath prefix
             2) The second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index.
             5) any following elements are labels in the case of multi-label classification
    """

    def __init__(self, row, root_datapath, use_feats, use_dets):
        self._data = row
        self._path = root_datapath
        self._path = os.path.join(root_datapath, row[1])

        self.use_feats = use_feats
        self.use_dets = use_dets

        if self.use_feats:
            self._feat_path = os.path.join(self._path, "feat")
        else:
            self._feat_path = None

        if self.use_dets:
            self._det_path = os.path.join(self._path, "det")
        else:
            self._det_path = None

    @property
    def path(self) -> str:
        return self._path

    @property
    def feat_path(self) -> str:
        return self._feat_path

    @property
    def det_path(self) -> str:
        return self._det_path

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive

    @property
    def start_frame(self) -> int:
        return int(self._data[3])

    @property
    def end_frame(self) -> int:
        return int(self._data[4])

    @property
    def label(self) -> int:
        return int(self._data[2])


class ROSVideoDataset(torch.utils.data.Dataset):
    r"""
    A highly efficient and adaptable dataset class for videos.
    Instead of loading every frame of a video,
    loads x RGB frames of a video (sparse temporal sampling) and evenly
    chooses those frames from start to end of the video, returning
    a list of x PIL images or ``FRAMES x CHANNELS x HEIGHT x WIDTH``
    tensors where FRAMES=x if the ``ImglistToTensor()``
    transform is used.

    More specifically, the frame range [START_FRAME, END_FRAME] is divided into NUM_SEGMENTS
    segments and FRAMES_PER_SEGMENT consecutive frames are taken from each segment.

    Note:
        A demonstration of using this class can be seen
        in ``demo.py``
        https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch

    Note:
        This dataset broadly corresponds to the frame sampling technique
        introduced in ``Temporal Segment Networks`` at ECCV2016
        https://arxiv.org/abs/1608.00859.

    Note:
        This class relies on receiving video data in a structure where
        inside a ``ROOT_DATA`` folder, each video lies in its own folder,
        where each video folder contains the frames of the video as
        individual files with a naming convention such as
        img_001.jpg ... img_059.jpg.
        For enumeration and annotations, this class expects to receive
        the path to a .txt file where each video sample has a row with four
        (or more in the case of multi-label, see README on Github)
        space separated values:
        ``VIDEO_FOLDER_PATH     START_FRAME      END_FRAME      LABEL_INDEX``.
        ``VIDEO_FOLDER_PATH`` is expected to be the path of a video folder
        excluding the ``ROOT_DATA`` prefix. For example, ``ROOT_DATA`` might
        be ``home\data\datasetxyz\videos\``, inside of which a ``VIDEO_FOLDER_PATH``
        might be ``jumping\0052\`` or ``sample1\`` or ``00053\``.

    Args:
        root_path: The root path in which video folders lie.
                   this is ROOT_DATA from the description above.
        annotationfile_path: The .txt annotation file containing
                             one row per video sample as described above.
        num_segments: The number of segments the video should
                      be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders as described above.
        transform: Transform pipeline that receives a list of PIL images/frames.
        test_mode: If True, frames are taken from the center of each
                   segment, instead of a random location in each segment.

    """

    def __init__(
        self,
        root_path: str,
        annotationfile_path: str,
        num_segments: int = 1,
        frames_per_segment: int = 1,
        imagefile_template: str = "{:06d}",
        transform=None,
        test_mode: bool = False,
    ):
        super(ROSVideoDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.test_mode = test_mode
        self.use_feats = True
        self.use_dets = True

        self._parse_annotationfile(self.use_feats, self.use_dets)
        self._sanity_check_samples()

    def _load_image(self, directory: str, idx: int) -> Image.Image:
        return Image.open(
            os.path.join(directory, "rgb", self.imagefile_template.format(idx) + ".png")
        ).convert("RGB")

    def _load_feats(self, directory: str, idx: int) -> Dict:
        feat_file = os.path.join(directory, self.imagefile_template.format(idx) + ".pk")
        feats = torch.load(feat_file)

        return feats

    def _parse_annotationfile(self, use_feats=False, use_dets=False):
        ann_file = open(self.annotationfile_path)
        self.video_list = [
            VideoRecord(x.strip().split(), self.root_path, use_feats, use_dets)
            for x in ann_file.readlines()[1:]
        ]

    def _sanity_check_samples(self):
        for record in self.video_list:
            if record.num_frames <= 0 or record.start_frame == record.end_frame:
                print(record.num_frames, record.start_frame, record.end_frame)
                print(
                    f"\nDataset Warning: video {record.path} seems to have zero RGB frames on disk!\n"
                )

            elif record.num_frames < (self.num_segments * self.frames_per_segment):
                print(
                    f"\nDataset Warning: video {record.path} has {record.num_frames} frames "
                    f"but the dataloader is set up to load "
                    f"(num_segments={self.num_segments})*(frames_per_segment={self.frames_per_segment})"
                    f"={self.num_segments * self.frames_per_segment} frames. Dataloader will throw an "
                    f"error when trying to load this video.\n"
                )

    def _get_start_indices(self, record: VideoRecord) -> "np.ndarray[int]":
        """For each segment, choose a start index from where frames are to be
        loaded from.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """
        # choose start indices that are perfectly evenly spread across the video frames.
        if self.test_mode:
            distance_between_indices = (record.num_frames - self.frames_per_segment + 1) / float(
                self.num_segments
            )

            start_indices = np.array(
                [
                    int(distance_between_indices / 2.0 + distance_between_indices * x)
                    for x in range(self.num_segments)
                ]
            )
        # randomly sample start indices that are approximately evenly spread across the video frames.
        else:
            max_valid_start_index = (
                record.num_frames - self.frames_per_segment + 1
            ) // self.num_segments

            start_indices = np.multiply(
                list(range(self.num_segments)), max_valid_start_index
            ) + np.random.randint(max_valid_start_index, size=self.num_segments)


        return start_indices

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple[
            "torch.Tensor[num_frames, channels, height, width]",
            Union[int, List[int]],
        ],
        Tuple[Any, Union[int, List[int]]],
    ]:
        """
        For video with id idx, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations across the video.

        Args:
            idx: Video sample index.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """
        record: VideoRecord = self.video_list[idx]
        frame_start_indices: "np.ndarray[int]" = self._get_start_indices(record)

        return self._get(record, frame_start_indices, idx)

    def _get(
        self, record: VideoRecord, frame_start_indices: "np.ndarray[int]", idx: int
    ) -> Union[
        Tuple[List[Image.Image], Union[int, List[int]]],
        Tuple[
            "torch.Tensor[num_frames, channels, height, width]",
            Union[int, List[int]],
        ],
        Tuple[Any, Union[int, List[int]]]
    ]:
        """Loads the frames of a video at the corresponding indices.

        Args:
            record: VideoRecord denoting a video sample.
            frame_start_indices: Indices from which to load consecutive frames from.
        Returns:
            A tuple of (video, label). Label is either a single
            integer or a list of integers in the case of multiple labels.
            Video is either 1) a list of PIL images if no transform is used
            2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
            if the transform "ImglistToTensor" is used
            3) or anything else if a custom transform is used.
        """
        data = {}
        frame_start_indices = frame_start_indices + record.start_frame
        
        # from each start_index, load self.frames_per_segment
        # consecutive frames
        for start_index in frame_start_indices:
            frame_index = int(start_index)

            # load self.frames_per_segment consecutive frames
            for _ in range(self.frames_per_segment):
                if not os.path.exists(os.path.join(record.feat_path, self.imagefile_template.format(frame_index) + ".pk")):
                    frame_index -= 1

                sample_data = self._load_feats(record.feat_path, frame_index)
                if record.det_path != None:
                    det_data = self._load_feats(record.det_path, frame_index)
                    sample_data["dets"] = det_data["feats"]
                    sample_data["dcls"] = det_data["objects"]
                    sample_data["bbox"] = det_data["boxes"]

                sample_data["frm"] = self._load_image(record.path, frame_index)
                if self.transform is not None:
                    sample_data["frm"] = self.transform(sample_data["frm"])

                for k in sample_data:
                    if k not in data:
                        data[k] = [sample_data[k]]
                    else:
                        data[k].append(sample_data[k])

                if frame_index < record.end_frame:
                    frame_index += 1

        return data, record.label, idx

    def __len__(self):
        return len(self.video_list)

class ROSFrameDataset(torch.utils.data.Dataset):
    """A dataset class to load labels per frame from ros data.

    Args:
    root_dir (string): Directory with all the images.
    data_list_file (string): File with list of filenames of all the samples
    transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, root_dir: str, data_list_file: str, transform=None):
        super(ROSFrameDataset, self).__init__()

        self.root_dir = root_dir
        self.frame_list = self._get_frame_list(data_list_file)
        self.transform = transform

    def _get_frame_list(self, data_file: str) -> List[str]:
        frame_list: List[str]
        with open(data_file) as f:
            lines = f.readlines()
            frame_list = [os.path.join(self.root_dir, line.strip()) for line in lines]

        return frame_list

    def _load_image(self, frame_file: str) -> Image.Image:
        frame = Image.open(frame_file).convert("RGB")
        if self.transform is not None:
            frame = self.transform(frame)

        return frame

    def _load_hand_pose(self, annotation_file: str) -> Tuple[List[float], List[float]]:
        with open(annotation_file) as f:
            lines = f.readlines()[0].strip().split()
            num_kpts = 21

            if int(float(lines[0])) == 1:
                lefth = [float(pos) for pos in lines[1 : (num_kpts * 3 + 1)]]
            else:
                lefth = np.zeros(num_kpts * 3).tolist()
            if int(float(lines[num_kpts * 3 + 1])) == 1:
                righth = [float(pos) for pos in lines[(num_kpts * 3 + 2) :]]
            else:
                righth = np.zeros(num_kpts * 3).tolist()

        return lefth, righth

    def _load_obj_data(self, annotation_file: str) -> Tuple[int, List[float]]:
        with open(annotation_file) as f:
            lines = f.readlines()[0].strip().split()
            obj_class = int(float(lines[0]))
            obj_pose = [float(c) for c in lines[1:]]

        return obj_class, obj_pose

    def _load_verb(self, annotation_file: str) -> int:
        with open(annotation_file) as f:
            lines = f.readlines()[0].strip().split()
            verb = int(float(lines[0]))

        return verb

    def __len__(self) -> int:
        return len(self.frame_list)

    def __getitem__(self, idx: int):
        """For frame with id idx, loads frame with corresponding labels - hand pose.

        Args:
            idx: Frame sample index.
        Returns:
            A tuple of (frame, label). Label is a dictionary consisting
            of hand pose and frame file name.
        """
        frm = self._load_image(self.frame_list[idx])
        v_path = self.frame_list[idx].split("/")
        path_list = "/".join(v_path[:-2])
        fname = v_path[-1].split(".")[0]

        l_hand, r_hand = self._load_hand_pose(os.path.join(path_list, "hand_pose", f"{fname}.txt"))

        return {
            "frm": frm,
            "l_hand": np.array(l_hand),
            "r_hand": np.array(r_hand),
            "fname": self.frame_list[idx],
        }


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
        action_files: Dict,
        data_dir: str = "data/ros",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        frames_per_segment: int = 1,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.frames_per_segment = frames_per_segment

        # data transformations (Normalization values recommended by 
        # torchvision model zoo)
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 27  # Action (interaction) classes

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
            self.data_train = ROSVideoDataset(
                self.hparams.data_dir,
                self.hparams.action_files["train_list"],
                frames_per_segment=self.frames_per_segment,
                transform=self.transforms,
            )
            self.data_val = ROSVideoDataset(
                self.hparams.data_dir,
                self.hparams.action_files["val_list"],
                frames_per_segment=self.frames_per_segment,
                transform=self.transforms,
                test_mode=True,
            )
            self.data_test = ROSVideoDataset(
                self.hparams.data_dir,
                self.hparams.action_files["test_list"],
                frames_per_segment=self.frames_per_segment,
                transform=self.transforms,
                test_mode=True,
            )

    def train_dataloader(self):
        dataloader = DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn_pad,
        )

        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn_pad,
        )

        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn_pad,
        )

        return dataloader
