import importlib.util
import logging
from typing import Iterable, Dict, Hashable, List, Union, Any

import numpy as np
import torch
import torchvision

from angel_system.interfaces.detect_activities import DetectActivities
from angel_system.impls.detect_activities.pytorchvideo_slow_fast_r50 import KINETICS_400_LABELS
from angel_system.impls.detect_activities.swinb.swin import swin_b
from angel_system.impls.detect_activities.swinb.utils import (
    get_start_end_idx, spatial_sampling, temporal_sampling
)


LOG = logging.getLogger(__name__)


class SwinBTransformer(DetectActivities):
    """
    ``DetectActivities`` implementation using the Shifted window (Swin)
    transformer from Learn.

    :param checkpoint_path: Path to a saved checkpoint file containing
        weights for the model.
    :param num_classes: Number of classes the model was trained on.
    :param use_cuda: Attempt to use a cuda device for inferences. If no
        device is found, CPU is used.
    :param cuda_device: When using CUDA use the device by the given ID. By
        default, this refers to GPU ID 0. This parameter is not used if
        `use_cuda` is false.
    :param det_threshold: Threshold for which predictions must exceed to
        create an activity detection.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int,
        use_cuda: bool = False,
        cuda_device: Union[int, str] = "cuda:0",
        det_threshold: float = 0.75,
    ):
        self._checkpoint_path = checkpoint_path
        self._num_classes = num_classes
        self._use_cuda = use_cuda
        self._cuda_device = cuda_device
        self._det_threshold = det_threshold

        # Set to None for lazy loading later.
        self._model: torch.nn.Module = None  # type: ignore
        self._model_device: torch.device = None  # type: ignore

        # Default configs from learn SwinVideo config
        self._mean = [0.45, 0.45, 0.45]
        self._std = [0.225, 0.225, 0.225]
        self._crop_size = 224
        self._num_frames = 32
        self._sampling_rate = 2
        self._frames_per_second = 30

        # Transfrom from learn/TimeSformer/video_classification.py
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    self._mean, self._std
                )
            ]
        )

    def get_model(self) -> "torch.nn.Module":
        """
        Lazy load the torch model in an idempotent manner.
        :raises RuntimeError: Use of CUDA was requested but is not available.
        """
        model = self._model
        if model is None:
            # Load the model with the checkpoint
            model = swin_b(self._checkpoint_path, self._num_classes)

            # Unfreeze head
            for n, p in model.named_parameters():
                if n.startswith("head"):
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            model = model.eval()
            model_device = torch.device('cpu')
            if self._use_cuda:
                if torch.cuda.is_available():
                    model_device = torch.device(device=self.cuda_device)
                    model = model.to(device=model_device)
                else:
                    raise RuntimeError(
                        "Use of CUDA requested but not available."
                    )
            self._model = model
            self._model_device = model_device

        return model

    def detect_activities(
        self,
        frame_iter: Iterable[np.ndarray]
    ) -> Iterable[str]:
        """
        Formats the given iterable of frames into the required input format
        for the swin model and then inputs them to the model for inferencing.
        """
        model = self.get_model()

        # Form the frames into the required format for the video model
        # Based off of the Learn swin CollateFn
        spatial_idx = 1 # only perform uniform crop and short size jitter
        clip_idx = -1

        frames = [self.transform(f) for f in frame_iter]
        frames = [torch.stack(frames)]

        fps, target_fps = 30, 30
        clip_size = self._sampling_rate * (self._num_frames) / target_fps * fps
        start_end_idx = [
            get_start_end_idx(len(x), clip_size, clip_idx=clip_idx, num_clips=1)
            for x in frames
        ]

        # This subsamples every n (sample rate) frames
        frames = [
            temporal_sampling(x, s, e, self._num_frames)
            for x, (s, e) in zip(frames, start_end_idx)
        ]

        # Crop and random short side scale jitter
        frames = [x.permute(1, 0, 2, 3) for x in frames]
        frames = [spatial_sampling(x, spatial_idx=spatial_idx, min_scale=224, max_scale=224, crop_size=224) for x in frames]
        frames = torch.stack(frames)

        # Predict!
        with torch.no_grad():
            preds = self._model(frames)

        # Get the top predicted classes
        post_act = torch.nn.Softmax(dim=1)
        preds: torch.Tensor = post_act(preds) # shape: (1, num_classes)
        top_preds = preds.topk(k=5)

        LABELS = ["pour water into measuring cup",
                  "pour water from cup into kettle",
                  "turn on scale",
                  "place bowl on scale",
                  "zero scale",
                  "measure 25 grams of coffee beans",
                  "misc"]

        # Map the predicted classes to the label names
        # top_preds.indices is a 1xk tensor
        pred_class_indices = top_preds.indices[0]

        # TODO: This will not work for models trained on data other than Kinetics400.
        # This is also copied from the pytorchvideo slow fast implementation.
        pred_class_names = [LABELS[int(i)] for i in pred_class_indices]

        # Filter out any detections below the threshold
        predictions = []
        pred_values = top_preds.values[0]
        for idx, p in enumerate(pred_class_names):
            if (pred_values[idx] > self._det_threshold):
                predictions.append(p)

        return predictions

    def get_config(self) -> dict:
        return {
            "use_cuda": self._use_cuda,
            "cuda_device": self._cuda_device,
        }

    @classmethod
    def is_usable(cls) -> bool:
        # check for optional dependencies
        torch_spec = importlib.util.find_spec('torch')
        torchvision_spec = importlib.util.find_spec('torchvision')
        if (
            torch_spec is not None and
            torchvision_spec is not None
        ):
            return True
        else:
            return False
