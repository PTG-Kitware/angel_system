import importlib.util
import logging
from typing import Iterable, Dict, Hashable, List, Union, Any
import pdb

import numpy as np
import torch
import torchvision

from angel_system.interfaces.mm_detect_activities import MMDetectActivities
from angel_system.impls.detect_activities.two_stage.two_stage import TwoStageModule


LOG = logging.getLogger(__name__)
H2O_CLASSES = ["background", "grab book", "grab espresso", "grab lotion", "grab spray", "grab milk", "grab cocoa", "grab chips", "grab cappuccino", "place book", "place espresso", "place lotion", "place spray", "place milk", "place cocoa", "place chips", "place cappuccino", "open lotion", "open milk", "open chips", "close lotion", "close milk", "close chips", "pour milk", "take out espresso", "take out cocoa", "take out chips", "take out cappuccino", "put in espresso", "put in cocoa", "put in cappuccino", "apply lotion", "apply spray", "read book", "read espresso", "spray spray", "squeeze lotion"]

class TwoStageDetector(MMDetectActivities):
    """
    ``MMDetectActivities`` implementation using the explicit spatio temporal 
    two-stage training models.

    The `detect_activities` method in this class checks that the correct
    number of frames are provided by checking that the length of `frame_iter`
    is equal to `self._num_frames` x `self._sampling_rate`. For example,
    if `self._sampling_rate` is 2 and `self._num_frames` is 32,
    the activity detector should pass 64 frames as input to
    `detect_activities`.

    :param checkpoint_path: Path to a saved checkpoint file containing
        weights for the model.
    :param num_classes: Number of classes the model was trained on. This
        should match the number of classes the model checkpoint was trained
        on.
    :param labels_file: Path to the labels file for the given checkpoint.
        The labels file is a text file with the class labels, one class
        per line. This should match the class labels the model checkpoint
        was trained on.
    :param num_frames: Number of frames passed to the model for inference.
    :param sampling_rate: Sampling rate for the frame input. This subsamples
        the input frames by this amount.
    :param torch_device: When using CUDA, use the device by the given ID. By
        default, this is set to `cpu`.
    :param det_threshold: Threshold for which predictions must exceed to
        create an activity detection.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int,
        labels_file: str,
        num_frames: int = 32,
        sampling_rate: int = 2,
        torch_device: str = "cpu",
        det_threshold: float = 0.75,
    ):
        self._checkpoint_path = checkpoint_path
        self._num_classes = num_classes
        self._torch_device = torch_device
        self._det_threshold = det_threshold
        self._num_frames = num_frames
        self._sampling_rate = sampling_rate
        self._labels_file = labels_file

        # Set to None for lazy loading later.
        self._model: torch.nn.Module = None  # type: ignore
        self._model_device: torch.device = None  # type: ignore

        # Pytorch default configs for imagenet pre-trained models
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        self._crop_size = 224
        self._frames_per_second = 30

        # Pytorch default transfroms for imagenet pre-trained models
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    self._mean, self._std
                )
            ]
        )

        # TODO: Set up the labels from the given labels file
        self._labels = H2O_CLASSES
        # with open(self._labels_file, "r") as f:
        #     for line in f:
        #         self._labels.append(line.rstrip())

    def get_model(self) -> torch.nn.Module:
        """
        Lazy load the torch model in an idempotent manner.
        :raises RuntimeError: Use of CUDA was requested but is not available.
        """
        model = self._model
        if model is None:
            # Load the model with the checkpoint
            model = TwoStageModule(self._checkpoint_path, self._num_classes)
            model = model.eval()

            # Transfer the model to the requested device
            if self._torch_device != 'cpu':
                if torch.cuda.is_available():
                    model_device = torch.device(device=self._torch_device)
                    model = model.to(device=model_device)
                else:
                    raise RuntimeError(
                        "Use of CUDA requested but not available."
                    )
            else:
                model_device = torch.device(self._torch_device)

            self._model = model
            self._model_device = model_device

        return model

    def detect_activities(
        self,
        frame_iter: Iterable[np.ndarray],
        aux_data_iter: Iterable[Dict]
    ) -> Iterable[str]:
        """
        Formats the given iterable of frames into the required input format
        for the swin model and then inputs them to the model for inferencing.
        """
        # Check that we got the right number of frames
        frame_iter = list(frame_iter)
        aux_data = dict(aux_data_iter)
        assert all([len(frame_iter) == len(aux_data[i]) for i in aux_data])
        # assert len(frame_iter) == (self._sampling_rate * self._num_frames)
        model = self.get_model()

        # Apply data pre-processing
        frames = [self.transform(f) for f in frame_iter]
        frames = torch.stack(frames)
        for k in aux_data:
            aux_data[k] = torch.Tensor(aux_data[k])

        # Move the inputs to the GPU if necessary
        if self._model.cuda:
            frames = frames.cuda()
            for k in aux_data:
                aux_data[k] = aux_data[k].cuda()

        # Predict!
        with torch.no_grad():
            preds = self._model(frames, aux_data)

        # Get the top predicted classes
        post_act = torch.nn.Softmax(dim=1)
        preds: torch.Tensor = post_act(preds) # shape: (1, num_classes)
        top_preds = preds.topk(k=5)

        # Map the predicted classes to the label names
        # top_preds.indices is a 1xk tensor
        pred_class_indices = top_preds.indices[0]

        pred_class_names = [self._labels[int(i)] for i in pred_class_indices]

        # Filter out any detections below the threshold
        predictions = []
        pred_values = top_preds.values[0]
        for idx, p in enumerate(pred_class_names):
            if (pred_values[idx] > self._det_threshold):
                predictions.append(p)

        # pdb.set_trace()
        return predictions

    def get_config(self) -> dict:
        return {
            "cuda_device": self._cuda_device,
            "num_classes": self._num_classes,
            "det_threshold": self._det_threshold,
            "checkpoint_path": self._checkpoint_path,
            "num_frames": self._num_frames,
            "sampling_rate": self._sampling_rate,
            "labels_file": self._labels_file,
        }

    @classmethod
    def is_usable(cls) -> bool:
        # Only torch/torchvision required
        return True
