import abc
from typing import Iterable, Hashable, Dict, Tuple, Any
import numpy as np

from smqtk_core import Configurable, Pluggable


class SpeechEmotionModel (Configurable, Pluggable):
    """
    Model to detect emotions from a set of input
    2D ``numpy.ndarray`` type arrays.
    """
    @abc.abstractmethod
    def __init__(self, model):
        """
        model: a serialized model. __init__ should implement loading model either using pickle or torch.
        """
        
    @abc.abstractmethod
    def classify(
      self,
      X: Iterable[np.ndarray],
      sampling_rate: int
    ) -> int:
        """
        Classify in the given modality representation.
        :param X: Iterable of some input modality as numpy arrays.
        :return: Prediction output corresponding to class number
        """

    def __call__(
      self,
      X: Iterable[np.ndarray],
      sampling_rate: int
    ) -> int:
        """
        Calls `classify_emotion() with the given iterable modality representation.`
        """
        return self.classify(X, sampling_rate)


class EyeGazeEncoderModel(Configurable, Pluggable):
    """
    Model to extract features given a sequence of eye gaze data
    takes in input of shape (6, SEQ_LEN) ``numpy.ndarray`` type arrays, consisting of ((gaze_x, gaze_y, gaze_z, head_x, head_y, head_z), time)
    """
    @abc.abstractmethod
    def __init__(self, model):
        """
        model: a serialized model. __init__ should implement loading model either using pickle or torch.
        """


    @abc.abstractmethod
    def encode(
      self,
      X: np.ndarray,
      sampling_rate: int
    ) -> np.ndarray:
        """
        Encode the given eye gaze data into a feature vector
        :param X: Iterable of some input modality as numpy arrays.
        :param sampling_rate: sampling rate to undersample the input X. used to accord to the model's input SEQ_LEN
        :return: encoded feature vector
        """

    def __call__(
      self,
      X: Iterable[np.ndarray],
      sampling_rate: int
    ) -> np.ndarray:
        """
        Calls `encode()` with the given eye gaze input
        """
        return self.encode(X, sampling_rate)
