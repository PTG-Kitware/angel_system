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