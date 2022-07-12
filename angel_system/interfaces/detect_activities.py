import abc
from typing import Iterable, Hashable, Dict, Tuple, Any
import numpy as np

from smqtk_core import Configurable, Pluggable


class DetectActivities (Configurable, Pluggable):
    """
    Algorithm that detects activities from a set of input
    frames as an iterable of ``numpy.ndarray`` type arrays.
    """

    @abc.abstractmethod
    def detect_activities(
      self,
      frame_iter: Iterable[np.ndarray]
    ) -> Dict[str, float]:
        """
        Detect activities in the given set of frames.
        :param frame_iter: Iterable of input frames as numpy arrays.
        :return: Dictionary mapping class labels to their prediction
            confidences for the given set of frames.
        """

    def __call__(
      self,
      frame_iter: Iterable[np.ndarray]
    ) -> Dict[str, float]:
        """
        Calls `detect_activities() with the given iterable set of frames.`
        """
        return self.detect_objects(frame_iter)
