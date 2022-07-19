import abc
from typing import Iterable, Hashable, Dict, Tuple, Any
import numpy as np

from smqtk_core import Configurable, Pluggable


class MMDetectActivities (Configurable, Pluggable):
    """
    Algorithm that detects activities from a set of input
    frames and multi-modal auxiliary data as iterables.
    """

    @abc.abstractmethod
    def detect_activities(
      self,
      frame_iter: Iterable[np.ndarray],
      aux_data_iter: Iterable[Dict]
    ) -> Dict[str, float]:
        """
        Detect activities in the given set of frames. And also takes
        multi-streamed data.
        :param frame_iter: Iterable of input frames as numpy arrays.
        :param aux_data_iter: Iterable of multi-modal auxiliary data.
        :return: Dictionary mapping class labels to their prediction
            confidences for the given set of frames.
        """

    def __call__(
      self,
      frame_iter: Iterable[np.ndarray],
      aux_data_iter: Iterable[Dict]
    ) -> Dict[str, float]:
        """
        Calls `detect_activities() with the given iterable set of frames.`
        """
        return self.detect_objects(frame_iter, aux_data_iter)
