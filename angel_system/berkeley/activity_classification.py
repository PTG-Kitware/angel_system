from typing import Sequence
from typing import Tuple

import numpy as np
import numpy.typing as npt


def predict(
    frames: Sequence[npt.NDArray[np.uint8]],
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    Predict activity classification for the sequence of temporally successive
    image frames.

    One activity classification is expected to cover the whole input window.
    The window frame may be allowed to be as small as one frame, or as large as
    the model and system resources can handle.

    It is currently expected that successive calls to this function will be
    called will be with a consistently sized `frames` instance.
    E.g. `frames` would be consistently the same length from call to call
    during the same runtime.

    Output confidences are expected to fall within the [0, 1] inclusive range.

    Use case example (with arbitrary input and return values):
        >>> frame_window_32: npt.NDArray = np.random.randint(0, 256, (32, 720, 1280, 3),
        ...                                                  dtype=np.uint8)
        >>> conf, labels = predict(frame_window_32)
        >>> print(conf)
        [ 0.1  0.1  0.2  0.6 ]
        >>> print(labels)
        [ "background", "measure-water", "measure-beans", "pour-water" ]

    NOTES FOR BERKELEY:
    * Additional input parameters and output values up for discussion.
    * We have a historical expectation that the label vector output is the same
      length and order from call to call. This is, yes, redundant information
      that could be extracted to a separate function that could be called just
      once.

    :param frames: Some sequence of RGB image frames in [H x W x C] shape for
        which and activity classification is desired.

    :return: Two vectors consisting of the predicted activity class labels and
        class confidences for the input window of image frames.
    """
