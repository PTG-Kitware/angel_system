from typing import Sequence
from typing import Tuple

import numpy as np
import numpy.typing as npt

from model import *
from detectron2.data.detection_utils import read_image
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
    preds = inference(frames)
    return preds



# for test:

import os

def Re_order(image_list, image_number):
    img_id_list = []
    for img in image_list:
        img_id = int(img.split('/')[-1].split('_')[1])
        img_id_list.append(img_id)
    img_id_arr = np.array(img_id_list)
    s = np.argsort(img_id_arr)
    new_list = []
    for i in range(image_number):
        idx = s[i]
        new_list.append(image_list[idx])
    return new_list


if __name__ == "__main__":
    # test to read images with dimension of [60, H, W, 3]

    path_root = '/shared/niudt/DATASET/PTG_Kitware/all_activities_6/images'
    img_list = os.listdir(path_root)
    image_list = []
    for img in img_list:
        if 'png' in img:
            image_list.append(os.path.join(path_root, img))
    # re-oder the input
    input_list = Re_order(image_list, len(image_list))



    idx = 0
    imgs = np.zeros([60, 720, 1280, 3])
    for img_path in input_list:
        img = read_image(img_path, format="RGB")
        imgs[idx,:,:,:] = img
        idx = idx + 1
        if idx == 60:
            break



    preds = predict(imgs)
    s = 1