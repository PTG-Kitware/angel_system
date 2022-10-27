import time
from typing import (
    List,
    Tuple,
)

import pandas
import torch

from angel_system.ptg_eval.common.load_data import time_from_name


def gt_predict(
    gt_file: str,
    frame_set_start_stamp: float,
    frame_set_end_stamp: float,
    predict_time: float = 1.0
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], List[str]]:
    """
    Creates a simulated classifier prediction given the ground truth annotation
    file and the frame timestamps. The frame stamps are used to determine which
    activity was being performed in the ground truth file. By default, there
    is an artificial delay to simulate the classifier's computation time.

    The return type is consistent with the UHO's predict function.

    :param gt_file: string path for the ground truth annotations to use. It is
        expected that this is a feather file.
    :param frame_set_start_stamp: The start time in seconds for this frame
        window.
    :param frame_set_end_stamp: The end time in seconds for this frame window.
    :param predict_time: How long to delay returning the result to simulate the
        computation time the classifier would take.

    :return: A tuple consisting of two elements:
            0: Classifier results: A tuple consisting of a [1 x n_classes] tensor
               representing the classifier's confidence for each class and a tensor
               containing the index of the max confidence in the confidence tensor.
            1: A list of length n_classes for mapping the classifier's confidence
               indices to class strings.
    """
    gt = pandas.read_feather(gt_file)

    # Create list of classifier classes
    anno_labels = gt['class'].tolist()
    classes = list(dict.fromkeys(anno_labels))
    classes.insert(0, "Background")

    start_frames = gt['start_frame'].tolist()
    end_frames = gt['end_frame'].tolist()

    start_frames_sec = [time_from_name(f) for f in start_frames]
    end_frames_sec = [time_from_name(f) for f in end_frames]

    # Find which window the given frame stamps line up with
    confidences = torch.zeros(len(classes))
    classifier_label_idx = 0
    activity = "Background"
    for anno_label, start, end in zip(anno_labels, start_frames_sec, end_frames_sec):
        if frame_set_start_stamp >= start and frame_set_end_stamp <= end:
            classifier_label_idx = classes.index(anno_label)
            activity = anno_label
            break
        elif (
            frame_set_start_stamp < start and
            start <= frame_set_end_stamp <= end
        ):
            # Frame window end frame is within this annotation window
            classifier_label_idx = classes.index(anno_label)
            activity = anno_label
            break
        elif (
            start <= frame_set_start_stamp <= end and
            frame_set_end_stamp > end
        ):
            # Frame window start frame is within this annotation window
            classifier_label_idx = classes.index(anno_label)
            activity = anno_label
            break

    confidences[classifier_label_idx] = 1.0
    class_idx = torch.tensor(classifier_label_idx)

    time.sleep(predict_time)

    return (confidences, class_idx), classes
