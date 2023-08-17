import logging
import warnings

from typing import Sequence
from typing import Tuple
from typing import List

import pandas as pd
import numpy as np

from angel_system.data.common.load_data import sanitize_str


log = logging.getLogger("ptg_eval_common")


def get_time_wind_range(
    start: float,
    end: float,
    dt: float,
    min_start_time: float,
    time_windows: Sequence[Tuple[float, float]],
) -> Tuple[int, int]:
    """
    Return slice indices of time windows that reside completely in
    start->end.

    :param start: Timestamp for the start of the time window in seconds.
    :param end: Timestamp for the end of the time window in seconds.
    :param dt: Length of the time window in seconds.
    :param min_start_time: The earliest timestamp between the
        ground truth and detections.
    :param time_windows: A list of all time windows in the format (start, end).

    :return: Start and end indicies Tuple(ind1, ind2) where
        time_windows[ind1:ind2] all live inside start->end.
    """
    # The start time of the ith window is min_start_time + dt*i.

    ind1_ = (start - min_start_time) / dt
    ind1 = int(np.ceil(ind1_))
    if ind1_ - ind1 + 1 < 1e-15:
        # We want to avoid the case where ind1_ is (j + eps) and it gets
        # rounded up to j + 1.
        ind1 -= 1

    # The end time of the ith window is min_start_time + dt*(i + 1).
    ind2_ = (end - min_start_time) / dt
    ind2 = int(np.floor(ind2_))
    if -ind2_ + ind2 + 1 < 1e-15:
        # We want to avoid the case where ind1_ is (j - eps) and it gets
        # rounded up to j - 1.
        ind1 += 1

    ind1 = max([ind1, 0])
    ind2 = min([ind2, len(time_windows) - 1])

    # assert ind2 >= ind1, "Invalid time window indexes"
    if ind2 < ind1:
        # Rounding issue
        warnings.warn(
            f"ind1: {ind1} is greater than ind2: {ind2} for start time {start} and end time {end}"
        )
        ind1 = ind2
    return ind1, ind2


def discretize_data_to_windows(
    labels: List[str],
    gt: pd.DataFrame,
    detections: pd.DataFrame,
    time_window: float,
    uncertainty_pad: float,
    min_start_time: float,
    max_end_time: float,
):
    """
    Reformat the ground truth and detection data into time windows and
    removed any invalid time windows.

    :param labels: Array of class labels (str).
    :param gt: Pandas dataframe of the ground truth.
    :param detections: Pandas dataframe of all detections per class.
    :param time_window: The span of time covered by detections will be discretized in
        equally-long windows of this size (seconds).
    :param uncertainty_pad: Time in seconds to pad ground truth detections by. Detections
        within the windows ground truth + or - uncertainty pad will not be scored.
    :param min_start_time: Minimum start time across the ground truth and detection sets
    :param max_end_time: Maximum end time across the ground truth and detection sets

    :return: Tuple(
        ground truth mask - Matrix of size (number of valid time windows x number classes)
            where True indicates a true class example, False inidcates a false class example.
            There should only be one True value per row,
        window class scores - Matrix of size (number of valid time windows x number classes)
            filled with the max confidence score per class for any detections in the time window,
        time_windows - A list of all time windows in the format (start, end)
        )
    """
    # ============================
    # Split by time window
    # ============================
    # Get time ranges

    assert (
        time_window > uncertainty_pad
    ), "Time window must be longer than the uncertainty pad"
    dt = time_window
    time_windows = np.arange(min_start_time, max_end_time, time_window)

    if time_windows[-1] < max_end_time:
        time_windows = np.append(time_windows, time_windows[-1] + time_window)
    time_windows = list(zip(time_windows[:-1], time_windows[1:]))
    time_windows = np.array(time_windows)
    window_class_scores = np.zeros((len(time_windows), len(labels)), dtype=float)

    # Mark which time windows are "valid".
    # Valid time windows are those that overlap with a predicted activity
    # classification.
    valid = np.zeros(len(time_windows), dtype=bool)
    for i in range(len(detections)):
        start_time = detections["start"][i]
        end_time = detections["end"][i]
        ind1, ind2 = get_time_wind_range(
            start_time,
            end_time,
            dt,
            min_start_time,
            time_windows,
        )

        correct_label = sanitize_str(detections["class_label"][i])
        correct_class_idx = labels.index(correct_label)

        if ind1 == ind2:
            # Bug fix: indexing ind1:ind2 does not work if they are equal
            valid[ind1] = True
            window_class_scores[ind1, correct_class_idx] = np.maximum(
                window_class_scores[ind1, correct_class_idx], detections["conf"][i]
            )
        else:
            valid[ind1:ind2] = True
            window_class_scores[ind1:ind2, correct_class_idx] = np.maximum(
                window_class_scores[ind1:ind2, correct_class_idx], detections["conf"][i]
            )

    gt_true_mask = np.zeros((len(time_windows), len(labels)), dtype=bool)
    for i in range(len(gt)):
        ind1, ind2 = get_time_wind_range(
            gt["start"][i], gt["end"][i], dt, min_start_time, time_windows
        )
        correct_label = sanitize_str(gt["class_label"][i])
        correct_class_idx = labels.index(correct_label)
        gt_true_mask[ind1:ind2, correct_class_idx] = True

    if not np.all(np.sum(gt_true_mask, axis=1) <= 1):
        raise AssertionError("Conflicting ground truth for same time windows")

    # If ground truth isn't specified for a particular window, we should assume
    # 'background'.
    bckg_class_idx = labels.index("background")
    ind = np.where(np.all(gt_true_mask == False, axis=1))[0]
    gt_true_mask[ind, bckg_class_idx] = True

    # Any time the ground truth class changes, we want to add in uncertainty
    # padding, but there should always be at least one time window at the
    # center of the ground-truth span.
    gt_label = np.argmax(gt_true_mask, axis=1)
    pad = int(np.round(uncertainty_pad / dt))
    if pad > 0:
        ind = np.where(np.diff(gt_label, axis=0) != 0)[0] + 1
        if ind[0] != 0:
            ind = np.hstack([1, ind])

        if ind[-1] != len(time_windows):
            ind = np.hstack([ind, len(time_windows)])

        for i in range(len(ind) - 1):
            ind1 = ind[i]
            ind2 = ind[i + 1]
            # time windows in range ind1:ind2 all have the same ground
            # truth class.

            ind1_ = ind1 + pad
            ind2_ = ind2 - pad
            indc = int(np.round((ind1 + ind2) / 2))
            ind1_ = min([ind1_, indc])
            ind2_ = max([ind2_, indc + 1])
            valid[ind1:ind1_] = False
            valid[ind2_:ind2] = False

    time_windows = time_windows[valid]
    window_class_scores = window_class_scores[valid]
    gt_true_mask = gt_true_mask[valid]

    return gt_true_mask, window_class_scores, time_windows
