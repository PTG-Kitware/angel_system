from ast import literal_eval
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
from tokenize import Double
import numpy as np
import PIL.Image
import os
import tqdm
import pickle
import pandas as pd
import logging
import re
import gc

gc.collect()

from angel_system.impls.detect_activities.swinb.swinb_detect_activities import SwinBTransformer
from angel_system.eval.support_functions import time_from_name
from angel_system.eval.visualization import EvalVisualization, plot_activities_confidence
from angel_system.eval.compute_scores import EvalMetrics


logging.basicConfig(level = logging.INFO)
log = logging.getLogger("ptg_eval")


def run_eval(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================
    # Load truth annotations
    # ============================
    #gt_label_to_ts_ranges = defaultdict(list)
    gt_f = pd.read_feather(args.activity_gt)
    # Keys: class, start_frame,  end_frame, exploded_ros_bag_path

    gt = []
    RE_FILENAME_TIME = re.compile(r"frame_\d+_(\d+)_\d+.\w+")
    for i, row in gt_f.iterrows():
        g = {
            'class': row["class"].lower().strip(),
            #'start': float(RE_FILENAME_TIME.match(row["start_frame"]).groups()[0]),
            'start': time_from_name(row["start_frame"]),
            #'end': float(RE_FILENAME_TIME.match(row["end_frame"]).groups()[0])
            'end': time_from_name(row["end_frame"])
        }
        gt.append(g)

    log.info(f"Loaded ground truth from {args.activity_gt}")
    gt = pd.DataFrame(gt)

    # ============================
    # Load detections from
    # extracted ros bag
    # ============================
    detections_input = [det for det in (literal_eval(s) for s in open(args.extracted_activity_detections))][0]
    detections = []

    for dets in detections_input:
        good_dets = {}
        for l, conf in zip(dets["label_vec"], dets["conf_vec"]):
            good_dets[l] = conf

        for l in dets["label_vec"]:
            d = {
                'class': l.lower().strip(),
                'start': dets["source_stamp_start_frame"],
                'end': dets["source_stamp_end_frame"],
                'conf': good_dets[l],
                'detect_intersection': np.nan
            }
            detections.append(d)
    detections = pd.DataFrame(detections)
    log.info(f"Loaded detections from {args.extracted_activity_detections}")

    # ============================
    # Load labels
    # ============================
    # grab all labels present in data
    labels = list(set([l.lower().strip().rstrip('.') for l in detections['class'].unique()]))

    log.debug(f"Labels: {labels}")

    # ============================
    # Split by time window
    # ============================
    # Get time ranges
    assert args.time_window > args.uncertainty_pad
    min_start_time = min(gt['start'].min(), detections['start'].min())
    max_end_time = max(gt['end'].max(), detections['end'].max())
    dt = args.time_window
    time_windows = np.arange(min_start_time, max_end_time, args.time_window)

    if time_windows[-1] < max_end_time:
        time_windows = np.append(time_windows, time_windows[-1] + args.time_window)
    time_windows = list(zip(time_windows[:-1], time_windows[1:]))
    time_windows = np.array(time_windows)
    dets_per_valid_time_w = np.zeros((len(time_windows), len(labels)),
                                     dtype=float)


    def get_time_wind_range(start, end):
        """Return slice indices of time windows that reside completely in
        start->end.

        time_windows[ind1:ind2] all live inside start->end.

        """
        # The start time of the ith window is min_start_time + dt*i.
        ind1_ = (start - min_start_time)/dt
        ind1 = int(np.ceil(ind1_))
        if ind1_ - ind1 + 1 < 1e-15:
            # We want to avoid the case where ind1_ is (j + eps) and it gets
            # rounded up to j + 1.
            ind1 -= 1

        # The end time of the ith window is min_start_time + dt*(i + 1).
        ind2_ = (end - min_start_time)/dt
        ind2 = int(np.floor(ind2_))
        if -ind2_ + ind2 + 1 < 1e-15:
            # We want to avoid the case where ind1_ is (j - eps) and it gets
            # rounded up to j - 1.
            ind1 += 1

        ind1 = max([ind1, 0])
        ind2 = min([ind2, len(time_windows)])

        return ind1, ind2


    # Valid time windows overlap with a detection.
    valid = np.zeros(len(time_windows), dtype=bool)
    for i in range(len(detections)):
        ind1, ind2 = get_time_wind_range(detections['start'][i],
                                         detections['end'][i])

        valid[ind1:ind2] = True
        correct_label = detections['class'][i].strip().rstrip('.')
        correct_class_idx = labels.index(correct_label)
        dets_per_valid_time_w[ind1:ind2, correct_class_idx] = np.maximum(dets_per_valid_time_w[ind1:ind2, correct_class_idx],
                                                              detections['conf'][i])

    gt_true_mask = np.zeros((len(time_windows), len(labels)), dtype=bool)
    for i in range(len(gt)):
        ind1, ind2 = get_time_wind_range(gt['start'][i], gt['end'][i])
        correct_label = gt['class'][i].strip().rstrip('.')
        correct_class_idx = labels.index(correct_label)
        gt_true_mask[ind1:ind2, correct_class_idx] = True

    if not np.all(np.sum(gt_true_mask, axis=1) <= 1):
        raise Exception('Conflicting ground truth for same time windows')

    # If ground truth isn't specified for a particular window, we should assume
    # 'background'.
    bckg_class_idx = labels.index('background')
    ind = np.where(np.all(gt_true_mask == False, axis=1))[0]
    gt_true_mask[ind, bckg_class_idx] = True

    # Any time the ground truth class changes, we want to add in uncertainty
    # padding, but there should always be at least one time window at the
    # center of the ground-truth span.
    gt_label = np.argmax(gt_true_mask, axis=1)
    pad = int(np.round(args.uncertainty_pad/dt))
    if pad > 0:
        ind = np.where(np.diff(gt_label, axis=0) != 0)[0] + 1
        if ind[0] != 0:
            ind = np.hstack([1, ind])

        if ind[-1] != len(time_windows):
            ind = np.hstack([ind, len(time_windows)])

        for i in range(len(ind) -1):
            ind1 = ind[i]
            ind2 = ind[i+1]
            # time windows in range ind1:ind2 all have the same ground
            # truth class.

            ind1_ = ind1 + pad
            ind2_ = ind2 - pad
            indc = int(np.round((ind1 + ind2)/2))
            ind1_ = min([ind1_, indc])
            ind2_ = max([ind2_, indc + 1])
            valid[ind1:ind1_] = False
            valid[ind2_:ind2] = False

    time_windows = time_windows[valid]
    dets_per_valid_time_w = dets_per_valid_time_w[valid]
    gt_true_mask = gt_true_mask[valid]

    # plot activity timelines
    plot_activities_confidence(labels=labels, gt=gt, dets=detections, output_dir=f"{output_dir}/plots")
    del gt
    del detections

    gc.collect()

    # ============================
    # Metrics
    # ============================
    metrics = EvalMetrics(labels, gt_true_mask, dets_per_valid_time_w, output_fn=f"{output_dir}/metrics.txt")
    metrics.precision()

    log.info(f"Saved metrics to {output_dir}/metrics.txt")

    # ============================
    # Plot
    # ============================
    vis = EvalVisualization(labels, gt_true_mask, dets_per_valid_time_w, output_dir=output_dir)
    vis.plot_pr_curve()
    vis.plot_roc_curve()

    log.info(f"Saved plots to {output_dir}/plots/")

    del gt_true_mask
    del dets_per_valid_time_w

    gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activity_gt",
        type=str,
        default="data/ros_bags/Annotated_folding_filter/labels_test_v1.2.feather",
        help="Feather file containing the ground truth annotations in the PTG-LEARN format. \
            The expected filename format is \'labels_test_v<label version>.feather\'"
    )
    parser.add_argument(
        "--extracted_activity_detections",
        type=str,
        default="data/ros_bags/_extracted/activity_detection_data.json",
        help="Text file containing the activity detections from an extracted ROS2 bag"
    )
    parser.add_argument(
        "--time_window",
        type=float,
        default=1,
        help="Time window in seconds to evaluate results on."
    )
    parser.add_argument(
        "--uncertainty_pad",
        type=float,
        default=0.5,
        help="Time in seconds to pad the groundtruth regions"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval",
        help="Folder to save results and plots to"
    )

    args = parser.parse_args()
    run_eval(args)

if __name__ == '__main__':
    main()
