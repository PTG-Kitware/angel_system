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
    time_windows = np.arange(min_start_time, max_end_time, args.time_window)

    if time_windows[-1] < max_end_time:
        time_windows = np.append(time_windows, time_windows[-1] + args.time_window)
    time_windows = list(zip(time_windows[:-1], time_windows[1:]))

    # Create masked matrix of detection confidences
    dets_per_valid_time_w = []
    gt_true_mask = []

    for time in time_windows:
        det_confs_in_w = []
        gt_tf_sample = [False] * len(labels)

        # Determine what detections we have that completely contain the time window
        det_overlap = detections.query(f'{time[0]} >= start and {time[1]} <= end')
        if det_overlap.empty:
            continue

        # Determine what gt we have that completely contain the time window
        gt_overlap = gt.query(f'{time[0]} >= start+{args.uncertainty_pad} and {time[1]} <= end-{args.uncertainty_pad}')
        if gt_overlap.empty:
            continue

        # Determine the highest conf for each class
        for label in labels:
            class_overlap = det_overlap.loc[det_overlap['class'] == label]
            det_confs_in_w.append(class_overlap['conf'].max())

        # Mark detection as correct
        for ii, r in gt_overlap.iterrows():
            correct_label = r['class'].strip().rstrip('.')
            correct_class_idx = labels.index(correct_label)

            gt_tf_sample[correct_class_idx] = True # tp
        
        dets_per_valid_time_w.append(det_confs_in_w)
        gt_true_mask.append(gt_tf_sample)

    dets_per_valid_time_w = np.array(dets_per_valid_time_w)
    gt_true_mask = np.array(gt_true_mask)

    # plot activity timelines
    plot_activities_confidence(labels=labels, gt=gt, dets=detections, output_dir=f"{output_dir}/plots")
    del gt
    del detections

    gc.collect()
    
    # ============================
    # Metrics
    # ============================
    metrics = EvalMetrics(labels, gt_true_mask, dets_per_valid_time_w, output_fn=f"{output_dir}/metrics.txt")
    #metrics.detect_intersection_per_activity_label()
    metrics.precision()

    log.info(f"Saved metrics to {output_dir}/metrics.txt")
    
    # ============================
    # Plot
    # ============================
    vis = EvalVisualization(labels, gt_true_mask, dets_per_valid_time_w, output_dir=output_dir)
    vis.plot_pr_curve()
    #vis.plot_roc_curve()
    #vis.plot_confusion_matrix()

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
