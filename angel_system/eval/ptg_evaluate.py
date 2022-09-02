from ast import literal_eval
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import numpy as np
import PIL.Image
import os
import tqdm
import pickle
import pandas as pd
import logging
import re

from angel_system.impls.detect_activities.swinb.swinb_detect_activities import SwinBTransformer
from angel_system.eval.support_functions import time_from_name
from angel_system.eval.visualization import EvalVisualization
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
    for i, row in gt_f.iterrows():
        g = {
            'class': row["class"].lower(), 
            'start': time_from_name(row["start_frame"]), 
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
                'class': l.lower(), 
                'start': dets["source_stamp_start_frame"], 
                'end': dets["source_stamp_end_frame"],
                'conf': good_dets[l],
                'iou': np.nan
            }
            detections.append(d)
    detections = pd.DataFrame(detections)

    # ============================
    # Load labels
    # ============================
    # Grab all labels based on gt file version
    label_re = re.compile(r'labels_test_(?P<version>v\d+\.\d+)(?P<class>(\w+)?).feather')
    label_ver = label_re.match(os.path.basename(args.activity_gt)).group('version')

    vlabels = pd.read_excel(args.labels, sheet_name=label_ver)
    vlabels['class'] = vlabels['class'].str.lower()
    
    # grab all labels present in data
    dlabels = list(set([l.lower() for l in gt['class'].unique()] + [l.lower() for l in detections['class'].unique()]))

    # Remove any labels that we don't actually have
    missing_labels = []
    for i, row in vlabels.iterrows():
        if row['class'] not in dlabels:
            missing_labels.append(i)
    labels = vlabels.drop(missing_labels)

    log.debug(f"Labels v{label_ver}: {labels}")

    # ============================
    # Metrics
    # ============================
    metrics =  EvalMetrics(labels=labels, gt=gt, dets=detections, output_fn=f"{output_dir}/metrics.txt")
    dets_label_to_ts_ranges = metrics.iou_per_activity_label()

    log.info(f"Saved metrics to {output_dir}/metrics.txt")
    
    # ============================
    # Plot
    # ============================
    vis = EvalVisualization(labels=labels, gt=gt, dets=detections, output_dir=output_dir)
    vis.plot_activities_confidence()
    vis.plot_pr_curve()

    log.info(f"Saved plots to {output_dir}/plots/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="model_files/activity_detector_annotation_labels.xlsx")
    parser.add_argument("--activity_gt", type=str, default="data/ros_bags/Annotated_folding_filter/labels_test_v1.2.feather", help="The feather file containing the ground truth annotations in the PTG-LEARN format")
    parser.add_argument("--extracted_activity_detections", type=str, default="data/ros_bags/Annotated_folding_filter/rosbag2_2022_08_08-18_56_31/_extracted/activity_detection_data.json", help="Text file containing the activity detections from an extracted ROS2 bag")
    parser.add_argument("--output_dir", type=str, default="eval", help="Folder to output results to. This will be populated as {output_dir}/{model_name}")

    args = parser.parse_args()
    run_eval(args)

if __name__ == '__main__':
    main()
