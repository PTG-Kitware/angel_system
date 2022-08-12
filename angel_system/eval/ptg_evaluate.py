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

from angel_system.impls.detect_activities.swinb.swinb_detect_activities import SwinBTransformer
from angel_system.eval.support_functions import time_from_name, GlobalValues, SliceResult
from angel_system.eval.visualization import plot_activity_confidence
from angel_system.eval.compute_scores import iou_per_activity_label


def run_eval(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================
    # Load truth annotations
    # ============================
    gt_label_to_ts_ranges = defaultdict(list)
    gt = pd.read_feather(args.activity_gt)
    # Keys: class, start_frame,  end_frame, exploded_ros_bag_path

    for i, row in gt.iterrows():
        label = row["class"]
        # "start" entry
        start_ts = time_from_name(row["start_frame"])
        end_ts = time_from_name(row["end_frame"])
        gt_label_to_ts_ranges[label].append({"time": (start_ts, end_ts), "conf": 1})

    print(f"Loaded ground truth from {args.activity_gt}\n")

    # ============================
    # Load images
    # ============================
    ros_bag_root_dir = Path(args.activity_gt).parent
    images_dir = os.path.join(ros_bag_root_dir, gt.iloc[0]["exploded_ros_bag_path"])
    GlobalValues.all_image_files = sorted(Path(images_dir).iterdir())
    GlobalValues.all_image_times = np.asarray([
        time_from_name(p.name) for p in GlobalValues.all_image_files
    ])
    print(f"Using images from {images_dir}\n")

    # ============================
    # Load detections from
    # extracted ros bag
    # ============================
    dets_label_to_ts_ranges = defaultdict(list)
    detections = [det for det in (literal_eval(s) for s in open(args.extracted_activity_detections))][0]

    for dets in detections:
        good_dets = {}
        for l, conf in zip(dets["label_vec"], dets["conf_vec"]):
            good_dets[l] = conf

        for l in dets["label_vec"]:
            dets_label_to_ts_ranges[l].append(
                {"time": (dets["source_stamp_start_frame"], dets["source_stamp_end_frame"]), "conf": good_dets[l]})

    # ============================
    # Plot
    # ============================
    for label, ts_range_pairs in gt_label_to_ts_ranges.items():
        if label in dets_label_to_ts_ranges:
            plot_activity_confidence(label=label, gt_ranges=ts_range_pairs, det_ranges=dets_label_to_ts_ranges, output_dir=output_dir)
        else:
            print(f"No detections found for \"{label}\"")

    print(f"Saved plots to {output_dir}/plots/")

    # ============================
    # Metrics
    # ============================
    mIOU, iou_per_label = iou_per_activity_label(gt_label_to_ts_ranges.keys(), gt_label_to_ts_ranges, dets_label_to_ts_ranges)

    # Save to file
    with open(f"{output_dir}/metrics.txt", "w") as f:
        f.write(f"IoU: {mIOU}\n")
        f.write(f"IoU Per Label:\n")
        for k, v in iou_per_label.items():
            f.write(f"\t{k}: {v}\n")

    print(f"Saved metrics to {output_dir}/metrics.txt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity_gt", type=str, default="data/ros_bags/Annotated_folding_filter/labels_test.feather", help="The feather file containing the ground truth annotations in the PTG-LEARN format")
    parser.add_argument("--extracted_activity_detections", type=str, default="data/ros_bags/Annotated_folding_filter/rosbag2_2022_08_08-18_56_31/_extracted/activity_detection_data.json", help="Text file containing the activity detections from an extracted ROS2 bag")
    parser.add_argument("--output_dir", type=str, default="eval", help="Folder to output results to. This will be populated as {output_dir}/{model_name}")

    args = parser.parse_args()
    run_eval(args)

if __name__ == '__main__':
    main()
