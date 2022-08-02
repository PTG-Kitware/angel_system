import scriptconfig as scfg
from ast import literal_eval
from collections import defaultdict
import csv
from pathlib import Path
import concurrent.futures
import tqdm
import numpy as np
import PIL.Image
import glob

from support_functions import time_from_name, run_swinb_model, GlobalValues
from visualization import plot_activity_confidence
from compute_scores import iou_per_activity_label


class EvalConfig(scfg.Config):
    default = {
        "images_dir_path": "../data/ros_bags/Annotated_folding_filter_rosbag-20220726T164959Z-002/filter_folding_rosbag/rosbag2_2022_07_21-20_22_19/_extracted/images",
        "conf_threshold": 0.8,
        "generate_detections": False,
        "activity_model": "../model_files/swinb_model_stage_base_ckpt_6.pth",
        "activity_labels": "../model_files/swinb_coffee_task_labels.txt",
        "activity_truth_csv": "../data/ros_bags/Annotated_folding_filter_rosbag-20220726T164959Z-002/filter_folding_rosbag_annotation.csv",
        "extracted_activity_detection_ros_bag": "../data/ros_bags/Annotated_folding_filter_rosbag-20220726T164959Z-002/filter_folding_rosbag/rosbag2_2022_07_28-16_00_06/_extracted/activity_detection_data.txt"
    }

def main(cmdline=True, **kw):
    config = EvalConfig()
    config.update_defaults(kw)

    GlobalValues.sample_image = np.asarray(PIL.Image.open(glob.glob(f"{config['images_dir_path']}/*.png")[0]))

    # Get annotations
    gt_label_to_ts_ranges = defaultdict(list)
    dets_label_to_ts_ranges = defaultdict(list)

    # Load truth annotations
    with open(config["activity_truth_csv"], 'r') as f:
        # IDs that have a starting event
        label_start_ts = dict()
        for row in csv.reader(f):
            if row[0].strip().startswith("#"):
                continue
            aid = row[0]
            label = row[9]
            if label not in label_start_ts:
                # "start" entry
                label_start_ts[label] = time_from_name(row[1])
            else:

                # "end" entry
                end_ts = time_from_name(row[1])
                gt_label_to_ts_ranges[label].append({"time": (label_start_ts[label], end_ts), "conf": 1})
                del label_start_ts[label]

    # Load detections
    if config["generate_detections"]:
        # Use model to create detections
        run_swinb_model() # TODO

        dets_label_to_ts_ranges = {} # TODO

    else:
        # Use extracted ROS bag
        detections = [det for det in (literal_eval(s) for s in open(config["extracted_activity_detection_ros_bag"]))][0]

        for dets in detections:
            good_dets = {}
            for l, conf in zip(dets["label_vec"], dets["conf_vec"]):
                good_dets[l] = conf

            for l in dets["label_vec"]:
                dets_label_to_ts_ranges[l].append(
                    {"time": (dets["source_stamp_start_frame"], dets["source_stamp_end_frame"]), "conf": good_dets[l]})

    label_to_slice_handle = dict()
    for label, ts_range_pairs in gt_label_to_ts_ranges.items():
        if label in dets_label_to_ts_ranges:
            plot_activity_confidence(label=label, gt_ranges=ts_range_pairs, det_ranges=dets_label_to_ts_ranges)
        else:
            print(f"No detections found for \"{label}\"")

    iou_per_activity_label(gt_label_to_ts_ranges.keys(), gt_label_to_ts_ranges, dets_label_to_ts_ranges)

if __name__ == '__main__':
    main()
