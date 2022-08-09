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
    model_name = Path(args.activity_model).stem
    output_dir = Path(os.path.join(args.output_dir, model_name))
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
    # Create detections from model
    # ============================
    # This model has a specific input frame quantity requirement.
    frame_input_size = 32 * 2  # 64
    window_stride = 1  # All possible frame-windows within the frame-range.

    save_file = Path(f"{output_dir}/slice_prediction_results_swinb-all_windows.pkl")

    with open(args.activity_labels, 'r') as l:
        detector = SwinBTransformer(
            checkpoint_path=args.activity_model,
            num_classes=len(l.readlines()),
            labels_file=args.activity_labels,
            num_frames=32,
            sampling_rate=2,
            torch_device="cuda:0",
        )

    # Detector for every 64-frame chunk, collecting slice prediction results per class.
    GlobalValues.clear_slice_values()

    def gen_results(i):
        """ Generate results for one slice of frames. """
        j = i + frame_input_size
        image_mats = np.asarray([np.asarray(PIL.Image.open(p)) for p in GlobalValues.all_image_files[i:j]])
        return SliceResult(
            (i, j),
            (time_from_name(GlobalValues.all_image_files[i].name),
             time_from_name(GlobalValues.all_image_files[j].name)),
            detector.detect_activities(image_mats)
        )

    if not save_file.is_file():
        print(f"Creating detection results from {model_name}")
        inputs = list(range(0, len(GlobalValues.all_image_files) - frame_input_size, window_stride))
        # # -- Serial version --
        # for slice_result in tqdm.tqdm(map(gen_results, inputs),
        #                    total=len(inputs),
        #                    ncols=120):
        #         GlobalValues.slice_index_ranges.append(slice_result.index_range)
        #         GlobalValues.slice_time_ranges.append(slice_result.time_range)
        #         GlobalValues.slice_preds.append(slice_result.preds)

        # -- Threaded version --
        with ThreadPoolExecutor(max_workers=3) as pool:
            # Starting indices across the whole frame range.
            for slice_result in tqdm.tqdm(
                pool.map(gen_results, inputs),
                total=len(inputs),
                ncols=120
            ):
                GlobalValues.slice_index_ranges.append(slice_result.index_range)
                GlobalValues.slice_time_ranges.append(slice_result.time_range)
                GlobalValues.slice_preds.append(slice_result.preds)

        # Save results to disk, this took a while! (~1.5 hours)
        print(f"Saving results to file: {save_file}")
        with open(save_file, 'wb') as ofile:
            pickle.dump({
                "slice_index_ranges":GlobalValues.slice_index_ranges,
                "slice_time_ranges":GlobalValues.slice_time_ranges,
                "slice_preds":GlobalValues.slice_preds,
            }, ofile, protocol=-1)
    else:
        # We have a results file.
        # Load computed results
        print(f"Loading results from file: {save_file}")
        with open(save_file, 'rb') as f:
            results_dict = pickle.load(f)

        # The [start, end) frame index ranges per slice
        GlobalValues.slice_index_ranges = results_dict['slice_index_ranges']  # List[Tuple[int, int]]

        # The [start, end) frame time pairs
        GlobalValues.slice_time_ranges = results_dict['slice_time_ranges']  # List[Tuple[float, float]]

        # Prediction results per slice
        GlobalValues.slice_preds = results_dict['slice_preds']  # List[Dict[str, float]]

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity_model", type=str, default="model_files/swinb_model_stage_base_ckpt_6.pth", help="The model checkpoint file")
    parser.add_argument("--activity_labels", type=str, default="model_files/swinb_coffee_task_labels.txt", help="File containing the activity labels separated by newlines")
    parser.add_argument("--activity_gt", type=str, default="data/ros_bags/Annotated_folding_filter/labels_test.feather", help="The feather file containing the ground truth annotations in the PTG-LEARN format")
    parser.add_argument("--extracted_activity_detections", type=str, default="data/ros_bags/Annotated_folding_filter/rosbag2_2022_08_08-18_56_31/_extracted/activity_detection_data.txt", help="Text file containing the activity detections from an extracted ROS2 bag")
    parser.add_argument("--output_dir", type=str, default="eval", help="Folder to output results to. This will be populated as {output_dir}/{model_name}")

    args = parser.parse_args()
    run_eval(args)

if __name__ == '__main__':
    main()
