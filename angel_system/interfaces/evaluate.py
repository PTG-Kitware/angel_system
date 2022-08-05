import scriptconfig as scfg
from ast import literal_eval
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import numpy as np
import PIL.Image
import glob
import os
import tqdm
import pickle

from angel_system.impls.detect_activities.swinb.swinb_detect_activities import SwinBTransformer
from angel_system.impls.eval.support_functions import time_from_name, GlobalValues, populate_global_values, SliceResult
from angel_system.impls.eval.visualization import plot_activity_confidence
from angel_system.impls.eval.compute_scores import iou_per_activity_label


class EvalConfig(scfg.Config):
    default = {
        "images_dir_path": "/home/local/KHQ/hannah.defazio/projects/PTG/angel_system/data/ros_bags/Annotated_folding_filter_rosbag-20220726T164959Z-002/filter_folding_rosbag/rosbag2_2022_07_21-20_22_19/_extracted/images",
        "conf_threshold": 0.8,
        "activity_model": "/home/local/KHQ/hannah.defazio/projects/PTG/angel_system/model_files/swinb_model_stage_base_ckpt_6.pth",
        "activity_labels": "/home/local/KHQ/hannah.defazio/projects/PTG/angel_system/model_files/swinb_coffee_task_labels.txt",
        "activity_truth_csv": "/home/local/KHQ/hannah.defazio/projects/PTG/angel_system/data/ros_bags/Annotated_folding_filter_rosbag-20220726T164959Z-002/filter_folding_rosbag_annotation.csv",
        "extracted_activity_detection_ros_bag": "/home/local/KHQ/hannah.defazio/projects/PTG/angel_system/data/ros_bags/Annotated_folding_filter_rosbag-20220726T164959Z-002/filter_folding_rosbag/rosbag2_2022_07_28-16_00_06/_extracted/activity_detection_data.txt",
        "output_dir": "/home/local/KHQ/hannah.defazio/projects/PTG/angel_system/eval"
    }

def main(cmdline=True, **kw):
    config = EvalConfig()
    config.update_defaults(kw)

    model_name = Path(config["activity_model"]).stem
    output_dir = Path(os.path.join(config["output_dir"], model_name))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================
    # Load images
    # ============================
    GlobalValues.all_image_files = sorted(Path(config["images_dir_path"]).iterdir())
    GlobalValues.all_image_times = np.asarray([
        time_from_name(p.name) for p in GlobalValues.all_image_files
    ])
    #populate_global_values(config["images_dir_path"])
    
    # ============================
    # Load truth annotations
    # ============================
    gt_label_to_ts_ranges = defaultdict(list)
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
    print(f"Loaded ground turth from {config['activity_truth_csv']}\n")

    # ============================
    # Create detections from model
    # ============================
    # This model has a specific input frame quantity requirement.
    frame_input_size = 32 * 2  # 64
    window_stride = 1  # All possible frame-windows within the frame-range.
    HYST_HISTORY_SIZE = 75

    save_file = Path(f"{output_dir}/slice_prediction_results_swinb-all_windows.pkl")

    with open(config["activity_labels"], 'r') as l:
        detector = SwinBTransformer(
            checkpoint_path=config["activity_model"],
            num_classes=len(l.readlines()),
            labels_file=config["activity_labels"],
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
    detections = [det for det in (literal_eval(s) for s in open(config["extracted_activity_detection_ros_bag"]))][0]

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
    label_to_slice_handle = dict()
    for label, ts_range_pairs in gt_label_to_ts_ranges.items():
        if label in dets_label_to_ts_ranges:
            plot_activity_confidence(label=label, gt_ranges=ts_range_pairs, det_ranges=dets_label_to_ts_ranges, output_dir=output_dir)
        else:
            print(f"No detections found for \"{label}\"")

    # ============================
    # Metrics
    # ============================
    mIOU, iou_pre_label = iou_per_activity_label(gt_label_to_ts_ranges.keys(), gt_label_to_ts_ranges, dets_label_to_ts_ranges)

if __name__ == '__main__':
    main()
