from pathlib import Path
import argparse

import logging

from angel_system.ptg_eval.common.load_data import (
    activities_from_dive_csv,
    activities_from_ros_export_json,
    activities_as_dataframe
)
from angel_system.ptg_eval.common.discretize_data import discretize_data_to_windows
from angel_system.ptg_eval.activity_classification.visualization import EvalVisualization
from angel_system.ptg_eval.activity_classification.compute_scores import EvalMetrics


logging.basicConfig(level = logging.INFO)
log = logging.getLogger("ptg_eval_activity")


def run_eval(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # labels, gt, detections = load_from_file(args.activity_gt, args.extracted_activity_detections)
    gt = activities_from_dive_csv(args.activity_gt)
    labels, detections = activities_from_ros_export_json(args.extracted_activity_detections)

    # Make gt/detections pd.DataFrame instance to be consistent with downstream
    # implementation.
    gt = activities_as_dataframe(gt)
    detections = activities_as_dataframe(detections)

    gt_true_mask, dets_per_valid_time_w, time_windows = (
        discretize_data_to_windows(labels, gt, detections,
                                   args.time_window, args.uncertainty_pad)
    )
    
    # ============================
    # Metrics
    # ============================
    metrics = EvalMetrics(labels, gt_true_mask, dets_per_valid_time_w, output_dir=output_dir)
    metrics.precision()

    log.info(f"Saved metrics to {metrics.output_fn}")

    # ============================
    # Plot
    # ============================
    vis = EvalVisualization(labels, gt_true_mask, dets_per_valid_time_w, output_dir=output_dir)
    vis.plot_activities_confidence(gt=gt, dets=detections)
    vis.plot_pr_curve()
    vis.plot_roc_curve()
    vis.confusion_mat()

    log.info(f"Saved plots to {vis.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activity_gt",
        type=str,
        help="CSV file containing the ground truth annotations as exported from DIVE."
    )
    parser.add_argument(
        "--extracted_activity_detections",
        type=str,
        help="JSON file containing the activity detections from an extracted ROS2 bag"
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
        help="Time in seconds to pad the ground-truth regions"
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
