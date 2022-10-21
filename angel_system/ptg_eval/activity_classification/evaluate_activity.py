from pathlib import Path
import argparse

import logging

from angel_system.ptg_eval.common.load_data import load_from_file
from angel_system.ptg_eval.common.discretize_data import discretize_data_to_windows
from angel_system.ptg_eval.activity_classification.visualization import EvalVisualization
from angel_system.ptg_eval.activity_classification.compute_scores import EvalMetrics


logging.basicConfig(level = logging.INFO)
log = logging.getLogger("ptg_eval_activity")


def run_eval(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, gt, detections = load_from_file(args.activity_gt, args.extracted_activity_detections)
    gt_true_mask, dets_per_valid_time_w = discretize_data_to_windows(labels, gt, detections,
                                                                     args.time_window, args.uncertainty_pad)
    
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
        help="Feather file containing the ground truth annotations in the PTG-LEARN format. \
              The expected filename format is \'labels_test_v<label version>.feather\'"
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
