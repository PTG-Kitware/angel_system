import argparse
import logging
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import numpy.typing as npt

from angel_system.data.common.load_data import (
    activities_from_dive_csv,
    activities_from_ros_export_json,
    activities_as_dataframe,
    add_inter_steps
)
from angel_system.data.common.discretize_data import discretize_data_to_windows
from angel_system.ptg_eval.activity_classification.visualization import (
    EvalVisualization,
)
from angel_system.ptg_eval.activity_classification.compute_scores import EvalMetrics


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ptg_eval_activity")


def run_eval(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_window = args.time_window
    uncertainty_pad = args.uncertainty_pad

    # Loop over gt/pred pairs, gathering input data for eval.
    gt_true_mask: Optional[npt.NDArray] = None
    dets_per_valid_time_w: Optional[npt.NDArray] = None
    labels: Optional[List[str]] = None
    for i, (gt_fpath, pred_fpath) in enumerate(args.activity_gt_pred_pair):
        log.info(f"Loading data from pair {i}")
        
        l_labels, detections = activities_from_ros_export_json(pred_fpath.as_posix())
        gt = activities_from_dive_csv(gt_fpath.as_posix())

        min_start_time = min(min(gt, key=lambda a: a.start).start,
                             min(detections, key=lambda a: a.start).start)
        max_end_time = max(max(gt, key=lambda a: a.end).end,
                           max(detections, key=lambda a: a.end).end)
        
        if args.add_inter_steps or args.add_before_after_steps:
            gt = add_inter_steps(gt, min_start_time, max_end_time,
                                add_inter_steps=args.add_inter_steps,
                                add_before_after_steps=args.add_before_finished_steps)
        
        if labels is None:
            labels = l_labels
        else:
            assert labels == l_labels, (
                f"Subsequent gt/pred pair has disjoint label sets/orders. "
                f"{labels} != {l_labels}"
            )
        cleaned_labels = []
        for label in labels:
            cleaned_labels.append(label.lower().strip().strip(".").strip())
        labels = cleaned_labels
        print(labels)

        # Make gt/detections pd.DataFrame instance to be consistent with downstream
        # implementation.
        gt = activities_as_dataframe(gt)
        detections = activities_as_dataframe(detections)

        # Local masks for this specific file pair
        (
            l_gt_true_mask,
            l_dets_per_valid_time_w,
            l_time_windows,
        ) = discretize_data_to_windows(
            labels, gt, detections, time_window, uncertainty_pad
        )

        # for each pair, output separate activity window plots
        log.info("Visualizing this detection set against respective ground-truth.")
        pair_out_dir = output_dir / f"pair_{gt_fpath.stem}_{pred_fpath.stem}"
        vis = EvalVisualization(labels, None, None, output_dir=pair_out_dir)
        vis.plot_activities_confidence(gt=gt, dets=detections)

        # Stack with global set
        if gt_true_mask is None:
            gt_true_mask = l_gt_true_mask
        else:
            gt_true_mask = np.concatenate([gt_true_mask, l_gt_true_mask])
        if dets_per_valid_time_w is None:
            dets_per_valid_time_w = l_dets_per_valid_time_w
        else:
            dets_per_valid_time_w = np.concatenate(
                [dets_per_valid_time_w, l_dets_per_valid_time_w]
            )

    assert labels is not None, "No consistent label set loaded."
    assert gt_true_mask is not None, "No ground truth loaded."
    assert dets_per_valid_time_w is not None, "No predictions loaded"

    # ============================
    # Metrics
    # ============================
    metrics = EvalMetrics(
        labels, gt_true_mask, dets_per_valid_time_w, output_dir=output_dir
    )
    metrics.precision_recall()

    log.info(f"Saved metrics to {metrics.output_dir}")

    # ============================
    # Plot
    # ============================
    vis = EvalVisualization(
        labels, gt_true_mask, dets_per_valid_time_w, output_dir=output_dir
    )
    vis.confusion_mat()

    log.info(f"Saved plots to {vis.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-pred-pair",
        help=(
            "Specification of a pair of filepaths that refer to the "
            "ground-truth CSV file and prediction result JSON file, "
            "respectively. This option may be repeated any number of times "
            "for independent pairs."
        ),
        dest="activity_gt_pred_pair",
        type=Path,
        nargs=2,
        default=[],
        action="append",
    )
    parser.add_argument(
        "--time_window",
        type=float,
        default=1,
        help="Time window in seconds to evaluate results on.",
    )
    parser.add_argument(
        "--uncertainty_pad",
        type=float,
        default=0.5,
        help="Time in seconds to pad the ground-truth regions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval",
        help="Folder to save results and plots to",
    )
    parser.add_argument(
        "--add_inter_steps",
        action='store_true',
        help="Adds interstitial steps to the ground truth",
    )
    parser.add_argument(
        "--add_before_finished_steps",
        action='store_true',
        help="Adds before and finished steps to the ground truth",
    )

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
