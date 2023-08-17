import os
import seaborn as sn
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple

from sklearn.metrics import (
    PrecisionRecallDisplay,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

import logging


log = logging.getLogger("ptg_eval")

plt.rcParams.update({"figure.max_open_warning": 0})
plt.rcParams.update({"font.size": 35})


class EvalVisualization:
    def __init__(
        self,
        labels: List[str],
        gt_true_mask: np.ndarray,
        window_class_scores: np.ndarray,
        output_dir: str = "",
    ):
        """
        :param labels: Array of class labels (str)
        :param gt_true_mask: Matrix of size (number of valid time windows x number classes) where True
            indicates a true class example, False inidcates a false class example. There should only be one
            True value per row
        :param window_class_scores: Matrix of size (number of valid time windows x number classes) filled with
            the max confidence score per class for any detections in the time window
        :param output_dir: Directory to write the plots to
        """
        self.output_dir = Path(os.path.join(output_dir, "plots/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(self.output_dir)

        self.labels = labels

        self.gt_true_mask = gt_true_mask
        self.window_class_scores = window_class_scores

    def confusion_mat(self):
        """
        Plot a confusion matrix of size (number of labels x number of labels)
        """

        log.debug("Plotting confusion matrix")
        plt.rcParams.update({"font.size": 55})
        fig, ax = plt.subplots(figsize=(100, 100))

        n_classes = len(self.labels)
        label_vec = np.arange(n_classes)
        true_idxs = np.where(self.gt_true_mask == True)[1]
        pred_idxs = np.argmax(self.window_class_scores, axis=1)

        cm = confusion_matrix(true_idxs, pred_idxs, labels=label_vec, normalize="true")

        sn.heatmap(cm, annot=True, fmt=".2f", ax=ax)
        ax.set(
            title="Confusion Matrix",
            xlabel="Predicted Label",
            ylabel="True Label",
        )
        fig.savefig(f"{self.output_dir}/confusion_mat.png", pad_inches=5)
        plt.close(fig)

    def plot_activities_confidence(
        self,
        gt: pd.DataFrame,
        dets: pd.DataFrame,
        min_start_time: float,
        max_end_time: float,
        custom_range: Tuple[float, float] = None,
        custom_range_color: str = "red",
    ):
        """
        Plot activity confidences over time

        :param gt: Pandas dataframe of the ground truth
        :param dets: Pandas dataframe of all detections per class
        :param min_start_time: Minimum start time across the ground truth and detection sets
        :param max_end_time: Maximum end time across the ground truth and detection sets
        :param custom_range: Optional tuple indicating the starting and ending times of an additional
                                range to highlight in addition to the `gt_ranges`.
        :param custom_range_color: The color of the additional range to be drawn. If not set, we will
                                    use "red".
        """
        log.debug("Plotting activity confidences")
        plt.rcParams.update({"font.size": 25})
        for label in self.labels:
            gt_ranges = gt.loc[gt["class_label"] == label]
            det_ranges = dets.loc[dets["class_label"] == label]

            if not gt_ranges.empty and not det_ranges.empty:
                # ============================
                # Setup figure
                # ============================
                total_time_delta = max_end_time - min_start_time
                pad = 0.05 * total_time_delta

                # Setup figure
                fig = plt.figure(figsize=(28, 12))
                ax = fig.add_subplot(111)
                ax.set_title(
                    f'Window Confidence over time for \n"{label}"\n in time '
                    f"range {min_start_time}:{max_end_time}"
                )
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Confidence")
                ax.set_ylim(0, 1.05)
                ax.set_xlim(0 - pad, (max_end_time - min_start_time + pad))

                # ============================
                # Ground truth
                # ============================
                # Bar plt to show bars where the "true" time ranges are for the activity.
                xs_bars = [p - min_start_time for p in gt_ranges["start"].tolist()]
                ys_gt_regions = [1] * gt_ranges.shape[0]
                bar_widths = [p for p in gt_ranges["end"] - gt_ranges["start"]]
                ax.bar(
                    xs_bars,
                    ys_gt_regions,
                    width=bar_widths,
                    align="edge",
                    color="lightgreen",
                    label="Ground truth",
                )

                if custom_range:
                    assert (
                        len(custom_range) == 2
                    ), "Assuming only two float values for custom range"
                    xs_bars2 = [custom_range[0]]
                    ys_height = [1.025]  # [0.1]
                    bar_widths2 = [custom_range[1] - custom_range[0]]
                    ys_bottom = [0]  # [1.01]
                    # TODO: Make this something that is added by clicking?
                    ax.bar(
                        xs_bars2,
                        ys_height,
                        width=bar_widths2,
                        bottom=ys_bottom,
                        align="edge",
                        color=custom_range_color,
                        alpha=0.5,
                    )

                # ============================
                # Detections
                # ============================
                xs2_bars = [p - min_start_time for p in det_ranges["start"].tolist()]
                ys2_det_regions = [p for p in det_ranges["conf"]]
                bar_widths2 = [p for p in det_ranges["end"] - det_ranges["start"]]
                ax.bar(
                    xs2_bars,
                    ys2_det_regions,
                    width=bar_widths2,
                    align="edge",
                    edgecolor="blue",
                    fill=False,
                    label="Detections",
                )

                ax.legend(loc="upper right")
                ax.plot

                # ============================
                # Save
                # ============================
                # plt.show()
                activity_plot_dir = Path(os.path.join(self.output_dir, "activities"))
                activity_plot_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(f"{str(activity_plot_dir)}/{label.replace(' ', '_')}.png")
                plt.close(fig)
            else:
                if gt_ranges.empty:
                    log.warning(f'No gt found for "{label}"')
                if det_ranges.empty:
                    log.warning(f'No detections found for "{label}"')
