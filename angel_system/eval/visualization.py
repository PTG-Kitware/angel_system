import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, AnchoredText
import numpy as np
import PIL
from pathlib import Path
import os

from angel_system.eval.support_functions import GlobalValues


def plot_activity_confidence(label, gt_ranges, det_ranges, output_dir, custom_range=None, custom_range_color="red"):
    """
    Plot activity confidences
    :param label: String label of the activity class predictions to render.
    :param gt_ranges: A sequence of tuples indicating the starting and ending time of ground-truth
                      time ranges the label activity occurred in the image sequence.
    :param custom_range: Optional tuple indicating the starting and ending times of an additional
                         range to highlight in addition to the `gt_ranges`.
    :param custom_range_color: The color of the additional range to be drawn. If not set, we will
                               use "red".
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Window Confidence over time for \"{label}\"")
    ax1.set_title("Real Detected Actions")
    ax2.set_title("Optimal Detected Actions")
    for ax in [ax1, ax2]:
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(GlobalValues.slice_time_ranges[0][0] - 1,
                    GlobalValues.slice_time_ranges[-1][1] + 1)
        # plt.yscale("log")

    # ============================
    # Ground truth
    # ============================
    # Bar plt to show bars where the "true" time ranges are for the activity.
    xs_bars = [p["time"][0] for p in gt_ranges]
    ys_gt_regions = [1 for _ in gt_ranges]
    bar_widths = [(p["time"][1]-p["time"][0]) for p in gt_ranges]
    ax1.bar(xs_bars, ys_gt_regions, width=bar_widths, align="edge", color="lightgreen", label="Ground truth")
    ax2.bar(xs_bars, ys_gt_regions, width=bar_widths, align="edge", color="lightgreen", label="Ground truth")

    if custom_range:
        assert len(custom_range) == 2, "Assuming only two float values for custom range"
        xs_bars2 = [custom_range[0]]
        ys_height = [1.025] #[0.1]
        bar_widths2 = [custom_range[1]-custom_range[0]]
        ys_bottom = [0] #[1.01]
        # TODO: Make this something that is added be clicking?
        ax1.bar(xs_bars2, ys_height,
               width=bar_widths2, bottom=ys_bottom, align="edge",
               color=custom_range_color, alpha=0.5)
        ax2.bar(xs_bars2, ys_height,
                width=bar_widths2, bottom=ys_bottom, align="edge",
                color=custom_range_color, alpha=0.5)

    # ============================
    # Optimal detections
    # ============================
    # Line plot to show detector confidence for a window slice.
    # Plotted point at the median frame-time of the window predicted over.
    xs_slice_median_time = [
        GlobalValues.all_image_times[int(np.average(idx_rng))]
        for idx_rng in GlobalValues.slice_index_ranges
    ]
    ys_pred_conf = [one_pred[label] for one_pred in GlobalValues.slice_preds]
    err_bar_widths = [(t_range[1] - t_range[0]) / 2.0 for t_range in GlobalValues.slice_time_ranges]
    
    errorbar = ax2.errorbar(xs_slice_median_time, ys_pred_conf,
                           # xerr=err_bar_widths,
                           linewidth=1,
                           # elinewidth=1, 
                           fmt=".b-", label="Detections")

    ax2.legend(loc="upper right")
    ax2.plot

    # ============================
    # Actual Detections
    # ============================
    det_ranges = det_ranges[label]

    xs2_bars = [p["time"][0] for p in det_ranges]
    ys2_det_regions = [p["conf"] for p in det_ranges]
    bar_widths2 = [(p["time"][1] - p["time"][0]) for p in det_ranges]
    ax1.bar(xs2_bars, ys2_det_regions, width=bar_widths2, align="edge", edgecolor="blue", fill=False, label="Detections")

    ax1.legend(loc="upper right")
    ax1.plot

    #plt.show()
    Path(os.path.join(output_dir, "plots/activities")).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output_dir}/plots/activities/{label.replace(' ', '_')}.png")
