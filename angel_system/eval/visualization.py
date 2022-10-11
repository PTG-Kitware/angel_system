import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, AnchoredText
import numpy as np
import PIL
from pathlib import Path
import os
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay


class EvalVisualization:
    def __init__(self, labels, gt, dets, output_dir):
        """
        :param labels: Pandas df with columns id (int) and class (str)
        :param gt: Dict of activity start and end time ground truth values, organized by label keys
        :param dets: Dict of activity start and end time detections with confidence values, organized by label keys
        :param output_dir: Directory to write the plots to
        """
        self.output_dir = Path(os.path.join(output_dir, "plots/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(self.output_dir)

        self.labels = labels
        self.gt = gt
        self.dets = dets

    def plot_activities_confidence(self, custom_range=None, custom_range_color="red"):
        """
        Plot activity confidences over time

        :param custom_range: Optional tuple indicating the starting and ending times of an additional
                             range to highlight in addition to the `gt_ranges`.
        :param custom_range_color: The color of the additional range to be drawn. If not set, we will
                                   use "red".
        """
        for i, row in self.labels.iterrows():
            label = row['class']

            gt_ranges = self.gt.loc[self.gt['class'] == label]
            det_ranges = self.dets.loc[self.dets['class'] == label]
            if not gt_ranges.empty and not det_ranges.empty:
                # ============================
                # Setup figure
                # ============================
                # Determine time range to plot
                min_start_time = min(gt_ranges['start'].min(), det_ranges['start'].min())
                max_end_time = max(gt_ranges['end'].max(), det_ranges['end'].max())
                total_time_delta = max_end_time - min_start_time
                pad = 0.05 * total_time_delta

                # Setup figure
                fig = plt.figure(num=None, figsize=(18.3, 14), dpi=80)
                plt.rc('font', **{'size': 35})
                plt.rc('axes', linewidth=4)
                ax = fig.add_subplot(111)
                ax.set_title(f"Window Confidence over time for \"{label}\"\n in time range {min_start_time}:{max_end_time}")
                ax.set_xlabel("Time (seconds)", fontsize=50)
                ax.set_ylabel("Confidence", fontsize=50)
                ax.set_ylim(0, 1.05)
                ax.set_xlim(0-pad, (max_end_time - min_start_time + pad))

                # ============================
                # Ground truth
                # ============================
                # Bar plt to show bars where the "true" time ranges are for the activity.
                xs_bars = [p - min_start_time for p in gt_ranges['start'].tolist()]
                ys_gt_regions = [1] * gt_ranges.shape[0]
                bar_widths = [p for p in gt_ranges['end']-gt_ranges['start']]
                ax.bar(xs_bars, ys_gt_regions, width=bar_widths, align="edge",
                       color="lightgreen", label="Ground truth", linewidth=16)

                if custom_range:
                    assert len(custom_range) == 2, "Assuming only two float values for custom range"
                    xs_bars2 = [custom_range[0]]
                    ys_height = [1.025] #[0.1]
                    bar_widths2 = [custom_range[1]-custom_range[0]]
                    ys_bottom = [0] #[1.01]
                    # TODO: Make this something that is added be clicking?
                    ax.bar(xs_bars2, ys_height,
                        width=bar_widths2, bottom=ys_bottom, align="edge",
                        color=custom_range_color, alpha=0.5)

                # ============================
                # Detections
                # ============================
                xs2_bars = [p - min_start_time for p in det_ranges['start'].tolist()]
                ys2_det_regions = [p for p in det_ranges['conf']]
                bar_widths2 = [p for p in det_ranges['end']-det_ranges['start']]
                ax.bar(xs2_bars, ys2_det_regions, width=bar_widths2, align="edge", edgecolor="blue", fill=False, label="Detections")

                ax.legend(loc="upper right", fontsize=40)
                ax.plot

                # ============================
                # Save
                # ============================
                #plt.show()
                activity_plot_dir = Path(os.path.join(self.output_dir, "activities"))
                activity_plot_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(f"{str(activity_plot_dir)}/{label.replace(' ', '_')}.png")
            else:
                log.warning(f"No detections/gt found for \"{label}\"")

    def plot_pr_curve(self, detect_intersection_thr=0.1):
        """
        Plot the PR curve for each label

        :param detect_intersection_thr: detection intersection threshold
        """
        # ============================
        # Setup figure
        # ============================
        fig, ax = plt.figure(num=None, figsize=(18.3, 14), dpi=80)
        plt.rc('font', **{'size': 22})
        plt.rc('axes', linewidth=4)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title("Precision vs. Recall")

        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.labels)))

        # ============================
        # Add F1 score
        # ============================
        fscores = np.linspace(0.2, 0.8, num=4)
        for f_score in fscores:
            x = np.linspace(0.001, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        # ============================
        # Get PR plot per class
        # ============================
        for i, row in self.labels.iterrows():
            id = row['id']
            label = row['class']

            det_ranges = self.dets.loc[self.dets['class'] == label]

            truth = [1 if det['detect_intersection'] > detect_intersection_thr else 0 for i, det in det_ranges.iterrows()]
            pred = [det['conf'] for i, det in det_ranges.iterrows()]

            PrecisionRecallDisplay.from_predictions(truth, pred).plot(ax=ax, name=f"class {id}", color=colors[i])

        # ============================
        # Save
        # ============================
        # Add legend and f1 curves to plot
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        ax.legend(handles=handles, labels=labels, loc="best")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

        fig.savefig(f"{self.output_dir}/PR.png")
