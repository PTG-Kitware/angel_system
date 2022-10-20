import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import logging


log = logging.getLogger("ptg_eval")

plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams.update({'font.size': 12})


class EvalVisualization:
    def __init__(self, labels, gt_true_mask, window_class_scores, output_dir=''):
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

    def plot_pr_curve(self):
        """
        Plot the PR curve for each label and the micro 
        average PR curve over all classes
        """
        log.debug("Plotting PR curves")
        pr_plot_dir = Path(os.path.join(self.output_dir, "pr"))
        pr_plot_dir.mkdir(parents=True, exist_ok=True)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.labels)))

        # ============================
        # Average values
        # ============================
        all_y_true = self.gt_true_mask.ravel()
        all_s = self.window_class_scores.ravel()
        precision_micro, recall_micro, _ = precision_recall_curve(all_y_true, all_s)
        average_precision_micro = average_precision_score(all_y_true, all_s, average="micro")

        # ============================
        # Get PR plot per class 
        # ============================
        for id, label in enumerate(self.labels):
            class_dets_per_time_w = self.window_class_scores[:, id]
            mask_per_class = self.gt_true_mask[:, id]

            class_dets_per_time_w.shape = (-1, 1)
            mask_per_class.shape = (-1, 1)

            display = PrecisionRecallDisplay.from_predictions(mask_per_class, class_dets_per_time_w)

            # ============================
            # Plot
            # ============================
            fig, ax = plt.subplots(figsize=(14, 8))

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_title("Precision vs. Recall")
            
            # ============================
            # Add F1 score 
            # ============================
            fscores = np.linspace(0.2, 0.8, num=4)
            for f_score in fscores:
                x = np.linspace(0.001, 1)
                y = f_score * x / (2 * x - f_score)
                (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2, linestyle='dashed')
                plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

            # plot average values
            av_display = PrecisionRecallDisplay(
                recall=recall_micro,
                precision=precision_micro,
                average_precision=average_precision_micro,
            )
            av_display.plot(ax=ax, name="Micro-averaged over all classes", 
                            color="navy", linestyle=":", linewidth=4)

            # plot class values
            display.plot(ax=ax, name=label, color=colors[id])

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

            fig.savefig(f"{pr_plot_dir}/{label.replace(' ', '_')}.png")
            plt.close(fig)

    def plot_roc_curve(self):
        """
        Plot the ROC curve for each label and the macro 
        average ROC curve over all classes
        """
        log.debug("Plotting ROC curves")
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.labels)))

        roc_plot_dir = Path(os.path.join(self.output_dir, "roc"))
        roc_plot_dir.mkdir(parents=True, exist_ok=True)

        # ============================
        # Get ROC and AUC per class 
        # ============================
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for id, label in enumerate(self.labels):
            class_dets_per_time_w = self.window_class_scores[:, id]
            mask_per_class = self.gt_true_mask[:, id]

            class_dets_per_time_w.shape = (-1, 1)
            mask_per_class.shape = (-1, 1)

            fpr[id], tpr[id], _ = roc_curve(mask_per_class, class_dets_per_time_w)
            roc_auc[id] = auc(fpr[id], tpr[id])

        # ============================
        # Average values
        # ============================
        n_classes = len(self.labels)

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # ============================
        # Plot
        # ============================
        for id, label in enumerate(self.labels):
            # ============================
            # Setup figure
            # ============================
            fig, ax = plt.subplots(figsize=(14, 8))

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_title("Receiver Operating Characteristic (ROC)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")

            # plot average values
            ax.plot(
                fpr["macro"],
                tpr["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )

            # plot class values
            ax.plot(fpr[id], tpr[id], color=colors[id], lw=2,
                    label=label + " (area = {0:0.2f})".format(roc_auc[id]))

            # ============================
            # Save
            # ============================
            ax.legend(loc="best")
            fig.savefig(f"{roc_plot_dir}/{label.replace(' ', '_')}.png")
            plt.close(fig)

    def confusion_mat(self):
        """
        Plot a confusion matrix of size (number of labels x number of labels)
        """
        log.debug("Plotting confusion matrix")
        fig, ax = plt.subplots(figsize=(20, 20))

        true_idxs = np.where(self.gt_true_mask==True)[1]
        pred_idxs = np.argmax(self.window_class_scores, axis=1)

        cm = confusion_matrix(true_idxs, pred_idxs, labels=range(len(self.labels)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=self.labels)
        disp.plot(ax=ax, xticks_rotation=90)
        
        plt.tight_layout()
        fig.savefig(f"{self.output_dir}/confusion_mat.png", pad_inches=5)
        plt.close(fig)

    def plot_activities_confidence(self, gt, dets, custom_range=None, custom_range_color="red"):
        """
        Plot activity confidences over time

        :param gt: Pandas dataframe of the ground truth
        :param dets: Pandas dataframe of all detections per class
        :param custom_range: Optional tuple indicating the starting and ending times of an additional
                                range to highlight in addition to the `gt_ranges`.
        :param custom_range_color: The color of the additional range to be drawn. If not set, we will
                                    use "red".
        """
        log.debug("Plotting activity confidences")
        for label in self.labels:
            gt_ranges = gt.loc[gt['class'] == label]
            det_ranges = dets.loc[dets['class'] == label]

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
                fig = plt.figure(figsize=(14, 6))
                ax = fig.add_subplot(111)
                ax.set_title(f"Window Confidence over time for \"{label}\"\n in time range {min_start_time}:{max_end_time}")
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Confidence")
                ax.set_ylim(0, 1.05)
                ax.set_xlim(0-pad, (max_end_time - min_start_time + pad))

                # ============================
                # Ground truth
                # ============================
                # Bar plt to show bars where the "true" time ranges are for the activity.
                xs_bars = [p - min_start_time for p in gt_ranges['start'].tolist()]
                ys_gt_regions = [1] * gt_ranges.shape[0]
                bar_widths = [p for p in gt_ranges['end']-gt_ranges['start']]
                ax.bar(xs_bars, ys_gt_regions, width=bar_widths, align="edge", color="lightgreen", label="Ground truth")

                if custom_range:
                    assert len(custom_range) == 2, "Assuming only two float values for custom range"
                    xs_bars2 = [custom_range[0]]
                    ys_height = [1.025] #[0.1]
                    bar_widths2 = [custom_range[1]-custom_range[0]]
                    ys_bottom = [0] #[1.01]
                    # TODO: Make this something that is added by clicking?
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

                ax.legend(loc="upper right")
                ax.plot

                # ============================
                # Save
                # ============================
                #plt.show()
                activity_plot_dir = Path(os.path.join(self.output_dir, "activities"))
                activity_plot_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(f"{str(activity_plot_dir)}/{label.replace(' ', '_')}.png")
                plt.close(fig)
            else:
                log.warning(f"No detections/gt found for \"{label}\"")
