import logging
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix


log = logging.getLogger("ptg_eval_activity")


class EvalMetrics():
    def __init__(self, labels, gt_true_mask, window_class_scores, output_dir=''):
        """
        :param labels: Array of class labels (str)
        :param gt_true_mask: Matrix of size (number of time windows x number classes) where 
            all time windows present have a valid classification scoring. 
            True indicates a true class example, False inidcates a false class example. 
            There should only be one True value per row.
        :param window_class_scores: Matrix of size (number of time windows x number classes)
            filled with confidence scores.
        :param output_dir: Directory to write the text files containing metrics
        """
        self.labels = labels
        self.gt_true_mask = gt_true_mask
        self.window_class_scores = window_class_scores

        self.output_dir = Path(os.path.join(output_dir, "metrics/"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(self.output_dir)

    def precision_recall(self, best_worst=5):
        """
        Calculate the average precision and recall per class
        and output to a file.

        :param best_worst: Number of best and worst precision values to 
            report to a file. The file will also include NaN values in
            a separate section. 
        """
        with open(f"{self.output_dir}/metrics.txt", "w") as f:
            n_classes = len(self.labels)
            label_vec = np.arange(n_classes)
            true_idxs = np.where(self.gt_true_mask==True)[1]
            pred_idxs = np.argmax(self.window_class_scores, axis=1)

            cm = confusion_matrix(true_idxs, pred_idxs,
                                  labels=label_vec,
                                  normalize="true")

            recall = np.diag(cm) / np.sum(cm, axis=1) # rows
            precision = np.diag(cm) / np.sum(cm, axis=0) # columns
            f.write(f'recall: {np.mean(recall)}\n')
            for r, l in zip(recall, self.labels):
                f.write(f"{l}: {r}\n")
            f.write('\n')
            f.write(f'precision: {np.mean(precision)}\n')
            for p, l in zip(precision, self.labels):
                f.write(f"{l}: {p}\n")

        with open(f"{self.output_dir}/best_worst_{best_worst}_precision.txt", "w") as f:
            cleaned_list = [(x, l) for x, l in zip(precision, self.labels) 
                            if str(x) != 'nan']
            sorted_precision = sorted(cleaned_list, reverse=True)

            best_x = sorted_precision[:5]
            worst_x = sorted_precision[-5:]

            nan_values = [(x, l) for x, l in zip(precision, self.labels) 
                          if str(x) == 'nan']

            f.write(f'best {best_worst} precision values:\n')
            for p, l in best_x:
                f.write(f'{l}: {p}\n')
            f.write('\n')
            f.write(f'worst {best_worst} precision values:\n')
            for p, l in worst_x:
                f.write(f'{l}: {p}\n')
            f.write('\n')
            f.write('NaN precision values:\n')
            for p, l in nan_values:
                f.write(f'{l}\n')
   