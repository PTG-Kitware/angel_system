import logging
from sklearn.metrics import average_precision_score
import numpy as np

log = logging.getLogger("ptg_eval")


class EvalMetrics():
    def __init__(self, labels, gt_true_mask, dets_per_valid_time_w, output_dir=''):
        """
        :param labels: Array of class labels (str)
        :param gt_true_mask: Matrix of size (number of valid time windows x number classes) where True
            indicates a true class example, False inidcates a false class example. There should only be one
            True value per row
        :param dets_per_valid_time_w: Matrix of size (number of valid time windows x number classes)
            filled with the max confidence score per class for any detections in the time window
        :param output_fn: Path (str) to a file
        """
        self.labels = labels
        self.gt_true_mask = gt_true_mask
        self.dets_per_valid_time_w = dets_per_valid_time_w

        self.output_fn = f"{output_dir}/metrics.txt"

    def precision(self):
        """
        Calculate the average precision per class and output to a file
        """
        with open(self.output_fn, "w") as f:
            f.write('precision: \n')
            for id, label in enumerate(self.labels):
                class_dets_per_time_w = self.dets_per_valid_time_w[:, id]
                mask_per_class = self.gt_true_mask[:, id]

                class_dets_per_time_w.shape = (-1, 1)
                mask_per_class.shape = (-1, 1)

                precision = average_precision_score(mask_per_class, class_dets_per_time_w)

                f.write(f'{self.labels[id]}: {precision}\n')
