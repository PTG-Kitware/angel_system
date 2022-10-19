import logging
from sklearn.metrics import average_precision_score
import numpy as np

log = logging.getLogger("ptg_eval")


class EvalMetrics():
    def __init__(self, labels, gt_true_mask, dets_per_valid_time_w, output_fn='metrics.txt'):
        """
        :param labels: Array of class labels (str)
        :param gt_true_pos_mask: Matrix of size (number of valid time windows x number classes) where True
            indicates a true class example, False inidcates a false class example. There should only be one
            True value per row
        :param dets_per_valid_time_w: Matrix of size (number of valid time windows x number classes)
            filled with the max confidence score per class for any detections in the time window
        :param output_fn: Path (str) to a file
        """
        self.labels = labels
        self.gt_true_mask = gt_true_mask
        self.dets_per_valid_time_w = dets_per_valid_time_w

        self.output_fn = output_fn

    def precision(self):
        """
        Calculate the average precision per class and output to a file
        """
        with open(self.output_fn, "w") as f:
            f.write('precision: \n')
            for id, label in enumerate(self.labels):
                class_dets_per_time_w = self.dets_per_valid_time_w[:, id]
                mask_per_class = self.gt_true_mask[:, id]

                ts = class_dets_per_time_w[mask_per_class]
                fs = class_dets_per_time_w[~mask_per_class]

                s = np.hstack([ts, fs]).T
                y_true = np.hstack([np.ones(len(ts), dtype=bool),
                        np.zeros(len(fs), dtype=bool)]).T
                s.shape = (-1, 1)
                y_true.shape = (-1, 1)

                precision = average_precision_score(y_true, s)
                # TODO add recall

                f.write(f'{self.labels[id]}: {precision}\n')
