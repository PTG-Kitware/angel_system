import logging
from sklearn.metrics import f1_score, precision_score, recall_score


log = logging.getLogger("ptg_eval")


class EvalMetrics():
    def __init__(self, labels, gt, dets, output_fn):
        self.labels = labels
        self.gt = gt
        self.dets = dets
        self.output_fn = output_fn

    def iou_per_activity_label(self):
        """
        Calculate the iou per activity label

        :param labels: Pandas df with columns id (int) and class (str)
        :param gt: Dict of activity start and end time ground truth values, organized by label keys
        :param dets: Dict of activity start and end time detections with confidence values, organized by label keys

        :return: Tuple(Average IoU across all classes, Dictionary mapping class labels to their average IoU scores, dets)
        """
        iou_per_label = {}
        for i, row in self.labels.iterrows():
            label = row['class']

            ious = []
            iou_counts = 0

            gt_ranges = self.gt.loc[self.gt['class'] == label]
            det_ranges = self.dets.loc[self.dets['class'] == label]

            for i, det_range in det_ranges.iterrows():
                # find overlapping gt if there is one
                gt_overlap = gt_ranges[(gt_ranges['end'] >= det_range['start']) & (det_range['end'] >= gt_ranges['end'])]
                
                if gt_overlap.empty:
                    # Insertion, didn't find any gt to calculate with
                    iou = 0
                    ious.append(iou)
                    iou_counts += 1

                    # Update dets
                    self.dets.loc[i, 'iou'] = iou

                    continue

                if gt_overlap.shape[0] > 1:
                    log.warning("Found more than one overlapping ground truth")
                gt_overlap = gt_overlap.iloc[0] # assuming only one gt in range

                det_area = (det_range['end'] - det_range['start'])
                
                # coordinates of the intersection interval
                i_left = max(det_range['start'], gt_overlap['start'])  # "right"-most left boundary
                i_right = min(det_range['end'], gt_overlap['end'])  # "left"-most right boundary
                intersection_area = i_right - i_left

                gt_area = intersection_area

                union_area = gt_area + det_area - intersection_area
                iou = intersection_area / union_area

                ious.append(iou)
                iou_counts += 1

                # Update dets
                self.dets.loc[i, 'iou'] = iou

            if not ious:
                # there are no detections for this label
                label_iou = 0
            else:
                label_iou = sum(ious) / iou_counts
            iou_per_label[label] = label_iou

        overall_iou = sum(iou_per_label.values()) / len(iou_per_label.values())

        # ============================
        # Save
        # ============================
        # Save to file
        with open(self.output_fn, "w") as f:
            f.write(f"IoU: {overall_iou}\n")
            f.write(f"IoU Per Label:\n")
            for k, v in iou_per_label.items():
                f.write(f"\t{k}: {v}\n")

        return self.dets
