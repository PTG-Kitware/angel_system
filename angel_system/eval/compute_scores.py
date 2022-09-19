import logging
from sklearn.metrics import f1_score, precision_score, recall_score


log = logging.getLogger("ptg_eval")


class EvalMetrics():
    def __init__(self, labels, gt, dets, output_fn):
        self.labels = labels
        self.gt = gt
        self.dets = dets
        self.output_fn = output_fn

    def detect_intersection_per_activity_label(self):
        """
        Calculate the detection intersection per activity label
        :param labels: Pandas df with columns id (int) and class (str)
        :param gt: Dict of activity start and end time ground truth values, organized by label keys
        :param dets: Dict of activity start and end time detections with confidence values, organized by label keys.
                     This will be modified in place to add a "detection intersection" key to each detection
        """
        detect_intersection_per_label = {}
        for i, row in self.labels.iterrows():
            label = row['class']

            detect_intersections = []
            detect_intersection_counts = 0

            gt_ranges = self.gt.loc[self.gt['class'] == label]
            det_ranges = self.dets.loc[self.dets['class'] == label]

            for i, det_range in det_ranges.iterrows():
                # find overlapping gt if there is one
                gt_overlap = gt_ranges[(gt_ranges['end'] >= det_range['start']) & (det_range['end'] >= gt_ranges['end'])]
                
                if gt_overlap.empty:
                    # Insertion, didn't find any gt to calculate with
                    detect_intersection = 0
                    detect_intersections.append(detect_intersection)
                    detect_intersection_counts += 1

                    # Update dets
                    self.dets.loc[i, 'detect_intersection'] = detect_intersection

                    continue

                if gt_overlap.shape[0] > 1:
                    log.warning("Found more than one overlapping ground truth")
                gt_overlap = gt_overlap.iloc[0] # assuming only one gt in range

                det_area = (det_range['end'] - det_range['start'])
                
                # coordinates of the intersection interval
                i_left = max(det_range['start'], gt_overlap['start'])  # "right"-most left boundary
                i_right = min(det_range['end'], gt_overlap['end'])  # "left"-most right boundary
                intersection_area = i_right - i_left

                detect_intersection = intersection_area / det_area

                detect_intersections.append(detect_intersection)
                detect_intersection_counts += 1

                # Update dets
                self.dets.loc[i, 'detect_intersection'] = detect_intersection

            if not detect_intersections:
                # there are no detections for this label
                label_detect_intersection = 0
            else:
                label_detect_intersection = sum(detect_intersections) / detect_intersection_counts
            detect_intersection_per_label[label] = label_detect_intersection

        overall_detect_intersection = sum(detect_intersection_per_label.values()) / len(detect_intersection_per_label.values())

        # ============================
        # Save
        # ============================
        # Save to file
        with open(self.output_fn, "w") as f:
            f.write(f"detection intersection: {overall_detect_intersection}\n")
            f.write(f"detection intersection Per Label:\n")
            for k, v in detect_intersection_per_label.items():
                f.write(f"\t{k}: {v}\n")
