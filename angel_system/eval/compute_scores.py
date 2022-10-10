import logging
from sklearn.metrics import precision_recall_fscore_support


log = logging.getLogger("ptg_eval")


class EvalMetrics():
    def __init__(self, labels, gt, dets, detect_intersection_thr=0.1, output_fn='metrics/txt'):
        self.labels = labels
        self.gt = gt
        self.dets = dets
        self.detect_intersection_thr = detect_intersection_thr
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
        # TODO: write to plot

    def precision_recall_f1(self):
        y_true = []
        y_pred = []

        # calulcating metrics based on detector frequency
        time_ranges = self.dets[['start', 'end']].drop_duplicates()

        for i, time in time_ranges.iterrows():
            # Determine best detection
            det_overlap = self.dets[(self.dets['start'] == time['start']) & (self.dets['end'] == time['end'])]
            best_det = det_overlap.loc[det_overlap['conf'].idxmax()]

            if best_det['detect_intersection'] > self.detect_intersection_thr:
                gt_overlap = self.gt[(self.gt['end'] >= time['start']) & (time['end'] >= self.gt['end'])]

                gt = gt_overlap.iloc[0]
                y_true.append(self.labels.loc[self.labels['class'] == gt['class']].iloc[0]['id'])
                y_pred.append(self.labels.loc[self.labels['class'] == best_det['class']].iloc[0]['id'])
            else:
                # If there is no gt, gt is background
                y_true.append(self.labels.loc[self.labels['class'] == "background"].iloc[0]['id'])
                y_pred.append(self.labels.loc[self.labels['class'] == best_det['class']].iloc[0]['id'])

        labels = [row['id'] for i, row in self.labels.iterrows()]
        label_names = [row['class'] for i, row in self.labels.iterrows()]
        avg_precision, avg_recall, avg_fscore, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted')
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, labels=labels)

        # ============================
        # Save
        # ============================
        # Save to file
        with open(self.output_fn, "a") as f:
            f.write("\n")
            for str_, avg_val, val in zip(['precision', 'recall', 'fscore'], [avg_precision, avg_recall, avg_fscore], [precision, recall, fscore]):
                f.write(f"{str_}: {avg_val}\n")
                f.write(f"{str_} per label: \n")
                for l, v in zip(label_names, val):
                    f.write(f"\t{l}: {v}\n")
                f.write("\n")
            