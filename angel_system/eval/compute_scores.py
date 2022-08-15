import logging


log = logging.getLogger("ptg_eval")


def iou_per_activity_label(labels, gt, dets):
    """
    Calculate the iou per activity label

    :param labels: List of String labels of the activity classes
    :param gt: Dict of activity start and end time ground truth values, organized by label keys
    :param dets: Dict of activity start and end time detections with confidence values, organized by label keys

    :return: Tuple(Average IoU across all classes, Dictionary mapping class labels to their average IoU scores)
    """
    iou_per_label = {}
    for label in labels:
        ious = []
        iou_counts = 0

        gt_ranges = gt[label]
        det_ranges = dets[label]

        for det_range in det_ranges:
            # find overlapping gt if there is one
            gt_overlap = [(gt_range["time"][0], gt_range["time"][1])
                          for gt_range in gt_ranges
                          if ((gt_range["time"][1] >= det_range["time"][0]) and (det_range["time"][1] >= gt_range["time"][1]))]

            if not gt_overlap:
                # Insertion, didn't find any gt to calculate with
                iou = 0
                ious.append(iou)
                iou_counts += 1
                continue

            if len(gt_overlap) > 1:
                logging.warning("Found more than one overlapping ground truth")
            gt_overlap = gt_overlap[0] # assuming only one gt in range

            det_area = (det_range["time"][1] - det_range["time"][0])

            # find intersection area
            if det_range["time"][0] <= gt_overlap[0] <= det_range["time"][1]:
                # gt starts after det
                intersection_start = gt_overlap[0]
            else:
                # gt starts before detection, ignore part of gt that happens before our detection range
                intersection_start = det_range["time"][0]

            if det_range["time"][0] <= gt_overlap[1] <= det_range["time"][1]:
                # gt ends before detection
                intersection_end = gt_overlap[1]
            else:
                # detection ends before gt, ignore part of gt that happens after the detection range
                intersection_end = det_range["time"][1]

            gt_area = (intersection_end - intersection_start)

            intersection_area = (intersection_end - intersection_start)
            union_area = gt_area + det_area - intersection_area
            iou = intersection_area / union_area

            ious.append(iou)
            iou_counts += 1

        if not ious:
            # there are no detections for this label
            label_iou = 0
        else:
            label_iou = sum(ious) / iou_counts
        iou_per_label[label] = label_iou

    overall_iou = sum(iou_per_label.values()) / len(iou_per_label.values())

    log.info(f"IoU: {overall_iou}")
    log.info(f"IoU Per Label:")
    for k, v in iou_per_label.items():
        log.info(f"\t{k}: {v}")

    return overall_iou, iou_per_label
