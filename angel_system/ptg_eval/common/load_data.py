import pandas as pd
import logging
import re
from ast import literal_eval


logging.basicConfig(level = logging.INFO)
log = logging.getLogger("ptg_eval_common")

RE_FILENAME_TIME = re.compile(r"frame_\d+_(\d+_\d+).\w+")


def time_from_name(fname):
    """
    Extract the float timestamp from the filename.

    :param fname: Filename of an image in the format
        frame_<frame number>_<seconds>_<nanoseconds>.<extension>

    :return: timestamp (float) in seconds
    """
    time = RE_FILENAME_TIME.match(fname).groups()[0].split('_')
    return float(time[0]) + (float(time[1]) * 1e-9)


def load_from_file(gt_fn, detections_fn):
    """
    Load the labels, ground truth, and detections from files

    :param gt_fn: Path to the ground truth feather file
    :param detections_fn: Path to the extracted activity detections 
        in json format

    :return: Tuple(array of class labels (str), pandas dataframe of ground truth, 
        pandas dataframe of detections)
    """
    # ============================
    # Load truth annotations
    # ============================
    gt_f = pd.read_feather(gt_fn)
    # Keys: class, start_frame,  end_frame, exploded_ros_bag_path

    gt = []
    for i, row in gt_f.iterrows():
        g = {
            'class': row["class"].lower().strip(),
            'start': time_from_name(row["start_frame"]),
            'end': time_from_name(row["end_frame"])
        }
        gt.append(g)

    log.info(f"Loaded ground truth from {gt_fn}")
    gt = pd.DataFrame(gt)

    # ============================
    # Load detections from
    # extracted ros bag
    # ============================
    detections_input = [det for det in (literal_eval(s) for s in open(detections_fn))][0]
    detections = []

    for dets in detections_input:
        good_dets = {}
        for l, conf in zip(dets["label_vec"], dets["conf_vec"]):
            good_dets[l] = conf

        for l in dets["label_vec"]:
            d = {
                'class': l.lower().strip(),
                'start': dets["source_stamp_start_frame"],
                'end': dets["source_stamp_end_frame"],
                'conf': good_dets[l]
            }
            detections.append(d)
    detections = pd.DataFrame(detections)
    log.info(f"Loaded detections from {detections_fn}")

    # ============================
    # Load labels
    # ============================
    # grab all labels present in data
    labels = list(set([l.lower().strip().rstrip('.') for l in detections['class'].unique()]))

    log.debug(f"Labels: {labels}")

    return labels, gt, detections
