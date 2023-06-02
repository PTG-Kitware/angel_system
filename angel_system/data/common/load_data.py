from ast import literal_eval
from dataclasses import asdict, fields
import logging
import json
import re
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import pandas as pd
import numpy as np

from angel_system.ptg_eval.common.structures import Activity


log = logging.getLogger("ptg_eval_common")

RE_FILENAME_TIME = re.compile(r"frame_(?P<frame>\d+)_(?P<ts>\d+(?:_|.)\d+).(?P<ext>\w+)")
def time_from_name(fname):
    """
    Extract the float timestamp from the filename.

    :param fname: Filename of an image in the format
        frame_<frame number>_<seconds>_<nanoseconds>.<extension>

    :return: timestamp (float) in seconds
    """
    fname = os.path.basename(fname)
    match = RE_FILENAME_TIME.match(fname)
    time = match.group('ts')
    if '_' in time:
        time = time.split('_')
        time = float(time[0]) + (float(time[1]) * 1e-9)
    elif '.' in time:
        time = float(time)

    frame = match.group('frame')
    return int(frame), time

def load_from_file(gt_fn, detections_fn) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
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
            'class_label': row["class"].lower().strip(),
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
                'class_label': l.lower().strip(),
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
    labels = list(set([l.lower().strip().rstrip('.') for l in detections['class_label'].unique()]))

    log.debug(f"Labels: {labels}")

    return labels, gt, detections

def activities_from_dive_csv(filepath: str) -> List[Activity]:
    """
    Load from a DIVE output CSV file a sequence of ground truth activity
    annotations.

    Class labels are converted to lower-case and stripped of any extraneous
    whitespace.

    :param filepath: Filesystem path to the CSV file.
    :return: List of loaded activity annotations.
    """
    print(f"Loading ground truth activities from: {filepath}")
    df = pd.read_csv(filepath)
    # There may be additional metadata rows. Filter out rows whose first column
    # value starts with a `#`.
    df = df[df[df.keys()[0]].str.contains('^[^#]')]
    # Create a mapping of detection/track ID to the activity annotation
    id_to_activity: Dict[int, Activity] = {}
    for row in df.iterrows():
        i, s = row
        a_id = int(s[0])
        frame, time = time_from_name(s[1])
        if a_id not in id_to_activity:
            id_to_activity[a_id] = Activity(
                s[9].lower().strip(), # class label
                time, # start
                np.inf, # end 
                frame, # start frame
                np.inf, # end frame
                1.0, # conf
            )
        else:
            # There's a struct in there, update it.
            a = id_to_activity[a_id]
            # Activity should not already have an end time assigned.
            assert a.end is np.inf, (
                f"More than 2 entries observed for activity track ID {a_id}."
            )
            id_to_activity[a_id] = Activity(
                a.class_label,
                a.start,
                time,
                a.start_frame,
                frame,
                a.conf,
            )
    # Assert that all activities have been assigned an associated end time.
    assert np.inf not in {a.end for a in id_to_activity.values()}, (
        f"Some activities in source CSV do not have corresponding end time "
        f"entries: {filepath}"
    )
    return list(id_to_activity.values())

def activities_from_ros_export_json(filepath: str) -> Tuple[List[str], List[Activity]]:
    """
    Load a number of predicted activities from a JSON file that is the result
    of bag extraction of a `ActivityDetection.msg` message topic.

    Class labels are converted to lower-case and stripped of any extraneous
    whitespace.

    See message file located here: ros/angel_msgs/msg/ActivityDetection.msg
    See bag extraction conversion logic located here: ros/angel_utils/scripts/bag_extractor.py

    :param filepath: Filesystem path to the JSON file.
    :return: List activity predicted labels and a list of loaded activity
        annotations (predicted).
    """
    log.info(f"Loading predicted activities from: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    activity_seq: List[Activity] = []
    label_vec = None
    for act_i, act_json in enumerate(data):
        # Cache/check stable labels vector.
        if label_vec is not None:
            assert act_json['label_vec'] == label_vec, (
                f"Inconsistent label set in loaded JSON activities. Activity "
                f"index {act_i} showed inconsistency."
            )
        else:
            label_vec = act_json['label_vec']

        # This activity window start/end times in seconds.
        act_start = act_json['source_stamp_start_frame']
        act_end = act_json['source_stamp_end_frame']

        # Create a separate activity item per prediction
        for lbl, conf in zip(label_vec, act_json['conf_vec']):
            activity_seq.append(Activity(
                lbl.lower().strip(),
                act_start, act_end, conf
            ))
    # normalize output label vec just like activity label treatment.
    label_vec = [lbl.lower().strip() for lbl in label_vec]
    return label_vec, activity_seq


def activities_as_dataframe(act_sequence: Sequence[Activity]) -> pd.DataFrame:
    """
    Transform a sequence of activity structures into a pandas dataframe whose
    keys are the attributes of the Activity structure.

    :param act_sequence: Sequence of activities to convert.
    :return: Converted data frame instance.
    """
    # `fields` introspection yields fields in order of specification in the
    # dataclass.
    # This method is much faster than using `asdict`, likely due to not using
    # deep-copy.
    return pd.DataFrame(
        {field.name: getattr(obj, field.name) for field in fields(obj)}
        for obj in act_sequence
    )
