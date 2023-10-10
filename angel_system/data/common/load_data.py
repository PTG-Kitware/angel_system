from ast import literal_eval
from dataclasses import asdict, fields
import logging
import json
import re
import os
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import pandas as pd
import numpy as np

from angel_system.data.common.structures import Activity, Step


log = logging.getLogger("ptg_data_common")


def sanitize_str(str_: str):
    """
    Convert string to lowercase and emove trailing whitespace and period.

    :param str_: Input text

    :return: ``str_`` converted to lowercase and stripped of trailing whitespace and period.
    :rtype: str
    """
    return str_.lower().strip(" .")


def Re_order(image_list, image_number):
    img_id_list = []
    for img in image_list:
        img_id, ts = time_from_name(img)
        img_id_list.append(img_id)
    img_id_arr = np.array(img_id_list)
    s = np.argsort(img_id_arr)
    new_list = []
    for i in range(image_number):
        idx = s[i]
        new_list.append(image_list[idx])
    return new_list


RE_FILENAME_TIME = re.compile(
    r"frame_(?P<frame>\d+)_(?P<ts>\d+(?:_|.)\d+).(?P<ext>\w+)"
)


def time_from_name(fname):
    """
    Extract the float timestamp from the filename.

    :param fname: Filename of an image in the format
        frame_<frame number>_<seconds>_<nanoseconds>.<extension>

    :return: timestamp (float) in seconds
    """
    fname = os.path.basename(fname)
    match = RE_FILENAME_TIME.match(fname)
    time = match.group("ts")
    if "_" in time:
        time = time.split("_")
        time = float(time[0]) + (float(time[1]) * 1e-9)
    elif "." in time:
        time = float(time)

    frame = match.group("frame")
    return int(frame), time


def load_from_file(
    gt_fn, detections_fn
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
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
            "class_label": sanitize_str(row["class"]),
            "start": time_from_name(row["start_frame"]),
            "end": time_from_name(row["end_frame"]),
        }
        gt.append(g)

    log.info(f"Loaded ground truth from {gt_fn}")
    gt = pd.DataFrame(gt)

    # ============================
    # Load detections from
    # extracted ros bag
    # ============================
    detections_input = [det for det in (literal_eval(s) for s in open(detections_fn))][
        0
    ]
    detections = []

    for dets in detections_input:
        good_dets = {}
        for l, conf in zip(dets["label_vec"], dets["conf_vec"]):
            good_dets[l] = conf

        for l in dets["label_vec"]:
            d = {
                "class_label": sanitize_str(l),
                "start": dets["source_stamp_start_frame"],
                "end": dets["source_stamp_end_frame"],
                "conf": good_dets[l],
            }
            detections.append(d)
    detections = pd.DataFrame(detections)
    log.info(f"Loaded detections from {detections_fn}")

    # ============================
    # Load labels
    # ============================
    # grab all labels present in data
    labels = list(set([sanitize_str(l) for l in detections["class_label"].unique()]))

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
    # df = df[df[df.keys()[0]].str.contains("^[^#]")]
    # Create a mapping of detection/track ID to the activity annotation
    id_to_activity: Dict[int, Activity] = {}

    for row in df.iterrows():
        i, s = row
        if "#" in str(s[0]):
            continue
        a_id = int(s[0])
        frame, time = time_from_name(s[1])

        if a_id not in id_to_activity:
            cls_lbl = int(float(s[9]))
            id_to_activity[a_id] = Activity(
                cls_lbl,  # class label
                time,  # start
                np.inf,  # end
                frame,  # start frame
                np.inf,  # end frame
                1.0,  # conf
            )
        else:
            # There's a struct in there, update it.
            a = id_to_activity[a_id]
            # Activity should not already have an end time assigned.
            assert (
                a.end is np.inf
            ), f"More than 2 entries observed for activity track ID {a_id}."
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


def steps_from_dive_csv(filepath: str, labels: List[str]) -> List[Activity]:
    """
    Load from a DIVE output CSV file a sequence of ground truth step
    annotations.

    Class labels are converted to lower-case and stripped of any extraneous
    whitespace.

    :param filepath: Filesystem path to the CSV file.
    :param labels: List of labels in the dataset

    :return: List of loaded step annotations.
    """
    print(f"Loading ground truth steps from: {filepath}")
    df = pd.read_csv(filepath)

    # There may be additional metadata rows. Filter out rows whose first column
    # value starts with a `#`.
    df = df[df[df.keys()[0]].str.contains("^[^#]")]
    # Create a mapping of detection/track ID to the step annotation
    id_to_step: Dict[int, Step] = {}

    cleaned_labels = [sanitize_str(l.split("(step")[0]) for l in labels]

    for row in df.iterrows():
        i, s = row
        s_id = int(s[0])
        frame, time = time_from_name(s[1])

        label = sanitize_str(s[9])
        label_idx = cleaned_labels.index(label)

        if s_id not in id_to_step:
            id_to_step[s_id] = Step(
                label_idx,  # current step id
                labels[label_idx],  # current step
                time,  # start
                np.inf,  # end
                1.0,  # conf
                False,  # completed
            )

        else:
            # There's a struct in there, update it.
            a = id_to_step[s_id]

            id_to_step[s_id] = Step(
                a.current_step_id,
                a.class_label,
                a.start,  # start
                time,  # end
                1.0,  # conf
                True,  # completed
            )

    # Assert that all activities have been assigned an associated end time.
    assert np.inf not in {a.end for a in id_to_step.values()}, (
        f"Some activities in source CSV do not have corresponding end time "
        f"entries: {filepath}"
    )

    return list(id_to_step.values())


def add_inter_steps_to_activity_gt(
    gt: List[Activity],
    min_start_time: float,
    max_end_time: float,
    add_inter_steps: bool = True,
    add_before_after_steps: bool = True,
):
    """
    Adds interstitial activities to the ground truth if ``add_inter_steps`` is True and
    the ground truth activities contain '(step X)'. The interstitial activities will use the step ids
    from '(step X)' and '(step Y)' to create a new label
    'In between step X and step Y' that represents  the time between step X and Y.

    Adds 'before' and 'finished' activities to the ground truth if ``add_before_after_steps`` is True
    and the ground truth activities contain '(step X)'. The 'before' activity will be created
    from ``min_start_time`` to the start time of the first activity. The 'finished' activity
    will be created from the end of the last activity to ``max_end_time``.

    :param gt: list of Activities
    :param min_start_time: Minimum start time across the ground truth and detection sets
    :param max_end_time: Maximum end time across the ground truth and detection sets
    :param add_inter_steps: If true, will add additional activities to ``gt`` that take place inbetween the exisiting activities
    :param add_before_after_steps: If true, will add additional activities that take place before the
        start of the first activity and after the end of the last activity

    :return: ``gt`` with any additional activities inserted
    """
    first_activity = min(gt, key=lambda a: a.start)
    last_activity = max(gt, key=lambda a: a.end)

    if add_inter_steps:
        for a_id, activity in enumerate(gt):
            sub_step_str = activity.class_label
            if "(step " not in sub_step_str:
                continue
            step = sub_step_str.split("(")[1][:-1]

            if a_id + 1 < len(gt):
                next_activity = gt[a_id + 1]
                next_sub_step_str = next_activity.class_label

                if "(step " not in next_sub_step_str:
                    continue
                next_step = next_sub_step_str.split("(")[1][:-1]

                inter_class = sanitize_str(f"In between {step} and {next_step}")

                gt.append(
                    Activity(
                        inter_class,
                        activity.end,
                        next_activity.start,
                        activity.end_frame,
                        next_activity.start_frame,
                        1,
                    )
                )

    if add_before_after_steps:
        gt.append(
            Activity(
                "not started",
                min_start_time,
                first_activity.start,
                np.inf,
                first_activity.start_frame,
                1,
            )
        )

        gt.append(
            Activity(
                "finished",
                last_activity.end,
                max_end_time,
                last_activity.end_frame,
                np.inf,
                1,
            )
        )

    return gt


def add_inter_steps_to_step_gt(
    gt: List[Step],
    labels: List[str],
    min_start_time: float,
    max_end_time: float,
    add_inter_steps: bool = True,
    add_before_after_steps: bool = True,
):
    """
    Adds interstitial steps to the ground truth if ``add_inter_steps`` is True and
    the ground truth steps contain '(step X)'. The interstitial steps will use the step ids
    from '(step X)' and '(step Y)' to create a new label
    'In between step X and step Y' that represents the time between step X and Y.

    Adds 'before' and 'finished' steps to the ground truth if ``add_before_after_steps`` is True
    and the ground truth steps contain '(step X)'. The 'before' step will be created
    from ``min_start_time`` to the start time of the first step. The 'finished' step
    will be created from the end of the last step to ``max_end_time``.

    :param gt: list of Activities
    :param min_start_time: Minimum start time across the ground truth and detection sets
    :param max_end_time: Maximum end time across the ground truth and detection sets
    :param add_inter_steps: If true, will add additional steps to ``gt`` that take place inbetween the exisiting steps
    :param add_before_after_steps: If true, will add additional steps that take place before the
        start of the first step and after the end of the last step

    :return: ``gt`` with any additional steps inserted
    """
    first_activity = min(gt, key=lambda a: a.start)
    last_activity = max(gt, key=lambda a: a.end)

    if add_inter_steps:
        for a_id, activity in enumerate(gt):
            sub_step_str = activity.class_label
            if "(step " not in sub_step_str:
                continue
            step = sub_step_str.split("(")[1][:-1]

            if a_id + 1 < len(gt):
                next_activity = gt[a_id + 1]
                next_sub_step_str = next_activity.class_label

                if "(step " not in next_sub_step_str:
                    continue
                next_step = next_sub_step_str.split("(")[1][:-1]

                inter_class = sanitize_str(f"In between {step} and {next_step}")

                gt.append(
                    Step(
                        labels.index(inter_class),
                        inter_class,
                        activity.end,
                        next_activity.start,
                        1,
                        True,
                    )
                )

    if add_before_after_steps:
        ns = "not started"
        gt.append(
            Step(
                labels.index(ns),
                ns,
                min_start_time,
                first_activity.start,
                1,
                True,
            )
        )

        f = "finished"
        gt.append(
            Step(
                labels.index(f),
                f,
                last_activity.end,
                max_end_time,
                1,
                True,
            )
        )

    return gt


def steps_from_ros_export_json(filepath: str) -> Tuple[List[str], List[Activity]]:
    """
    Load a number of predicted steps from a JSON file that is the result
    of bag extraction of a `TaskUpdate.msg` message topic.

    Class labels are converted to lower-case and stripped of any extraneous
    whitespace.

    See message file located here: ros/angel_msgs/msg/TaskUpdate.msg
    See bag extraction conversion logic located here: ros/angel_utils/scripts/bag_extractor.py

    :param filepath: Filesystem path to the JSON file.
    :return: List activity predicted labels and a list of loaded activity
        annotations (predicted).
    """
    log.info(f"Loading predicted steps from: {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)
    step_seq: List[Step] = []
    for step_json in data:
        # This activity window start/end times in seconds.
        time_stamp = float(step_json["header"]["time_sec"]) + (
            float(step_json["header"]["time_nanosec"]) * 1e-9
        )
        latest_sensor_ts = float(step_json["latest_sensor_input_time"])
        current_step_id = step_json["current_step_id"]
        current_step = step_json["current_step"]
        conf_vec = step_json["hmm_step_confidence"]
        completed_vec = step_json["completed_steps"]

        step_seq.append(
            Step(
                current_step_id,  # current id
                sanitize_str(current_step),  # class_label
                latest_sensor_ts,  # start
                latest_sensor_ts,  # end
                conf_vec[current_step_id],  # conf
                completed_vec[current_step_id],  # completed
            )
        )
    return step_seq


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
    with open(filepath, "r") as f:
        data = json.load(f)
    activity_seq: List[Activity] = []
    label_vec = None
    for act_i, act_json in enumerate(data):
        # Cache/check stable labels vector.
        if label_vec is not None:
            assert act_json["label_vec"] == label_vec, (
                f"Inconsistent label set in loaded JSON activities. Activity "
                f"index {act_i} showed inconsistency."
            )
        else:
            label_vec = act_json["label_vec"]

        # This activity window start/end times in seconds.
        act_start = act_json["source_stamp_start_frame"]
        act_end = act_json["source_stamp_end_frame"]

        # Create a separate activity item per prediction
        for lbl, conf in zip(label_vec, act_json["conf_vec"]):
            activity_seq.append(
                Activity(
                    sanitize_str(lbl),
                    act_start,
                    act_end,  # time
                    np.inf,
                    np.inf,  # frame
                    conf,
                )
            )
    # normalize output label vec just like activity label treatment.
    label_vec = [sanitize_str(lbl) for lbl in label_vec]
    return label_vec, activity_seq


def objs_as_dataframe(sequence: Sequence) -> pd.DataFrame:
    """
    Transform a sequence of structures into a pandas dataframe whose
    keys are the attributes of the Activity structure.

    :param sequence: Sequence of objects to convert.
    :return: Converted data frame instance.
    """
    # `fields` introspection yields fields in order of specification in the
    # dataclass.
    # This method is much faster than using `asdict`, likely due to not using
    # deep-copy.
    return pd.DataFrame(
        {field.name: getattr(obj, field.name) for field in fields(obj)}
        for obj in sequence
    )


def find_matching_gt_activity(gt_activity, fn):
    fn = os.path.basename(fn)
    frame, time = time_from_name(fn)

    """
    gt_activity = {
        sub_step_str: [{
            'start': 123456,
            'end': 657899
        }]
    }
    """

    matching_gt = {}
    for sub_step_str, times in gt_activity.items():
        for gt_time in times:
            if gt_time["start"] <= time <= gt_time["end"]:
                return sub_step_str

    return "None"
