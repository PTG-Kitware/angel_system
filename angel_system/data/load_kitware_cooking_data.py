import pandas as pd
import numpy as np

from angel_system.data.common.load_data import (
    time_from_name,
    sanitize_str,
    activities_from_dive_csv,
)


def activity_data_loader(gt_dir, video):
    """
    :return: gt_activity = {
                sub_step_str: [{
                    'start': 123456,
                    'end': 657899
                }]
            }
    """
    gt_activity_fn = f"{gt_dir}/{video}_activity_labels_v_{labels_ver}.csv"
    assert os.path.isfile(gt_activity_fn)

    gt = activities_from_dive_csv(gt_activity_fn)
    gt_activity = {}
    for act in gt:
        class_label = act.class_label
        if class_label not in gt_activity.keys():
            gt_activity[class_label] = []
        gt_activity[class_label].append(
            {"start": act.start_frame, "end": act.end_frame}  # start_time,  # end_time
        )

    print(f"Loaded ground truth from {gt_activity_fn}")
    return gt_activity
