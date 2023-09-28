import pandas as pd
import numpy as np

from angel_system.data.common.load_data import (
    time_from_name,
    sanitize_str,
    activities_from_dive_csv,
)


def cooking_activity_data_loader(gt_activity_fn):
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
    """
    gt_activity = {
        sub_step_str: [{
            'start': 123456,
            'end': 657899
        }]
    }
    """

    return gt_activity


def coffee_activity_data_loader(video="all_activities_11"):
    import pandas as pd

    # Load ground truth activity
    root_dir = (
        "/data/users/hannah.defazio/ptg_nas/data_copy/coffee_labels/"
        # "/media/hannah.defazio/Padlock_DT6/Data/notpublic/PTG/Coffee/coffee_labels"
    )
    # root_dir = "/Padlock_DT/Coffee/coffee_labels"
    gt_dir = f"{root_dir}/Labels"

    gt_activity_fn = f"{gt_dir}/{video}.csv"

    return cooking_activity_data_loader(gt_activity_fn)


def tea_activity_data_loader(video="all_activities_11"):
    import pandas as pd

    # Load ground truth activity
    gt_dir = "/data/users/hannah.defazio/ptg_nas/data_copy/tea_labels"
    labels_ver = 1

    gt_activity_fn = f"{gt_dir}/{video}_activity_labels_v_{labels_ver}.csv"
    print(f"Loaded ground truth from {gt_activity_fn}")

    return cooking_activity_data_loader(gt_activity_fn)
