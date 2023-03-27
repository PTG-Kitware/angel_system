import pandas as pd
import numpy as np

from angel_system.berkeley.utils.data.dataloaders.common import time_from_name


def coffee_activity_data_loader(video='all_activities_11'):
    # Load ground truth activity 
    #root_dir = '/media/hannah.defazio/Padlock_DT/Data/notpublic/PTG/Coffee/coffee_labels'
    root_dir = '/Padlock_DT/Coffee/coffee_labels'
    gt_dir = f'{root_dir}/Labels'

    gt_activity_fn = f'{gt_dir}/{video}.feather'

    gt = pd.read_feather(gt_activity_fn) # Keys: class, start_frame,  end_frame, exploded_ros_bag_path
    gt_activity = {}
    for i, row in gt.iterrows():
        class_label = row["class"].lower().strip().strip('.')
        if class_label not in gt_activity.keys():
            gt_activity[class_label] = []
        gt_activity[class_label].append({
            'start': time_from_name(row["start_frame"]),
            'end': time_from_name(row["end_frame"])
        })
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


# TODO: training data -> kwcoco
