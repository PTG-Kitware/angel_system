import os
import argparse
import glob
import yaml
import numpy as np
from pathlib import Path

from angel_system.activity_hmm.core import load_and_discretize_data


# ----------------------------------------------------------------------------
def load_data(args):
    # Load from real system.
    dt = 0.25
    gt = []
    for sdir in os.listdir(args.base_dir):
        activity_gt = glob.glob('%s/%s/*.csv' % (args.base_dir, sdir))[0]
        extracted_activity_detections = glob.glob('%s/%s/*.json' % (args.base_dir, sdir))[0]
        gt.append([sdir, load_and_discretize_data(activity_gt,
                                                extracted_activity_detections,
                                                dt, 0.5)])

    # For the purpose of fitting mean and std outside of the HMM, we can just
    # append everything together.

    X = []
    activity_ids = []
    time_windows = []
    valid = []
    for gt_ in gt:
        time_windowsi, class_str, Xi, activity_idsi, validi = gt_[1]
        time_windows.append(time_windowsi)
        X.append(Xi)
        activity_ids.append(activity_idsi)
        valid.append(validi)

    time_windows = np.vstack(time_windows)
    X = np.vstack(X)
    valid = np.hstack(valid)
    activity_ids = np.hstack(activity_ids)

    # ----------------------------------------------------------------------------
    # What was loaded is activity_id ground truth, but we want step ground truth.

    # Map activities to steps
    with open(args.config_fname, 'r') as stream:
        config = yaml.safe_load(stream)

    dest = config['activity_mean_and_std_file']

    activity_id_to_step = {}
    for step in config['steps']:
        if isinstance(step['activity_id'], str):
            a_ids = [int(s) for s in step['activity_id'].split(',')]
        else:
            a_ids = [step['activity_id']]

        for i in a_ids:
            activity_id_to_step[i] = step['id']

    activity_id_to_step[0] = 0
    steps = sorted(list(set(activity_id_to_step.values())))
    assert steps == list(range(max(steps) + 1))

    true_step = [activity_id_to_step[activity_id] for activity_id in activity_ids]

    return time_windows, true_step, dest
# ----------------------------------------------------------------------------
def fit(time_windows, true_step):
    # Fit HMM.
    num_classes = max(true_step) + 1
    class_mean_conf = []
    class_std_conf = []
    med_class_duration = []
    true_mask = np.diag(np.ones(num_classes, dtype=bool))[true_step]
    for i in range(num_classes):
        class_mean_conf.append(np.mean(X[true_mask[:, i], :], axis=0))
        class_std_conf.append(np.std(X[true_mask[:, i], :], axis=0))

        indr = np.where(np.diff(true_mask[:, i].astype(np.int8)) < 0)[0]
        indl = np.where(np.diff(true_mask[:, i].astype(np.int8)) > 0)[0]

        if true_mask[0, i] and indl[0] != 0:
            indl = np.hstack([0, indl])

        if true_mask[-1, i] and indr[-1] != len(true_mask) - 1:
            indr = np.hstack([indr, len(true_mask) - 1])

        # wins has shape (2, num_instances) where wins[0, i] indicates when the ith
        # instance starts and wins[1, i] indicates when the ith instance ends.
        wins = np.array(list(zip(indl, indr))).T

        # During (seconds) of each instance.
        twins = time_windows[wins[1], 1] - time_windows[wins[0], 0]

        med_class_duration.append(np.mean(twins))

    med_class_duration = np.array(med_class_duration)
    class_mean_conf = np.array(class_mean_conf)
    class_std_conf = np.array(class_std_conf)

    # ----------------------------------------------------------------------------
    if False:
        # Fit dummy mean and cov.
        num_steps = len(steps)
        num_activities = len(activity_id_to_step)
        class_mean_conf2 = np.zeros((num_steps, num_activities))
        class_std_conf = np.ones((num_steps, num_activities))*0.05

        for key in activity_id_to_step:
            class_mean_conf2[activity_id_to_step[key], key] = 1

        ind = np.argmax(class_mean_conf2*class_mean_conf, axis=1)
        class_mean_conf2[:] = 0
        for i, ii in enumerate(ind):
            class_mean_conf2[i, ii] = 1

        class_mean_conf = class_mean_conf2

    return class_mean_conf, class_std_conf

# ----------------------------------------------------------------------------
def save(dest, class_mean_conf, class_std_conf):
    np.save(dest, [class_mean_conf, class_std_conf])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_fname",
        dest="config_fname",
        type=Path,
        default='angel_system/config/tasks/task_steps_config-recipe_coffee_trimmed_v3.yaml'
    )
    parser.add_argument(
        "--base_dir",
        dest="base_dir",
        type=Path,
        default='/mnt/data10tb/ptg/hmm_training_data'
    )

    args = parser.parse_args()

    time_windows, true_step, dest = load_data(args)
    class_mean_conf, class_std_conf = fit(time_windows, true_step)
    save(dest, class_mean_conf, class_std_conf)


if __name__ == '__main__':
    main()