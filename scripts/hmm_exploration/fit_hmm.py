import numpy as np
import pandas as pd
import glob
import os
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import  accuracy_score
import scipy
import yaml

from angel_system.activity_hmm.core import ActivityHMM, ActivityHMMRos, \
    get_skip_score, score_raw_detections, load_and_discretize_data
from angel_system.impls.detect_activities.swinb.swinb_detect_activities import SwinBTransformer
from angel_system.ptg_eval.common.load_data import time_from_name
from angel_system.ptg_eval.activity_classification.visualization import EvalVisualization
from angel_system.ptg_eval.activity_classification.compute_scores import EvalMetrics

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
        AutoMinorLocator
    HAS_MATLOTLIB = True
except ModuleNotFoundError:
    HAS_MATLOTLIB = False


os.chdir('/home/local/KHQ/matt.brown/libraries/angel_system')


# ----------------------------------------------------------------------------
# Load from real system.
activity_gt = '/mnt/data10tb/ptg/labels_test_v1.5.feather'
extracted_activity_detections = '/mnt/data10tb/ptg/activity_detection_data.json'
dt = 0.25
ret = load_and_discretize_data(activity_gt, extracted_activity_detections,
                               dt, 0.5)
time_windows, class_str, X, Z, valid = ret
valid[Z == 0] = False


# Fit HMM.
num_classes = max(Z) + 1
true_mask = np.diag(np.ones(num_classes, dtype=bool))[Z]
class_mean_conf = []
class_std_conf = []
med_class_duration = []
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

#plt.imshow(class_mean_conf); plt.colorbar()
#plt.imshow(class_std_conf); plt.colorbar()
#plt.imshow(class_mean_conf/class_std_conf); plt.colorbar()
#plt.imshow(X.T, interpolation='nearest', aspect='auto'); plt.plot(Z, 'r.')

np.save("/mnt/data2tb/libraries/angel_system/config/activity_labels/recipe_coffee_mean_std.npy", [class_mean_conf, class_std_conf])


config_fname = '/mnt/data2tb/libraries/angel_system/config/tasks/task_steps_config-recipe_coffee.yaml'
live_model = ActivityHMMRos(config_fname)

if False:
    # Verify reading and writing.
    mean, std = live_model.get_hmm_mean_and_std()
    print(std - class_std_conf)
    live_model.set_hmm_mean_and_std(class_mean_conf, class_std_conf)

    config_fname2 = '/mnt/data2tb/libraries/angel_system/config/tasks/task_steps_config-recipe_coffee2.yaml'
    live_model.save_task_yaml(config_fname2)

    live_model2 = live_model = ActivityHMMRos(config_fname2)


# Score
Z_ = []
for i in range(len(time_windows)):
    label_vec = range(X.shape[1])
    live_model.add_activity_classification(label_vec, X[i], time_windows[i, 0],
                                           time_windows[i, 1])
    class_str_ = live_model.get_current_state()
    Z_.append(live_model.class_str.index(class_str_))

Z_ = np.array(Z_)

plt.plot(Z[valid], 'g.')
plt.plot(Z_[valid], 'b.')

print('Time when estimated correct step %0.1f%%' % (np.mean(Z == Z_)*100))
print('Time when estimated later step %0.1f%%' % (np.mean(Z_ > Z)*100))
print('Time when estimated earlier step %0.1f%%' % (np.mean(Z_ < Z)*100))
