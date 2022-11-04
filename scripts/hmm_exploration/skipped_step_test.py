import numpy as np
import pandas as pd
import glob
import os
import copy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import  accuracy_score
import scipy
import yaml
from scipy.optimize import minimize, root_scalar

import angel_system
from angel_system.activity_hmm.core import ActivityHMM, ActivityHMMRos, \
    get_skip_score, score_raw_detections, load_and_discretize_data
from angel_system.activity_hmm.eval import plot_precision_recall, \
    accuracy_from_true_false_examples
from angel_system.impls.detect_activities.swinb.swinb_detect_activities import SwinBTransformer
from angel_system.ptg_eval.common.load_data import time_from_name

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
        AutoMinorLocator
    HAS_MATLOTLIB = True
except ModuleNotFoundError:
    HAS_MATLOTLIB = False


os.chdir('/home/local/KHQ/matt.brown/libraries/angel_system')


# ----------------------------------------------------------------------------
config_fname = '/mnt/data2tb/libraries/angel_system/config/tasks/task_steps_config-recipe_coffee_trimmed_v3.yaml'

# Map activities to steps
with open(config_fname, 'r') as stream:
    config = yaml.safe_load(stream)

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


# Load from real system.
dt = 0.25
base_dir = '/mnt/data10tb/ptg/hmm_test_data'
gt = []
for sdir in os.listdir(base_dir):
    activity_gt = glob.glob('%s/%s/*.csv' % (base_dir, sdir))[0]
    extracted_activity_detections = glob.glob('%s/%s/*.json' % (base_dir, sdir))[0]
    ret = load_and_discretize_data(activity_gt, extracted_activity_detections,
                                   dt, 0.5)
    time_windows, class_str, X, activity_ids, valid = ret
    true_step = [activity_id_to_step[activity_id] for activity_id in activity_ids]
    gt.append([sdir, time_windows, class_str, X, true_step, valid])
# ----------------------------------------------------------------------------


sdir, time_windows, class_str, X, true_step, valid = gt[0]
time_windows = time_windows - time_windows.min()

base_path = os.path.split(os.path.abspath(angel_system.__file__))[0]
config_fname = base_path + '/../config/tasks/task_steps_config-recipe_coffee_trimmed_v3.yaml'


print(f'Loading HMM with recipe {config_fname}')
live_model = ActivityHMMRos(config_fname)

if False:
    mean, std = live_model.get_hmm_mean_and_std()
    live_model.set_hmm_mean_and_std(mean, std)
    std = std + 0.2
    std = live_model.get_hmm_mean_and_std()[1]


N = len(time_windows)
#N = 1000
true_example = []
false_example = []
thresh = 0.5
correct = 0

running_step_finished_conf = []
running_unfiltered_step_conf = []
for i in range(N):
    print('State', i + 1)
    live_model.add_activity_classification(range(live_model.num_activities),
                                           X[i], time_windows[i, 0],
                                           time_windows[i, 1])

    ret = live_model.analyze_current_state()
    times, state_sequence, step_finished_conf, raw_step_conf = ret

    running_step_finished_conf.append(step_finished_conf)
    running_unfiltered_step_conf.append(raw_step_conf)

    truth_step_finished = [s in set(true_step[:i]) for s in range(1, live_model.model.num_steps)]
    truth_step_finished = np.array(truth_step_finished)

    correct += np.all(truth_step_finished == (step_finished_conf > thresh))

    if np.any(truth_step_finished):
        true_example.append(min(step_finished_conf[truth_step_finished]))

    if np.any(~truth_step_finished):
        false_example.append(max(step_finished_conf[~truth_step_finished]))

    dsp_str = []
    for i in range(len(step_finished_conf)):
        if truth_step_finished[i]:
            dsp_str.append('%.2f+' % step_finished_conf[i])
        else:
            dsp_str.append('%.2f-' % step_finished_conf[i])

    print(', '.join(dsp_str))

running_step_finished_conf = np.array(running_step_finished_conf).T
running_unfiltered_step_conf = np.array(running_unfiltered_step_conf).T
true_example = np.array(true_example)
false_example = np.array(false_example)

plt.close('all')
fig = plt.figure(num=None, figsize=(14, 10), dpi=80)
plt.rc('font', **{'size': 22})
plt.rc('axes', linewidth=4)
plt.subplot(2, 1, 1)
plt.imshow(running_step_finished_conf,
           extent=[times[0], times[-1], live_model.num_steps-0.5, -0.5],
           aspect='auto', interpolation='nearest')
plt.plot(times, true_step[:N], 'r', linewidth=3)
plt.yticks(range(1, live_model.num_steps))
plt.title('HMM Step-Completion Confidence (Red=Truth)', fontsize=20)
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
plt.tick_params(
        axis="y",
        which="major",
        grid_color='lightgrey')
plt.grid(axis='y', which='major')

plt.colorbar()
plt.subplot(2, 1, 2)
plt.imshow(running_unfiltered_step_conf,
           extent=[times[0], times[-1], live_model.num_steps-0.5, -0.5],
           aspect='auto', interpolation='nearest')
plt.plot(times, true_step[:N], 'r', linewidth=3)
plt.yticks(range(0, live_model.num_steps))
plt.colorbar()
plt.xlabel('Time (s)', fontsize=20)
plt.title('Raw Step Confidence (Red=Truth)', fontsize=20)
fig.tight_layout()

plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
plt.tick_params(
        axis="y",
        which="major",
        grid_color='lightgrey')
plt.grid(axis='y', which='major')

plt.show()
plt.savefig('/mnt/data10tb/ptg/analysis.png')

plot_precision_recall(true_example, false_example)
accuracy_from_true_false_examples(true_example, false_example)
