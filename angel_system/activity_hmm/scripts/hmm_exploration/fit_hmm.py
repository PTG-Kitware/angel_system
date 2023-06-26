import numpy as np
import pandas as pd
import glob
import os
import copy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
import scipy
import yaml
from scipy.optimize import minimize, root_scalar

from angel_system.activity_hmm.core import (
    ActivityHMM,
    ActivityHMMRos,
    get_skip_score,
    score_raw_detections,
    load_and_discretize_data,
)
from angel_system.impls.detect_activities.swinb.swinb_detect_activities import (
    SwinBTransformer,
)
from angel_system.data.common.load_data import time_from_name

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

    HAS_MATLOTLIB = True
except ModuleNotFoundError:
    HAS_MATLOTLIB = False


os.chdir("/home/local/KHQ/matt.brown/libraries/angel_system")


# ----------------------------------------------------------------------------
config_fname = "/angel_system/config/tasks/task_steps_config-recipe_m2_apply_tourniquet_v0.052.yaml"

if False:
    # Old way.
    # Load from real system.
    dt = 0.25
    base_dir = "/mnt/data10tb/ptg/hmm_training_data"
    gt = []
    for sdir in os.listdir(base_dir):
        activity_gt = glob.glob("%s/%s/*.csv" % (base_dir, sdir))[0]
        extracted_activity_detections = glob.glob("%s/%s/*.json" % (base_dir, sdir))[0]
        gt.append(
            [
                sdir,
                load_and_discretize_data(
                    activity_gt, extracted_activity_detections, dt, 0.5
                ),
            ]
        )

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

    # ------------------------------------------------------------------------
    # What was loaded is activity_id ground truth, but we want step ground truth.

    # Map activities to steps
    with open(config_fname, "r") as stream:
        config = yaml.safe_load(stream)

    activity_id_to_step = {}
    for step in config["steps"]:
        if isinstance(step["activity_id"], str):
            a_ids = [int(s) for s in step["activity_id"].split(",")]
        else:
            a_ids = [step["activity_id"]]

        for i in a_ids:
            activity_id_to_step[i] = step["id"]

    activity_id_to_step[0] = 0
    steps = sorted(list(set(activity_id_to_step.values())))
    assert steps == list(range(max(steps) + 1))

    true_step = [activity_id_to_step[activity_id] for activity_id in activity_ids]
    # ------------------------------------------------------------------------
else:
    fname = "/angel_workspace/ros_bags/activity_hmm_training_data.pkl"
    with open(fname, "rb") as of:
        X, true_step = pickle.load(of)

    valid = np.ones(len(activity_ids), dtype=bool)


# ----------------------------------------------------------------------------


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
    try:
        twins = time_windows[wins[1], 1] - time_windows[wins[0], 0]

        med_class_duration.append(np.mean(twins))
    except NameError:
        med_class_duration.append(1)

med_class_duration = np.array(med_class_duration)
class_mean_conf = np.array(class_mean_conf)
class_std_conf = np.array(class_std_conf)

# ----------------------------------------------------------------------------
if False:
    # Fit dummy mean and cov.
    num_steps = len(steps)
    num_activities = len(activity_id_to_step)
    class_mean_conf2 = np.zeros((num_steps, num_activities))
    class_std_conf = np.ones((num_steps, num_activities)) * 0.05

    for key in activity_id_to_step:
        class_mean_conf2[activity_id_to_step[key], key] = 1

    ind = np.argmax(class_mean_conf2 * class_mean_conf, axis=1)
    class_mean_conf2[:] = 0
    for i, ii in enumerate(ind):
        class_mean_conf2[i, ii] = 1

    class_mean_conf = class_mean_conf2
# ----------------------------------------------------------------------------

np.save(config["activity_mean_and_std_file"], [class_mean_conf, class_std_conf])

# ----------------------------------------------------------------------------

# Analyse how quickly we can move to next step.
config_fname = "/mnt/data2tb/libraries/angel_system/config/tasks/task_steps_config-recipe_coffee_trimmed_v2.yaml"
live_model = ActivityHMMRos(config_fname)
model = live_model.noskip_model


_, X, Z_true, _, _ = model.sample(1000)
_, Z, _, _ = model.decode(X)

ind = Z_true > 0
Z_true[ind] == Z[ind]

Z = Z[Z != 0]


# ----------------------------------------------------------------------------
# Fit median durations.

config_fname = "/mnt/data2tb/libraries/angel_system/config/tasks/task_steps_config-recipe_coffee.yaml"
live_model = ActivityHMMRos(config_fname)

dt = live_model.dt
class_str = live_model.class_str
med_class_duration = live_model.med_class_duration
num_steps_can_jump_fwd = live_model.num_steps_can_jump_fwd
num_steps_can_jump_bck = live_model.num_steps_can_jump_bck
class_mean_conf = live_model.class_mean_conf
class_std_conf = live_model.class_std_conf


def err(med_class_duration):
    model = ActivityHMM(
        dt,
        class_str,
        med_class_duration=med_class_duration,
        num_steps_can_jump_fwd=num_steps_can_jump_fwd,
        num_steps_can_jump_bck=num_steps_can_jump_bck,
        class_mean_conf=class_mean_conf,
        class_std_conf=class_std_conf,
    )

    log_prob = model.decode(X)[0]
    print(log_prob, med_class_duration)
    return -log_prob


res = minimize(err, med_class_duration, bounds=[(0, 10) for _ in range(num_classes)])

med_class_duration = res.x
# ----------------------------------------------------------------------------

# Investigate which steps we might get stuck in.
config_fname = "/mnt/data2tb/libraries/angel_system/config/tasks/task_steps_config-recipe_coffee_trimmed.yaml"
live_model = ActivityHMMRos(config_fname)
model = live_model.model
model.model.transmat_ += 1e-12
model.model.startprob_ += 1e-12

fract = []
med_class_duration_ = []
for curr_step_ind in range(0, len(live_model.class_str) - 1):
    print("Processing step:", curr_step_ind)

    N = 1000
    conf = model.sample_for_step(curr_step_ind, N)

    log_prob1 = -(
        ((conf - class_mean_conf[curr_step_ind]) / class_std_conf[curr_step_ind]) ** 2
    )
    log_prob1 = np.sum(log_prob1, axis=1)
    log_prob2 = -(
        (
            (conf - class_mean_conf[curr_step_ind + 1])
            / class_std_conf[curr_step_ind + 1]
        )
        ** 2
    )
    log_prob2 = np.sum(log_prob2, axis=1)

    llr = log_prob1 - log_prob2

    # Log of likelihood ratio for how much more likely it is that we move to
    # the next step when we should.
    llr0 = np.percentile(llr, 10)

    def err(n):
        if n < 0:
            return 1e10

        # llr to move on due to transition probability
        err = np.log(1 - (0.5) ** (1 / n)) - np.log((0.5) ** (1 / n))
        err += llr0
        return err

    n0 = 5 / dt
    n = np.logspace(-5, np.log10(n0), 10000)
    errs = [abs(err(n_)) for n_ in n[:-1]]
    ind = np.argmin(errs)
    n = root_scalar(err, x0=n[ind - 1], x1=n[ind + 1]).root
    n = min([n, n0])
    med_class_duration_.append(dt * n)

    fract.append(np.mean(llr > 0))

[(i, med_class_duration_[i]) for i in range(len(med_class_duration_))]
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# What happens with a bunch of background in a row.
config_fname = "/mnt/data2tb/libraries/angel_system/config/tasks/task_steps_config-recipe_coffee_trimmed.yaml"
live_model = ActivityHMMRos(config_fname)

curr_step = 17
start_time = 0
end_time = 1
for _ in range(1, 500):
    conf_vec = live_model.model.sample_for_step(curr_step, 1).ravel()
    label_vec = list(range(live_model.num_activities))
    live_model.add_activity_classification(label_vec, conf_vec, start_time, end_time)
    start_time += 1
    end_time += 1

ret = live_model.analyze_current_state()
times, state_sequence, step_finished_conf = ret
state_sequence = state_sequence[state_sequence != 0]
plt.close("all")
plt.subplot(2, 1, 1)
plt.plot(state_sequence, ".")
plt.subplot(2, 1, 2)
plt.plot(np.diff(state_sequence), ".")
# ----------------------------------------------------------------------------


config_fname = "/mnt/data2tb/libraries/angel_system/config/tasks/task_steps_config-recipe_coffee.yaml"
live_model = ActivityHMMRos(config_fname)

if False:
    # Analyze mean and std.

    model = live_model.model
    model.model.transmat_ += 1e-12
    model.model.startprob_ += 1e-12

    num_steps = len(live_model.class_str)
    num_classes = len(live_model.class_str)

    # Gaussian is ~ exp(-((x-mu)/sigma)^2), so log Gaussian ~ -((x-mu)/sigma)^2
    # so, log likelihood of a potential solution is sum over all classifiers of
    # this.
    ideal_conf = []
    for curr_step_ind in range(0, len(live_model.class_str)):
        print("Processing step:", curr_step_ind)
        live_model.clear_history()
        start_time = 0
        end_time = 1
        for i in range(len(ideal_conf)):
            live_model.add_activity_classification(None, ideal_conf[i], i, i + 1 - 1e-6)

        X0 = live_model.X
        if X0 is not None:
            X0 = X0.copy()

        Xi = class_mean_conf[curr_step_ind]

        if X0 is not None:
            X = np.vstack([X0, Xi])
        else:
            X = np.atleast_2d(Xi)

        X_ = live_model.model.pad_in_hidden_background(X)
        Z_ = model.model.decode(X_)[1]

        if curr_step_ind >= 1:
            assert model.inv_map[Z_[-2]] == curr_step_ind - 1

        def err(Xi):
            X[-1] = Xi
            X_ = live_model.model.pad_in_hidden_background(X)
            log_probs = []
            for step in range(num_steps):
                Z_[-1] = model.fwd_map[step]

                if step == curr_step_ind:
                    log_prob0 = model.calc_log_prob_(X_, Z_)
                else:
                    log_probs.append(model.calc_log_prob_(X_, Z_))

            log_prob1 = max([1e-14, max(log_probs)])
            s = log_prob0 / log_prob1
            return -s

        res = minimize(err, Xi, bounds=[(0, 1) for _ in range(num_classes)])

        assert -res.fun > 1
        Xi = res.x
        print(-err(Xi))
        X[-1] = Xi
        Z = model.decode(X)[1]
        assert Z[-1] == curr_step_ind
        ideal_conf.append(res.x)


if False:
    # Verify reading and writing.
    mean, std = live_model.get_hmm_mean_and_std()
    print(std - class_std_conf)
    live_model.set_hmm_mean_and_std(class_mean_conf, class_std_conf)

    config_fname2 = "/mnt/data2tb/libraries/angel_system/config/tasks/task_steps_config-recipe_coffee2.yaml"
    live_model.save_task_yaml(config_fname2)

    live_model2 = live_model = ActivityHMMRos(config_fname2)


# Score
Z_ = []
for i in range(len(time_windows)):
    label_vec = range(X.shape[1])
    live_model.add_activity_classification(
        label_vec, X[i], time_windows[i, 0], time_windows[i, 1]
    )
    class_str_ = live_model.get_current_state()
    Z_.append(live_model.class_str.index(class_str_))

Z_ = np.array(Z_)

plt.plot(Z[valid], "g.")
plt.plot(Z_[valid], "b.")

print("Time when estimated correct step %0.1f%%" % (np.mean(Z == Z_) * 100))
print("Time when estimated later step %0.1f%%" % (np.mean(Z_ > Z) * 100))
print("Time when estimated earlier step %0.1f%%" % (np.mean(Z_ < Z) * 100))
