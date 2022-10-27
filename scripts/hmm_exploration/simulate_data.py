import numpy as np
import pandas as pd
import glob
import os
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import  accuracy_score
import scipy

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



class_str = ['background',
             'measure 12 ounces of water in the liquid measuring cup',
             'pour the water from the liquid measuring cup into the electric kettle',
             'turn on the kettle',
             'place the dripper on top of the mug',
             'take the coffee filter and fold it in half to create a semi-circle',
             'fold the filter in half again to create a quarter-circle',
             'place the folded filter into the dripper such that the the point of the quarter-circle rests in the center of the dripper',
             'spread the filter open to create a cone inside the dripper',
             'turn on the kitchen scale',
             'Place a bowl on the scale',
             'Zero the scale',
             'Add coffee beans to the bowl until the scale reads 25 grams',
             'Pour the measured coffee beans into the coffee grinder',
             'Set timer for 20 seconds',
             'Turn on the timer',
             'Grind the coffee beans by pressing and holding down on the black part of the lid',
             'Pour the grounded coffee beans into the filter cone prepared in step 2',
             'Turn on the thermometer',
             'Place the end of the thermometer into the water',
             'Set timer to 30 seconds',
             'Pour a small amount of water over the grounds in order to wet the grounds',
             'Slowly pour the water over the grounds in a circular motion. Do not overfill beyond the top of the paper filter',
             'Allow the rest of the water in the dripper to drain',
             'Remove the dripper from the cup',
             'Remove the coffee grounds and paper filter from the dripper',
             'Discard the coffee grounds and paper filter',
             'finished']

#class_str = ['background', 'step1', 'step2', 'step3', 'step4']

class_map = {class_str[i]:i for i in range(len(class_str))}
N = len(class_str)

# Time step between detection updates.
dt = 1    # s

# Define the median time spent in each step.
med_class_duration = np.ones(N)*5

class_mean_conf = np.ones(N)*0.5
class_std_conf = np.ones(N)*0.1

# ----------------------------------------------------------------------------
# Simulate one dataset and export.
model = ActivityHMM(dt, class_str, med_class_duration=med_class_duration,
                     num_steps_can_jump_fwd=0, num_steps_can_jump_bck=0,
                     class_mean_conf=class_mean_conf,
                     class_std_conf=class_std_conf)

times, X, Z, X_, Z_ = model.sample(500)

det_json_fname = '/mnt/data10tb/ptg/activity_detection_data2.json'
gt_feather_fname = '/mnt/data10tb/ptg/simulated.feather'
model.save_to_disk(times, X, Z, det_json_fname, gt_feather_fname)
# ----------------------------------------------------------------------------

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


model = ActivityHMM(dt, class_str,
                    med_class_duration=med_class_duration,
                    num_steps_can_jump_fwd=0, num_steps_can_jump_bck=0,
                    class_mean_conf=class_mean_conf,
                    class_std_conf=class_std_conf)
model_skip = ActivityHMM(dt, class_str,
                         med_class_duration=med_class_duration,
                         num_steps_can_jump_fwd=10, num_steps_can_jump_bck=10,
                         class_mean_conf=class_mean_conf,
                         class_std_conf=class_std_conf)

plt.close('all')
score_raw_detections(X[valid], Z[valid], plot_results=True)

Z2 = model.decode(X)[1]
tp = int(sum(Z[valid] == Z2[valid]))
fp = fn = int(len(Z[valid]) - tp)
p = tp/(tp + fp)
r = tp/(tp + fn)
plt.plot(r, p, 'o', markersize=18, label='HMM (No Skipping)')

Z2 = model_skip.decode(X)[1]
tp = int(sum(Z[valid] == Z2[valid]))
fp = fn = int(len(Z[valid]) - tp)
p = tp/(tp + fp)
r = tp/(tp + fn)
plt.plot(r, p, 'o', markersize=18, label='HMM (Allow Skipping)')
plt.title('Classify Active Steps', fontsize=40)
plt.gcf().tight_layout()
plt.legend(fontsize=20, loc=0)
plt.savefig('/mnt/data10tb/ptg/det_pr_vs_hmm.png')

plt.plot(Z2[valid], 'g-')
plt.plot(Z[valid], 'bo')
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Verify covariance is being respectfed.
times, X, Z, X_, Z_ = model.sample(5000)
tp = np.array([X[i, Z[i]] for i in range(len(Z))])
fp = [[X[i, j] for j in range(X.shape[1]) if Z[i] != j]
      for i in range(len(X))]
fp = np.array(fp).ravel()
print('True examples mean:', np.mean(tp), 'std:', np.std(tp))
print('False examples mean:', np.mean(fp), 'std:',
      np.sqrt(np.mean(np.mean(fp**2))))
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
config_fname = '/mnt/data2tb/libraries/angel_system/config/tasks/task_steps_config-recipe_coffee.yaml'
live_model = ActivityHMMRos(config_fname)

curr_step = 0
start_time = 0
end_time = 1

for _ in range(25):
    conf_vec = np.zeros(len(live_model.model.class_str));
    conf_vec[curr_step] = 0.5
    live_model.add_activity_classification(live_model.model.class_str,
                                           conf_vec, start_time, end_time)
    curr_step += 1
    start_time += 1
    end_time += 1

    print(live_model.get_current_state())


live_model.revert_to_step(10)
print(live_model.get_current_state())
print(live_model.model.class_str[10])

#plt.imshow(live_model.X)

live_model.get_current_state()
live_model
live_model.times
start_time
start_time=10000; end_time=10001
live_model.add_activity_classification(live_model.model.class_str,
                                       conf_vec, start_time, end_time)
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Detector P/R vs SNR.
x1 = np.random.normal(size=10000)
x2 = [max(np.random.normal(size=30)) for _ in range(100000)]

snrs = [2, 3, 3.5, 4, 5, 6, 8]

fig = plt.figure(num=None, figsize=(14, 10), dpi=80)
plt.rc('font', **{'size': 22})
plt.rc('axes', linewidth=4)
for snr in snrs:
    tp = x1 + snr
    fp = x2
    s = np.hstack([tp, fp]).T
    y_tue = np.hstack([np.ones(len(tp), dtype=bool),
                       np.zeros(len(fp), dtype=bool)]).T
    s.shape = (-1, 1)
    y_tue.shape = (-1, 1)
    precision, recall, thresholds = precision_recall_curve(y_tue, s)
    a = max([accuracy_score(y_tue, s > t) for t in np.linspace(0, 1, 100)])

    print('snr:', snr, 'accuracy', a)

    thresholds = np.hstack([thresholds[0], thresholds])
    auc = -np.trapz(precision, recall)
    plt.plot(recall, precision, linewidth=6,
             label='Detection SNR=%0.1f (AP=%0.2f)' % (snr, auc))

plt.xlabel('Recall', fontsize=40)
plt.ylabel('Precision', fontsize=40)
plt.title('Action Classifier P/R', fontsize=40)
plt.xlim([0, 1.01])
plt.ylim([0, 1.01])
fig.tight_layout()

plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
plt.tick_params(
        axis="y",
        which="major",
        grid_color='lightgrey')
plt.tick_params(
        axis="y",
        which="minor",
        grid_linestyle='--',
        grid_color='lightgrey')
plt.grid(axis='y', which='both')
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
plt.tick_params(
        axis="x",
        which="major",
        grid_color='lightgrey')
plt.tick_params(
        axis="x",
        which="minor",
        grid_linestyle='--',
        grid_color='lightgrey')
plt.grid(axis='x', which='both')
plt.legend(fontsize=20, loc=0)

plt.show()
plt.savefig('/mnt/data10tb/ptg/det_pr_vs_det_snr.png')

# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
model = ActivityHMM(dt, class_str, med_class_duration=med_class_duration,
                     num_steps_can_jump_fwd=0, num_steps_can_jump_bck=0,
                     class_mean_conf=class_mean_conf,
                     class_std_conf=class_std_conf)
self = model

model1 = ActivityHMM(dt, class_str, med_class_duration=med_class_duration,
                      num_steps_can_jump_fwd=0, num_steps_can_jump_bck=0,
                      class_mean_conf=class_mean_conf,
                      class_std_conf=class_std_conf)

model2 = ActivityHMM(dt, class_str, med_class_duration=med_class_duration,
                      num_steps_can_jump_fwd=1, num_steps_can_jump_bck=0,
                      class_mean_conf=class_mean_conf,
                      class_std_conf=class_std_conf)

model3 = ActivityHMM(dt, class_str, med_class_duration=med_class_duration,
                      num_steps_can_jump_fwd=1000, num_steps_can_jump_bck=1000,
                      class_mean_conf=class_mean_conf,
                      class_std_conf=class_std_conf)

# Measure performance of classifier of which step is active at each time.

if False:
    num_class_states = []
    for i in range(100):
        print(i)
        X, Z = model.sample(500)[1:3]
        num_class_states.append([sum(Z == i_) for i_ in range(len(model.class_str))])

    num_class_states = np.array(num_class_states)
    #plt.plot(np.sort(num_class_states[:, :10].ravel()))
    plt.hist(num_class_states[:, :10].ravel(), 20)

# Z is the truth state.
X, Z = model.sample(1000)[1:3]

ind = np.where(Z == len(model.class_str) - 1)[0][0]
Z = Z[:ind+10]


num_trials = 50

tp = []
fp = []
tp_hmm1 = 0
fp_hmm1 = 0
tp_hmm2 = 0
fp_hmm2 = 0
tp_hmm3 = 0
fp_hmm3 = 0
for i in range(num_trials):
    print('%i/%i' % (i + 1, num_trials))

    # Simulate trial
    while True:
        X, Z = model.sample(1000)[1:3]
        try:
            # Possible we never made it to the last class
            ind = np.where(Z == len(model.class_str) - 1)[0][0]
            break
        except IndexError:
            continue

    Z = Z[:ind+10]
    X = X[:ind+10]

    # Classify based on individual detections alone.
    mask = np.zeros_like(X, dtype=bool)
    np.put_along_axis(mask, np.atleast_2d(Z).T, True, axis=1)
    tp.append(X[mask])
    fp.append(X[~mask])

    # Classify using hmm decoding.
    Z_ = model1.decode(X)[1]
    tp_ = int(sum(Z_ == Z))
    tp_hmm1 += tp_
    fp_hmm1 += int(len(Z) - tp_)

    Z_ = model2.decode(X)[1]
    tp_ = int(sum(Z_ == Z))
    tp_hmm2 += tp_
    fp_hmm2 += int(len(Z) - tp_)

    Z_ = model3.decode(X)[1]
    tp_ = int(sum(Z_ == Z))
    tp_hmm3 += tp_
    fp_hmm3 += int(len(Z) - tp_)

# Calculate precision-recall curve for pure detections.
tp = np.hstack(tp)
fp = np.hstack(fp)
s = np.hstack([tp, fp]).T
y_tue = np.hstack([np.ones(len(tp), dtype=bool),
                   np.zeros(len(fp), dtype=bool)]).T
s.shape = (-1, 1)
y_tue.shape = (-1, 1)
precision, recall, thresholds = precision_recall_curve(y_tue, s)
auc = -np.trapz(precision, recall)



plt.close('all')
score_raw_detections(X, Z, plot_results=True)

fn_hmm1 = fp_hmm1
p = tp_hmm1/(tp_hmm1 + fp_hmm1)
r = tp_hmm1/(tp_hmm1 + fn_hmm1)
plt.plot(r, p, 'o', markersize=18, label='HMM (Assume No Skipping)')

fn_hmm2 = fp_hmm2
p = tp_hmm2/(tp_hmm2 + fp_hmm2)
r = tp_hmm2/(tp_hmm2 + fn_hmm2)
plt.plot(r, p, 'o', markersize=18, label='HMM (Can Skip 1)')

fn_hmm3 = fp_hmm3
p = tp_hmm3/(tp_hmm3 + fp_hmm3)
r = tp_hmm3/(tp_hmm3 + fn_hmm3)
plt.plot(r, p, 'o', markersize=18, label='HMM (Any Order)')


plt.legend(fontsize=20, loc=0)
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
plt.tick_params(
        axis="y",
        which="major",
        grid_color='lightgrey')
plt.tick_params(
        axis="y",
        which="minor",
        grid_linestyle='--',
        grid_color='lightgrey')
plt.grid(axis='y', which='both')
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
plt.tick_params(
        axis="x",
        which="major",
        grid_color='lightgrey')
plt.tick_params(
        axis="x",
        which="minor",
        grid_linestyle='--',
        grid_color='lightgrey')
plt.grid(axis='x', which='both')
fig.tight_layout()

plt.show()
plt.savefig('/mnt/data10tb/ptg/pr.png')
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Test live system.
config_fname = '/mnt/data2tb/libraries/angel_system/config/tasks/task_steps_config-recipe_coffee.yaml'
live_model = ActivityHMMRos(config_fname)
times, X0, Z0, X0_, Z0_ = live_model.model.sample(5000)
ind = np.round(np.linspace(0, len(times), int(len(times)) - 1)).astype(int)
results = []
for i in range(len(ind) - 1):
    start_time = times[ind[i]]
    end_time = times[ind[i + 1]]
    print(start_time, end_time)
    conf_vec = np.mean(X0[ind[i]:(ind[i + 1] + 1)], axis=0)
    live_model.add_activity_classification(live_model.model.class_str,
                                           conf_vec, start_time, end_time)

    Z1 = live_model.model.class_str.index(live_model.get_current_state())
    results.append([scipy.stats.mode(Z0[ind[i]:(ind[i + 1] + 1)]).mode[0],
                    Z1])
    print(results[-1])

# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
config_fname = '/mnt/data2tb/libraries/angel_system/config/tasks/task_steps_config-recipe_coffee.yaml'
live_model = ActivityHMMRos(config_fname)
start_time = 0
end_time = 1
i = 1
conf_vec = np.ones(len(live_model.model.class_str))*(-10000);   conf_vec[i] = 1
live_model.add_activity_classification(live_model.model.class_str,
                                       conf_vec, start_time, end_time)
live_model.get_current_state()
start_time += 1
end_time += 1
i += 1

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Measure performance of “Did user complete every step?”

for snr in [8]:
    class_mean_conf = np.ones(N)*0.1*snr
    class_std_conf = np.ones(N)*0.1

    model = ActivityHMM(dt, class_str, med_class_duration=med_class_duration,
                        num_steps_can_jump_fwd=0, num_steps_can_jump_bck=0,
                        class_mean_conf=class_mean_conf,
                        class_std_conf=class_std_conf)

    model_skip = ActivityHMM(dt, class_str, med_class_duration=med_class_duration,
                             num_steps_can_jump_fwd=1, num_steps_can_jump_bck=0,
                             class_mean_conf=class_mean_conf,
                             class_std_conf=class_std_conf)


    num_trials = 500
    tp_hmm = []
    fp_hmm = []
    for i in range(num_trials):
        print('%i/%i' % (i + 1, num_trials))

        # Simulate trial
        X0, Z0, X0_, Z0_ = model.sample(1000)[1:]

        if False:
            # Sanity checking low scores.
            print(min([X0_[i, Z0_[i]] for i in range(len(Z0_))]))
            print(max([X0_[i, Z0_[i]] for i in range(len(Z0_))]))

            print(min([X0[i, Z0[i]] for i in range(len(Z0))]))
            print(max([X0[i, Z0[i]] for i in range(len(Z0))]))

            log_prob1, Z1, X_, Z1_ = model.decode(X0)
            print(min([X_[i, Z1_[i]] for i in range(len(Z1_))]))
            print(max([X_[i, Z1_[i]] for i in range(len(Z1_))]))


            print('log_prob', model.calc_log_prob_(X0_, Z0_))
            log_prob1, Z1, X_, Z1_ = model.decode(X0)
            s = [X_[i, Z1_[i]] for i in range(len(Z1_))]
            print('log_prob', model.calc_log_prob_(X_, Z1_, verbose=True))
            s = [X0_[i, Z0_[i]] for i in range(len(Z0_))]

        s = get_skip_score(model, model_skip, X0)

        fp_hmm.append(s)

        # Remove one step for the skipped-step example.
        inds = list(set(Z0))
        skip_ind = inds[np.random.randint(1, len(inds))]
        inds = np.array([True if Z0[i] != skip_ind else False for i in range(len(Z0))])

        X = X0[inds]
        Z = Z0[inds]

        s = get_skip_score(model, model_skip, X)

        tp_hmm.append(s)


    if False:
        tp = np.array(tp_hmm)
        fp = np.array(fp_hmm)
        plt.plot((tp[:, 0] - tp[:, 1])/tp[:, 1], 'go', markersize=2)
        plt.plot((fp[:, 0] - fp[:, 1])/fp[:, 1], 'ro', markersize=2)

        plt.plot(tp[:, 0]- tp[:, 1], 'go', markersize=10)
        plt.plot(fp[:, 0] - fp[:, 1], 'ro', markersize=10)

        plt.plot(fp[:, 0]/fp[:, 1], fp[:, 1], 'ro', markersize=2)
        plt.plot(tp[:, 0]/tp[:, 1], tp[:, 1], 'go', markersize=2)

        plt.plot(fp[:, 0] - fp[:, 1], fp[:, 1], 'ro', markersize=2)
        plt.plot(tp[:, 0] - tp[:, 1], tp[:, 1], 'go', markersize=2)

        plt.plot(fp[:, 0], fp[:, 1], 'ro', markersize=2)
        plt.plot(tp[:, 0], tp[:, 1], 'go', markersize=2)


    tp = -np.diff(np.array(tp_hmm), axis=1).ravel()
    fp = -np.diff(np.array(fp_hmm), axis=1).ravel()

    plt.close('all')
    plt.plot(fp, 'ro', markersize=2)
    plt.plot(tp, 'go', markersize=2)

    s = np.hstack([tp, fp]).T
    y_tue = np.hstack([np.ones(len(tp), dtype=bool),
                       np.zeros(len(fp), dtype=bool)]).T
    s.shape = (-1, 1)
    y_tue.shape = (-1, 1)
    precision, recall, thresholds = precision_recall_curve(y_tue, s)
    thresholds = np.hstack([thresholds[0], thresholds])
    auc = -np.trapz(precision, recall)


    save_dir = '/mnt/data10tb/ptg/skip_step_classifier_pr'
    snr = np.mean(np.diag(model.model.means_)/np.sqrt(model.model._covars_))
    print(snr)
    np.savetxt('%s/pr_snr=%0.5f.txt' % (save_dir, snr),
               np.vstack([precision, recall, thresholds]))


    fig = plt.figure(num=None, figsize=(14, 10), dpi=80)
    plt.rc('font', **{'size': 22})
    plt.rc('axes', linewidth=4)
    for fname in sorted(glob.glob('%s/*.txt' % save_dir)):
        snr = float(os.path.splitext(os.path.split(fname)[1])[0].split('pr_snr=')[1])
        # if snr < 4:
        #     continue

        precision, recall, thresholds = np.loadtxt(fname)
        plt.plot(recall, precision, linewidth=6, label='Detection SNR=%0.1f' % snr)

    plt.plot([0, 1], [0.5, 0.5], 'k--', linewidth=4, label='Guessing')

    plt.xlabel('Recall', fontsize=40)
    plt.ylabel('Precision', fontsize=40)
    plt.title('"Was Step Skipped?" Classifier', fontsize=40)
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    fig.tight_layout()

    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.tick_params(
            axis="y",
            which="major",
            grid_color='lightgrey')
    plt.tick_params(
            axis="y",
            which="minor",
            grid_linestyle='--',
            grid_color='lightgrey')
    plt.grid(axis='y', which='both')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.tick_params(
            axis="x",
            which="major",
            grid_color='lightgrey')
    plt.tick_params(
            axis="x",
            which="minor",
            grid_linestyle='--',
            grid_color='lightgrey')
    plt.grid(axis='x', which='both')
    plt.legend(fontsize=20, loc=0)

    plt.show()
    plt.savefig('/mnt/data10tb/ptg/skipped_step_classifier.png')
# ----------------------------------------------------------------------------
