import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import precision_recall_curve
import copy

from angel_system.impls.detect_activities.swinb.swinb_detect_activities import SwinBTransformer
from angel_system.eval.support_functions import time_from_name
from angel_system.eval.visualization import EvalVisualization
from angel_system.eval.compute_scores import EvalMetrics

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
        AutoMinorLocator
except ModuleNotFoundError:
    pass


class activity_hmm(object):
    def __init__(self, dt, class_str, med_class_duration,
                 num_steps_can_jump_fwd=0, num_steps_can_jump_bck=0,
                 class_mean_conf=None, class_std_conf=None):
        """

        Parameters
        ----------
        dt : float
            Time (seconds) between time steps. We will interpolate
            classifications to a time series of this fixed time step.

        class_str : list of str
            String characterizing of each class in the order that the steps
            should be carried out. The first element should be 'background'.

        med_class_duration : array of float
            The expected median duration of each activity (seconds). The ith
            element of this array sets how long we tend to stay in class[i] via
            the transition probability tii (from state i to state i). The
            probability of remaining in state i for N timesteps is given by
            tii^N.

        num_steps_can_jump_fwd : int
            In our modeling of how the sequence of classes can play out, we
            constrain such that a certain number of steps can be skipped forward.
            If set to 0, this enforces that the output estimated sequence will
            include every step. Otherwise, there can be gaps of missing steps
            of size 'num_steps_can_jump_fwd'.

        num_steps_can_jump_bck : int
            In our modeling of how the sequence of classes can play out, we
            constrain to allow jumping backwards a certain number of steps.
            If set to 0, this enforces that the output estimated sequence will
            move monotonically forward. Otherwise, the sequence could jump
            backwards 'num_steps_can_jump_bck' steps.

        class_mean_conf : array of float | None
            If provided, this defines the mean value of Guassian model for
            confidence emission for each class when that class is actually
            active. The remaining confidence will be distributed to the means
            of the remaining classes not active.

        class_std_conf : array of float | None
            If provided, this defines the standard deviation of the Guassian
            model for confidence emission for each class when that class is
            actually active.

        """
        assert class_str[0].lower() == 'background'
        assert num_steps_can_jump_fwd >= 0
        assert num_steps_can_jump_bck >= 0

        self.class_str = class_str
        self.dt = dt

        # The first element of 'class_str' should be 'background', which captures
        # all other possible classes not covered by the remaining elements of
        # 'class_str'. However, we want finer control such that if we enter another
        # state i and then move into a background state, we might only want to
        # allow certain possible transitions out of that background state. E.g., if
        # num_steps_can_jump_fwd = 0 and num_steps_can_jump_bck = 0, we would only
        # want transition to back to state i or to state i + 1. To achieve this
        # within the HMM framework, we need to define a background state between
        # each non-background state of 'class_str'.

        class_str_ = []
        bckg_mask = []

        # 'class_str_map' is defined such that
        # class_str[i] == class_str_[class_str_map[i]] is True.
        class_str_map = []

        inv_map = []

        k = 0
        for i, clss in enumerate(class_str):
            if i == 0:
                class_str_.append('background')
                class_str_map.append(len(class_str_) - 1)
                bckg_mask.append(True)
                inv_map.append(i)
                continue

            class_str_.append(clss)
            class_str_map.append(len(class_str_) - 1)
            inv_map.append(i)
            bckg_mask.append(False)
            bckg_mask.append(True)
            class_str_.append('background%i' % k)
            inv_map.append(0)
            k += 1

        class_str_map = np.array(class_str_map, dtype=int)
        self.class_str_map = class_str_map
        bckg_mask = np.array(bckg_mask, dtype=bool)
        self.bckg_mask = bckg_mask
        self.class_str_ = class_str_
        self.inv_map = inv_map

        N = len(class_str)
        N_ = len(class_str_)
        model = GaussianHMM(n_components=N_, covariance_type='spherical')
        self.model = model
        model.n_features = N_

        # Define the starting probability.
        model.startprob_ = np.zeros(N_)
        model.startprob_[0] = 1
        for i in range(1, num_steps_can_jump_fwd + 2):
            if i < 0 or i >= N_:
                continue

            model.startprob_[i] = 1

        model.startprob_ /= np.sum(model.startprob_)

        # Indices into N_-length arrays (e.g., class_str_) that correspond to the
        # original N-length arrays (e.g., class_str).
        not_bckg_mask_ = ~bckg_mask.copy()
        not_bckg_mask_[0] = True


        if class_mean_conf is not None and class_std_conf is not None:
            # We define an N x N mean matrix where element (i, j) is the mean value
            # emmitted for class j when class i is the state.
            class_conf_mean_mat = np.zeros((N_, N_))
            class_conf_cov_mat = np.zeros((N_, N_))
            sphere_conf_cov_mat = np.zeros(N_)

            # Square to turn from std to cov. We'll reuse the name.
            class_std_conf = class_std_conf**2

            ki = 0
            for i in range(N_):
                if not_bckg_mask_[i]:
                    class_conf_mean_mat[i] = (1 - class_mean_conf[ki])/(N - 1)

                    class_conf_mean_mat[i, i] = class_mean_conf[ki]
                    # np.sum(class_conf_mean_mat[i])

                    class_conf_cov_mat[i] = class_std_conf[ki]
                    sphere_conf_cov_mat[i] = class_std_conf[ki]

                    ki += 1
                else:
                    class_conf_mean_mat[i] = class_conf_mean_mat[0]
                    class_conf_cov_mat[i] = class_conf_cov_mat[0]
                    sphere_conf_cov_mat[i] = sphere_conf_cov_mat[0]

            #np.sum(class_conf_mean_mat[not_bckg_mask_][:, not_bckg_mask_], axis=1)
            #class_conf_mean_mat[~not_bckg_mask_][:, ~not_bckg_mask_].ravel()

            class_conf_cov_mat[np.diag_indices_from(class_conf_cov_mat)] += 1e-6


            model.means_ = class_conf_mean_mat
            model.covars_ = sphere_conf_cov_mat


        # -------------- Define transition probabilities -------------------------
        # Median number of timesteps spent in each step.
        n = med_class_duration/dt

        # If we are in an activity (class) with probability of transitioning out
        # tii, then the probability that we are still in this class after n
        # timesteps is tii^n.
        tdiag = (0.5)**(1/n)

        # Only valid states are possible.
        valid = np.zeros((N_, N_), dtype=bool)
        ki = 0
        for i in range(N_):
            # We are currently in state i, which other states are valid
            # transitions.
            i0 = i
            i1 = i + 2

            if i == (i//2)*2:
                # i is even, so it is a background state.
                i0 -= 1
                i1 += 0

                if num_steps_can_jump_fwd > 0:
                    i1 += num_steps_can_jump_fwd*2

                if num_steps_can_jump_bck > 0:
                    i0 -= 2*num_steps_can_jump_bck
            else:
                i0 += 0
                i1 += 0

                if num_steps_can_jump_fwd > 0:
                    i1 += num_steps_can_jump_fwd*2 + 1

                if num_steps_can_jump_bck > 0:
                    i0 -= 2*num_steps_can_jump_bck


            for ii in range(i0, i1):
                if ii < 0 or ii >= N_:
                    continue

                valid[i, ii] = True

        self.valid_trans = valid

        #print(valid)

        model.transmat_ = np.zeros((N_, N_))
        ki = 0
        for i in range(N_):
            if ~bckg_mask[i]:
                # specify which indices are valid
                sum(valid[i])

                model.transmat_[i, valid[i]] = (1 - tdiag[ki])/(sum(valid[i]) - 1)
                model.transmat_[i, i] = tdiag[ki]
                # np.sum(model.transmat_[i])

                ki += 1
            else:
                model.transmat_[i, valid[i]] = 1/(sum(valid[i]))

    def sample(self, N):
        """Return simulated classifier output X and truth state Z.

        Parameters
        ----------
        N : int
            Number of samples (states) to simulate.

        Return
        ------
        X : (n_samples, num_classes)
            Simulated detector confidences for each timestep (row) and each
            possible step being detected (column).
        Z : (n_samples,)
            Truth state associated with each time step.
        """
        X_, Z_ = self.model.sample(N)
        X_ = np.abs(X_)
        X = np.zeros((N, len(self.class_str)))

        X[:, 0] = np.mean(X_[:, self.bckg_mask], axis=1)
        X[:, 1:] = X_[:, ~self.bckg_mask]

        Z = np.array([self.inv_map[i] for i in Z_])

        X = (X.T/np.sum(X, axis=1)).T

        return X, Z

    def decode(self, X, force_skip_step=None):
        """
        Parameters
        ----------
        X : (n_samples, num_classes)
            Simulated detector confidences for each timestep (row) and each
            possible step being detected (column).
        force_skip_step : None | int
            Enforce that the solution skips a particular step.

        Return
        ------
        Z : (n_samples,)
            Truth state associated with each time step.
        """
        X_ = np.zeros((len(X), len(self.class_str_)))

        for i in np.where(self.bckg_mask)[0]:
            X_[:, i] = X[:, 0]

        X_[:, ~self.bckg_mask] = X[:, 1:]

        if force_skip_step is not None:
            if force_skip_step == 0 orforce_skip_step < len(self.class_str):
                raise Exception('\'force_skip_step\' must be an integer '
                                'between 1 and %i' % (len(self.class_str) - 1))

            # Make a copy of the model so we can adjust it.
            model = GaussianHMM(n_components=self.model.n_components,
                                covariance_type=self.model.covariance_type)
            model.n_features = self.model.n_features
            model.startprob_ = self.model.startprob_.copy()
            model.means_ = self.model.means_
            model.covars_ = self.model.covars_ + 1e-6
            model.transmat_ = self.model.transmat_

            ind = self.inv_map.index(force_skip_step)

            model.transmat_[ind, ind] = 0

            # These are the transition probabilities from each ith state
            # (ith element) into the state we are looking to skip.
            model.transmat_[:, ind]

            # We add two to skip the background state after.
            model.transmat_[:, ind + 2] += model.transmat_[:, ind]

            # Now noone can ever transition into 'force_skip_step'.
            model.transmat_[:, ind] =  0
        else:
            model = self.model

        log_prob, Z_ = model.decode(X_)


        state_sequence = np.array([self.inv_map[i] for i in Z_])

        return log_prob, state_sequence

    def did_skip_step(self, Z):
        """
        Parameters
        ----------
        Z : (n_samples,)
            Truth state associated with each time step.

        Return
        ------
        missing_ind : list of int
            Indices of missing steps (if any).

        missing_class : list of str
            String steps that were missed (if any).
        """
        inds = set(Z)

        if 0 in inds:
            # Remove the background state, it isn't relevant.
            inds.remove(0)

        missing_ind = set(range(1, len(self.class_str))).difference(inds)
        missing_ind = sorted(list(missing_ind))
        missing_class = [self.class_str[i] for i in missing_ind]

        return missing_ind, missing_class

    def save_to_disk(self, X, Z):



def score_detections(X, Z, plot_results=False):
    mask = np.zeros_like(X, dtype=bool)
    np.put_along_axis(mask, np.atleast_2d(Z).T, True, axis=1)

    tp = X[mask]
    fp = X[~mask]

    s = np.hstack([tp, fp]).T
    y_tue = np.hstack([np.ones(len(tp), dtype=bool),
                       np.zeros(len(fp), dtype=bool)]).T
    s.shape = (-1, 1)
    y_tue.shape = (-1, 1)

    precision, recall, thresholds = precision_recall_curve(y_tue, s)

    auc = -np.trapz(precision, recall)

    if plot_results:
        fig = plt.figure(num=None, figsize=(14, 10), dpi=80)
        plt.rc('font', **{'size': 22})
        plt.rc('axes', linewidth=4)
        plt.plot(recall, precision, linewidth=6, label='Raw Detections')
        plt.xlabel('Recall', fontsize=40)
        plt.ylabel('Precision', fontsize=40)
        fig.tight_layout()

    return auc



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

class_mean_conf = np.ones(N)*0.3
class_std_conf = np.ones(N)*0.1

# ----------------------------------------------------------------------------
model = activity_hmm(dt, class_str, med_class_duration=med_class_duration,
                     num_steps_can_jump_fwd=0, num_steps_can_jump_bck=0,
                     class_mean_conf=class_mean_conf,
                     class_std_conf=class_std_conf)
self = model

model1 = activity_hmm(dt, class_str, med_class_duration=med_class_duration,
                      num_steps_can_jump_fwd=0, num_steps_can_jump_bck=0,
                      class_mean_conf=class_mean_conf,
                      class_std_conf=class_std_conf)

model2 = activity_hmm(dt, class_str, med_class_duration=med_class_duration,
                      num_steps_can_jump_fwd=1, num_steps_can_jump_bck=0,
                      class_mean_conf=class_mean_conf,
                      class_std_conf=class_std_conf)

model3 = activity_hmm(dt, class_str, med_class_duration=med_class_duration,
                      num_steps_can_jump_fwd=1000, num_steps_can_jump_bck=1000,
                      class_mean_conf=class_mean_conf,
                      class_std_conf=class_std_conf)

# Measure performance of classifier of which step is active at each time.

if False:
    num_class_states = []
    for i in range(100):
        print(i)
        X, Z = model.sample(500)
        num_class_states.append([sum(Z == i_) for i_ in range(len(model.class_str))])

    num_class_states = np.array(num_class_states)
    #plt.plot(np.sort(num_class_states[:, :10].ravel()))
    plt.hist(num_class_states[:, :10].ravel(), 20)

# Z is the truth state.
X, Z = model.sample(1000)

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
        X, Z = model.sample(1000)
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
score_detections(X, Z, plot_results=True)

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
# Measure performance of “Did user complete every step?”

model_all = activity_hmm(dt, class_str, med_class_duration=med_class_duration,
                         num_steps_can_jump_fwd=0, num_steps_can_jump_bck=0,
                         class_mean_conf=class_mean_conf,
                         class_std_conf=class_std_conf)
model_skip = activity_hmm(dt, class_str, med_class_duration=med_class_duration,
                          num_steps_can_jump_fwd=1, num_steps_can_jump_bck=0,
                          class_mean_conf=class_mean_conf,
                          class_std_conf=class_std_conf)

num_trials = 50
tp_hmm = []
fp_hmm = []
for i in range(num_trials):
    print('%i/%i' % (i + 1, num_trials))

    # Simulate trial
    X, Z = model_all.sample(1000)

    # Remove one step for the skipped-step example.
    inds = list(set(Z))
    skip_ind = inds[np.random.randint(1, len(inds))]
    inds = np.array([True if Z[i] != skip_ind else False for i in range(len(Z))])

    X_skip = X[inds]
    Z_skip = Z[inds]

    # Analyze the example that really did have a skip in it.
    log_prob1, Z1 = model_all.decode(X_skip)


    log_prob2, Z2 = model_skip.decode(X_skip)

    if len(model_all.did_skip_step(Z2)) == 0:
        # The maximum-likelihood solution doesn't skip any steps, so we need to
        # investigate forcing a step skip.


    # Likelihood ratio.
    s = np.exp(log_prob1 - log_prob2)

    tp_hmm.append(s)

    # Analyze the example that did not have a skip in it.
    log_prob1 = model_skip.decode(X)[0]
    log_prob2 = model_all.decode(X)[0]

    # Likelihood ratio.
    s = np.exp(log_prob1 - log_prob2)

    fp_hmm.append(s)


s = np.hstack([tp, fp_hmm]).T
y_tue = np.hstack([np.ones(len(tp_hmm), dtype=bool),
                   np.zeros(len(fp_hmm), dtype=bool)]).T
s.shape = (-1, 1)
y_tue.shape = (-1, 1)
precision, recall, thresholds = precision_recall_curve(y_tue, s)
auc = -np.trapz(precision, recall)

fig = plt.figure(num=None, figsize=(14, 10), dpi=80)
plt.rc('font', **{'size': 22})
plt.rc('axes', linewidth=4)
plt.plot(recall, precision, linewidth=6, label='Raw Detections')
plt.xlabel('Recall', fontsize=40)
plt.ylabel('Precision', fontsize=40)
plt.title('"Was Step Skipped?" Classifier', fontsize=40)
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

plt.show()
plt.savefig('/mnt/data10tb/ptg/skipped_step_classifier.png')


# ----------------------------------------------------------------------------
