import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import precision_recall_curve
import pandas as pd
import json
import yaml

from angel_system.ptg_eval.common.load_data import time_from_name

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
        AutoMinorLocator
    HAS_MATLOTLIB = True
except ModuleNotFoundError:
    HAS_MATLOTLIB = False


class ActivityHMMRos:
    def __init__(self, config_fname):
        """

        Parameters
        ----------
        config_fname : str
            Path to the configuration yaml.

        """
        with open(config_fname, 'r') as stream:
            config = yaml.safe_load(stream)

        # Verify that all of the sections of the config exist.
        for key in ['version', 'activity_labels', 'title', 'steps', 'hmm']:
            if key not in config:
                raise AssertionError(f'config \'{config_fname}\' does not '
                                     f'have the required \'{key}\' defined')

        default_mean_conf = config['hmm']['default_mean_conf']
        default_std_conf = config['hmm']['default_std_conf']

        self.dt = config['hmm']['dt']
        self.X = None

        # Times associated with the center of the window associated with each
        # element of self.X.
        self.times = None

        # Start with a background class.
        class_str = ['background']
        med_class_duration = [1]
        class_mean_conf = [default_mean_conf]
        class_std_conf = [default_std_conf]

        for i in range(len(config['steps'])):
            ii = config['steps'][i]['id']
            if i != ii:
                raise Exception(f"The {i}th step in '{config_fname}' should have 'id' "
                                f"{i} but it has 'id' {ii}")

            class_str.append(config['steps'][i]['description'])
            med_class_duration.append(config['steps'][i]['median_duration_seconds'])

            try:
                class_mean_conf.append(config['steps'][i]['class_mean_conf'])
            except KeyError:
                class_mean_conf.append(default_mean_conf)

            try:
                class_std_conf.append(config['steps'][i]['class_std_conf'])
            except KeyError:
                class_std_conf.append(default_std_conf)

        class_mean_conf = np.array(class_mean_conf)
        self.class_mean_conf = class_mean_conf

        # This is the model that enforces steps are done in order without
        # skipping steps.
        self.noskip_model = ActivityHMM(self.dt, class_str, med_class_duration,
                                 num_steps_can_jump_fwd=0,
                                 num_steps_can_jump_bck=0,
                                 class_mean_conf=class_mean_conf,
                                 class_std_conf=class_std_conf)

        num_steps_can_jump_fwd = config['hmm']['num_steps_can_jump_fwd']
        num_steps_can_jump_bck = config['hmm']['num_steps_can_jump_bck']
        self.model = ActivityHMM(self.dt, class_str,
                                      med_class_duration=med_class_duration,
                                      num_steps_can_jump_fwd=num_steps_can_jump_fwd,
                                      num_steps_can_jump_bck=num_steps_can_jump_bck,
                                      class_mean_conf=class_mean_conf,
                                      class_std_conf=class_std_conf)

    def add_activity_classification(self, label_vec, conf_vec, start_time,
                                    end_time):
        """Provide activity classification results for time period.

        Parameters
        ----------
        label_vec : array-like of str | array-like of int
            DESCRIPTION.
        conf_vec : array-like of float
            Classifier's reperoted confidence associated with each class label
            in 'label_vec'.
        start_time : float
            Time (seconds) of the start of the window that the activity
            classification applies to.
        end_time : float
            Time (seconds) of the end of the window that the activity
            classification applies to.

        Returns
        -------
        None.

        """
        n = int(np.round((end_time - start_time)/self.dt))
        n = max([n, 1])
        n += 1
        times_ = np.linspace(start_time, end_time, n)
        times_ = (times_[1:] + times_[:-1])/2
        X = np.tile(np.atleast_2d(conf_vec), (len(times_), 1))

        if self.X is None:
            self.times = times_
            self.X = np.atleast_2d(X)
            return

        DT = times_[0] - self.times[-1]

        if DT < 0:
            raise Exception('Set a new classification time starting at time '
                            '%0.4f s that is %0.4f s in the past relative to '
                            'most-recent update for time %0.4f' % (start_time,
                                                                   DT,
                                                                   self.times[-1]))

        if DT > self.dt:
            last_time = self.times[-1]

            # If there is a long idle period, at most, let's add in 5 seconds
            # of buffer.
            last_time = max([last_time, start_time - 5])

            n = int(np.round((start_time - last_time)/self.dt))
            n = max([n, 1])
            n += 1
            times_fill = np.linspace(last_time, start_time, n)
            times_fill = (times_fill[1:] + times_fill[:-1])/2
            X_fill = np.tile(self.class_mean_conf, (len(times_fill), 1))

            self.times = np.hstack([self.times, times_fill])
            self.X = np.vstack([self.X, X_fill])

        self.times = np.hstack([self.times, times_])
        self.X = np.vstack([self.X, X])

    def revert_to_step(self, step_ind):
        """Erase history back to when we were at the specified step.

        Parameters
        ----------
        step_ind : int
            Step to revert with integer index that matches with 'id' field in
            the recipe file.

        Returns
        -------
        None.

        """
        log_prob1, Z, X_, Z_ = self.model.decode(self.X)
        ind = np.where(Z == step_ind)[0]
        if len(ind) == 0:
            raise ValueError(f'Found no previous steps <= {step_ind}')

        self.X = self.X[:ind[-1] + 1]
        self.times = self.times[:ind[-1] + 1]

    def get_skip_score(self):
        s = get_skip_score(self.noskip_model, self.model, self.X)
        return s[0] - s[1]

    def get_current_state(self):
        log_prob1, Z, X_, Z_ = self.model.decode(self.X)
        return self.model.class_str[Z[-1]]

    def analyze_current_state(self):
        """Return information about the current state.

        Returns
        -------
        Z_can_skip_ : int
            Index of most likely current step assuming steps could have been
            skipped according to 'num_steps_can_jump_fwd' and
            'num_steps_can_jump_bck'.
        Z_no_skip : int
            Index of most likely current step assuming steps that no steps were
            skipped.
        skip_score : float
            Score indicating how much more likely it is that a step was skipped
            than no steps being skipped.
        """
        log_prob1, Z_no_skip, X_, Z_no_skip_ = self.noskip_model.decode(self.X)
        log_prob1 = self.model.calc_log_prob_(X_, Z_no_skip_)

        log_prob2, Z_can_skip, _, Z_can_skip_ = self.model.decode(self.X)

        #log_prob2 = model_skip.calc_log_prob_(X_, Z_can_skip_)

        if len(self.model.did_skip_step(Z_can_skip)[0]) == 0:
            # The maximum likelihood solution doesn't skip a step, so we need to
            # explicitly check various assumptions of forced skipped steps.

            skipped_step_check = range(1, len(self.noskip_model.class_str))
            log_prob2 = []
            #skipped = []
            Z2s_ = []
            for j in skipped_step_check:
                log_prob2_, Z2_ = self.noskip_model.decode(X, force_skip_step=j)[:2]
                Z2s_.append(Z2_)
                log_prob2.append(log_prob2_)
                #skipped.append(model.did_skip_step(Z2_)[0])

            ind = np.argmax(log_prob2)
            Z_can_skip_ = Z2s_[ind]
            log_prob2 = log_prob2[ind]

        #score = (log_prob2 - log_prob1)/log_prob2

        skip_score = log_prob2 - log_prob1

        return Z_can_skip, Z_no_skip, skip_score


class ActivityHMM(object):
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
        med_class_duration = np.array(med_class_duration)
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
        fwd_map = []

        k = 0
        # Assuming class_str[0] is always 'background'
        class_str_.append('background')
        class_str_map.append(len(class_str_) - 1)
        bckg_mask.append(True)
        inv_map.append(0)
        fwd_map.append(0)

        # Initialize the remainder of the classes
        for i, clss in enumerate(class_str[1:], start=1):
            class_str_.append(clss)
            class_str_map.append(len(class_str_) - 1)
            fwd_map.append(len(inv_map))
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
        self.fwd_map = fwd_map

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
        not_bckg_mask = ~bckg_mask

        # Mask capturing the elements of X_ that correspond to the elements of
        # X.
        orig_mask = not_bckg_mask.copy()
        orig_mask[0] = True

        if class_mean_conf is not None:
            class_mean_conf = np.array(class_mean_conf)

            # We define an N x N mean matrix where element (i, j) is the mean
            # value emmitted for class j when class i is the state.
            if np.any(class_mean_conf > 1):
                raise ValueError('\'class_mean_conf\' must be between 0-1')

            if np.ndim(class_mean_conf) == 1:
                class_mean_conf = np.diag(class_mean_conf)
            elif np.ndim(class_mean_conf) > 2:
                raise ValueError('np.ndim(class_mean_conf) must be 1 or 2')

            class_conf_mean_mat = np.zeros((N_, N_))
            ki = 0
            for i in range(N_):
                if orig_mask[i]:
                    class_conf_mean_mat[i, orig_mask] = class_mean_conf[ki]
                    class_conf_mean_mat[i, ~orig_mask] = class_conf_mean_mat[i, 0]
                    ki += 1
                else:
                    class_conf_mean_mat[i] = class_conf_mean_mat[0]

            model.means_ = class_conf_mean_mat

        if class_std_conf is not None:
            class_std_conf = np.array(class_std_conf)

            # Square to turn from std to cov.
            class_std_conf2 = class_std_conf**2

            if np.ndim(class_std_conf) == 1:
                class_std_conf2 = np.tile(class_std_conf2, (N, 1))
            elif np.ndim(class_mean_conf) > 2:
                raise ValueError('np.ndim(class_std_conf) must be 1 or 2')

            # Full covariance
            model.covariance_type = 'diag'
            conf_cov_mat = np.zeros((N_, N_))

            ki = 0
            for i in range(N_):
                if orig_mask[i]:
                    conf_cov_mat[i, orig_mask] = class_std_conf2[ki]
                    conf_cov_mat[i, ~orig_mask] = conf_cov_mat[i, 0]
                    ki += 1
                else:
                    conf_cov_mat[i] = conf_cov_mat[0]

            model.covars_ = conf_cov_mat + 1e-9

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
                i1 += 1

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

        X[:, 0] = X_[:, 0]

        # Assign the true non-visibile-background confidence to the visible
        # background confidence.
        for i in range(N):
            if self.bckg_mask[Z_[i]]:
                X[i, 0] = X_[i, Z_[i]]

        X[:, 1:] = X_[:, ~self.bckg_mask]

        Z = np.array([self.inv_map[i] for i in Z_])

        if False:
            print(min([X[i, Z[i]] for i in range(len(Z))]))

        #X = (X.T/np.sum(X, axis=1)).T

        times = np.linspace(0, self.dt*(N - 1), N)

        return times, X, Z, X_, Z_

    def decode(self, X, force_skip_step=None):
        """
        Parameters
        ----------
        X : (n_samples, num_classes)
            Detector confidences for each timestep (row) and each
            possible step being detected (column).
        force_skip_step : None | int
            Enforce that the solution skips a particular step.

        Return
        ------
        log_prob : float
            Logarithm of probability.
        Z : (n_samples,)
            Truth state associated with each time step.
        """
        X_ = np.zeros((len(X), len(self.class_str_)))

        for i in np.where(self.bckg_mask)[0]:
            X_[:, i] = X[:, 0]

        X_[:, ~self.bckg_mask] = X[:, 1:]

        if force_skip_step is not None:
            if force_skip_step == 0 or force_skip_step >= len(self.class_str):
                raise ValueError('\'force_skip_step\' must be an integer '
                                 'between 1 and %i' %
                                 (len(self.class_str) - 1))

            # Make a copy of the model so we can adjust it.
            model = GaussianHMM(n_components=self.model.n_components,
                                covariance_type=self.model.covariance_type)
            model.n_features = self.model.n_features
            model.startprob_ = self.model.startprob_.copy()
            model.means_ = self.model.means_.copy()
            model._covars_ = self.model._covars_.copy()
            model.transmat_ = self.model.transmat_.copy()

            ind = self.inv_map.index(force_skip_step)

            model.transmat_[ind, ind] = 0
            model.transmat_[ind] /= sum(model.transmat_[ind])

            if ind == 1:
                model.startprob_[0] = 1
                model.startprob_[1] = 0

            valid = self.valid_trans.copy()
            tdiag = np.diag(self.model.transmat_)
            bckg_mask = self.bckg_mask
            N_ = len(self.class_str_)

            if ind < model.transmat_.shape[1] - 3:
                # These are the transition probabilities from each ith state
                # (ith element) into the state we are looking to skip.
                # model.transmat_[:, ind]

                # We add two to skip the background state immediately after to
                # the next real step.
                model.transmat_[:, ind + 2] += model.transmat_[:, ind]

                # Now noone can ever transition into 'force_skip_step'.
                model.transmat_[:, ind] =  0
            else:
                # We are at the second-to-last element (the last element is
                # background after the last step).
                N = model.transmat_.shape[1]
                ind2 = np.where(model.transmat_[:, ind] > 0)[0]

                for i in ind2:
                    # Take the probability allotated to move it into the last-
                    # viable background state ind - 1.

                    model.transmat_[i, ind - 1] += model.transmat_[i, ind]
                    model.transmat_[i, ind] =  0
        else:
            model = self.model

        log_prob, Z_ = model.decode(X_)

        Z = np.array([self.inv_map[i] for i in Z_])

        return log_prob, Z, X_, Z_

    def calc_log_prob_(self, X_, Z_, verbose=False):
        #log_prob0, Z_ = self.model.decode(X_)

        # We can't allow multiple possible backgrounds to have a high score.


        log_prob_ = self.model._compute_log_likelihood(X_)

        # We can't allow multiple possible backgrounds to have a high score.


        log_prob = log_prob_[0, Z_[0]] + np.log(self.model.startprob_[Z_[0]])

        if verbose:
            print('Log(prob) of first state', log_prob)

        for i in range(1, len(Z_)):
            log_prob += log_prob_[i, Z_[i]]

            if verbose:
                print('Log(prob) step %i being in state %i with detector'
                      'confidence %0.6f:' % (i, Z_[i], X_[i, Z_[i]]),
                      log_prob_[i, Z_[i]])

            if self.model.transmat_[Z_[i - 1], Z_[i]] == 0:
                print('Cannot move from', Z_[i - 1], 'to', Z_[i])
                raise Exception()

            # We moved from Z_[i - 1] to Z_[i].
            log_prob += np.log(self.model.transmat_[Z_[i - 1], Z_[i]])

            if verbose:
                print('Log(prob) for the transition from step %i being in '
                      'state %i to step %i being in state %i:' % (i - 1,
                      Z_[i - 1], i, Z_[i]),
                      np.log(self.model.transmat_[Z_[i - 1], Z_[i]]))

        return log_prob

    def predict_proba(self, X):
        """
        Parameters
        ----------
        X : (n_samples, num_classes)
            Simulated detector confidences for each timestep (row) and each
            possible step being detected (column).

        Return
        ------
        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample from ``X``.
        """
        X_ = np.zeros((len(X), len(self.class_str_)))

        for i in np.where(self.bckg_mask)[0]:
            X_[:, i] = X[:, 0]

        X_[:, ~self.bckg_mask] = X[:, 1:]

        posteriors_ = self.model.predict_proba(X_)

        bckg_mask = self.bckg_mask.copy()
        bckg_mask[0] = False

        posteriors_[:, 0] = posteriors_[:, 0] + np.sum(posteriors_[:, bckg_mask], axis=1)

        posteriors = posteriors_[:, ~bckg_mask]

        return posteriors

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

    def save_to_disk(self, times, X, Z, det_json_fname, gt_feather_fname):
        """Save siumulated data to disk.

        Parameters
        ----------
        times : (n_samples)
            Time associated with each row/element of X and Z.
        X : (n_samples, num_classes)
            Simulated detector confidences for each timestep (row) and each
            possible step being detected (column).

        Z : (n_samples,)
            Truth state associated with each time step.

        det_json_fname : str
            File path for output detection json.

        gt_feather_fname : str
            File path for output ground truth feather file.
        """
        time_bins = (times[1:] + times[:-1])/2
        time_bins = np.hstack([(times[0] - times[1])/2, time_bins,
                               times[-1] + (times[-1] - times[-2])/2])
        time_bins -= time_bins[0]

        # Save ground truth.
        img_fnames = []
        for i, t in enumerate(time_bins):
            s = int(np.floor(t))
            micro_ = (t - s)*1e6
            micro = int(np.floor(micro_))
            img_fnames.append('frame_%i_%i_%i.png' % (i + 1, s, micro))

        data = []
        istart = 0
        iend = 0
        stop_i = len(Z) - 1
        while istart <= stop_i:
            if iend == stop_i or Z[iend + 1] != Z[istart]:
                data.append([self.class_str[Z[istart]],
                             img_fnames[istart],
                             img_fnames[iend + 1],
                             'simulated'])
                #print(istart, iend, Z[istart], Z[iend], Z[iend+1], iend - istart)
                iend = iend + 1
                istart = iend
            else:
                iend = iend + 1

        gt = pd.DataFrame(data,columns=['class', 'start_frame', 'end_frame',
                                        'exploded_ros_bag_path'])
        gt.to_feather(gt_feather_fname)

        # Save detections
        detections = []
        for i in range(len(X)):
            det = {}
            t = time_bins[i]
            s = int(np.floor(t))
            nano = int((t - s)*1e9)
            det['header'] = {'time_sec': s,
                             'time_nanosec': nano,
                             'frame_id': 'Activity detection'}
            det['source_stamp_start_frame'] = time_bins[i]
            det['source_stamp_end_frame'] = time_bins[i + 1]
            det['label_vec'] = self.class_str
            det['conf_vec'] = X[i].tolist()
            detections.append(det)

        with open(det_json_fname, "w") as write_file:
            json.dump(detections, write_file)


def get_skip_score(model, model_skip, X, brute_search=False):
    """

    Parameters
    ----------
    model : TYPE
        Model that enforces no steps are skipped.
    model_skip : TYPE
        Model that allows steps to be skipped.
    X : (n_samples, num_classes)
        Detector confidences for each timestep (row) and each possible step
        being detected (column).
    brute_search: bool
        Whether to use a slower search method. If the model that allows for
        skip stepping yields a solution that does not actually skip a step,
        then log_prob1 will equal log_prob2, and their difference will be zero
        indicating that step skipping is equally likely to completein of all
        steps. In these cases, if 'brute_search' is set to True, sequences
        of potential solutions are checked, each enforecd skipping to skip one
        step, and the most-likely of these is used as the skipped-step example.
        This will yield a log_prob1 < log_prob2, indicating that step skipping
        is actually less likely than completing all steps.

    Returns
    -------
    log_prob2 : TYPE
        Logarithm of likelihood of the solution where steps are skipped.
    log_prob1 : TYPE
        Logarithm of likelihood of the solution where steps are not skipped.

    """
    # Solution that doesn't skip.

    log_prob1, Z1, X_, Z1_ = model.decode(X)
    log_prob1 = model_skip.calc_log_prob_(X_, Z1_)

    log_prob2, Z2, _, Z2_ = model_skip.decode(X)

    #log_prob2 = model_skip.calc_log_prob_(X_, Z2_)

    if len(model_skip.did_skip_step(Z2)[0]) == 0:
        # The maximum likelihood solution doesn't skip a step, so we need to
        # explicitly check various assumptions of forced skipped steps.

        skipped_step_check = range(1, len(model.class_str))
        log_prob2 = []
        #skipped = []
        for j in skipped_step_check:
            log_prob2_, Z2_ = model.decode(X, force_skip_step=j)[:2]
            log_prob2.append(log_prob2_)
            #skipped.append(model.did_skip_step(Z2_)[0])

        ind = np.argmax(log_prob2)
        log_prob2 = log_prob2[ind]

    #score = (log_prob2 - log_prob1)/log_prob2

    return log_prob2, log_prob1


def score_raw_detections(X, Z, plot_results=False):
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
        plt.rc('font', **{'size': 28})
        plt.rc('axes', linewidth=4)
        plt.plot(recall, precision, linewidth=6, label='Raw Detections')
        plt.xlabel('Recall', fontsize=40)
        plt.ylabel('Precision', fontsize=40)
        fig.tight_layout()

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

    return auc


def load_and_discretize_data(activity_gt: str,
                             extracted_activity_detections: str,
                             time_window: float, uncertain_pad: float):
    """Loads unstructured detection and ground truth data and discretize.

    Parameters
    ----------
    activity_gt : str
        Path to activity ground truth feather file.
    extracted_activity_detections : str
        Path to detection json.
    time_window : float
        Time (seconds) to discretize to.
    uncertain_pad : float
        Time uncertainty padding (seconds) during change of class.

    Returns
    -------
    time_windows : Numpy 2-D array of float
        time_windows[i, 0] encodes the start time for the ith time window and
        time_windows[i, 1] the end time of the ith window (seconds).
    labels : list of str
        labels[j] encodes the string name for the ith activity class.
    dets_per_valid_time_w : Numpy 2-D array
        dets_per_valid_time_w[i, j] encodes the classifier's confidence that
        activity j occurred in the ith time window.
    gt_label : array-like of int
        gt_label[i] encodes the ground truth class integer that occurred at in
        the ith time window.
    valid : array-like of bool
        Encodes which time windows are valid for scoring. An invalid time
        window may occur when it is too close to a change in ground-truth
        class.

    """
    gt_f = pd.read_feather(activity_gt)
    # Keys: class, start_frame,  end_frame, exploded_ros_bag_path

    gt = []
    for i, row in gt_f.iterrows():
        g = {
            'class': row["class"].lower().strip(),
            'start': time_from_name(row["start_frame"]),
            'end': time_from_name(row["end_frame"])
        }
        gt.append(g)

    print(f"Loaded ground truth from {activity_gt}")
    gt = pd.DataFrame(gt)

    with open(extracted_activity_detections) as json_file:
        detections_input = json.load(json_file)

    detections = []

    for dets in detections_input:
        good_dets = {}
        for l, conf in zip(dets["label_vec"], dets["conf_vec"]):
            good_dets[l] = conf

        for l in dets["label_vec"]:
            d = {
                'class': l.lower().strip(),
                'start': dets["source_stamp_start_frame"],
                'end': dets["source_stamp_end_frame"],
                'conf': good_dets[l],
                'detect_intersection': np.nan
            }
            detections.append(d)
    detections = pd.DataFrame(detections)
    print(f"Loaded detections from {extracted_activity_detections}")

    # ============================
    # Load labels
    # ============================
    labels0 = [l.lower().strip().rstrip('.') for l in detections['class']]
    labels = []
    labels_ = set()
    for label in labels0:
        if label not in labels_:
            labels_.add(label)
            labels.append(label)

    print(f"Labels: {labels}")

    # ============================
    # Split by time window
    # ============================
    # Get time ranges
    min_start_time = min(gt['start'].min(), detections['start'].min())
    max_end_time = max(gt['end'].max(), detections['end'].max())
    dt = time_window
    time_windows = np.arange(min_start_time, max_end_time, time_window)

    if time_windows[-1] < max_end_time:
        time_windows = np.append(time_windows, time_windows[-1] + time_window)
    time_windows = list(zip(time_windows[:-1], time_windows[1:]))
    time_windows = np.array(time_windows)
    dets_per_valid_time_w = np.zeros((len(time_windows), len(labels)),
                                     dtype=float)


    def get_time_wind_range(start, end):
        """Return slice indices of time windows that reside completely in
        start->end.
        time_windows[ind1:ind2] all live inside start->end.
        """
        # The start time of the ith window is min_start_time + dt*i.
        ind1_ = (start - min_start_time)/dt
        ind1 = int(np.ceil(ind1_))
        if ind1_ - ind1 + 1 < 1e-15:
            # We want to avoid the case where ind1_ is (j + eps) and it gets
            # rounded up to j + 1.
            ind1 -= 1

        # The end time of the ith window is min_start_time + dt*(i + 1).
        ind2_ = (end - min_start_time)/dt
        ind2 = int(np.floor(ind2_))
        if -ind2_ + ind2 + 1 < 1e-15:
            # We want to avoid the case where ind1_ is (j - eps) and it gets
            # rounded up to j - 1.
            ind1 += 1

        ind1 = max([ind1, 0])
        ind2 = min([ind2, len(time_windows)])

        return ind1, ind2


    # Valid time windows overlap with a detection.
    valid = np.zeros(len(time_windows), dtype=bool)
    for i in range(len(detections)):
        ind1, ind2 = get_time_wind_range(detections['start'][i],
                                         detections['end'][i])

        valid[ind1:ind2] = True
        correct_label = detections['class'][i].strip().rstrip('.')
        correct_class_idx = labels.index(correct_label)
        dets_per_valid_time_w[ind1:ind2, correct_class_idx] = np.maximum(dets_per_valid_time_w[ind1:ind2, correct_class_idx],
                                                              detections['conf'][i])

    gt_true_mask = np.zeros((len(time_windows), len(labels)), dtype=bool)
    for i in range(len(gt)):
        ind1, ind2 = get_time_wind_range(gt['start'][i], gt['end'][i])
        correct_label = gt['class'][i].strip().rstrip('.')
        correct_class_idx = labels.index(correct_label)
        gt_true_mask[ind1:ind2, correct_class_idx] = True

    if not np.all(np.sum(gt_true_mask, axis=1) <= 1):
        raise ValueError('Conflicting ground truth for same time windows')

    # If ground truth isn't specified for a particular window, we should assume
    # 'background'.
    bckg_class_idx = labels.index('background')
    ind = np.where(np.all(gt_true_mask == False, axis=1))[0]
    gt_true_mask[ind, bckg_class_idx] = True

    # Any time the ground truth class changes, we want to add in uncertainty
    # padding, but there should always be at least one time window at the
    # center of the ground-truth span.
    gt_label = np.argmax(gt_true_mask, axis=1)
    pad = int(np.round(uncertain_pad/dt))
    if pad > 0:
        ind = np.where(np.diff(gt_label, axis=0) != 0)[0] + 1
        if ind[0] != 0:
            ind = np.hstack([1, ind])

        if ind[-1] != len(time_windows):
            ind = np.hstack([ind, len(time_windows)])

        for i in range(len(ind) -1):
            ind1 = ind[i]
            ind2 = ind[i+1]
            # time windows in range ind1:ind2 all have the same ground
            # truth class.

            ind1_ = ind1 + pad
            ind2_ = ind2 - pad
            indc = int(np.round((ind1 + ind2)/2))
            ind1_ = min([ind1_, indc])
            ind2_ = max([ind2_, indc + 1])
            valid[ind1:ind1_] = False
            valid[ind2_:ind2] = False

    return time_windows, labels, dets_per_valid_time_w, gt_label, valid
