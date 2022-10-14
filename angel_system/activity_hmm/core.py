import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import precision_recall_curve
import pandas as pd
import json
import yaml

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

        default_mean_conf = config['hmm']['class_mean_conf']
        default_std_conf = config['hmm']['class_std_conf']

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
                raise Exception('The %i-th step in \'%s\' should have \'id\' '
                                '%i but it has \'id\' %i' %
                                (i, config_fname, i, ii))

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

        self.model = ActivityHMM(self.dt, class_str, med_class_duration,
                                 num_steps_can_jump_fwd=0,
                                 num_steps_can_jump_bck=0,
                                 class_mean_conf=class_mean_conf,
                                 class_std_conf=class_std_conf)

        self.model_skip = ActivityHMM(self.dt, class_str,
                                      med_class_duration=med_class_duration,
                                      num_steps_can_jump_fwd=1,
                                      num_steps_can_jump_bck=0,
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
        X = np.tile(np.atleast_2d(conf_vec), (n,1))

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

        if self.times[-1]:
            last_time = self.times[-1]

            # If there is a long idle period, at most, let's add in 5 seconds
            # of buffer.
            last_time = max([last_time, start_time - 5])

            n = int(np.round((start_time - last_time)/self.dt))
            n = max([n, 1])
            n += 1
            times_fill = np.linspace(last_time, start_time, n)
            times_fill = (times_fill[1:] + times_fill[:-1])/2
            X_fill = np.tile(self.class_mean_conf, (n,1))

            self.times = np.hstack([self.times, times_fill])
            self.X = np.vstack([self.X, X_fill])

        self.times = np.hstack([self.times, times_])
        self.X = np.vstack([self.X, X])

    def get_skip_score(self):
        s = get_skip_score(self.model, self.model_skip, self.X)
        return s[0] - s[1]

    def get_current_state(self):
        log_prob1, Z, X_, Z_ = self.model.decode(self.X)
        return self.model.class_str[Z[-1]]


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
        #not_bckg_mask_[0] = True


        if class_mean_conf is not None and class_std_conf is not None:
            class_mean_conf = np.array(class_mean_conf)
            class_std_conf = np.array(class_std_conf)

            # We define an N x N mean matrix where element (i, j) is the mean value
            # emmitted for class j when class i is the state.
            if np.any(class_mean_conf > 1):
                raise Exception('\'class_mean_conf\' must be between 0-1')

            class_conf_mean_mat = np.zeros((N_, N_))
            sphere_conf_cov_mat = np.zeros(N_)

            # Square to turn from std to cov. We'll reuse the name.
            class_std_conf = class_std_conf**2

            #class_conf_mean_mat[:] = (1 - class_mean_conf[0])/(N - 1)

            ki = 0
            for i in range(N_):
                if not_bckg_mask_[i]:
                    #class_conf_mean_mat[i] = (1 - class_mean_conf[ki])/(N - 1)

                    class_conf_mean_mat[i, i] = class_mean_conf[ki]
                    # np.sum(class_conf_mean_mat[i])

                    sphere_conf_cov_mat[i] = class_std_conf[ki]

                    ki += 1
                else:
                    #class_conf_mean_mat[i, ~bckg_mask] = (1 - class_mean_conf[0])/(N - 1)
                    class_conf_mean_mat[i, bckg_mask] = class_mean_conf[0]

                    sphere_conf_cov_mat[i] = class_std_conf[0]

            #np.sum(class_conf_mean_mat[not_bckg_mask_][:, not_bckg_mask_], axis=1)
            #class_conf_mean_mat[~not_bckg_mask_][:, ~not_bckg_mask_].ravel()

            #class_conf_cov_mat[np.diag_indices_from(class_conf_cov_mat)] += 1e-6

            model.means_ = class_conf_mean_mat
            model._covars_ = sphere_conf_cov_mat


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
                raise Exception('\'force_skip_step\' must be an integer '
                                'between 1 and %i' % (len(self.class_str) - 1))

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
        plt.rc('font', **{'size': 22})
        plt.rc('axes', linewidth=4)
        plt.plot(recall, precision, linewidth=6, label='Raw Detections')
        plt.xlabel('Recall', fontsize=40)
        plt.ylabel('Precision', fontsize=40)
        fig.tight_layout()

    return auc
