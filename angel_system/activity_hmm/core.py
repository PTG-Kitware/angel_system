import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import precision_recall_curve
import pandas as pd
import json
import yaml
import os
import scipy
import time

from angel_system.data.common.load_data import time_from_name
from angel_system.data.common.load_data import activities_from_dive_csv

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

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
        # Initialize variables
        self.X = None
        self.default_std_conf = None
        self.default_mean_conf = None

        # Times associated with the center of the window associated with each
        # element of self.X.
        self.times = None

        with open(config_fname, "r") as stream:
            config = yaml.safe_load(stream)

        # Verify that all of the top-level sections of the config exist.
        for key in ["version", "activity_labels", "title", "steps", "hmm"]:
            if key not in config:
                raise AssertionError(
                    f"config '{config_fname}' does not have the required '{key}' defined"
                )

        try:
            self.activity_labels_fname = config["activity_labels"]
        except KeyError:
            raise AssertionError(
                f"config '{config_fname}' does not have the required "
                f"'activity_labels' defined"
            )

        try:
            self.activity_mean_and_cov_fname = config["activity_mean_and_std_file"]
            loaded_mean_and_cov = True
        except KeyError:
            loaded_mean_and_cov = False
            self.activity_mean_and_cov_fname = None

        if self.activity_mean_and_cov_fname is not None:
            ret = np.load(self.activity_mean_and_cov_fname, allow_pickle=True)
            class_mean_conf, class_std_conf = ret
        else:
            self.default_mean_conf = config["hmm"]["default_mean_conf"]
            self.default_std_conf = config["hmm"]["default_std_conf"]
            class_mean_conf = []
            class_std_conf = []

        try:
            self.task_title = config["title"]
        except KeyError:
            raise AssertionError(
                f"config '{config_fname}' does not have the required 'title' defined"
            )

        try:
            self.dt = config["hmm"]["dt"]
        except KeyError:
            raise AssertionError(
                f"config '{config_fname}' does not have the required 'dt' under "
                f"'hmm' defined"
            )

        class_str = []
        med_class_duration = []
        activity_per_step = []

        steps = config["steps"]

        # Verify that all steps have sufficient information defined.
        for i in range(len(steps)):
            if "id" not in steps[i]:
                raise AssertionError(
                    f"The {i}th step in '{config_fname}' does not define the 'id'"
                )

            for key in ["description", "activity_id", "median_duration_seconds"]:
                if key not in steps[i]:
                    raise AssertionError(
                        f"The step with id: {steps[i]['id']} "
                        f"in '{config_fname}' does not "
                        f"define the required field '{key}'"
                    )

            # Strip inline comments.
            steps[i]["description"] = steps[i]["description"].split("#")[0].rstrip()

        # Step 0 must be the background step. The recipe yaml may explicitly
        # define it as such, or it may leave it out (and imply it), starting by
        # defining step 1.
        if steps[0]["id"] == 0:
            # If step with id=0 is defined, it better be background.
            if steps[0]["description"].lower() not in ["background", "background."]:
                raise AssertionError(
                    f"'{config_fname}' defines a step with "
                    "id=0, but the first step should start "
                    "with id=1 with id=0 implied but not "
                    "explicitly defined to be a background "
                    "state"
                )
        else:
            # Create the implied background step.
            steps.insert(
                0,
                {
                    "id": 0,
                    "activity_id": 0,
                    "description": "Background",
                    "median_duration_seconds": 5,
                },
            )
            if not loaded_mean_and_cov:
                steps[0]["mean_conf"] = self.default_mean_conf
                steps[0]["std_conf"] = self.default_std_conf

        for i in range(len(steps)):
            ii = steps[i]["id"]

            if i != ii:
                raise AssertionError(
                    f"The {i}th step in '{config_fname}' should have 'id' {i} "
                    f"but it has 'id' {ii}"
                )

            if i == 0:
                # This must be the background step.
                if not steps[0]["description"].lower() in ["background", "background."]:
                    raise AssertionError(
                        f"The step with id=0 must be the "
                        "background state with "
                        "description='Background'"
                    )
            elif steps[i]["description"].lower() in ["background", "background."]:
                raise AssertionError(f"The background state must have id=0")

            activity_per_step.append(steps[i]["activity_id"])

            class_str_ = steps[i]["description"]
            class_str_ = class_str_.split("#")[0]
            class_str_ = class_str_.rstrip()
            class_str.append(class_str_)

            med_class_duration.append(steps[i]["median_duration_seconds"])

            if loaded_mean_and_cov:
                pass
            else:
                try:
                    class_mean_conf.append(steps[i]["class_mean_conf"])
                except KeyError:
                    class_mean_conf.append(self.default_mean_conf)

                try:
                    class_std_conf.append(steps[i]["class_std_conf"])
                except KeyError:
                    class_std_conf.append(self.default_std_conf)

        self.activity_per_step = activity_per_step
        self.class_str = class_str
        self.med_class_duration = med_class_duration

        self.num_steps_can_jump_fwd = config["hmm"]["num_steps_can_jump_fwd"]
        self.num_steps_can_jump_bck = config["hmm"]["num_steps_can_jump_bck"]

        self.set_hmm_mean_and_std(class_mean_conf, class_std_conf)

    @property
    def num_steps(self):
        """Number of steps in the recipe."""
        return self.model.num_steps

    @property
    def num_activities(self):
        """Return number of dimensions in classification vector to be recieved."""
        return self.model.num_activities

    @property
    def class_mean_conf(self):
        return self._class_mean_conf

    @property
    def class_std_conf(self):
        return self._class_std_conf

    def get_hmm_mean_and_std(self):
        """Return the mean and standard deviation of activity classifications."""
        return self.model.get_hmm_mean_and_std()

    def set_hmm_mean_and_std(self, class_mean_conf, class_std_conf):
        """Set the mean and standard deviation of activity classifications."""
        self._class_mean_conf = np.array(class_mean_conf)
        self._class_std_conf = np.array(class_std_conf)
        self.model = ActivityHMM(
            self.dt,
            self.class_str,
            med_class_duration=self.med_class_duration,
            num_steps_can_jump_fwd=self.num_steps_can_jump_fwd,
            num_steps_can_jump_bck=self.num_steps_can_jump_bck,
            class_mean_conf=self.class_mean_conf,
            class_std_conf=self.class_std_conf,
        )

        self.unconstrained_model = ActivityHMM(
            self.dt,
            self.class_str,
            self.med_class_duration,
            num_steps_can_jump_fwd=self.num_steps,
            num_steps_can_jump_bck=self.num_steps,
            class_mean_conf=self.class_mean_conf,
            class_std_conf=self.class_std_conf,
        )

        # This is the model that enforces steps are done in order without
        # skipping steps.
        self.noskip_model = ActivityHMM(
            self.dt,
            self.class_str,
            self.med_class_duration,
            num_steps_can_jump_fwd=0,
            num_steps_can_jump_bck=0,
            class_mean_conf=self.class_mean_conf,
            class_std_conf=self.class_std_conf,
        )

    def add_activity_classification(self, label_vec, conf_vec, start_time, end_time):
        """Provide activity classification results for time period.

        Parameters
        ----------
        label_vec : array-like of int
            Activity id (activity_id in recipe) associated with each value in
            conf_vec.
        conf_vec : array-like of float
            Classifier's reported confidence associated with each class label
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
        assert end_time > start_time
        t = (end_time + start_time) / 2

        label_vec = list(label_vec)
        X = [conf_vec[label_vec.index(i)] for i in range(self.num_activities)]

        if self.X is None:
            self.times = np.array([t])
            self.X = np.atleast_2d(conf_vec)
            return

        DT = t - self.times[-1]

        if DT <= 0:
            raise AssertionError(
                "Set a new classification time starting at "
                "time %0.4f s that is %0.4f s in the past "
                "relative to most-recent update for time "
                "%0.4f" % (start_time, DT, self.times[-1])
            )

        self.times = np.hstack([self.times, t])
        self.X = np.vstack([self.X, conf_vec])

    def clear_history(self):
        """Erase all history of classifier outputs.

        Returns
        -------
        None.

        """
        self.X = None
        self.times = None

    def revert_to_step(self, step_ind):
        """Erase history back to when we were at the specified step.

        Parameters
        ----------
        step_ind : int
            Step to revert with integer index that matches with 'id' field in
            the recipe file. The actual index reverted to may be earlier than
            provided if the index is within a region of skipped steps.

        Returns
        -------
        None.

        """
        log_prob1, Z, _, Z_ = self.model.decode(self.X)
        ind = np.where(np.logical_and(Z <= step_ind, Z != 0))[0]
        if len(ind) == 0:
            raise ValueError(f"Found no previous steps <= {step_ind}")

        self.X = self.X[: ind[-1] + 1]
        self.times = self.times[: ind[-1] + 1]

    def get_skip_score(self):
        s = get_skip_score(self.noskip_model, self.model, self.X)
        return s[0] - s[1]

    def get_current_state(self):
        """Return HMM's most-likely current step."""
        log_prob1, Z, _, Z_ = self.model.decode(self.X)
        return self.model.class_str[Z[-1]]

    def analyze_current_state(self):
        """Return information about the current state.

        Returns
        -------
        times : Times associated with each state (block of time with associated
            activity) in the HMM history..

        state_sequence : list of int
            List of integers (same length as times) encoding the most-likely
            step, as estimated by the HMM, that the user was in at each time in
            'times'. Value 0 indicates background, and value 1 indicates the
            first step of the recipe.
        step_finished_conf : list of float
            Array of length equal to the number of steps in the recipe
            (not including background) indicating the confidence that the user
            was in that step at some point in the history.
        unfiltered_step_conf : list of float
            Array of length equal to the number of steps in the recipe
            (including background) indicating the confidence that the user
            is currently in that step. Values range from 0-1.
        """
        log_prob0, state_sequence, _, state_sequence_ = self.model.decode(self.X)
        log_prob0 = self.unconstrained_model.calc_log_prob_(self.X, state_sequence_)

        # log_prob1, state_sequence1, _, state_sequence1_ = self.noskip_model.decode(self.X)

        states = set(state_sequence)
        step_finished_conf = [s in states for s in range(1, self.model.num_steps)]
        step_finished_conf = np.array(step_finished_conf, dtype=float)

        # for i in range(len(step_finished_conf)):
        #     if step_finished_conf[i] == 1:
        #         step = i + 1
        #         #print('Forcing skip on step', step)
        #         fksip_model = self.model.get_model_force_skip_step(step)
        #         log_prob2, state_sequence2_ = fksip_model.decode(self.X)
        #         log_prob2 = self.unconstrained_model.calc_log_prob_(self.X,
        #                                                             state_sequence2_)

        #         step_finished_conf[i] = np.exp((log_prob2 - log_prob0)/log_prob0)

        log_prob_ = self.model.model._compute_log_likelihood(self.X[-1:])
        log_prob_ = log_prob_[:, self.model.fwd_map].ravel()
        log_prob_ -= log_prob_.max()
        prob = np.exp(log_prob_)
        prob /= sum(prob)

        unfiltered_step_conf = prob

        return self.times, state_sequence, step_finished_conf, unfiltered_step_conf

    def save_task_yaml(self, fname, save_weights_inline=False):
        if save_weights_inline:
            mean, std = self.get_hmm_mean_and_std()

        with open(fname, "w") as f:
            f.write(
                "# Schema version.\n"
                'version: "1.0"\n'
                "\n"
                "# Reference to the activity classification labels configuration that we will\n"
                "# reference into.\n"
                f'activity_labels: "{self.activity_labels_fname}"\n\n'
            )

            if not save_weights_inline:
                f.write(
                    "# Reference to the file defining the mean and standard deviation of the\n"
                    "# activity classifications to be used by the HMM. For N activities, both the\n"
                    "# mean and standard deviation should be N x N matrices such that when activity\n"
                    "# i is actually occuring, the classifier will emit confidence\n"
                    "# mean[i, j] +/- std[i, j] for activity j.\n"
                    f'activity_mean_and_std_file: "{self.activity_mean_and_cov_fname}"\n\n'
                )

            f.write(
                "# Task title for display purposes.\n"
                f'title: "{self.task_title}"\n'
                "\n"
                "# Layout of the steps that define this task.\n"
                "steps:\n"
                "  # Item format:\n"
                "  # - id: Identifying integer for the step.\n"
                "  # - activity_id: The ID of an activity classification associated with this\n"
                "  #                step. This must reference an ID within the `activity_labels`\n"
                "  #                configuration file referenced above.\n"
                "  # - description: Human semantic description of this step.\n"
                "  # - median_duration_seconds: Median expected time this task will\n"
                "  #                            consume in seconds.\n"
                "  # - mean_conf: mean value of classifier confidence for true examples.\n"
                "  # - std_conf: standard deviation of confidence for both true and false\n"
                "  #             examples.\n"
            )

            for i in range(1, len(self.activity_per_step)):
                if i == 1:
                    f.write(
                        f"  - id: {i}   # Must start at 1, 0 is reserved for background.\n"
                    )
                else:
                    f.write(f"  - id: {i}\n")

                f.write(
                    f"    activity_id: {self.activity_per_step[i]}\n"
                    f"    description: >-\n"
                    f"      {self.class_str[i]}\n"
                    f"    median_duration_seconds: {self.med_class_duration[i]}\n"
                )

                if save_weights_inline:
                    mean_ = str(mean[i]).replace("\n", "")
                    f.write(f"    mean_conf: {mean_}\n")
                    std_ = str(std[i]).replace("\n", "")
                    f.write(f"    std_conf: {std_}\n")

            # Write the final details about the HMM.
            f.write(
                f"\n# Hidden markov model configuration parameters\n"
                f"hmm:\n"
                f"  # Time (seconds) between time steps of HMM. Sets the temporal precision of\n"
                f"  # the HMM analysis at the expense of processing costs.\n"
                f"  dt: {self.dt}\n\n"
                f"  # Constrain whether HMM sequence can skip steps or jump backwards. When both\n"
                f"  # values are set to 0, forward progress without skipping steps is enforced.\n"
                f"  num_steps_can_jump_fwd: {self.num_steps_can_jump_fwd}\n"
                f"  num_steps_can_jump_bck: {self.num_steps_can_jump_bck}\n"
                f"\n"
            )

            if self.default_mean_conf is not None:
                f.write(
                    f"  # Default classifier mean confidence to use if not explicitly provided for a\n"
                    f"  # step.\n"
                    f"  default_mean_conf: {self.default_mean_conf}\n\n"
                )

            if self.default_std_conf is not None:
                f.write(
                    f"  # Default classifier standard deviation of confidence to use if not\n"
                    f"  # explicitly provided for a step.\n"
                    f"  default_std_conf: {self.default_std_conf}\n"
                )


class ActivityHMM(object):
    """Models a time sequence of activities as a hidden Markov model.

    We assume that at any given time, the person that we are analyzing their
    egocentric video is engaged in one activity from a set of N possible
    activities captured by 'class_str'. The first element of 'class_str' is
    'background', which captures any activity not covered by the remaining
    elements in 'class_str'. We discretize our analysis of the person's
    progression through activities into fixed time steps of size 'dt' seconds
    (e.g., 1 s), and we assume that the person is engaged in one and only one
    activity within each dt-sized time window. Our task is to consider some
    associated observations (e.g., activity classifier outputs), one set per
    time window, and to infer what the person was most-likely doing in each
    time window (i.e., infer the hidden states). We are looking to filter the
    raw activity classifier outputs with additional temporal constraints that
    we expect to hold in the real world, such as the fact that when a person is
    engaged in an activity, they tend to continue for a particular amount of
    time (inertia). Also, certain activity tend to follow certain other
    activities, and some transitions between activities are very unlikely.

    If the person is actually engaged in activity i at timestep s, then we
    assume the likehoold the person will switch to activity j at timestep s+1
    is only a function of which activity they were engaged in at timestep n
    (i.e., the Markov property). This is encoded by an NxN transition
    probability matrix where element (i, j) is the probability that the
    person will transition to activity j at timestep s+1 given that they were
    doing activity i in timestep s. The current implementation identifies
    forbidden transitions and sets their probably to zero, and all remaining
    allowable transitions are assumed equally probable. With this, element
    (i, i) is the probably of remaining in the same activity between timesteps.

    To define what the diagonal values for the transition probably matrix
    should be, we consider an estimate for the median duration that one would
    continue doing one activity. If we devide this median duration by the
    timestep dt, we get the median duration in number of timesteps. We know the
    probability of staying for at least the median duration is 0.5, which is
    should be equal to t[i, i]^N because you stayed in the state for N steps.
    Therefore, we set the transition probability to the Nth root of the median
    duration in units of timesteps.

    Further, we model the signal we get to indicate what the underlying
    activities were can be modeled as a Gaussian mixture with matrices
    'class_mean_conf' and 'class_std_conf'. That is, if the person is actually
    engaged in activity i, then the signal received (e.g., from a raw activity
    classifier) will have mean value class_mean_conf[i] with standard deviation
    class_std_conf[i].

    All of this motivates our use a Gaussian mixture hidden Markov model
    (G-HMM) to  take raw measurements (e.g., activitiy classifications) and to
    filter to the most-likely sequence for the true activities engaged in as a
    function of time.

    One important note is that setting the median duration, which implicitly
    sets the diagonal values of the transition probably matrix, does not
    gaurentee anything about the effect on the duration recovered by the G-HMM.
    There is nothing explicit in the modeling that considers "you have been in
    this state too long, I will push you along to another state." If there is a
    strong signal in the raw activity confidence that indicates the person is
    very likely still in the state, the median duration will not force them out
    after they have been there "too long." Likewise, if the raw confidences
    strongly indicate that the person transitions to state i and in the
    immediate next timestep transitions to step j, the ith activity's median
    duration setting won't preven this. Median duration only acts to tweak the
    amount of raw confidence required to activate a transition (in a maximum-
    likelihood sense). Another framework would be required to encode, for
    example, that it would be extremely unlikely to remain in a specific
    activity for less than tl seconds or more than th seconds.

    """

    def __init__(
        self,
        dt,
        class_str,
        med_class_duration,
        num_steps_can_jump_fwd=0,
        num_steps_can_jump_bck=0,
        class_mean_conf=None,
        class_std_conf=None,
    ):
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
            actually active. If class_std_conf has shape (n_features,), then
            it represent the standard deviation of each feature assumed to
            apply equally to all components of the mixture model (steps). If
            class_std_conf has shape (n_steps, n_features), then
            class_std_conf[i] represents the standard deviations of the
            n_features when the ith  step is active. If class_std_conf has
            shape (n_steps, n_features, n_features), then class_std_conf[i]
            represents the covariance matrix between the n_features when the
            ith step is active. Note 3 dimensional, class_std_conf represents
            covariance versus the other shapes encoding standard deviation,
            which internally will get squared to become diagonal covariances.
        """
        self.cov_eps = 1e-9
        med_class_duration = np.array(med_class_duration)
        assert class_str[0].lower() == "background"
        assert num_steps_can_jump_fwd >= 0
        assert num_steps_can_jump_bck >= 0

        self.class_str = class_str
        self.dt = dt
        self.num_steps_can_jump_fwd = num_steps_can_jump_fwd
        self.num_steps_can_jump_bck = num_steps_can_jump_bck

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
        class_str_.append("background")
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

            if i < len(class_str[1:]):
                bckg_mask.append(True)
                class_str_.append("background%i" % k)
                inv_map.append(0)
                k += 1

        self.num_steps = len(class_str)

        class_str_map = np.array(class_str_map, dtype=int)
        self.class_str_map = class_str_map
        bckg_mask = np.array(bckg_mask, dtype=bool)
        self.bckg_mask = bckg_mask
        self.class_str_ = class_str_
        self.inv_map = inv_map
        self.fwd_map = fwd_map

        N = len(class_str)
        N_ = len(class_str_)
        model = GaussianHMM(n_components=N_, covariance_type="spherical")
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

        # Mask capturing the steps of Z_ that correspond to the visible steps
        # of Z.
        orig_mask = not_bckg_mask.copy()
        orig_mask[0] = True

        if class_mean_conf is not None:
            class_mean_conf = np.array(class_mean_conf)

            # We define an N x N mean matrix where element (i, j) is the mean
            # value emmitted for class j when class i is the state.
            if np.any(class_mean_conf > 1):
                raise ValueError("'class_mean_conf' must be between 0-1")

            if np.ndim(class_mean_conf) == 1:
                num_activities = len(class_mean_conf)
                class_mean_conf = np.diag(class_mean_conf)
            elif np.ndim(class_mean_conf) == 2:
                num_activities = class_mean_conf.shape[1]
            else:
                raise ValueError("np.ndim(class_mean_conf) must be 1 or 2")

            self.num_activities = num_activities

            class_conf_mean_mat = np.zeros((N_, num_activities))
            ki = 0
            for i in range(N_):
                if orig_mask[i]:
                    class_conf_mean_mat[i] = class_mean_conf[ki]
                    ki += 1
                else:
                    class_conf_mean_mat[i] = class_conf_mean_mat[0]

            model.means_ = class_conf_mean_mat

        if class_std_conf is not None:
            class_std_conf = np.array(class_std_conf)
            if np.ndim(class_std_conf) == 3:
                # Full covariance
                model.covariance_type = "full"
                conf_cov_mat = np.zeros((N_, num_activities, num_activities))

                ki = 0
                for i in range(N_):
                    if orig_mask[i]:
                        conf_cov_mat[i] = class_std_conf[ki]
                        ki += 1
                    else:
                        conf_cov_mat[i] = conf_cov_mat[0]

                model.covars_ = conf_cov_mat + self.cov_eps
            else:
                # Square to turn from std to cov.
                class_std_conf2 = class_std_conf**2

                if np.ndim(class_std_conf) == 1:
                    class_std_conf2 = np.tile(class_std_conf2, (N, 1))

                model.covariance_type = "diag"
                conf_cov_mat = np.zeros((N_, num_activities))

                ki = 0
                for i in range(N_):
                    if orig_mask[i]:
                        conf_cov_mat[i] = class_std_conf2[ki]
                        ki += 1
                    else:
                        conf_cov_mat[i] = conf_cov_mat[0]

                model.covars_ = conf_cov_mat + self.cov_eps

        # -------------- Define transition probabilities -------------------------
        # Median number of timesteps spent in each step.
        n = med_class_duration / dt

        # If we are in an activity (class) with probability of transitioning out
        # tii, then the probability that we are still in this class after n
        # timesteps is tii^n.
        tdiag = (0.5) ** (1 / n)

        # Only valid states are possible.
        valid = np.zeros((N_, N_), dtype=bool)
        ki = 0
        for i in range(N_):
            # We are currently in state i, which other states are valid
            # transitions.
            i0 = (i // 2) * 2
            if i0 == i:
                # i is even, so this is a background step. Background steps can
                # only ever move one forward. We don't want to allow hoping
                # from background to background.
                i1 = (i // 2) * 2 + 2
            else:
                i1 = (i // 2) * 2 + 4
                if num_steps_can_jump_fwd > 0:
                    i1 += num_steps_can_jump_fwd * 2

                if num_steps_can_jump_bck > 0:
                    i0 -= 2 * num_steps_can_jump_bck

            for ii in range(i0, i1):
                if ii < 0 or ii >= N_:
                    continue

                valid[i, ii] = True

        self.valid_trans = valid

        # print(valid)

        model.transmat_ = np.zeros((N_, N_))
        ki = 0
        for i in range(N_):
            if ~bckg_mask[i]:
                # specify which indices are valid
                model.transmat_[i, valid[i]] = (1 - tdiag[ki]) / (sum(valid[i]) - 1)
                model.transmat_[i, i] = tdiag[ki]
                # np.sum(model.transmat_[i])

                ki += 1
            else:
                model.transmat_[i, valid[i]] = 1 / (sum(valid[i]))

    def get_hmm_mean_and_std(self):
        """Return the mean and covariance of the model."""
        mean = self.model.means_.copy()
        cov = self.model._covars_.copy() - self.cov_eps
        mean = mean[self.fwd_map]
        std = np.sqrt(cov[self.fwd_map])
        return mean, std

    def sample_for_step(self, step_ind, N):
        """Return simulated classifier associated with the specified step.

        Parameters
        ----------
        step_ind : int
            Recipe step index. Zero encodes background.
        N : int
            Number of samples (states) to simulate.

        Return
        ------
        X : (n_samples, num_classes)
            Simulated detector confidences for each timestep (row) and each
            possible step being detected (column).
        Z : (n_samples,)calc_log_prob_hmm(model, X, Z_, verbose=False)
            Truth state associated with each time step.
        """
        mean = self.model.means_[self.fwd_map[step_ind]]
        std = np.sqrt(self.model._covars_[self.fwd_map[step_ind]])
        n1 = scipy.stats.norm(loc=mean, scale=std)
        random_state = np.uint32(time.time() * 100000)
        return np.array([n1.rvs(random_state=random_state) for _ in range(N)])

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
        Z : (n_samples,)calc_log_prob_hmm(model, X, Z_, verbose=False)
            Truth state associated with each time step.
        """
        X, Z_ = self.model.sample(N)
        X = np.abs(X)

        Z = np.array([self.inv_map[i] for i in Z_])

        if False:
            print(min([X[i, Z[i]] for i in range(len(Z))]))

        # X = (X.T/np.sum(X, axis=1)).T

        times = np.linspace(0, self.dt * (N - 1), N)

        # TODO stop return X, it is now redundant.
        return times, X, Z, X, Z_

    def get_model_force_skip_step(self, force_skip_step):
        """Return model that requires we skip a specific step.

        force_skip_step : None | int
            Enforce that the solution skips a particular step.
        """
        if force_skip_step == 0 or force_skip_step >= len(self.class_str):
            raise ValueError(
                "'force_skip_step' must be an integer "
                "between 1 and %i" % (len(self.class_str) - 1)
            )

        # Make a copy of the model so we can adjust it.
        model = GaussianHMM(
            n_components=self.model.n_components,
            covariance_type=self.model.covariance_type,
        )
        model.n_features = self.model.n_features
        model.startprob_ = self.model.startprob_.copy()
        model.means_ = self.model.means_.copy()
        model._covars_ = self.model._covars_.copy()
        model.transmat_ = self.model.transmat_.copy()

        ind = self.inv_map.index(force_skip_step)

        model.transmat_[ind, ind] = 0
        model.transmat_[ind] /= sum(model.transmat_[ind])

        if ind == 1:
            model.startprob_[:] = 0
            model.startprob_[3] = 0.5
            model.startprob_[4] = 0.5

        valid = self.valid_trans.copy()
        tdiag = np.diag(self.model.transmat_)
        bckg_mask = self.bckg_mask
        N_ = len(self.class_str_)

        orig_mask = ~self.bckg_mask.copy()
        orig_mask[0] = True
        tdiag = np.diag(self.model.transmat_[orig_mask][:, orig_mask])

        # Only valid states are possible.
        valid = np.zeros_like(self.valid_trans)
        ki = 0
        for i in range(len(valid)):
            # We are currently in state i, which other states are valid
            # transitions.
            i0 = (i // 2) * 2
            i1 = (i // 2) * 2 + 2

            if i0 == i:
                # i is even, so this is a background step. Background steps can
                # only ever move one forward. We don't want to allow hoping
                # from background to background.
                pass
            else:
                if self.num_steps_can_jump_fwd > 0:
                    i1 += self.num_steps_can_jump_fwd * 2

                if self.num_steps_can_jump_bck > 0:
                    i0 -= 2 * self.num_steps_can_jump_bck

            for ii in range(i0, i1):
                if ii < 0 or ii >= len(valid):
                    continue

                valid[i, ii] = True

        valid[:, ind] = False
        valid[ind, ind] = True

        # print(valid)

        model.transmat_ = np.zeros_like(self.model.transmat_)
        ki = 0
        for i in range(N_):
            if ~bckg_mask[i]:
                # specify which indices are valid
                model.transmat_[i, valid[i]] = (1 - tdiag[ki]) / (sum(valid[i]) - 1)
                model.transmat_[i, i] = tdiag[ki]
                # np.sum(model.transmat_[i])

                ki += 1
            else:
                model.transmat_[i, valid[i]] = 1 / (sum(valid[i]))

        return model

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
        if force_skip_step is not None:
            model = self.get_model_force_skip_step(force_skip_step)
        else:
            model = self.model

        log_prob, Z_ = model.decode(X)

        Z = np.array([self.inv_map[i] for i in Z_])

        # TODO stop return X, it is now redundant.
        return log_prob, Z, X, Z_

    def calc_log_prob_(self, X, Z_, verbose=False):
        """Calculate the log likelihood of a particular solution to an HMM.

        Parameters
        ----------
        X : Numpy array (n_samples, num_classes)
            Simulated detector confidences for each timestep (row) and each
            possible step being detected (column).
        Z : Numpy array (n_samples,)
            State associated with each time step..
        verbose : bool, optional
            Verbose logging. The default is False.

        Returns
        -------
        log_prob : float
            Log likelihood.

        """
        return calc_log_prob_hmm(self.model, X, Z_, verbose)

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

    def save_sequence_to_disk(self, times, X, Z, det_json_fname, gt_feather_fname):
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
        time_bins = (times[1:] + times[:-1]) / 2
        time_bins = np.hstack(
            [
                (times[0] - times[1]) / 2,
                time_bins,
                times[-1] + (times[-1] - times[-2]) / 2,
            ]
        )
        time_bins -= time_bins[0]

        # Save ground truth.
        img_fnames = []
        for i, t in enumerate(time_bins):
            s = int(np.floor(t))
            micro_ = (t - s) * 1e6
            micro = int(np.floor(micro_))
            img_fnames.append("frame_%i_%i_%i.png" % (i + 1, s, micro))

        data = []
        istart = 0
        iend = 0
        stop_i = len(Z) - 1
        while istart <= stop_i:
            if iend == stop_i or Z[iend + 1] != Z[istart]:
                data.append(
                    [
                        self.class_str[Z[istart]],
                        img_fnames[istart],
                        img_fnames[iend + 1],
                        "simulated",
                    ]
                )
                # print(istart, iend, Z[istart], Z[iend], Z[iend+1], iend - istart)
                iend = iend + 1
                istart = iend
            else:
                iend = iend + 1

        gt = pd.DataFrame(
            data, columns=["class", "start_frame", "end_frame", "exploded_ros_bag_path"]
        )
        gt.to_feather(gt_feather_fname)

        # Save detections
        detections = []
        for i in range(len(X)):
            det = {}
            t = time_bins[i]
            s = int(np.floor(t))
            nano = int((t - s) * 1e9)
            det["header"] = {
                "time_sec": s,
                "time_nanosec": nano,
                "frame_id": "Activity detection",
            }
            det["source_stamp_start_frame"] = time_bins[i]
            det["source_stamp_end_frame"] = time_bins[i + 1]
            det["label_vec"] = self.class_str
            det["conf_vec"] = X[i].tolist()
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

    log_prob1, Z1, _, Z1_ = model.decode(X)
    log_prob1 = model_skip.calc_log_prob_(X, Z1_)

    log_prob2, Z2, _, Z2_ = model_skip.decode(X)

    # log_prob2 = model_skip.calc_log_prob_(X, Z2_)

    if len(model_skip.did_skip_step(Z2)[0]) == 0:
        # The maximum likelihood solution doesn't skip a step, so we need to
        # explicitly check various assumptions of forced skipped steps.

        skipped_step_check = range(1, len(model.class_str))
        log_prob2 = []
        # skipped = []
        for j in skipped_step_check:
            log_prob2_, Z2_ = model.decode(X, force_skip_step=j)[:2]
            log_prob2.append(log_prob2_)
            # skipped.append(model.did_skip_step(Z2_)[0])

        ind = np.argmax(log_prob2)
        log_prob2 = log_prob2[ind]

    # score = (log_prob2 - log_prob1)/log_prob2

    return log_prob2, log_prob1


def score_raw_detections(X, Z, plot_results=False):
    mask = np.zeros_like(X, dtype=bool)
    np.put_along_axis(mask, np.atleast_2d(Z).T, True, axis=1)

    tp = X[mask]
    fp = X[~mask]

    s = np.hstack([tp, fp]).T
    y_tue = np.hstack([np.ones(len(tp), dtype=bool), np.zeros(len(fp), dtype=bool)]).T
    s.shape = (-1, 1)
    y_tue.shape = (-1, 1)

    precision, recall, thresholds = precision_recall_curve(y_tue, s)

    auc = -np.trapz(precision, recall)

    if plot_results:
        fig = plt.figure(num=None, figsize=(14, 10), dpi=80)
        plt.rc("font", **{"size": 28})
        plt.rc("axes", linewidth=4)
        plt.plot(recall, precision, linewidth=6, label="Raw Detections")
        plt.xlabel("Recall", fontsize=40)
        plt.ylabel("Precision", fontsize=40)
        fig.tight_layout()

        plt.xlim([0, 1.01])
        plt.ylim([0, 1.01])
        fig.tight_layout()

        plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.tick_params(axis="y", which="major", grid_color="lightgrey")
        plt.tick_params(
            axis="y", which="minor", grid_linestyle="--", grid_color="lightgrey"
        )
        plt.grid(axis="y", which="both")
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
        plt.tick_params(axis="x", which="major", grid_color="lightgrey")
        plt.tick_params(
            axis="x", which="minor", grid_linestyle="--", grid_color="lightgrey"
        )
        plt.grid(axis="x", which="both")
        plt.legend(fontsize=20, loc=0)

    return auc


def calc_log_prob_hmm(model, X, Z_, verbose=False):
    """Calculate the log likelihood of a particular solution to an HMM.

    Parameters
    ----------
    model : hmmlearn.hmm.GaussianHMM
        HMM model.
    X : Numpy array (n_samples, num_classes)
        Simulated detector confidences for each timestep (row) and each
        possible step being detected (column).
    Z_ : Numpy array (n_samples,)
        State associated with each time step..
    verbose : bool, optional
        Verbose logging. The default is False.

    Returns
    -------
    log_prob : float
        Log likelihood.

    """
    # log_prob0, Z_ = self.model.decode(X)

    # We can't allow multiple possible backgrounds to have a high score.

    log_prob_ = model._compute_log_likelihood(X)

    # We can't allow multiple possible backgrounds to have a high score.

    log_prob = log_prob_[0, Z_[0]] + np.log(model.startprob_[Z_[0]])

    if verbose:
        print("Log(prob) of first state", log_prob)

    for i in range(1, len(Z_)):
        log_prob += log_prob_[i, Z_[i]]

        if verbose:
            print(
                "Log(prob) step %i being in state %i with detector"
                "confidence %0.6f:" % (i, Z_[i], X[i, Z_[i]]),
                log_prob_[i, Z_[i]],
            )

        if model.transmat_[Z_[i - 1], Z_[i]] == 0:
            raise AssertionError(f"Cannot move from {Z_[i - 1]} to {Z_[i]}")

        # We moved from Z_[i - 1] to Z_[i].
        log_prob += np.log(model.transmat_[Z_[i - 1], Z_[i]])

        if verbose:
            print(
                "Log(prob) for the transition from step %i being in "
                "state %i to step %i being in state %i:" % (i - 1, Z_[i - 1], i, Z_[i]),
                np.log(model.transmat_[Z_[i - 1], Z_[i]]),
            )

    return log_prob


def load_and_discretize_data(
    activity_gt: str,
    extracted_activity_detections: str,
    time_window: float,
    uncertain_pad: float,
):
    """Loads unstructured detection and ground truth data and discretize.

    Parameters
    ----------
    activity_gt : str
        Path to activity ground truth feather or Dive csv file.
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
    ext = os.path.splitext(activity_gt)[1]

    if ext == ".csv":
        gt0 = activities_from_dive_csv(activity_gt)

        gt = []
        for i, row in enumerate(gt0):
            g = {
                "class": row.class_label.lower().strip(),
                "start": row.start,
                "end": row.end,
            }
            gt.append(g)
    elif ext == ".feather":
        gt_f = pd.read_feather(activity_gt)
        # Keys: class, start_frame,  end_frame, exploded_ros_bag_path

        gt = []
        for i, row in gt_f.iterrows():
            g = {
                "class": row["class"].lower().strip(),
                "start": time_from_name(row["start_frame"]),
                "end": time_from_name(row["end_frame"]),
            }
            gt.append(g)
    else:
        raise Exception(f"Unhandled file extension {ext}")

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
                "class": l.lower().strip(),
                "start": dets["source_stamp_start_frame"],
                "end": dets["source_stamp_end_frame"],
                "conf": good_dets[l],
                "detect_intersection": np.nan,
            }
            detections.append(d)
    detections = pd.DataFrame(detections)
    print(f"Loaded detections from {extracted_activity_detections}")

    # ============================
    # Load labels
    # ============================
    labels0 = [l.lower().strip().rstrip(".") for l in detections["class"]]
    labels = []
    labels_ = set()
    for label in labels0:
        if label not in labels_:
            labels_.add(label)
            labels.append(label)

    # ============================
    # Split by time window
    # ============================
    # Get time ranges
    min_start_time = min(gt["start"].min(), detections["start"].min())
    max_end_time = max(gt["end"].max(), detections["end"].max())
    dt = time_window
    time_windows = np.arange(min_start_time, max_end_time, time_window)

    if time_windows[-1] < max_end_time:
        time_windows = np.append(time_windows, time_windows[-1] + time_window)
    time_windows = list(zip(time_windows[:-1], time_windows[1:]))
    time_windows = np.array(time_windows)
    dets_per_valid_time_w = np.zeros((len(time_windows), len(labels)), dtype=float)

    def get_time_wind_range(start, end):
        """Return slice indices of time windows that reside completely in
        start->end.
        time_windows[ind1:ind2] all live inside start->end.
        """
        # The start time of the ith window is min_start_time + dt*i.
        ind1_ = (start - min_start_time) / dt
        ind1 = int(np.ceil(ind1_))
        if ind1_ - ind1 + 1 < 1e-15:
            # We want to avoid the case where ind1_ is (j + eps) and it gets
            # rounded up to j + 1.
            ind1 -= 1

        # The end time of the ith window is min_start_time + dt*(i + 1).
        ind2_ = (end - min_start_time) / dt
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
        ind1, ind2 = get_time_wind_range(detections["start"][i], detections["end"][i])

        valid[ind1:ind2] = True
        correct_label = detections["class"][i].strip().rstrip(".")
        correct_class_idx = labels.index(correct_label)
        dets_per_valid_time_w[ind1:ind2, correct_class_idx] = np.maximum(
            dets_per_valid_time_w[ind1:ind2, correct_class_idx], detections["conf"][i]
        )

    gt_true_mask = np.zeros((len(time_windows), len(labels)), dtype=bool)
    for i in range(len(gt)):
        ind1, ind2 = get_time_wind_range(gt["start"][i], gt["end"][i])
        correct_label = gt["class"][i].strip().rstrip(".")
        correct_class_idx = labels.index(correct_label)
        gt_true_mask[ind1:ind2, correct_class_idx] = True

    if not np.all(np.sum(gt_true_mask, axis=1) <= 1):
        raise ValueError("Conflicting ground truth for same time windows")

    # If ground truth isn't specified for a particular window, we should assume
    # 'background'.
    bckg_class_idx = labels.index("background")
    ind = np.where(np.all(gt_true_mask == False, axis=1))[0]
    gt_true_mask[ind, bckg_class_idx] = True

    # Any time the ground truth class changes, we want to add in uncertainty
    # padding, but there should always be at least one time window at the
    # center of the ground-truth span.
    gt_label = np.argmax(gt_true_mask, axis=1)
    pad = int(np.round(uncertain_pad / dt))
    if pad > 0:
        ind = np.where(np.diff(gt_label, axis=0) != 0)[0] + 1
        if ind[0] != 0:
            ind = np.hstack([1, ind])

        if ind[-1] != len(time_windows):
            ind = np.hstack([ind, len(time_windows)])

        for i in range(len(ind) - 1):
            ind1 = ind[i]
            ind2 = ind[i + 1]
            # time windows in range ind1:ind2 all have the same ground
            # truth class.

            ind1_ = ind1 + pad
            ind2_ = ind2 - pad
            indc = int(np.round((ind1 + ind2) / 2))
            ind1_ = min([ind1_, indc])
            ind2_ = max([ind2_, indc + 1])
            valid[ind1:ind1_] = False
            valid[ind2_:ind2] = False

    return time_windows, labels, dets_per_valid_time_w, gt_label, valid
