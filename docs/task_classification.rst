Step and Task Classification
============================
The system observes the user as they are trying to complete a **task** (recipe).
The task is comprised of a sequence of **steps**, where the order of the steps matters.
There may be multiple viable sequences of steps to acceptably complete the task, but the point is that there are
unacceptable sequences.
The system collects 30 fps video from the headset as well as the hand poses extracted by the headset.
An object detector analyses the video stream to identify and localize particular objects.
An **activity classifier** considers batches of video frames (e.g., 32) along with hand pose and object detections to
determine the activity that the user is doing within this narrow timeframe.
We define the scope of *activity* in this context to be what can be inferred solely from the timespan that the activity
classifier considers.
For example, the same activity may be required at two different steps during the task, and it is not the purview of the
activity classifier to differentiate between the steps.
Instead, we rely on a `Hidden Markov Model <https://en.wikipedia.org/wiki/Hidden_Markov_model>`_ (HMM) to take the
time series of activity classifications and consider constraints on the order of steps and the median time expected to
be within each step.

Activity Classifier
+++++++++++++++++++

Hidden Markov Model (HMM)
+++++++++++++++++++
The `step HMM <https://github.com/PTG-Kitware/angel_system/blob/master/angel_system/activity_hmm/core.py#L162>`_ is
based on a `Guassian HMM <https://hmmlearn.readthedocs.io/en/stable/api.html#gaussianhmm>`_ to filter activity
classifications to recover the estimated true sequence of steps as a function of time.
The Gaussian HMM has *num_steps × num_activities* **mean** and **covariance** matrices encoding that when actually in
step i, the confidence emitted by the activity classifier for activity ``j`` is given by
``mean[i, j] +/- covariance[i, j]``.
When the mean and covariance matrices are fit to real instances of activity classifier outputs, they can capture the
fact that some activities are more-accurately identified by the classifier than others, and some pairs of activities
are easily confused.

Internally, the HMM adds a background state between each defined task states.
This allows for a user temporarily stopping a step and then returning while respecting step order.
For example, you can move from step ``i`` to its background state ``i-b``, but from there, you can move back to ``i``
or move to ``i+1`` without being considered as skipping a step.

Simulating Data
+++++++++++++++
The simulating scripts are bit unstructured at the moment, but generation code
lives `here <https://github.com/PTG-Kitware/angel_system/blob/master/scripts/hmm_exploration/simulate_data.py#L71>_.
