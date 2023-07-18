from dataclasses import dataclass


@dataclass(frozen=True)
class Activity:
    """
    Representation of an activity prediction.

    This structure is expected to be utilized for both ground truth and
    predicted activities.
    """

    # String label of the class the activity is representing.
    class_label: str
    # Start and end time in floating-point seconds.
    start: float
    end: float
    # Start and end frame
    start_frame: int
    end_frame: int
    # Confidence of a predicted activity. This should be in the [0,1] range.
    # Ground truth traditionally sets this to `1.0`.
    conf: float


@dataclass(frozen=True)
class Step:
    """
    Representation of a step prediction.

    This structure is expected to be utilized for both ground truth and
    predicted activities.
    """

    # The index of the step currently in progress.
    # A value of `-1` indicates that no step has been started yet.
    current_step_id: int
    # String of the step currently in progress.
    class_label: str
    # Start time in floating point seconds.
    start: float
    # End time in floating point seconds.
    end: float
    # Confidence of a predicted step. This should be in the [0,1] range.
    # Ground truth traditionally sets this to `1.0`.
    conf: float
    # Indicated if the step is complete
    completed: bool
