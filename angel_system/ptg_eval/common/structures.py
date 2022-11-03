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
    # Confidence of a predicted activity. This should be in the [0,1] range.
    # Ground truth traditionally sets this to `1.0`.
    conf: float
