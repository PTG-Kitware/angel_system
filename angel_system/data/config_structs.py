"""
Structures related to configuration files.
"""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import cast
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import yaml


@dataclass
class ObjectLabel:
    """
    A single object label configuration with an ID and the label.
    """

    id: int
    label: str


@dataclass
class ObjectLabelSet:
    """
    A defined set of object labels.
    """

    version: float
    title: str
    labels: Tuple[ObjectLabel]

    def __post_init__(self):
        # coerce nested label objects into the ObjectLabel type.
        if self.labels and not isinstance(self.labels[0], ObjectLabel):
            raw_labels = cast(Sequence[Dict], self.labels)
            self.labels = tuple(ObjectLabel(**rl) for rl in raw_labels)


def load_object_label_set(filepath: PathLike) -> ObjectLabelSet:
    """
    Load from YAML file an object label set configuration.

    :param filepath: Filepath to load from.

    :return: Structure containing the loaded configuration.
    """
    with open(filepath) as infile:
        data = yaml.safe_load(infile)
    return ObjectLabelSet(**data)


@dataclass
class ActivityLabel:
    """
    One activity classification ID and paired label information.
    """

    # Identifier integer for this activity label
    id: int
    # Concise string label for this activity. Should not contain any spaces.
    label: str
    # Full sentence description of this activity.
    full_str: str
    # Optional integer representing how many times an activity should be
    # repeated to be considered "full"
    # TODO: This parameter has ambiguous and violated meaning (not used as
    #       intended if at all).
    repeat: Optional[int] = None


@dataclass
class ActivityLabelSet:
    version: str
    title: str
    labels: Tuple[ActivityLabel]

    def __post_init__(self):
        # coerce nested label objects into the ObjectLabel type.
        if self.labels and not isinstance(self.labels[0], ActivityLabel):
            raw_labels = cast(Sequence[Dict], self.labels)
            self.labels = tuple(ActivityLabel(**rl) for rl in raw_labels)


def load_activity_label_set(filepath: PathLike) -> ActivityLabelSet:
    """
    Load from YAML file an activity label set configuration.

    :param filepath: Filepath to load from.

    :return: Structure containing the loaded configuration.
    """
    with open(filepath) as infile:
        data = yaml.safe_load(infile)
    return ActivityLabelSet(**data)


@dataclass
class TaskStep:
    """
    A single task step with activity components.
    """

    id: int
    label: str
    full_str: str
    activity_ids: Tuple[int]


@dataclass
class LinearTask:
    """
    A linear task with steps composed of activities.
    """

    version: str
    title: str
    labels: Tuple[TaskStep]

    def __post_init__(self):
        # Coerce pathlike input (str) into a Path instance if not already.
        if self.labels and not isinstance(self.labels[0], TaskStep):
            raw = cast(Sequence[Dict], self.labels)
            self.labels = tuple(TaskStep(**r) for r in raw)


def load_linear_task_config(filepath: PathLike) -> LinearTask:
    """
    Load from YAML file a linear task configuration.

    :param filepath: Filepath to load from.

    :return: Structure containing the loaded configuration.
    """
    with open(filepath) as infile:
        data = yaml.safe_load(infile)
    return LinearTask(**data)


@dataclass
class OneTaskConfig:
    """
    Specification of where one task configuration is located.
    """

    id: int
    label: str
    config_file: Path
    active: bool

    def __post_init__(self):
        # Coerce pathlike input (str) into a Path instance if not already.
        # Interpret relative paths now to absolute based on current working
        # directory.
        if not isinstance(self.config_file, Path):
            self.config_file = Path(self.config_file).absolute()


@dataclass
class MultiTaskConfig:
    """
    A collection of linear task configurations.
    """

    version: str
    title: str
    tasks: Tuple[OneTaskConfig]

    def __post_init__(self):
        # coerce nested task objects into OneTaskConfig types
        if self.tasks and not isinstance(self.tasks[0], OneTaskConfig):
            raw = cast(Sequence[Dict], self.tasks)
            self.tasks = tuple(OneTaskConfig(**r) for r in raw)


def load_multi_task_config(filepath: PathLike):
    """
    Relative file paths are currently interpreted relative to the current
    working directory and resolved to be absolute.

    :param filepath: Filepath to load from.

    :return: Structure containing the loaded configuration.
    """
    with open(filepath) as infile:
        data = yaml.safe_load(infile)
    return MultiTaskConfig(**data)


def load_active_task_configs(cfg: MultiTaskConfig) -> Dict[str, LinearTask]:
    """
    Load task configurations that are enabled in the multitask configuration.

    :param cfg: Multitask configuration to base loading on.

    :raises FileNotFoundError: Configured task configuration file did not refer
        to an open-able file.

    :return: Mapping of task label from the input configuration to the
        LinearTask instance loaded.
    """
    return {
        ct.label: load_linear_task_config(ct.config_file)
        for ct in cfg.tasks
        if ct.active
    }
