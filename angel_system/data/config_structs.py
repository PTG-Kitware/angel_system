"""
Structures related to configuration files.
"""

from dataclasses import dataclass, field
from os import PathLike
from typing import cast
from typing import Dict
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
        if self.labels and not isinstance(self.labels, ObjectLabel):
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
