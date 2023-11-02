from typing import *


class BoundingBoxes:
    def __init__(
        self,
        left: List[int],
        right: List[int],
        top: List[int],
        bottom: List[int],
        item: List[Any],
    ):
        """
        Wrapper of bounding boxes and a contained entity corresponding to each bounding box.
        The item is intentionally kept ambiguous to provide flexibility (e.g. can pass in
        an object label that corresponds to each bounding box or a tuple of an object label and
        its confidence score).
        """
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.item = item
