import logging
import heapq
from scipy.spatial import distance
import threading
from typing import *

from angel_system.data.common.bounding_boxes import BoundingBoxes

LOG = logging.getLogger(__name__)


class Centroid2DStrategyQueue:
    """
    Little class to handle priority queueing of detected object bounding boxes
    based on their centroid (center coordinate of the bounding box).
    Items are stored in a priority queue based on a timestamp integer.
    When items are popped from the queue, the `last_n` items *before* a provided
    timestamp are returned.


    Typical Example Usage:
    q = Centroid2DStrategyQueue(n=1, k=2)
    q.add(timestamp=1, BoundingBoxes(..., [('obj1', 'obj2', 'obj3')]))
    q.add(timestamp=2, BoundingBoxes(..., [('obj1', 'obj2', 'obj3')]))
    q.get_n_before(2)
    """

    def __init__(
        self,
        n: int,
        center_x: int,
        center_y: int,
        k: int = 1,
        log_func: Optional[Callable[..., None]] = None,
    ):
        """
        Additional arguments are passed to the logging method
        :param n: Whenever objects are retrieved, return the last n entries.
        :param k: Acquires the top k objects that are the most centered given their centroid.
        :param log_func: Optional callable to be invoked to receive the
            message. If this is `None`, the local Logger instance to this
            module is used.
        """
        self._log_func = log_func

        self.n = n
        self.k = k

        # This is the main priority queue. Each item should be a Tuple[int, Any] in which
        # the elements correspond to (Integer Timestamp, Any Object). An example of the queued
        # object's second element could be a Tuple of the top K detected objects.
        self.pq = []
        self.center_x = center_x
        self.center_y = center_y
        self.lock = threading.Lock()

    def get_queue(self):
        return self.pq

    def add(self, timestamp: int, bounding_boxed_item: BoundingBoxes):
        self.lock.acquire()
        k_most_centered_objects = self._get_k_most_center_objects(bounding_boxed_item)
        heapq.heappush(self.pq, (timestamp, k_most_centered_objects))
        self.lock.release()

    def get_n_before(self, timestamp: int) -> List[Any]:
        """
        Gets the self.n items before the provided timestamp.
        """
        items = []
        self.lock.acquire()
        while self.pq:
            next_timestamp, _ = self.pq[0]
            if next_timestamp < timestamp:
                items.append(heapq.heappop(self.pq))
            else:
                break
        self.lock.release()
        if self._log_func:
            self._log_func(
                f"Read up to {self.n} items from queue"
                + "; ".join([f"{item} @ Time={time}" for time, item in items])
            )
        return items[-self.n :] if items else items

    def _get_k_most_center_objects(self, bb: BoundingBoxes) -> List[Any]:
        """
        Acquires the top k objects with respect to centroid distance from the center pixel.
        Returns a list of Tuples of (centroid distance, top k most centered objects)
        """
        k_most_centered_objects = []

        # Sort the bounding boxes in order of distance from centroid to center pixel.
        zipped = zip(bb.item, bb.left, bb.right, bb.top, bb.bottom)
        for item, left, right, top, bottom in zipped:
            centroid_x, centroid_y = self._get_centroid(left, right, top, bottom)
            dist = distance.euclidean(
                [centroid_x, centroid_y], [self.center_x, self.center_y]
            )
            heapq.heappush(k_most_centered_objects, (dist, item))

        # Return the top k centered objects based on centroid distance.
        result = []
        for _ in range(self.k):
            if not k_most_centered_objects:
                break
            result.append(heapq.heappop(k_most_centered_objects))
        return result

    def _get_centroid(
        self, left: int, right: int, top: int, bottom: int
    ) -> Tuple[int, int]:
        """
        Calculates the center 2D pixel of a 2D bounding box.
        """
        width_center = left + int((right - left) / 2)
        height_center = top + int((bottom - top) / 2)
        return [width_center, height_center]
