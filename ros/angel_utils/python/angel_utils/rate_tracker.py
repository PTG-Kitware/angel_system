import time


class RateTracker:
    """
    Keep track of the rate at which something is cycling.

    This class is currently *not* thread-safe.

    The C++ sibling of this utility is located [here (impl)](ros/angel_utils/src/rate_tracker.cpp)
    and [here (header)](ros/angel_utils/include/angel_utils/rate_tracker.hpp).
    Changes to this API and implementation should be reflected there.

    :param window_size: Number of tick delta's to retain to compute moving
        averages.
    """

    def __init__(self, window_size: int = 10):
        self._window_size = window_size
        self._first_tick = True
        self._last_measure_time = None
        self._time_vec = []
        self._time_vec_i = 0

    def _has_measurements(self) -> bool:
        """ If we have any time measurements to act on or not. """
        return self._time_vec.size() > 0

    def tick(self) -> None:
        """
        Perform a measurement of time since the last tick.
        """
        now = time.monotonic()
        if self._first_tick:
            self._first_tick = False
        else:
            time_since_last_tick = now - self._last_measure_time
            # Insert time measurement appropriately into the window
            if len(self._time_vec) < self._window_size:
                self._time_vec.append(time_since_last_tick)
            else:
                # _time_vec is full, so now we start rotating new measurement
                # insertion. _time_vec_i start's at 0 when we first enter here,
                # which would overwrite the oldest time measurement at the
                # beginning of the vector.
                self._time_vec[self._time_vec_i] = time_since_last_tick
                self._time_vec_i = (self._time_vec_i + 1) % len(self._time_vec)
        self._last_measure_time = now

    def get_delta_avg(self) -> float:
        """
        Get the average time delta between ticks within our window of
        measurement.

        If there have been no measurements taken yet (by calling `tick()`) then
        a `-1` value is returned.

        :return: Average time in seconds between tick measurements.
        """
        avg_time = -1
        if len(self._time_vec) > 0:
            window_total = sum(self._time_vec)
            avg_time = window_total / len(self._time_vec)
        return avg_time

    def get_rate_avg(self) -> float:
        """
        Get the average tick rate from the window of measurements.

        If there have been no measurements taken yet (by calling `tick()`) then
        a `-1` value is returned.

        :return: Average rate in Hz between tick measurements.
        """
        avg_rate = -1
        if len(self._time_vec) > 0:
            avg_rate = 1.0 / self.get_delta_avg()
        return avg_rate
