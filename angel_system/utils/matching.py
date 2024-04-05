import itertools
import math
from typing import Callable
from typing import Sequence
from typing import List
from typing import Optional
from typing import Reversible
from typing import TypeVar


T = TypeVar("T")


def descending_match_with_tolerance(
    key_times: Sequence[int],
    values: Reversible[T],
    tol: int,
    time_from_value_fn: Optional[Callable[[T], int]] = None,
) -> List[Optional[T]]:
    """
    Match `values` to `key_times` until there are no more `key_times` to
    left to match to.

    Key and value times provided are assumed to be in strictly ascending
    temporal order.

    Some key times may have no match that is within the tolerance given.
    In such a case, `None` is associated with the key time.

    The matching logic used is effectively greedy, however, the expected
    use-case is that of nanoseconds and with a tolerance that is significantly
    less than the normal distance between key times.

    :param key_times: Integer time values (usually nanoseconds) in ascending
        value order to match against.
    :param values: Reversible values to match against key times.
        These values should either be integer times themselves, be able to be
        associated to an integer via the `time_from_value_fn`. This reversible
        will be iterated over at most once.
    :param tol: Integer time matching tolerance.
    :param time_from_value_fn: Optional callable to provide a translation from
        the value type to the time (nanoseconds) of that object. By default, we
        consider the value *to be* the time.

    :returns: A list that is parallel in association to the `key_times`
        sequence. Slots may be valued with None (no match) or with a value from
        the `values` iterable. Values in this list should also be in relative
        ascending order.
    """
    # Values will be stored in descending temporal order, will reverse before
    # returning.
    match_list: List[Optional[T]] = []

    value_riter = reversed(values)
    carried_value: Optional[T] = None
    for kt in reversed(key_times):
        closest_delta = math.inf
        closest_value = None
        for value in value_riter:
            # in python, if-statements are faster than passthrough functions...
            # print(f"value: {value}")
            if time_from_value_fn is not None:
                vt = time_from_value_fn(value)
            else:
                vt = value
                
            t_delta = abs(kt - vt)
            # print(f"t_delta: {t_delta}")
            # print(f"kt: {kt}")
            # print(f"vt: {vt}")
            if t_delta <= tol and t_delta < closest_delta:
                closest_delta = t_delta
                closest_value = value
            elif vt < kt:
                # Not close enough to match the current frame any more and is
                # now earlier than the current frame, so retain the candidate
                # time for potentially matching against the next frame time.
                carried_value = value
                break  # keep the hand iter state
        if carried_value is not None:
            # Chain the last yielded, but matchable hand back into the
            # hand iterable (still proceeding in reversed time
            # direction).
            value_riter = itertools.chain([carried_value], value_riter)
            carried_value = None  # clear carried value
        # Could be a time, or could be None still indicating no match.
        # print(f"carried_value: {carried_value}")
        # print(f"closest_value: {closest_value}")
        match_list.append(closest_value)

    # hands were added in reverse order (descending time), so flip it back into
    # ascending time order.
    match_list.reverse()
    return match_list
