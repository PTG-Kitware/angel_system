from functools import lru_cache
from typing import Dict

from angel_msgs.msg import SystemCommands


__all__ = [
    "task_int_to_str"
]


@lru_cache()
def _get_sys_cmds_task_map() -> Dict[int, str]:
    """
    Generate the TASK enumeration value to string mapping based on the contents
    of the SystemCommands structure at the time of the call.

    This should allow this functionality to be dynamic to any changes in the
    SystemCommands message.

    :returns: Dictionary mapping integer SystemCommands.TASK_* values into
        their enumeration names.
    """
    return {v: k for k, v in SystemCommands.__dict__.items() if k.startswith("TASK_")}


def task_int_to_str(i: int) -> str:
    """
    Convert the integer value intended to represent one of the "TASK_*"
    enumeration values in the `angel_msgs.msg.SystemCommands` message type into
    the enumeration property string name.

    :param i: Task enumeration value.
    :return: String name of the enumeration value.
    """
    return _get_sys_cmds_task_map()[i]
