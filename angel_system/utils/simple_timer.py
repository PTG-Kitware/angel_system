import logging
import time
from typing import Any, Callable, Optional


LOG = logging.getLogger(__name__)


class SimpleTimer:
    """
    Little class to wrap the timing of things. To be use with the ``with``
    statement.
    """

    def __init__(self, msg: str, log_func: Optional[Callable[..., None]] = None):
        """
        Additional arguments are passed to the logging method
        :param msg: Message to be displayed before and after the context block.
        :param log_func: Optional callable to be invoked to receive the
            message. If this is `None`, the local Logger instance to this
            module is used.
        :param args: Additional arguments to be passed as string formatting of
            the given `msg`.
        """
        self._log_func = log_func
        self._msg = msg
        self._s = 0.0

    def __enter__(self) -> None:
        if self._log_func:
            self._log_func(self._msg)
        else:
            LOG.info(self._msg)
        self._s = time.time()

    def __exit__(self, *_: Any) -> None:
        if self._log_func:
            self._log_func(f"{self._msg} -> {time.time() - self._s:.9f} s")
        else:
            LOG.info(f"{self._msg} -> {time.time() - self._s:.9f} s")
