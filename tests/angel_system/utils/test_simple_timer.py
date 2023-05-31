import unittest.mock as mock

from angel_system.utils.simple_timer import SimpleTimer


def test_simple_timer_basic() -> None:
    """
    Test that constructing and using with just a basic message does not fail.
    """
    t = SimpleTimer("some message")
    with t:
        pass


def test_simple_timer_log_fn() -> None:
    """
    Test that the logging function given is called twice when context invoking
    the simple timer.
    """
    log_fn = mock.MagicMock()
    t = SimpleTimer("test message", log_func=log_fn)
    with t:
        pass
    assert log_fn.call_count == 2
