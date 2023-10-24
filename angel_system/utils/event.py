from threading import Event


class WaitAndClearEvent(Event):
    """
    Simple subclass that adds a wait-and-clear method to simultaneously wait
    for the lock and clear the set flag upon successful waiting.
    """

    def wait_and_clear(self, timeout=None):
        """Block until the internal flag is true, then clear the flag if it was
        set.

        If the internal flag is true on entry, return immediately. Otherwise,
        block until another thread calls set() to set the flag to true, or until
        the optional timeout occurs.

        When the timeout argument is present and not None, it should be a
        floating point number specifying a timeout for the operation in seconds
        (or fractions thereof).

        This method returns the internal flag on exit, so it will always return
        True except if a timeout is given and the operation times out.

        """
        with self._cond:
            signaled = self._flag
            if not signaled:
                signaled = self._cond.wait(timeout)
            self._flag = False
            return signaled
