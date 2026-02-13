"""Session timer — fires callback after inactivity timeout."""

import threading
from collections.abc import Callable


class SessionTimer:
    """Timer that fires a callback after a period of inactivity."""

    def __init__(self, timeout_seconds: float, on_timeout: Callable[[], None]) -> None:
        self._timeout = timeout_seconds
        self._on_timeout = on_timeout
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._timer is not None and self._timer.is_alive()

    def reset(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._timeout, self._on_timeout)
            self._timer.daemon = True
            self._timer.start()

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
