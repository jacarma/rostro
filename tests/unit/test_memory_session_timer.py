"""Tests for session timer."""

import time
from unittest.mock import MagicMock

from rostro.memory.session_timer import SessionTimer


class TestSessionTimer:
    def test_callback_fires_after_timeout(self) -> None:
        callback = MagicMock()
        timer = SessionTimer(timeout_seconds=0.1, on_timeout=callback)
        timer.reset()
        time.sleep(0.3)
        callback.assert_called_once()
        timer.stop()

    def test_reset_postpones_callback(self) -> None:
        callback = MagicMock()
        timer = SessionTimer(timeout_seconds=0.2, on_timeout=callback)
        timer.reset()
        time.sleep(0.1)
        timer.reset()  # restart the timer
        time.sleep(0.1)
        callback.assert_not_called()  # hasn't fired yet
        time.sleep(0.2)
        callback.assert_called_once()
        timer.stop()

    def test_stop_cancels_timer(self) -> None:
        callback = MagicMock()
        timer = SessionTimer(timeout_seconds=0.1, on_timeout=callback)
        timer.reset()
        timer.stop()
        time.sleep(0.3)
        callback.assert_not_called()

    def test_is_active(self) -> None:
        callback = MagicMock()
        timer = SessionTimer(timeout_seconds=1.0, on_timeout=callback)
        assert not timer.is_active
        timer.reset()
        assert timer.is_active
        timer.stop()
        assert not timer.is_active

    def test_no_callback_without_reset(self) -> None:
        callback = MagicMock()
        timer = SessionTimer(timeout_seconds=0.1, on_timeout=callback)
        time.sleep(0.3)
        callback.assert_not_called()
        timer.stop()
