"""Tests for activation module."""

from unittest.mock import MagicMock, patch

import numpy as np

from rostro.activation.vad import VADActivation, VADConfig, VADState


class TestVADConfig:
    """Tests for VADConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VADConfig()
        assert config.volume_threshold == 0.02
        assert config.silence_duration_ms == 1500
        assert config.min_speech_duration_ms == 300
        assert config.sample_window_ms == 50

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "volume_threshold": 0.05,
            "silence_duration_ms": 2000,
            "min_speech_duration_ms": 500,
        }
        config = VADConfig.from_dict(data)
        assert config.volume_threshold == 0.05
        assert config.silence_duration_ms == 2000
        assert config.min_speech_duration_ms == 500


class TestVADState:
    """Tests for VADState enum."""

    def test_states_exist(self):
        """Test that all states exist."""
        assert VADState.DORMANT
        assert VADState.LISTENING
        assert VADState.PROCESSING


class TestVADActivation:
    """Tests for VADActivation."""

    def test_initial_state(self, vad_config, audio_config):
        """Test initial state is DORMANT."""
        vad = VADActivation(vad_config, audio_config)
        assert vad.state == VADState.DORMANT

    def test_state_property(self, vad_config, audio_config):
        """Test state property access."""
        vad = VADActivation(vad_config, audio_config)
        assert vad.state == VADState.DORMANT

    @patch("rostro.activation.vad.AudioCapture")
    def test_start_begins_monitoring(self, mock_capture, vad_config, audio_config):
        """Test that start begins audio capture."""
        vad = VADActivation(vad_config, audio_config)
        callback = MagicMock()

        vad.start(on_speech_complete=callback)

        # Should create and start audio capture
        mock_capture.return_value.start.assert_called_once()

    @patch("rostro.activation.vad.AudioCapture")
    def test_stop_ends_monitoring(self, mock_capture, vad_config, audio_config):
        """Test that stop ends audio capture."""
        vad = VADActivation(vad_config, audio_config)
        callback = MagicMock()

        vad.start(on_speech_complete=callback)
        vad.stop()

        mock_capture.return_value.stop.assert_called()

    def test_pause_clears_recording(self, vad_config, audio_config):
        """Test that pause clears recorded audio."""
        vad = VADActivation(vad_config, audio_config)
        vad._recorded_audio = [np.zeros(100, dtype=np.int16)]

        vad.pause()

        assert len(vad._recorded_audio) == 0
        assert vad.state == VADState.DORMANT
