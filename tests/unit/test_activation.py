"""Tests for activation module."""

from unittest.mock import MagicMock, patch

import numpy as np

from rostro.activation.vad import VADActivation, VADConfig, VADState


class TestVADConfig:
    """Tests for VADConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VADConfig()
        assert config.confidence_threshold == 0.7
        assert config.start_secs == 0.2
        assert config.min_volume == 0.002
        assert config.silence_duration_ms == 1500
        assert config.min_speech_duration_ms == 300

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "confidence_threshold": 0.5,
            "start_secs": 0.3,
            "min_volume": 0.005,
            "silence_duration_ms": 2000,
            "min_speech_duration_ms": 500,
        }
        config = VADConfig.from_dict(data)
        assert config.confidence_threshold == 0.5
        assert config.start_secs == 0.3
        assert config.min_volume == 0.005
        assert config.silence_duration_ms == 2000
        assert config.min_speech_duration_ms == 500

    def test_from_dict_defaults(self):
        """Test from_dict uses defaults for missing keys."""
        config = VADConfig.from_dict({})
        assert config.confidence_threshold == 0.7
        assert config.start_secs == 0.2
        assert config.min_volume == 0.002


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

    @patch("rostro.activation.vad.SileroVAD")
    @patch("rostro.activation.vad.AudioCapture")
    def test_start_loads_silero_and_begins_monitoring(
        self, mock_capture, mock_silero, vad_config, audio_config
    ):
        """Test that start loads Silero and begins audio capture."""
        vad = VADActivation(vad_config, audio_config)
        callback = MagicMock()

        vad.start(on_speech_complete=callback)

        mock_silero.assert_called_once()
        mock_capture.return_value.start.assert_called_once()

    @patch("rostro.activation.vad.SileroVAD")
    @patch("rostro.activation.vad.AudioCapture")
    def test_stop_ends_monitoring(self, mock_capture, mock_silero, vad_config, audio_config):
        """Test that stop ends audio capture."""
        vad = VADActivation(vad_config, audio_config)
        callback = MagicMock()

        vad.start(on_speech_complete=callback)
        vad.stop()

        mock_capture.return_value.stop.assert_called()

    def test_pause_clears_recording_and_buffers(self, vad_config, audio_config):
        """Test that pause clears recorded audio and Silero buffers."""
        vad = VADActivation(vad_config, audio_config)
        vad._recorded_audio = [np.zeros(100, dtype=np.int16)]
        vad._silero_buffer = np.ones(200, dtype=np.float32)
        vad._voice_frame_count = 5
        mock_silero = MagicMock()
        vad._silero = mock_silero

        vad.pause()

        assert len(vad._recorded_audio) == 0
        assert len(vad._silero_buffer) == 0
        assert vad._voice_frame_count == 0
        assert len(vad._pre_roll) == 0
        assert vad.state == VADState.DORMANT
        mock_silero.reset_states.assert_called_once()

    def test_resume_resets_buffers(self, vad_config, audio_config):
        """Test that resume resets Silero buffers and state."""
        vad = VADActivation(vad_config, audio_config)
        vad._audio_capture = MagicMock()
        vad._on_speech_complete = MagicMock()
        mock_silero = MagicMock()
        vad._silero = mock_silero
        vad._voice_frame_count = 3

        vad.resume()

        assert len(vad._silero_buffer) == 0
        assert vad._voice_frame_count == 0
        assert len(vad._pre_roll) == 0
        mock_silero.reset_states.assert_called_once()

    def test_downsample_to_16k_passthrough(self, vad_config, audio_config):
        """Test downsampling is passthrough when already at 16kHz."""
        audio_config.sample_rate = 16000
        vad = VADActivation(vad_config, audio_config)

        audio_int16 = np.array([0, 1000, -1000, 32767], dtype=np.int16)
        result = vad._downsample_to_16k(audio_int16)

        assert result.dtype == np.float32
        assert len(result) == 4
        np.testing.assert_allclose(result[0], 0.0, atol=1e-5)

    def test_downsample_to_16k_from_24k(self, vad_config, audio_config):
        """Test downsampling from 24kHz to 16kHz."""
        audio_config.sample_rate = 24000
        vad = VADActivation(vad_config, audio_config)

        # 2400 samples at 24kHz (100ms) -> 1600 samples at 16kHz
        audio_int16 = np.zeros(2400, dtype=np.int16)
        result = vad._downsample_to_16k(audio_int16)

        assert result.dtype == np.float32
        assert len(result) == 1600

    def test_process_audio_ignores_during_cooldown(self, vad_config, audio_config):
        """Test that audio is ignored during resume cooldown."""
        import time

        vad = VADActivation(vad_config, audio_config)
        vad._silero = MagicMock()
        vad._resume_time = time.time()  # Just resumed

        state_callback = MagicMock()
        vad._on_state_change = state_callback

        audio = np.ones(1600, dtype=np.int16) * 10000
        vad._process_audio(audio)

        # Should not change state
        state_callback.assert_not_called()

    def test_process_audio_quiet_skips_silero(self, vad_config, audio_config):
        """Test that very quiet audio skips Silero inference."""
        audio_config.sample_rate = 16000
        vad = VADActivation(vad_config, audio_config)
        mock_silero = MagicMock()
        vad._silero = mock_silero
        vad._resume_time = 0.0

        # Very quiet audio (below min_volume)
        audio = np.zeros(1600, dtype=np.int16)
        vad._process_audio(audio)

        # Silero should not be called
        mock_silero.assert_not_called()

    def test_process_audio_runs_silero_on_loud_audio(self, vad_config, audio_config):
        """Test that loud audio triggers Silero inference."""
        audio_config.sample_rate = 16000
        vad = VADActivation(vad_config, audio_config)
        mock_silero = MagicMock(return_value=0.3)  # Below threshold
        vad._silero = mock_silero
        vad._resume_time = 0.0

        # Loud audio
        audio = (np.random.randn(1600) * 10000).astype(np.int16)
        vad._process_audio(audio)

        # Silero should be called (1600 samples -> 3 frames of 512 + 64 leftover)
        assert mock_silero.call_count == 3

    def test_speech_confirmation_requires_consecutive_frames(self, vad_config, audio_config):
        """Test that speech start requires consecutive voice frames."""
        audio_config.sample_rate = 16000
        vad_config.start_secs = 0.2
        vad = VADActivation(vad_config, audio_config)
        mock_silero = MagicMock(return_value=0.9)  # Above threshold
        vad._silero = mock_silero
        vad._resume_time = 0.0
        # Manually set frames_needed (normally set in start())
        # 0.2s / 0.1s per callback = 2 consecutive voice callbacks needed
        vad._frames_needed_for_start = 2

        state_changes: list[VADState] = []
        vad._on_state_change = lambda s: state_changes.append(s)

        # First callback: should stay DORMANT (only 1 voice callback, need 2)
        audio = (np.random.randn(1600) * 10000).astype(np.int16)
        vad._process_audio(audio)
        assert vad.state == VADState.DORMANT

        # Second callback: should transition to LISTENING (2 consecutive)
        vad._process_audio(audio)
        assert vad.state == VADState.LISTENING
        assert VADState.LISTENING in state_changes

    def test_pre_roll_captures_speech_onset(self, vad_config, audio_config):
        """Test that pre-roll buffer captures audio before speech confirmation."""
        audio_config.sample_rate = 16000
        vad = VADActivation(vad_config, audio_config)
        vad._resume_time = 0.0
        vad._frames_needed_for_start = 1  # Confirm immediately

        # First: feed some quiet audio to fill pre-roll
        quiet_audio = np.zeros(1600, dtype=np.int16)
        vad._silero = MagicMock(return_value=0.1)
        vad._process_audio(quiet_audio)
        vad._process_audio(quiet_audio)
        assert len(vad._pre_roll) == 2

        # Now feed loud audio that triggers speech
        vad._silero = MagicMock(return_value=0.9)
        loud_audio = (np.random.randn(1600) * 10000).astype(np.int16)
        vad._process_audio(loud_audio)

        assert vad.state == VADState.LISTENING
        # recorded_audio should include pre-roll chunks + the triggering chunk
        assert len(vad._recorded_audio) == 3

    def test_silence_ends_recording(self, vad_config, audio_config):
        """Test that silence after speech triggers finalization."""
        import time

        audio_config.sample_rate = 16000
        vad_config.silence_duration_ms = 0  # Immediate end-of-speech for testing
        vad_config.min_speech_duration_ms = 0
        vad = VADActivation(vad_config, audio_config)
        vad._resume_time = 0.0

        callback = MagicMock()
        vad._on_speech_complete = callback

        # Put VAD in LISTENING state manually
        vad._state = VADState.LISTENING
        vad._speech_start_time = time.time() - 1.0
        vad._silence_start_time = time.time() - 1.0
        vad._recorded_audio = [(np.ones(1600, dtype=np.int16) * 5000)]

        # Feed quiet audio -> should trigger finalization
        mock_silero = MagicMock(return_value=0.1)
        vad._silero = mock_silero
        quiet = np.zeros(1600, dtype=np.int16)
        vad._process_audio(quiet)

        callback.assert_called_once()
        assert vad.state == VADState.DORMANT
