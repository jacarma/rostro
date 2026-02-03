"""Tests for audio module."""

import numpy as np

from rostro.audio.capture import AudioCapture
from rostro.audio.config import AudioConfig


class TestAudioConfig:
    """Tests for AudioConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AudioConfig()
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_duration_ms == 100

    def test_chunk_size_calculation(self):
        """Test chunk size calculation."""
        config = AudioConfig(sample_rate=16000, chunk_duration_ms=100)
        # 16000 samples/sec * 0.1 sec = 1600 samples
        assert config.chunk_size == 1600

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "sample_rate": 44100,
            "channels": 2,
            "chunk_duration_ms": 50,
        }
        config = AudioConfig.from_dict(data)
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.chunk_duration_ms == 50


class TestAudioCapture:
    """Tests for AudioCapture."""

    def test_compute_rms_silence(self):
        """Test RMS computation for silence."""
        silence = np.zeros(1000, dtype=np.int16)
        rms = AudioCapture.compute_rms(silence)
        assert rms == 0.0

    def test_compute_rms_loud(self):
        """Test RMS computation for loud audio."""
        # Max amplitude
        loud = np.full(1000, 32767, dtype=np.int16)
        rms = AudioCapture.compute_rms(loud)
        assert rms > 0.9  # Should be close to 1.0

    def test_compute_rms_medium(self):
        """Test RMS computation for medium audio."""
        # Half amplitude
        medium = np.full(1000, 16000, dtype=np.int16)
        rms = AudioCapture.compute_rms(medium)
        assert 0.4 < rms < 0.6  # Should be around 0.5

    def test_to_wav_bytes(self):
        """Test converting samples to WAV bytes."""
        audio = np.zeros(1600, dtype=np.int16)
        wav_bytes = AudioCapture.to_wav_bytes(audio, sample_rate=16000)

        # Check WAV header
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"

    def test_is_active_when_not_started(self, audio_config):
        """Test is_active when capture not started."""
        capture = AudioCapture(audio_config)
        assert not capture.is_active
