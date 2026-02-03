"""Shared test fixtures."""

from unittest.mock import patch

import numpy as np
import pytest

from rostro.activation.vad import VADConfig
from rostro.audio.config import AudioConfig
from rostro.avatar.face_pack import FacePack


@pytest.fixture
def audio_config() -> AudioConfig:
    """Create a test audio configuration."""
    return AudioConfig(
        sample_rate=16000,
        channels=1,
        chunk_duration_ms=100,
    )


@pytest.fixture
def vad_config() -> VADConfig:
    """Create a test VAD configuration."""
    return VADConfig(
        volume_threshold=0.02,
        silence_duration_ms=1500,
        min_speech_duration_ms=300,
        sample_window_ms=50,
    )


@pytest.fixture
def face_pack() -> FacePack:
    """Create a test face pack."""
    return FacePack.default()


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Create sample audio data."""
    # Generate 1 second of silence
    return np.zeros(16000, dtype=np.int16)


@pytest.fixture
def loud_audio() -> np.ndarray:
    """Create loud sample audio data."""
    # Generate 1 second of loud noise
    return (np.random.randn(16000) * 10000).astype(np.int16)


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch("openai.OpenAI") as mock:
        yield mock
