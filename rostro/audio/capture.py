"""Audio capture from microphone."""

import io
import wave
from collections.abc import Callable
from typing import Any

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

from rostro.audio.config import AudioConfig


class AudioCapture:
    """Handles microphone audio capture."""

    def __init__(self, config: AudioConfig) -> None:
        """Initialize audio capture.

        Args:
            config: Audio configuration.
        """
        self.config = config
        self._stream: sd.InputStream | None = None
        self._callback: Callable[[np.ndarray[Any, np.dtype[np.int16]]], None] | None = None

    def start(self, callback: Callable[[np.ndarray[Any, np.dtype[np.int16]]], None]) -> None:
        """Start capturing audio.

        Args:
            callback: Function to call with each audio chunk (numpy array).
        """
        self._callback = callback
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="int16",
            blocksize=self.config.chunk_size,
            device=self.config.input_device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop capturing audio."""
        if self._stream is not None:
            print("[AudioCapture] Stopping stream...")
            self._stream.stop()
            print("[AudioCapture] Closing stream...")
            self._stream.close()
            print("[AudioCapture] Stream closed")
            self._stream = None
        self._callback = None

    def _audio_callback(
        self,
        indata: np.ndarray[Any, np.dtype[np.int16]],
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Internal callback for sounddevice stream.

        Args:
            indata: Input audio data.
            frames: Number of frames.
            time: Timing information.
            status: Stream status flags.
        """
        if self._callback is not None:
            # Make a copy since indata buffer is reused
            self._callback(indata.copy().flatten())

    @property
    def is_active(self) -> bool:
        """Check if capture is active."""
        return self._stream is not None and self._stream.active

    @staticmethod
    def compute_rms(audio: np.ndarray[Any, np.dtype[np.int16]]) -> float:
        """Compute RMS (root mean square) of audio samples.

        Args:
            audio: Audio samples as int16.

        Returns:
            RMS value normalized to 0.0-1.0 range.
        """
        # Normalize int16 to float
        audio_float = audio.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float**2))
        return float(rms)

    @staticmethod
    def to_wav_bytes(
        audio: np.ndarray[Any, np.dtype[np.int16]],
        sample_rate: int,
        channels: int = 1,
    ) -> bytes:
        """Convert audio samples to WAV format bytes.

        Args:
            audio: Audio samples as int16.
            sample_rate: Sample rate in Hz.
            channels: Number of channels.

        Returns:
            WAV file bytes.
        """
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())

        return buffer.getvalue()
