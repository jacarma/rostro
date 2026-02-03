"""Voice Activity Detection (VAD) for activation."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from rostro.audio.capture import AudioCapture
from rostro.audio.config import AudioConfig


class VADState(Enum):
    """VAD state machine states."""

    DORMANT = auto()  # Monitoring, low CPU
    LISTENING = auto()  # Voice detected, recording
    PROCESSING = auto()  # Silence detected, sending to STT


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""

    volume_threshold: float = 0.02  # RMS threshold (0.0-1.0)
    silence_duration_ms: int = 1500  # Silence before end-of-speech
    min_speech_duration_ms: int = 300  # Ignore very short sounds
    sample_window_ms: int = 50  # Analysis window size

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VADConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            VADConfig instance.
        """
        return cls(
            volume_threshold=float(data.get("volume_threshold", 0.02)),
            silence_duration_ms=int(data.get("silence_duration_ms", 1500)),
            min_speech_duration_ms=int(data.get("min_speech_duration_ms", 300)),
            sample_window_ms=int(data.get("sample_window_ms", 50)),
        )


class VADActivation:
    """Voice Activity Detection based activation system."""

    def __init__(
        self,
        vad_config: VADConfig,
        audio_config: AudioConfig,
    ) -> None:
        """Initialize VAD activation.

        Args:
            vad_config: VAD configuration.
            audio_config: Audio configuration.
        """
        self.vad_config = vad_config
        self.audio_config = audio_config
        self._state = VADState.DORMANT
        self._audio_capture: AudioCapture | None = None
        self._recorded_audio: list[np.ndarray[Any, np.dtype[np.int16]]] = []
        self._speech_start_time: float = 0.0
        self._silence_start_time: float = 0.0
        self._on_speech_complete: Callable[[bytes], None] | None = None
        self._on_state_change: Callable[[VADState], None] | None = None

    @property
    def state(self) -> VADState:
        """Get current VAD state."""
        return self._state

    def start(
        self,
        on_speech_complete: Callable[[bytes], None],
        on_state_change: Callable[[VADState], None] | None = None,
    ) -> None:
        """Start VAD monitoring.

        Args:
            on_speech_complete: Callback with WAV audio bytes when speech ends.
            on_state_change: Optional callback when VAD state changes.
        """
        self._on_speech_complete = on_speech_complete
        self._on_state_change = on_state_change
        self._audio_capture = AudioCapture(self.audio_config)
        self._audio_capture.start(self._process_audio)
        self._set_state(VADState.DORMANT)

    def stop(self) -> None:
        """Stop VAD monitoring."""
        if self._audio_capture is not None:
            self._audio_capture.stop()
            self._audio_capture = None
        self._set_state(VADState.DORMANT)

    def pause(self) -> None:
        """Pause VAD (e.g., while assistant is speaking)."""
        print("[VAD] Pausing...")
        if self._audio_capture is not None:
            print("[VAD] Stopping audio capture...")
            self._audio_capture.stop()
            print("[VAD] Audio capture stopped")
        self._set_state(VADState.DORMANT)
        self._recorded_audio.clear()
        print("[VAD] Paused")

    def resume(self) -> None:
        """Resume VAD monitoring."""
        if self._audio_capture is not None and self._on_speech_complete is not None:
            self._audio_capture.start(self._process_audio)
            self._set_state(VADState.DORMANT)

    def _set_state(self, new_state: VADState) -> None:
        """Update state and notify callback.

        Args:
            new_state: New VAD state.
        """
        if self._state != new_state:
            self._state = new_state
            if self._on_state_change is not None:
                self._on_state_change(new_state)

    def _process_audio(self, audio_chunk: np.ndarray[Any, np.dtype[np.int16]]) -> None:
        """Process an audio chunk from the microphone.

        Args:
            audio_chunk: Audio samples.
        """
        rms = AudioCapture.compute_rms(audio_chunk)
        current_time = time.time()
        is_voice = rms > self.vad_config.volume_threshold

        if self._state == VADState.DORMANT:
            if is_voice:
                # Voice detected, start recording
                self._speech_start_time = current_time
                self._recorded_audio = [audio_chunk]
                self._set_state(VADState.LISTENING)

        elif self._state == VADState.LISTENING:
            self._recorded_audio.append(audio_chunk)

            if is_voice:
                # Reset silence timer
                self._silence_start_time = current_time
            else:
                if self._silence_start_time == 0.0:
                    self._silence_start_time = current_time

                silence_duration_ms = (current_time - self._silence_start_time) * 1000

                if silence_duration_ms >= self.vad_config.silence_duration_ms:
                    # Check minimum speech duration
                    speech_duration_ms = (current_time - self._speech_start_time) * 1000

                    if speech_duration_ms >= self.vad_config.min_speech_duration_ms:
                        # Speech complete, process it
                        self._set_state(VADState.PROCESSING)
                        self._finalize_recording()
                    else:
                        # Too short, ignore
                        self._recorded_audio.clear()
                        self._set_state(VADState.DORMANT)

                    self._silence_start_time = 0.0

    def _finalize_recording(self) -> None:
        """Finalize and send the recorded audio."""
        if not self._recorded_audio:
            self._set_state(VADState.DORMANT)
            return

        # Concatenate all audio chunks
        full_audio = np.concatenate(self._recorded_audio)
        self._recorded_audio.clear()

        # Convert to WAV bytes
        wav_bytes = AudioCapture.to_wav_bytes(
            full_audio,
            self.audio_config.sample_rate,
            self.audio_config.channels,
        )

        # Send to callback (which puts it in a queue, so this is safe)
        if self._on_speech_complete is not None:
            self._on_speech_complete(wav_bytes)

        self._set_state(VADState.DORMANT)
