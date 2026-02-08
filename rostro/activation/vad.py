"""Voice Activity Detection (VAD) for activation."""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from rostro.activation.silero import SILERO_FRAME_SAMPLES, SILERO_SAMPLE_RATE, SileroVAD
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

    confidence_threshold: float = 0.7  # Silero confidence threshold (0.0-1.0)
    start_secs: float = 0.2  # Consecutive voice frames to confirm speech start
    min_volume: float = 0.002  # RMS pre-filter (skip Silero inference on silence)
    silence_duration_ms: int = 1500  # Silence before end-of-speech
    min_speech_duration_ms: int = 300  # Ignore very short sounds

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VADConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            VADConfig instance.
        """
        return cls(
            confidence_threshold=float(data.get("confidence_threshold", 0.7)),
            start_secs=float(data.get("start_secs", 0.2)),
            min_volume=float(data.get("min_volume", 0.002)),
            silence_duration_ms=int(data.get("silence_duration_ms", 1500)),
            min_speech_duration_ms=int(data.get("min_speech_duration_ms", 300)),
        )


# How often to reset Silero internal state during silence (seconds)
_MODEL_RESET_INTERVAL_S = 5.0

# Pre-roll buffer duration in seconds (captures speech onset during confirmation)
_PRE_ROLL_SECS = 0.8


class VADActivation:
    """Voice Activity Detection based activation system."""

    # Seconds to ignore audio after resuming (echo suppression)
    RESUME_COOLDOWN_S = 1.0

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
        self._resume_time: float = 0.0
        self._on_speech_complete: Callable[[bytes], None] | None = None
        self._on_state_change: Callable[[VADState], None] | None = None

        # Silero model (lazy-loaded in start())
        self._silero: SileroVAD | None = None

        # Buffer for accumulating resampled 16kHz samples for Silero
        self._silero_buffer = np.array([], dtype=np.float32)

        # Speech start confirmation: count consecutive voice frames
        self._voice_frame_count: int = 0
        self._frames_needed_for_start: int = 0

        # Pre-roll: keep recent audio chunks to capture speech onset
        pre_roll_chunks = int(_PRE_ROLL_SECS * 1000 / audio_config.chunk_duration_ms)
        self._pre_roll: deque[np.ndarray[Any, np.dtype[np.int16]]] = deque(
            maxlen=max(pre_roll_chunks, 1)
        )

        # Periodic model reset tracking
        self._last_silence_time: float = 0.0
        self._last_model_reset_time: float = 0.0

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

        # Lazy-load Silero model
        if self._silero is None:
            print("[VAD] Loading Silero VAD model...")
            self._silero = SileroVAD()
            print("[VAD] Silero VAD loaded")

        # Calculate how many consecutive voice callbacks confirm speech start
        callbacks_for_start = self.vad_config.start_secs / (
            self.audio_config.chunk_duration_ms / 1000
        )
        self._frames_needed_for_start = max(int(callbacks_for_start), 1)

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
        self._silero_buffer = np.array([], dtype=np.float32)
        self._voice_frame_count = 0
        self._pre_roll.clear()
        if self._silero is not None:
            self._silero.reset_states()
        print("[VAD] Paused")

    def resume(self) -> None:
        """Resume VAD monitoring."""
        if self._audio_capture is not None and self._on_speech_complete is not None:
            self._resume_time = time.time()
            self._silero_buffer = np.array([], dtype=np.float32)
            self._voice_frame_count = 0
            self._pre_roll.clear()
            if self._silero is not None:
                self._silero.reset_states()
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

    def _downsample_to_16k(
        self, audio_int16: np.ndarray[Any, np.dtype[np.int16]]
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Downsample audio from capture sample rate to 16kHz float32.

        Args:
            audio_int16: Audio samples at capture sample rate.

        Returns:
            Float32 audio at 16kHz, normalized to [-1, 1].
        """
        audio_float = audio_int16.astype(np.float32) / 32768.0

        if self.audio_config.sample_rate == SILERO_SAMPLE_RATE:
            return audio_float

        # Linear interpolation for downsampling
        n_out = int(len(audio_float) * SILERO_SAMPLE_RATE / self.audio_config.sample_rate)
        x_in = np.linspace(0, 1, len(audio_float))
        x_out = np.linspace(0, 1, n_out)
        result: np.ndarray[Any, np.dtype[np.float32]] = np.interp(x_out, x_in, audio_float).astype(
            np.float32
        )
        return result

    def _process_audio(self, audio_chunk: np.ndarray[Any, np.dtype[np.int16]]) -> None:
        """Process an audio chunk from the microphone.

        Args:
            audio_chunk: Audio samples at capture sample rate (int16).
        """
        current_time = time.time()

        # Ignore audio during cooldown after resume (echo suppression)
        if current_time - self._resume_time < self.RESUME_COOLDOWN_S:
            return

        # Pre-filter: skip Silero inference if audio is too quiet
        rms = AudioCapture.compute_rms(audio_chunk)
        if rms < self.vad_config.min_volume:
            is_voice = False
            # Track silence for periodic model reset
            if self._last_silence_time == 0.0:
                self._last_silence_time = current_time
            elif (
                current_time - self._last_model_reset_time >= _MODEL_RESET_INTERVAL_S
                and self._silero is not None
            ):
                self._silero.reset_states()
                self._last_model_reset_time = current_time
        else:
            self._last_silence_time = 0.0

            # Downsample and buffer for Silero
            resampled = self._downsample_to_16k(audio_chunk)
            self._silero_buffer = np.concatenate((self._silero_buffer, resampled))

            # Process all complete 512-sample frames
            is_voice = False
            while len(self._silero_buffer) >= SILERO_FRAME_SAMPLES and self._silero is not None:
                frame = self._silero_buffer[:SILERO_FRAME_SAMPLES]
                self._silero_buffer = self._silero_buffer[SILERO_FRAME_SAMPLES:]
                confidence = self._silero(frame)
                if confidence >= self.vad_config.confidence_threshold:
                    is_voice = True

        if self._state == VADState.DORMANT:
            # Always keep pre-roll buffer updated
            self._pre_roll.append(audio_chunk)

            if is_voice:
                self._voice_frame_count += 1
                if self._voice_frame_count >= self._frames_needed_for_start:
                    # Speech confirmed — start recording with pre-roll
                    self._speech_start_time = current_time
                    self._recorded_audio = list(self._pre_roll)
                    self._pre_roll.clear()
                    self._set_state(VADState.LISTENING)
                    self._voice_frame_count = 0
            else:
                self._voice_frame_count = 0

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

        # Convert to WAV bytes (at capture sample rate for STT)
        wav_bytes = AudioCapture.to_wav_bytes(
            full_audio,
            self.audio_config.sample_rate,
            self.audio_config.channels,
        )

        # Send to callback (which puts it in a queue, so this is safe)
        if self._on_speech_complete is not None:
            self._on_speech_complete(wav_bytes)

        self._set_state(VADState.DORMANT)
