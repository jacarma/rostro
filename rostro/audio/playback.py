"""Audio playback."""

import io
import threading
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

from rostro.audio.config import AudioConfig


class AudioPlayback:
    """Handles audio playback with volume analysis for lip-sync."""

    def __init__(self, config: AudioConfig) -> None:
        """Initialize audio playback.

        Args:
            config: Audio configuration.
        """
        self.config = config
        self._is_playing = False
        self._current_audio: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._playback_position = 0
        self._playback_thread: threading.Thread | None = None
        self._volume_callback: Callable[[float], None] | None = None
        self._stop_event = threading.Event()

    def play(
        self,
        audio_data: bytes,
        format: str = "mp3",
        volume_callback: Callable[[float], None] | None = None,
    ) -> None:
        """Play audio data.

        Args:
            audio_data: Audio bytes (MP3 or WAV format).
            format: Audio format ("mp3" or "wav").
            volume_callback: Optional callback with current volume (0.0-1.0).
        """
        print(f"[Playback] Received {len(audio_data)} bytes of {format} audio")

        # Convert audio to numpy array
        try:
            audio_array = self._decode_audio(audio_data, format)
            print(f"[Playback] Decoded to {len(audio_array)} samples, "
                  f"duration: {len(audio_array)/self.config.sample_rate:.2f}s")
        except Exception as e:
            print(f"[Playback] Decode error: {e}")
            raise

        self._current_audio = audio_array
        self._playback_position = 0
        self._volume_callback = volume_callback
        self._stop_event.clear()
        self._is_playing = True

        # Start playback in a separate thread
        self._playback_thread = threading.Thread(target=self._playback_loop)
        self._playback_thread.start()

    def stop(self) -> None:
        """Stop audio playback."""
        self._stop_event.set()
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=1.0)
        self._is_playing = False
        self._current_audio = None

    def _playback_loop(self) -> None:
        """Internal playback loop with volume tracking."""
        if self._current_audio is None:
            print("[Playback] No audio data")
            return

        audio = self._current_audio
        sample_rate = self.config.sample_rate
        chunk_size = self.config.chunk_size

        print(f"[Playback] Starting: {len(audio)} samples, {len(audio)/sample_rate:.2f}s")

        try:
            # Start playback
            sd.play(audio, sample_rate, device=self.config.output_device)

            # Track volume while playing
            total_samples = len(audio)
            position = 0

            while sd.get_stream() and sd.get_stream().active and not self._stop_event.is_set():
                # Estimate current position based on time
                if self._volume_callback is not None and position < total_samples:
                    end = min(position + chunk_size, total_samples)
                    chunk = audio[position:end]
                    rms = float(np.sqrt(np.mean(chunk**2)))
                    self._volume_callback(rms)
                    position = end

                sd.sleep(int(self.config.chunk_duration_ms))

            sd.wait()  # Wait for playback to finish
            print("[Playback] Finished successfully")

        except Exception as e:
            print(f"[Playback] Error: {e}")
        finally:
            self._is_playing = False
            if self._volume_callback is not None:
                self._volume_callback(0.0)

    def _decode_audio(
        self, audio_data: bytes, format: str
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Decode audio bytes to numpy array.

        Args:
            audio_data: Audio bytes.
            format: Audio format ("mp3" or "wav").

        Returns:
            Audio samples as float32 array.
        """
        if format == "wav":
            return self._decode_wav(audio_data)
        elif format == "mp3":
            return self._decode_mp3(audio_data)
        else:
            raise ValueError(f"Unsupported audio format: {format}")

    def _decode_wav(self, audio_data: bytes) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Decode WAV audio and resample if needed.

        Args:
            audio_data: WAV bytes.

        Returns:
            Audio samples as float32 array at config sample rate.
        """
        import wave

        buffer = io.BytesIO(audio_data)
        with wave.open(buffer, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            source_rate = wav_file.getframerate()
            raw_data = wav_file.readframes(n_frames)

        # Convert to numpy array
        if sample_width == 2:
            audio = np.frombuffer(raw_data, dtype=np.int16)
            audio_float = audio.astype(np.float32) / 32768.0
        elif sample_width == 1:
            audio = np.frombuffer(raw_data, dtype=np.uint8)
            audio_float = (audio.astype(np.float32) - 128) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Convert stereo to mono if needed
        if n_channels == 2:
            audio_float = audio_float.reshape(-1, 2).mean(axis=1)

        # Resample if source rate differs from config rate
        if source_rate != self.config.sample_rate:
            audio_float = self._resample(audio_float, source_rate, self.config.sample_rate)

        return audio_float

    def _resample(
        self,
        audio: np.ndarray[Any, np.dtype[np.float32]],
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Resample audio to target sample rate using linear interpolation.

        Args:
            audio: Source audio samples.
            source_rate: Source sample rate.
            target_rate: Target sample rate.

        Returns:
            Resampled audio.
        """
        if source_rate == target_rate:
            return audio

        # Calculate new length
        duration = len(audio) / source_rate
        new_length = int(duration * target_rate)

        # Linear interpolation
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(new_indices, old_indices, audio)

        return cast(np.ndarray[Any, np.dtype[np.float32]], resampled.astype(np.float32))

    def _decode_mp3(self, audio_data: bytes) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Decode MP3 audio using pygame.

        Args:
            audio_data: MP3 bytes.

        Returns:
            Audio samples as float32 array.
        """
        # Use pygame's mixer for MP3 decoding
        import os
        import tempfile

        # Write to temp file (pygame needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            # Use pygame to load and convert
            import pygame.mixer

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.config.sample_rate, channels=1)

            sound = pygame.mixer.Sound(temp_path)
            raw_array = pygame.sndarray.array(sound)

            # Convert to float32
            audio_float = raw_array.astype(np.float32) / 32768.0

            # Convert stereo to mono if needed
            if len(audio_float.shape) > 1:
                audio_float = audio_float.mean(axis=1)

            return audio_float
        finally:
            os.unlink(temp_path)

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing
