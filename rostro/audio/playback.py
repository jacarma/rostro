"""Audio playback with queue-based sequential playback."""

import io
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

from rostro.audio.config import AudioConfig


@dataclass
class AudioItem:
    """An item in the playback queue."""

    audio: np.ndarray[Any, np.dtype[np.float32]]
    sample_rate: int


class AudioPlayback:
    """Handles audio playback with a queue for sequential playback.

    The queue operates autonomously:
    - When audio is enqueued and nothing is playing, playback starts immediately
    - When playback finishes, the next item in the queue plays automatically
    - Volume callbacks are called during playback for lip-sync
    """

    def __init__(self, config: AudioConfig) -> None:
        """Initialize audio playback.

        Args:
            config: Audio configuration.
        """
        self.config = config
        self._queue: queue.Queue[AudioItem] = queue.Queue()
        self._is_playing = False
        self._stop_event = threading.Event()
        self._playback_thread: threading.Thread | None = None
        self._volume_callback: Callable[[float], None] | None = None
        self._on_queue_empty: Callable[[], None] | None = None
        self._lock = threading.Lock()

    def set_volume_callback(self, callback: Callable[[float], None] | None) -> None:
        """Set the volume callback for lip-sync.

        Args:
            callback: Function called with volume level (0.0-1.0) during playback.
        """
        self._volume_callback = callback

    def set_on_queue_empty(self, callback: Callable[[], None] | None) -> None:
        """Set callback for when the queue becomes empty after playback.

        Args:
            callback: Function called when all queued audio has finished playing.
        """
        self._on_queue_empty = callback

    def play(
        self,
        audio_data: bytes,
        format: str = "mp3",
        volume_callback: Callable[[float], None] | None = None,
    ) -> None:
        """Enqueue audio for playback.

        If nothing is playing, playback starts immediately.
        If something is playing, the audio is queued and will play after.

        Args:
            audio_data: Audio bytes (MP3 or WAV format).
            format: Audio format ("mp3" or "wav").
            volume_callback: Optional callback with current volume (0.0-1.0).
        """
        # Update volume callback if provided
        if volume_callback is not None:
            self._volume_callback = volume_callback

        print(f"[Playback] Received {len(audio_data)} bytes of {format} audio")

        # Decode audio
        try:
            audio_array = self._decode_audio(audio_data, format)
            duration = len(audio_array) / self.config.sample_rate
            print(f"[Playback] Decoded: {len(audio_array)} samples, {duration:.2f}s")
        except Exception as e:
            print(f"[Playback] Decode error: {e}")
            raise

        # Create queue item
        item = AudioItem(audio=audio_array, sample_rate=self.config.sample_rate)

        # Enqueue
        self._queue.put(item)
        print(f"[Playback] Enqueued, queue size: {self._queue.qsize()}")

        # Start playback thread if not running
        self._ensure_playback_thread()

    def _ensure_playback_thread(self) -> None:
        """Ensure the playback thread is running."""
        with self._lock:
            if self._playback_thread is None or not self._playback_thread.is_alive():
                self._stop_event.clear()
                self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
                self._playback_thread.start()

    def _playback_loop(self) -> None:
        """Main playback loop - processes queue items sequentially."""
        print("[Playback] Playback thread started")

        while not self._stop_event.is_set():
            try:
                # Wait for next item (with timeout to check stop_event)
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    # Check if we should exit
                    if self._queue.empty():
                        break
                    continue

                # Play this item
                self._is_playing = True
                self._play_item(item)
                self._queue.task_done()

            except Exception as e:
                print(f"[Playback] Loop error: {e}")

        # Loop exited - queue is empty
        self._is_playing = False
        if self._volume_callback is not None:
            self._volume_callback(0.0)

        print("[Playback] Playback thread finished")

        # Notify that queue is empty
        if self._on_queue_empty is not None:
            try:
                self._on_queue_empty()
            except Exception as e:
                print(f"[Playback] on_queue_empty callback error: {e}")

    def _play_item(self, item: AudioItem) -> None:
        """Play a single audio item.

        Args:
            item: The audio item to play.
        """
        audio = item.audio
        sample_rate = item.sample_rate
        chunk_size = self.config.chunk_size

        print(f"[Playback] Playing: {len(audio)} samples, {len(audio) / sample_rate:.2f}s")

        try:
            # Start playback
            sd.play(audio, sample_rate, device=self.config.output_device)

            # Track volume while playing
            total_samples = len(audio)
            position = 0

            while not self._stop_event.is_set():
                stream = sd.get_stream()
                if stream is None or not stream.active:
                    break

                # Calculate and report volume for lip-sync
                if self._volume_callback is not None and position < total_samples:
                    end = min(position + chunk_size, total_samples)
                    chunk = audio[position:end]
                    rms = float(np.sqrt(np.mean(chunk**2)))
                    self._volume_callback(rms)
                    position = end

                sd.sleep(int(self.config.chunk_duration_ms))

            sd.wait()  # Wait for playback to finish
            print("[Playback] Item finished")

        except Exception as e:
            print(f"[Playback] Play error: {e}")

    def stop(self) -> None:
        """Stop all playback and clear the queue."""
        print("[Playback] Stopping...")
        self._stop_event.set()

        # Clear the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

        # Stop current playback
        sd.stop()

        # Wait for thread to finish
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None

        self._is_playing = False
        if self._volume_callback is not None:
            self._volume_callback(0.0)

        print("[Playback] Stopped")

    def wait(self) -> None:
        """Wait for all queued audio to finish playing."""
        if self._playback_thread is not None:
            self._playback_thread.join()

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing or queued."""
        return self._is_playing or not self._queue.empty()

    @property
    def is_thread_alive(self) -> bool:
        """Check if the playback thread is still running."""
        return self._playback_thread is not None and self._playback_thread.is_alive()

    @property
    def queue_size(self) -> int:
        """Get the number of items in the queue."""
        return self._queue.qsize()

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
