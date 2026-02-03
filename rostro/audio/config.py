"""Audio configuration."""

from dataclasses import dataclass
from typing import Any


@dataclass
class AudioConfig:
    """Configuration for audio I/O."""

    sample_rate: int = 16000  # Hz (required by Whisper)
    channels: int = 1  # Mono
    chunk_duration_ms: int = 100  # Buffer size for processing
    input_device: int | None = None  # None = system default
    output_device: int | None = None  # None = system default

    @property
    def chunk_size(self) -> int:
        """Calculate chunk size in samples."""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            AudioConfig instance.
        """
        input_dev = data.get("input_device")
        output_dev = data.get("output_device")
        return cls(
            sample_rate=int(data.get("sample_rate", 16000)),
            channels=int(data.get("channels", 1)),
            chunk_duration_ms=int(data.get("chunk_duration_ms", 100)),
            input_device=int(input_dev) if input_dev is not None else None,
            output_device=int(output_dev) if output_dev is not None else None,
        )
