"""Base protocol definitions for AI providers."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Message:
    """A message in a conversation."""

    role: str  # "system", "user", or "assistant"
    content: str


class STTProvider(Protocol):
    """Protocol for Speech-to-Text providers."""

    def transcribe(self, audio: bytes, format: str = "wav") -> str:
        """Transcribe audio to text.

        Args:
            audio: Raw audio bytes.
            format: Audio format (e.g., "wav", "mp3").

        Returns:
            Transcribed text.
        """
        ...


class TTSProvider(Protocol):
    """Protocol for Text-to-Speech providers."""

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Optional voice identifier.

        Returns:
            Audio bytes.
        """
        ...

    def list_voices(self) -> list[str]:
        """List available voices.

        Returns:
            List of voice identifiers.
        """
        ...


class LLMProvider(Protocol):
    """Protocol for LLM chat completion providers."""

    def complete(self, messages: list[Message], **kwargs: object) -> str:
        """Generate a completion for the given messages.

        Args:
            messages: List of conversation messages.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Generated response text.
        """
        ...

    def stream(self, messages: list[Message], **kwargs: object) -> Iterator[str]:
        """Stream a completion for the given messages.

        Args:
            messages: List of conversation messages.
            **kwargs: Additional provider-specific parameters.

        Yields:
            Response text chunks.
        """
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation providers."""

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for the given text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...
