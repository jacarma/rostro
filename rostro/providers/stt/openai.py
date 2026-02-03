"""OpenAI Whisper Speech-to-Text provider."""

import io

from openai import OpenAI


class OpenAISTTProvider:
    """OpenAI Whisper API implementation for speech-to-text."""

    def __init__(self, model: str = "whisper-1", api_key: str | None = None) -> None:
        """Initialize the OpenAI STT provider.

        Args:
            model: Whisper model to use.
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def transcribe(self, audio: bytes, format: str = "wav") -> str:
        """Transcribe audio to text using Whisper.

        Args:
            audio: Raw audio bytes.
            format: Audio format (e.g., "wav", "mp3").

        Returns:
            Transcribed text.
        """
        audio_file = io.BytesIO(audio)
        audio_file.name = f"audio.{format}"

        response = self.client.audio.transcriptions.create(
            model=self.model,
            file=audio_file,
        )

        return response.text
