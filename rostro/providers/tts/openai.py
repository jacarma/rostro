"""OpenAI Text-to-Speech provider."""

from typing import Literal, cast

from openai import OpenAI

VoiceType = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


class OpenAITTSProvider:
    """OpenAI TTS API implementation for text-to-speech."""

    AVAILABLE_VOICES: list[str] = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def __init__(
        self,
        model: str = "tts-1",
        voice: str = "nova",
        api_key: str | None = None,
    ) -> None:
        """Initialize the OpenAI TTS provider.

        Args:
            model: TTS model to use (tts-1 or tts-1-hd).
            voice: Default voice to use.
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.model = model
        self.default_voice = voice
        self.client = OpenAI(api_key=api_key)

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Voice to use. If None, uses default voice.

        Returns:
            Audio bytes in WAV format (24kHz, 16-bit, mono).
        """
        voice_to_use = cast(VoiceType, voice or self.default_voice)

        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice_to_use,
            input=text,
            response_format="wav",
        )

        return response.content

    def list_voices(self) -> list[str]:
        """List available voices.

        Returns:
            List of voice identifiers.
        """
        return self.AVAILABLE_VOICES.copy()
