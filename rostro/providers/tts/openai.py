"""OpenAI Text-to-Speech provider."""

from typing import Literal, cast

from openai import OpenAI

VoiceType = Literal[
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
]


class OpenAITTSProvider:
    """OpenAI TTS API implementation for text-to-speech."""

    AVAILABLE_VOICES: list[str] = [
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "onyx",
        "nova",
        "sage",
        "shimmer",
        "verse",
    ]

    def __init__(
        self,
        model: str = "tts-1",
        voice: str = "nova",
        instructions: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the OpenAI TTS provider.

        Args:
            model: TTS model to use (tts-1, tts-1-hd, or gpt-4o-mini-tts).
            voice: Default voice to use.
            instructions: Voice style instructions (only for gpt-4o-mini-tts).
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.model = model
        self.default_voice = voice
        self.instructions = instructions
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

        kwargs: dict[str, object] = {
            "model": self.model,
            "voice": voice_to_use,
            "input": text,
            "response_format": "wav",
        }
        if self.instructions:
            kwargs["instructions"] = self.instructions

        response = self.client.audio.speech.create(**kwargs)  # type: ignore[arg-type]

        return response.content

    def list_voices(self) -> list[str]:
        """List available voices.

        Returns:
            List of voice identifiers.
        """
        return self.AVAILABLE_VOICES.copy()
