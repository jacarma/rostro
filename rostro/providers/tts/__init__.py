"""Text-to-Speech provider adapters."""

from rostro.providers.tts.google import GoogleTTSProvider
from rostro.providers.tts.openai import OpenAITTSProvider

__all__ = ["GoogleTTSProvider", "OpenAITTSProvider"]
