"""Google Cloud Text-to-Speech provider using WaveNet voices."""

import os
from pathlib import Path

from google.cloud import texttospeech
from google.oauth2 import service_account


class GoogleTTSProvider:
    """Google Cloud TTS API implementation using WaveNet voices."""

    # Spanish (Spain) voices - es-ES
    AVAILABLE_VOICES: list[str] = [
        # WaveNet voices
        "es-ES-Wavenet-E",  # Male
        "es-ES-Wavenet-F",  # Female
        "es-ES-Wavenet-G",  # Male
        "es-ES-Wavenet-H",  # Female
        # Neural2 voices (higher quality)
        "es-ES-Neural2-A",  # Female
        "es-ES-Neural2-E",  # Female
        "es-ES-Neural2-F",  # Male
        "es-ES-Neural2-G",  # Male
        "es-ES-Neural2-H",  # Female
        # Studio voices (highest quality)
        "es-ES-Studio-C",  # Female
        "es-ES-Studio-F",  # Male
        # Chirp HD voices (newest, very natural)
        "es-ES-Chirp-HD-F",  # Female
        "es-ES-Chirp-HD-D",  # Male
    ]

    def __init__(
        self,
        voice: str = "es-ES-Wavenet-C",
        language_code: str = "es-ES",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        credentials_path: str | None = None,
    ) -> None:
        """Initialize the Google Cloud TTS provider.

        Args:
            voice: Voice name (e.g., es-ES-Wavenet-C).
            language_code: Language code (e.g., es-ES).
            speaking_rate: Speaking rate (0.25 to 4.0, default 1.0).
            pitch: Pitch adjustment in semitones (-20.0 to 20.0, default 0.0).
            credentials_path: Path to service account JSON. If None, uses
                GOOGLE_APPLICATION_CREDENTIALS env var or default location.
        """
        self.default_voice = voice
        self.language_code = language_code
        self.speaking_rate = speaking_rate
        self.pitch = pitch

        # Initialize client with credentials
        if credentials_path:
            creds_path = Path(credentials_path)
        elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            creds_path = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        else:
            # Default location in project
            project_root = Path(__file__).parent.parent.parent.parent
            creds_path = project_root / "config" / "credentials" / "google-tts.json"

        if creds_path.exists():
            credentials = service_account.Credentials.from_service_account_file(
                str(creds_path)
            )  # type: ignore[no-untyped-call]
            self.client = texttospeech.TextToSpeechClient(credentials=credentials)
        else:
            # Try default credentials (e.g., from gcloud auth)
            self.client = texttospeech.TextToSpeechClient()

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize.
            voice: Voice to use. If None, uses default voice.

        Returns:
            Audio bytes in WAV format (24kHz, 16-bit, mono).
        """
        voice_name = voice or self.default_voice

        # Extract language code from voice name (e.g., es-ES-Wavenet-C -> es-ES)
        parts = voice_name.split("-")
        if len(parts) >= 2:
            lang_code = f"{parts[0]}-{parts[1]}"
        else:
            lang_code = self.language_code

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice_params = texttospeech.VoiceSelectionParams(
            language_code=lang_code,
            name=voice_name,
        )

        # Request LINEAR16 (raw PCM) at 24kHz to match OpenAI TTS output
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
            speaking_rate=self.speaking_rate,
            pitch=self.pitch,
        )

        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )

        # Google returns raw LINEAR16 PCM, we need to wrap it in WAV header
        return self._add_wav_header(response.audio_content, sample_rate=24000)

    def _add_wav_header(self, pcm_data: bytes, sample_rate: int = 24000) -> bytes:
        """Add WAV header to raw PCM data.

        Args:
            pcm_data: Raw PCM audio data (16-bit, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            Complete WAV file as bytes.
        """
        import struct

        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(pcm_data)
        chunk_size = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            chunk_size,
            b"WAVE",
            b"fmt ",
            16,  # Subchunk1Size for PCM
            1,   # AudioFormat (1 = PCM)
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )

        return header + pcm_data

    def list_voices(self) -> list[str]:
        """List available voices.

        Returns:
            List of voice identifiers.
        """
        return self.AVAILABLE_VOICES.copy()
