"""Runtime controller that orchestrates all components."""

import queue
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from rostro.activation.vad import VADActivation, VADConfig, VADState
from rostro.audio.config import AudioConfig
from rostro.audio.playback import AudioPlayback
from rostro.avatar.engine import AvatarEngine, AvatarState
from rostro.conversation.engine import ConversationEngine
from rostro.providers.base import Message
from rostro.providers.llm.openai import OpenAILLMProvider
from rostro.providers.stt.openai import OpenAISTTProvider
from rostro.providers.tts.openai import OpenAITTSProvider


class RuntimeState(Enum):
    """Runtime controller states."""

    STARTING = auto()
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    ERROR = auto()
    STOPPING = auto()


@dataclass
class ErrorConfig:
    """Error handling configuration."""

    max_retries: int = 3
    retry_delay_ms: int = 1000
    timeout_ms: int = 30000
    fallback_message: str = "Sorry, I'm having connection issues. Please try again."

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ErrorConfig":
        """Create from dictionary."""
        return cls(
            max_retries=data.get("max_retries", 3),
            retry_delay_ms=data.get("retry_delay_ms", 1000),
            timeout_ms=data.get("timeout_ms", 30000),
            fallback_message=data.get(
                "fallback_message",
                "Sorry, I'm having connection issues. Please try again.",
            ),
        )


class RuntimeController:
    """Main controller that orchestrates all Rostro components."""

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the runtime controller.

        Args:
            config_path: Path to configuration YAML file.
        """
        # Load environment variables
        load_dotenv()

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize state
        self._state = RuntimeState.STARTING
        self._running = False

        # Component references (initialized in start())
        self._avatar: AvatarEngine | None = None
        self._vad: VADActivation | None = None
        self._playback: AudioPlayback | None = None
        self._conversation: ConversationEngine | None = None
        self._stt: OpenAISTTProvider | None = None
        self._tts: OpenAITTSProvider | None = None
        self._llm: OpenAILLMProvider | None = None

        # Queue for audio from VAD (to avoid threading issues with Pygame)
        self._speech_queue: queue.Queue[bytes] = queue.Queue()

        # Error handling config
        self._error_config = ErrorConfig.from_dict(self.config.get("error_handling", {}))

    def _load_config(self, config_path: Path | None) -> dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file.

        Returns:
            Configuration dictionary.
        """
        if config_path is None:
            # Try default location
            config_path = Path("config/default.yaml")

        if config_path.exists():
            with open(config_path) as f:
                loaded: dict[str, Any] = yaml.safe_load(f)
                return loaded

        # Return minimal default config
        return {
            "providers": {
                "llm": {"provider": "openai", "model": "gpt-4o-mini"},
                "stt": {"provider": "openai", "model": "whisper-1"},
                "tts": {"provider": "openai", "voice": "nova"},
            },
            "audio": {},
            "activation": {"mode": "vad", "vad": {}},
            "avatar": {"face_pack": "default", "resolution": [800, 600], "fps": 30},
            "persona": {"system_prompt": "You are a friendly assistant."},
        }

    def start(self) -> None:
        """Start all components and begin the main loop."""
        self._state = RuntimeState.STARTING
        self._running = True

        # Initialize audio config
        audio_config = AudioConfig.from_dict(self.config.get("audio", {}))

        # Initialize providers
        providers_config = self.config.get("providers", {})

        llm_config = providers_config.get("llm", {})
        self._llm = OpenAILLMProvider(model=llm_config.get("model", "gpt-4o-mini"))

        stt_config = providers_config.get("stt", {})
        self._stt = OpenAISTTProvider(model=stt_config.get("model", "whisper-1"))

        tts_config = providers_config.get("tts", {})
        self._tts = OpenAITTSProvider(voice=tts_config.get("voice", "nova"))

        # Initialize conversation engine
        self._conversation = ConversationEngine.from_config(self.config)

        # Initialize audio playback
        self._playback = AudioPlayback(audio_config)

        # Initialize avatar
        assets_path = Path("assets")
        self._avatar = AvatarEngine.from_config(
            self.config.get("avatar", {}),
            assets_path if assets_path.exists() else None,
        )
        self._avatar.initialize()

        # Initialize VAD
        vad_config = VADConfig.from_dict(self.config.get("activation", {}).get("vad", {}))
        self._vad = VADActivation(vad_config, audio_config)
        self._vad.start(
            on_speech_complete=self._on_speech_complete,
            on_state_change=self._on_vad_state_change,
        )

        self._state = RuntimeState.IDLE
        if self._avatar:
            self._avatar.state = AvatarState.IDLE

        print("Rostro started. Press ESC to quit.")

        # Main loop
        self._main_loop()

    def stop(self) -> None:
        """Stop all components and shutdown."""
        self._state = RuntimeState.STOPPING
        self._running = False

        if self._vad:
            self._vad.stop()

        if self._playback:
            self._playback.stop()

        if self._avatar:
            self._avatar.shutdown()

        print("Rostro stopped.")

    def _main_loop(self) -> None:
        """Main render/event loop."""
        while self._running and self._avatar:
            # Check for pending speech to process
            try:
                audio_bytes = self._speech_queue.get_nowait()
                self._process_speech(audio_bytes)
            except queue.Empty:
                pass

            if not self._avatar.run_frame():
                break

        self.stop()

    def _on_vad_state_change(self, vad_state: VADState) -> None:
        """Handle VAD state changes.

        Args:
            vad_state: New VAD state.
        """
        if self._avatar is None:
            return

        if vad_state == VADState.DORMANT:
            if self._state == RuntimeState.LISTENING:
                self._state = RuntimeState.IDLE
                self._avatar.state = AvatarState.IDLE
        elif vad_state == VADState.LISTENING:
            self._state = RuntimeState.LISTENING
            self._avatar.state = AvatarState.LISTENING
        elif vad_state == VADState.PROCESSING:
            self._state = RuntimeState.PROCESSING
            self._avatar.state = AvatarState.THINKING

    def _on_speech_complete(self, audio_bytes: bytes) -> None:
        """Handle completed speech from VAD (called from audio thread).

        Args:
            audio_bytes: WAV audio bytes of user speech.
        """
        # Put audio in queue to be processed by main thread
        # This avoids threading issues with Pygame
        self._speech_queue.put(audio_bytes)

    def _process_speech(self, audio_bytes: bytes) -> None:
        """Process speech audio (called from main thread).

        Args:
            audio_bytes: WAV audio bytes of user speech.
        """
        if self._avatar is None:
            return

        self._state = RuntimeState.PROCESSING
        self._avatar.state = AvatarState.THINKING

        try:
            # Transcribe speech
            user_text = self._transcribe_with_retry(audio_bytes)
            if not user_text.strip():
                self._return_to_idle()
                return

            print(f"User: {user_text}")

            # Add to conversation
            if self._conversation:
                self._conversation.add_user_message(user_text)

            # Generate response
            response_text = self._complete_with_retry()
            if not response_text:
                self._handle_error()
                return

            print(f"Assistant: {response_text}")

            # Add response to conversation
            if self._conversation:
                self._conversation.add_assistant_message(response_text)

            # Synthesize and speak
            self._speak_response(response_text)

        except Exception as e:
            print(f"Error processing speech: {e}")
            self._handle_error()

    def _transcribe_with_retry(self, audio_bytes: bytes) -> str:
        """Transcribe audio with retry logic.

        Args:
            audio_bytes: WAV audio bytes.

        Returns:
            Transcribed text.
        """
        if self._stt is None:
            return ""

        for attempt in range(self._error_config.max_retries):
            try:
                return self._stt.transcribe(audio_bytes, format="wav")
            except Exception as e:
                if attempt < self._error_config.max_retries - 1:
                    time.sleep(self._error_config.retry_delay_ms / 1000)
                else:
                    raise e

        return ""

    def _complete_with_retry(self) -> str:
        """Generate LLM completion with retry logic.

        Returns:
            Generated response text.
        """
        if self._llm is None or self._conversation is None:
            return ""

        messages = self._conversation.build_messages()

        for attempt in range(self._error_config.max_retries):
            try:
                # Convert to base Message type for protocol
                base_messages = [Message(role=m.role, content=m.content) for m in messages]
                return self._llm.complete(base_messages)
            except Exception as e:
                if attempt < self._error_config.max_retries - 1:
                    time.sleep(self._error_config.retry_delay_ms / 1000)
                else:
                    raise e

        return ""

    def _speak_response(self, text: str) -> None:
        """Synthesize and speak a response.

        Args:
            text: Text to speak.
        """
        print("[Speak] Starting speech synthesis...")

        if self._tts is None or self._playback is None or self._avatar is None:
            print("[Speak] Missing components, aborting")
            return

        # Pause VAD while speaking
        if self._vad:
            self._vad.pause()

        self._state = RuntimeState.SPEAKING
        self._avatar.state = AvatarState.SPEAKING

        try:
            # Synthesize audio
            print("[Speak] Calling TTS API...")
            audio_bytes = self._tts.synthesize(text)
            print(f"[Speak] Got {len(audio_bytes)} bytes from TTS")

            # Play with volume callback for lip sync
            print("[Speak] Starting playback...")
            self._playback.play(
                audio_bytes,
                format="wav",
                volume_callback=self._on_playback_volume,
            )

            # Wait for playback to complete
            print("[Speak] Waiting for playback to finish...")
            while self._playback.is_playing:
                if self._avatar and not self._avatar.run_frame():
                    break
                time.sleep(0.01)
            print("[Speak] Playback finished")

        except Exception as e:
            print(f"[Speak] Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self._return_to_idle()

    def _on_playback_volume(self, volume: float) -> None:
        """Handle volume updates during playback.

        Args:
            volume: Current volume level 0.0-1.0.
        """
        if self._avatar:
            self._avatar.set_volume(volume)

    def _return_to_idle(self) -> None:
        """Return to idle state and resume VAD."""
        self._state = RuntimeState.IDLE
        if self._avatar:
            self._avatar.state = AvatarState.IDLE
            self._avatar.set_volume(0.0)

        # Resume VAD
        if self._vad:
            self._vad.resume()

    def _handle_error(self) -> None:
        """Handle an error state."""
        self._state = RuntimeState.ERROR
        if self._avatar:
            self._avatar.state = AvatarState.ERROR

        # Speak fallback message if possible
        try:
            if self._tts and self._playback:
                audio_bytes = self._tts.synthesize(self._error_config.fallback_message)
                self._playback.play(audio_bytes, format="wav")
                while self._playback.is_playing:
                    if self._avatar and not self._avatar.run_frame():
                        break
                    time.sleep(0.01)
        except Exception:
            print(f"Fallback: {self._error_config.fallback_message}")

        self._return_to_idle()
