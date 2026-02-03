"""Integration tests for the full pipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from rostro.avatar.engine import AvatarEngine, AvatarState
from rostro.avatar.face_pack import FacePack
from rostro.conversation.engine import ConversationEngine


@pytest.mark.integration
class TestConversationPipeline:
    """Integration tests for conversation flow."""

    def test_conversation_flow(self):
        """Test basic conversation flow."""
        engine = ConversationEngine(system_prompt="You are a helpful assistant.")

        # Simulate conversation
        engine.add_user_message("Hello")
        engine.add_assistant_message("Hi there!")
        engine.add_user_message("How are you?")

        messages = engine.build_messages()

        # Should have system + 3 messages
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[1].content == "Hello"
        assert messages[2].content == "Hi there!"
        assert messages[3].content == "How are you?"


@pytest.mark.integration
class TestAvatarPipeline:
    """Integration tests for avatar rendering."""

    def test_avatar_state_transitions(self):
        """Test avatar state transitions."""
        engine = AvatarEngine()

        # Test all state transitions
        states = [
            AvatarState.IDLE,
            AvatarState.LISTENING,
            AvatarState.THINKING,
            AvatarState.SPEAKING,
            AvatarState.ERROR,
            AvatarState.IDLE,
        ]

        for state in states:
            engine.state = state
            assert engine.state == state

    def test_face_pack_loading(self):
        """Test loading face pack from directory."""
        with TemporaryDirectory() as tmpdir:
            # Create manifest
            manifest = Path(tmpdir) / "manifest.yaml"
            manifest.write_text(
                """
name: "Test Pack"
version: "1.0"
author: "Test"
type: programmatic
colors:
  face: "#FFDDCC"
  eyes: "#112233"
  mouth: "#CC4455"
  background: "#1A1A1A"
"""
            )

            pack = FacePack.load(Path(tmpdir))
            engine = AvatarEngine(face_pack=pack)

            assert engine.face_pack.name == "Test Pack"
            assert engine.face_pack.colors.face == "#FFDDCC"
