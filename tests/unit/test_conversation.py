"""Tests for conversation module."""

from rostro.conversation.engine import ConversationEngine, Message


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestConversationEngine:
    """Tests for ConversationEngine."""

    def test_default_values(self):
        """Test default initialization."""
        engine = ConversationEngine()
        assert engine.system_prompt == ""
        assert engine.max_history == 10
        assert len(engine.history) == 0

    def test_add_user_message(self):
        """Test adding a user message."""
        engine = ConversationEngine()
        engine.add_user_message("Hello")

        assert len(engine.history) == 1
        assert engine.history[0].role == "user"
        assert engine.history[0].content == "Hello"

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        engine = ConversationEngine()
        engine.add_assistant_message("Hi there!")

        assert len(engine.history) == 1
        assert engine.history[0].role == "assistant"
        assert engine.history[0].content == "Hi there!"

    def test_build_messages_empty(self):
        """Test building messages with empty history."""
        engine = ConversationEngine()
        messages = engine.build_messages()
        assert len(messages) == 0

    def test_build_messages_with_system_prompt(self):
        """Test building messages with system prompt."""
        engine = ConversationEngine(system_prompt="You are helpful.")
        engine.add_user_message("Hello")

        messages = engine.build_messages()
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "You are helpful."
        assert messages[1].role == "user"

    def test_build_messages_preserves_order(self):
        """Test that messages are in correct order."""
        engine = ConversationEngine()
        engine.add_user_message("First")
        engine.add_assistant_message("Second")
        engine.add_user_message("Third")

        messages = engine.build_messages()
        assert len(messages) == 3
        assert messages[0].content == "First"
        assert messages[1].content == "Second"
        assert messages[2].content == "Third"

    def test_clear_history(self):
        """Test clearing history."""
        engine = ConversationEngine()
        engine.add_user_message("Hello")
        engine.add_assistant_message("Hi")

        engine.clear_history()
        assert len(engine.history) == 0

    def test_history_trimming(self):
        """Test that history is trimmed to max_history."""
        engine = ConversationEngine(max_history=3)

        for i in range(5):
            engine.add_user_message(f"Message {i}")

        assert len(engine.history) == 3
        assert engine.history[0].content == "Message 2"
        assert engine.history[2].content == "Message 4"

    def test_from_config(self):
        """Test creating from config dict."""
        config = {"persona": {"system_prompt": "You are a friendly assistant."}}
        engine = ConversationEngine.from_config(config)
        assert engine.system_prompt == "You are a friendly assistant."

    def test_from_config_empty(self):
        """Test creating from empty config."""
        engine = ConversationEngine.from_config({})
        assert engine.system_prompt == ""
