"""Conversation engine for managing dialogue."""

from dataclasses import dataclass, field


@dataclass
class Message:
    """A message in a conversation."""

    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class ConversationEngine:
    """Manages conversation state and prompt building."""

    system_prompt: str = ""
    max_history: int = 10
    history: list[Message] = field(default_factory=list)
    _memory_context: str = field(default="", repr=False)

    def set_memory_context(self, context: str) -> None:
        """Set the memory context to inject into the system prompt.

        Args:
            context: Memory context string (topic index, memories, etc.).
        """
        self._memory_context = context

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation.

        Args:
            content: User message content.
        """
        self.history.append(Message(role="user", content=content))
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation.

        Args:
            content: Assistant message content.
        """
        self.history.append(Message(role="assistant", content=content))
        self._trim_history()

    def build_messages(self) -> list[Message]:
        """Build the full message list for the LLM.

        Returns:
            List of messages including system prompt and history.
        """
        messages: list[Message] = []

        # Build system prompt with optional memory context
        system = self.system_prompt
        if self._memory_context:
            system = system + "\n\n" + self._memory_context if system else self._memory_context

        if system:
            messages.append(Message(role="system", content=system))

        # Add conversation history
        messages.extend(self.history)

        return messages

    def get_last_assistant_message(self) -> str | None:
        """Get the last assistant message content.

        Returns:
            Last assistant message content, or None if no assistant messages.
        """
        for msg in reversed(self.history):
            if msg.role == "assistant":
                return msg.content
        return None

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history.clear()

    def _trim_history(self) -> None:
        """Trim history to max_history messages."""
        if len(self.history) > self.max_history:
            # Keep only the most recent messages
            self.history = self.history[-self.max_history :]

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "ConversationEngine":
        """Create conversation engine from configuration.

        Args:
            config: Configuration dictionary with persona settings.

        Returns:
            Configured ConversationEngine instance.
        """
        persona = config.get("persona", {})
        if isinstance(persona, dict):
            system_prompt = str(persona.get("system_prompt", ""))
        else:
            system_prompt = ""

        return cls(system_prompt=system_prompt)
