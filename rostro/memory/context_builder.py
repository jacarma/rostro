"""Context builder — assembles dynamic system prompt layers."""

from datetime import datetime, timezone

from rostro.memory.general_store import MemoryResult

MEMORY_INSTRUCTIONS = (
    "You have persistent memory. When the user shares decisions, "
    "discoveries, or conclusions, acknowledge them naturally. You may "
    "also propose your own observations — if the user confirms them, "
    "they will be saved as memory."
)


class ContextBuilder:
    """Builds the memory-related portion of the system prompt."""

    def __init__(self) -> None:
        self._topic_index: list[tuple[str, str, str]] = []
        self._memories: dict[str, MemoryResult] = {}
        self._topic_name: str | None = None
        self._topic_content: str | None = None

    def set_topic_index(self, entries: list[tuple[str, str, str]]) -> None:
        self._topic_index = entries

    def add_memories(self, memories: list[MemoryResult]) -> None:
        for m in memories:
            if m.text not in self._memories:
                self._memories[m.text] = m

    def set_memories(self, memories: list[MemoryResult]) -> None:
        self._memories.clear()
        self.add_memories(memories)

    def set_topic_content(self, topic: str, content: str) -> None:
        self._topic_name = topic
        self._topic_content = content
        self._memories.clear()

    def reset(self) -> None:
        self._topic_index = []
        self._memories.clear()
        self._topic_name = None
        self._topic_content = None

    def build(self) -> str:
        parts: list[str] = []

        if self._topic_index:
            lines = []
            for ts, topic, summary in self._topic_index:
                relative = self.to_relative_time(ts)
                lines.append(f"- {topic} ({relative}): {summary}")
            parts.append("Recent topics:\n" + "\n".join(lines))

        if self._topic_content is not None:
            parts.append(f"Context about {self._topic_name}:\n{self._topic_content}")
        elif self._memories:
            lines = [f"- {m.text}" for m in self._memories.values()]
            parts.append("Things you know about the user:\n" + "\n".join(lines))

        parts.append(MEMORY_INSTRUCTIONS)

        return "\n\n".join(parts)

    @staticmethod
    def to_relative_time(iso_timestamp: str) -> str:
        try:
            ts = datetime.fromisoformat(iso_timestamp)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = now - ts
            seconds = int(delta.total_seconds())
            if seconds < 60:
                return "just now"
            elif seconds < 3600:
                mins = seconds // 60
                return f"{mins} minute{'s' if mins != 1 else ''} ago"
            elif seconds < 86400:
                hours = seconds // 3600
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
            else:
                days = seconds // 86400
                return f"{days} day{'s' if days != 1 else ''} ago"
        except Exception:
            return "recently"
