"""Memory manager — orchestrates all memory components."""

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from rostro.memory.context_builder import ContextBuilder
from rostro.memory.digester import Digester
from rostro.memory.general_store import GeneralStore
from rostro.memory.session_timer import SessionTimer
from rostro.memory.topic_detector import TopicDetector
from rostro.memory.topic_store import TopicStore
from rostro.providers.base import EmbeddingProvider, Message


@dataclass
class MemoryConfig:
    """Memory system configuration."""

    session_timeout_minutes: float = 7
    topics_dir: Path = field(default_factory=lambda: Path("data/topics"))
    db_path: Path = field(default_factory=lambda: Path("data/memory.db"))
    topic_split_threshold_lines: int = 50
    min_conclusions_for_new_topic: int = 3
    embedding_similarity_threshold: float = 0.7
    max_memories_per_search: int = 5
    index_max_entries: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryConfig":
        return cls(
            session_timeout_minutes=data.get("session_timeout_minutes", 7),
            topics_dir=Path(data.get("topics_dir", "data/topics")),
            db_path=Path(data.get("db_path", "data/memory.db")),
            topic_split_threshold_lines=data.get("topic_split_threshold_lines", 50),
            min_conclusions_for_new_topic=data.get("min_conclusions_for_new_topic", 3),
            embedding_similarity_threshold=data.get("embedding_similarity_threshold", 0.7),
            max_memories_per_search=data.get("max_memories_per_search", 5),
            index_max_entries=data.get("index_max_entries", 20),
        )


class MemoryManager:
    """Orchestrates all memory components. Single entry point for controller."""

    def __init__(self, config: MemoryConfig, llm: object, embedder: EmbeddingProvider) -> None:
        self._config = config
        self._llm = llm

        self._general = GeneralStore(
            db_path=config.db_path,
            embedder=embedder,
            similarity_threshold=config.embedding_similarity_threshold,
            max_results=config.max_memories_per_search,
        )
        self._topics = TopicStore(
            topics_dir=config.topics_dir,
            split_threshold=config.topic_split_threshold_lines,
        )
        self._detector = TopicDetector(llm=llm)
        self._digester = Digester(llm=llm)
        self._context = ContextBuilder()
        self._timer = SessionTimer(
            timeout_seconds=config.session_timeout_minutes * 60,
            on_timeout=self._on_session_timeout,
        )

        self._active_topic: str | None = None
        self._history: list[dict[str, str]] = []
        self._lock = threading.Lock()

        self._load_topic_index()

    def on_user_message(self, text: str) -> None:
        """Called for each user message. Triggers memory search and topic detection."""
        self._timer.reset()

        with self._lock:
            if self._active_topic is None:
                # Search general memory
                results = self._general.search(text)
                if results:
                    self._context.add_memories(results)

                # Try to detect topic
                existing = self._topics.list_topics()
                topic = self._detector.detect(text, existing)
                if topic is not None:
                    self._active_topic = topic
                    content = self._topics.read(topic)
                    if content is not None:
                        self._context.set_topic_content(topic, content)

    def get_context(self) -> str:
        """Returns the memory context string to inject into system prompt."""
        with self._lock:
            return self._context.build()

    def set_history(self, history: list[dict[str, str]]) -> None:
        """Update the conversation history reference for digestion."""
        with self._lock:
            self._history = history

    def stop(self) -> None:
        """Stop the session timer."""
        self._timer.stop()

    def _on_session_timeout(self) -> None:
        """Called when session times out. Digest and clear."""
        with self._lock:
            if not self._history:
                self._reset_session()
                return

            conclusions = self._digester.extract(self._history)

            if conclusions:
                for conclusion in conclusions:
                    self._general.save(conclusion, topic=self._active_topic)

                if self._active_topic is not None:
                    for conclusion in conclusions:
                        self._topics.append(self._active_topic, conclusion)
                    if self._topics.needs_split(self._active_topic):
                        self._trigger_split(self._active_topic)
                elif len(conclusions) >= self._config.min_conclusions_for_new_topic:
                    conversation_text = " ".join(
                        msg["content"] for msg in self._history if msg["role"] == "user"
                    )
                    topic = self._detector.detect(conversation_text, self._topics.list_topics())
                    if topic:
                        for conclusion in conclusions:
                            self._topics.append(topic, conclusion)

                self._update_topic_index(conclusions)

            self._reset_session()

    def _reset_session(self) -> None:
        """Clear session state for fresh conversation."""
        self._active_topic = None
        self._history = []
        self._context.reset()
        self._load_topic_index()

    def _load_topic_index(self) -> None:
        """Load topic index from file."""
        index_path = self._config.topics_dir / "index.txt"
        if not index_path.exists():
            return
        entries = []
        for line in index_path.read_text().strip().splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                entries.append((parts[0], parts[1], parts[2]))
        self._context.set_topic_index(entries)

    def _update_topic_index(self, conclusions: list[str]) -> None:
        """Update the topic index file after digestion."""
        topic = self._active_topic or "general"
        summary = conclusions[0] if conclusions else "conversation"
        if len(summary) > 60:
            summary = summary[:57] + "..."

        index_path = self._config.topics_dir / "index.txt"
        index_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        if index_path.exists():
            lines = index_path.read_text().strip().splitlines()

        now = datetime.now(UTC).isoformat()
        new_line = f"{now}|{topic}|{summary}"
        lines.insert(0, new_line)
        lines = lines[: self._config.index_max_entries]
        index_path.write_text("\n".join(lines) + "\n")

    def _trigger_split(self, topic: str) -> None:
        """Trigger async topic file split via LLM."""
        thread = threading.Thread(target=self._do_split, args=(topic,), daemon=True)
        thread.start()

    def _do_split(self, topic: str) -> None:
        """Perform topic split in background thread."""
        content = self._topics.read(topic)
        if content is None:
            return

        prompt = (
            "Split this topic file into 2-4 subtopics. "
            "Group related entries together. Use the original topic as prefix "
            f"(e.g. '{topic}-subtopic').\n\n"
            "Respond ONLY with JSON:\n"
            '{"subtopics": {"name": ["entry1", "entry2"], ...}}\n\n'
            f"Content:\n{content}"
        )
        messages = [Message(role="system", content=prompt)]
        try:
            response = self._llm.complete(messages)  # type: ignore[attr-defined]
            data = json.loads(response)
            subtopics_data = data.get("subtopics", {})
            if not subtopics_data:
                return

            today = date.today().isoformat()
            subtopics: dict[str, str] = {}
            for name, entries in subtopics_data.items():
                title = name.replace("-", " ").title()
                entry_lines = [f"- {e} ({today})" for e in entries]
                subtopics[name] = f"# {title}\n\n" + "\n".join(entry_lines) + "\n"

            with self._lock:
                self._topics.replace_with_subtopics(topic, subtopics)
        except Exception as e:
            print(f"[Memory] Topic split failed for '{topic}': {e}")
