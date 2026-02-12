# Memory System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add persistent memory to Rostro so it remembers conclusions, decisions, and discoveries across conversations.

**Architecture:** Three memory layers — topic files (markdown), general memory (SQLite + embeddings), and a recent topics index. A session timer triggers digestion (extract conclusions via LLM) on inactivity, then clears conversation history. See `docs/plans/2026-02-12-memory-system-design.md` for full design.

**Tech Stack:** Python 3.11+, SQLite (stdlib), numpy (existing dep), OpenAI embeddings API (existing provider), GPT-4o-mini (existing provider)

---

### Task 1: GeneralStore — SQLite + embeddings

The foundation. Other modules depend on this for saving and searching memories.

**Files:**
- Create: `rostro/memory/general_store.py`
- Create: `tests/unit/test_memory_general_store.py`

**Step 1: Write the failing tests**

```python
"""Tests for general memory store."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rostro.memory.general_store import GeneralStore


@pytest.fixture
def mock_embedder():
    """Mock embedding provider returning fixed-size vectors."""
    embedder = MagicMock()
    # Return a normalized vector so cosine similarity works predictably
    vec = np.random.randn(1536).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    embedder.embed.return_value = vec.tolist()
    return embedder


@pytest.fixture
def store(tmp_path, mock_embedder):
    """Create a GeneralStore with temp database."""
    db_path = tmp_path / "memory.db"
    return GeneralStore(
        db_path=db_path,
        embedder=mock_embedder,
        similarity_threshold=0.7,
        max_results=5,
    )


class TestGeneralStore:
    def test_init_creates_db_and_table(self, store, tmp_path):
        db_path = tmp_path / "memory.db"
        assert db_path.exists()
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor]
        conn.close()
        assert "memories" in tables

    def test_save_memory(self, store, mock_embedder):
        store.save("Likes bomba rice", topic="cooking")
        mock_embedder.embed.assert_called_once_with("Likes bomba rice")
        assert store.count() == 1

    def test_save_multiple(self, store):
        store.save("Fact one")
        store.save("Fact two", topic="cooking")
        assert store.count() == 2

    def test_search_returns_results(self, store, mock_embedder):
        # Make embed return the same vector for both save and search
        # so cosine similarity = 1.0
        vec = np.random.randn(1536).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        mock_embedder.embed.return_value = vec.tolist()

        store.save("Allergic to tree nuts")
        results = store.search("nut allergy")
        assert len(results) >= 1
        assert results[0].text == "Allergic to tree nuts"

    def test_search_empty_db(self, store):
        results = store.search("anything")
        assert results == []

    def test_search_respects_max_results(self, store, mock_embedder):
        vec = np.random.randn(1536).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        mock_embedder.embed.return_value = vec.tolist()

        for i in range(10):
            store.save(f"Memory {i}")
        results = store.search("query")
        assert len(results) <= 5

    def test_search_result_has_fields(self, store, mock_embedder):
        vec = np.random.randn(1536).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        mock_embedder.embed.return_value = vec.tolist()

        store.save("Prefers homemade broth", topic="cooking")
        results = store.search("broth")
        assert len(results) == 1
        result = results[0]
        assert result.text == "Prefers homemade broth"
        assert result.topic == "cooking"
        assert result.score > 0.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_memory_general_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rostro.memory.general_store'`

**Step 3: Write minimal implementation**

```python
"""General memory store with SQLite and embeddings."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass
class MemoryResult:
    """A memory search result."""

    text: str
    topic: str | None
    score: float
    created_at: str


class GeneralStore:
    """Long-term memory store using SQLite and vector embeddings."""

    def __init__(
        self,
        db_path: Path,
        embedder: object,
        similarity_threshold: float = 0.7,
        max_results: int = 5,
    ) -> None:
        self._db_path = db_path
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._max_results = max_results
        self._init_db()

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                topic TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        conn.commit()
        conn.close()

    def save(self, text: str, topic: str | None = None) -> None:
        embedding = self._embedder.embed(text)  # type: ignore[union-attr]
        blob = np.array(embedding, dtype=np.float32).tobytes()
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT INTO memories (text, embedding, topic, created_at) VALUES (?, ?, ?, ?)",
            (text, blob, topic, now),
        )
        conn.commit()
        conn.close()

    def search(self, query: str) -> list[MemoryResult]:
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute("SELECT text, embedding, topic, created_at FROM memories").fetchall()
        conn.close()

        if not rows:
            return []

        query_vec = np.array(self._embedder.embed(query), dtype=np.float32)  # type: ignore[union-attr]
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        results: list[MemoryResult] = []
        for text, blob, topic, created_at in rows:
            mem_vec = np.frombuffer(blob, dtype=np.float32)
            mem_norm = np.linalg.norm(mem_vec)
            if mem_norm == 0:
                continue
            score = float(np.dot(query_vec, mem_vec) / (query_norm * mem_norm))
            if score >= self._threshold:
                results.append(MemoryResult(text=text, topic=topic, score=score, created_at=created_at))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: self._max_results]

    def count(self) -> int:
        conn = sqlite3.connect(self._db_path)
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        return count
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_memory_general_store.py -v`
Expected: All PASS

**Step 5: Type check**

Run: `mypy rostro/memory/general_store.py`
Expected: Success

**Step 6: Commit**

```bash
git add rostro/memory/general_store.py tests/unit/test_memory_general_store.py
git commit -m "feat(memory): add GeneralStore with SQLite and embeddings"
```

---

### Task 2: TopicStore — CRUD for topic files

Manages topic markdown files: read, append, list, and auto-split.

**Files:**
- Create: `rostro/memory/topic_store.py`
- Create: `tests/unit/test_memory_topic_store.py`

**Step 1: Write the failing tests**

```python
"""Tests for topic file store."""

from pathlib import Path

import pytest

from rostro.memory.topic_store import TopicStore


@pytest.fixture
def store(tmp_path):
    return TopicStore(topics_dir=tmp_path, split_threshold=5)


class TestTopicStore:
    def test_list_topics_empty(self, store):
        assert store.list_topics() == []

    def test_append_creates_file(self, store, tmp_path):
        store.append("cooking", "Likes bomba rice")
        path = tmp_path / "cooking.md"
        assert path.exists()
        content = path.read_text()
        assert "Likes bomba rice" in content
        assert "# Cooking" in content

    def test_append_multiple_entries(self, store, tmp_path):
        store.append("cooking", "Likes bomba rice")
        store.append("cooking", "Uses homemade broth")
        content = (tmp_path / "cooking.md").read_text()
        assert "Likes bomba rice" in content
        assert "Uses homemade broth" in content

    def test_list_topics_returns_names(self, store):
        store.append("cooking", "entry")
        store.append("travel", "entry")
        topics = store.list_topics()
        assert sorted(topics) == ["cooking", "travel"]

    def test_read_topic(self, store):
        store.append("cooking", "Likes bomba rice")
        content = store.read("cooking")
        assert "Likes bomba rice" in content

    def test_read_nonexistent_topic(self, store):
        assert store.read("nonexistent") is None

    def test_needs_split_false_when_small(self, store):
        store.append("cooking", "One entry")
        assert not store.needs_split("cooking")

    def test_needs_split_true_when_over_threshold(self, store):
        for i in range(6):
            store.append("cooking", f"Entry {i}")
        assert store.needs_split("cooking")

    def test_line_count(self, store):
        store.append("cooking", "A")
        store.append("cooking", "B")
        assert store.line_count("cooking") == 4  # header + blank + 2 entries

    def test_replace_topic_with_subtopics(self, store, tmp_path):
        store.append("cooking", "Original entry")
        subtopics = {
            "cooking-baking": "# Cooking Baking\n\n- Baking entry",
            "cooking-rice": "# Cooking Rice\n\n- Rice entry",
        }
        store.replace_with_subtopics("cooking", subtopics)
        assert not (tmp_path / "cooking.md").exists()
        assert (tmp_path / "cooking-baking.md").exists()
        assert (tmp_path / "cooking-rice.md").exists()
        assert "cooking-baking" in store.list_topics()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_memory_topic_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Topic file store — CRUD for markdown topic files."""

from datetime import date
from pathlib import Path


class TopicStore:
    """Manages topic markdown files in a directory."""

    def __init__(self, topics_dir: Path, split_threshold: int = 50) -> None:
        self._dir = topics_dir
        self._split_threshold = split_threshold
        self._dir.mkdir(parents=True, exist_ok=True)

    def list_topics(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("*.md"))

    def read(self, topic: str) -> str | None:
        path = self._dir / f"{topic}.md"
        if not path.exists():
            return None
        return path.read_text()

    def append(self, topic: str, entry: str) -> None:
        path = self._dir / f"{topic}.md"
        today = date.today().isoformat()
        line = f"- {entry} ({today})\n"

        if not path.exists():
            title = topic.replace("-", " ").title()
            path.write_text(f"# {title}\n\n{line}")
        else:
            with open(path, "a") as f:
                f.write(line)

    def needs_split(self, topic: str) -> bool:
        return self.line_count(topic) > self._split_threshold

    def line_count(self, topic: str) -> int:
        path = self._dir / f"{topic}.md"
        if not path.exists():
            return 0
        return len(path.read_text().splitlines())

    def replace_with_subtopics(self, topic: str, subtopics: dict[str, str]) -> None:
        old_path = self._dir / f"{topic}.md"
        if old_path.exists():
            old_path.unlink()
        for name, content in subtopics.items():
            (self._dir / f"{name}.md").write_text(content)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_memory_topic_store.py -v`
Expected: All PASS

**Step 5: Type check and commit**

Run: `mypy rostro/memory/topic_store.py`

```bash
git add rostro/memory/topic_store.py tests/unit/test_memory_topic_store.py
git commit -m "feat(memory): add TopicStore for topic file CRUD"
```

---

### Task 3: TopicDetector — detect topic via LLM

Classifies user messages into existing topics or null.

**Files:**
- Create: `rostro/memory/topic_detector.py`
- Create: `tests/unit/test_memory_topic_detector.py`

**Step 1: Write the failing tests**

```python
"""Tests for topic detector."""

import json
from unittest.mock import MagicMock

import pytest

from rostro.memory.topic_detector import TopicDetector


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def detector(mock_llm):
    return TopicDetector(llm=mock_llm)


class TestTopicDetector:
    def test_detect_existing_topic(self, detector, mock_llm):
        mock_llm.complete.return_value = '{"topic": "cooking"}'
        result = detector.detect("I want to make paella", ["cooking", "travel"])
        assert result == "cooking"

    def test_detect_no_topic(self, detector, mock_llm):
        mock_llm.complete.return_value = '{"topic": null}'
        result = detector.detect("Hello, how are you?", ["cooking", "travel"])
        assert result is None

    def test_detect_new_topic(self, detector, mock_llm):
        mock_llm.complete.return_value = '{"topic": "fitness"}'
        result = detector.detect("I started going to the gym", ["cooking", "travel"])
        assert result == "fitness"

    def test_detect_with_no_existing_topics(self, detector, mock_llm):
        mock_llm.complete.return_value = '{"topic": "cooking"}'
        result = detector.detect("Let me tell you about my recipe", [])
        assert result == "cooking"

    def test_detect_handles_malformed_json(self, detector, mock_llm):
        mock_llm.complete.return_value = "not json at all"
        result = detector.detect("test message", [])
        assert result is None

    def test_detect_handles_llm_error(self, detector, mock_llm):
        mock_llm.complete.side_effect = Exception("API error")
        result = detector.detect("test message", [])
        assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_memory_topic_detector.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Topic detector — classifies user messages into topics via LLM."""

import json

from rostro.providers.base import Message


class TopicDetector:
    """Detects conversation topic from user messages."""

    def __init__(self, llm: object) -> None:
        self._llm = llm

    def detect(self, user_message: str, existing_topics: list[str]) -> str | None:
        topics_str = ", ".join(existing_topics) if existing_topics else "(none yet)"
        prompt = (
            "You classify user messages into conversation topics. "
            "Existing topics: " + topics_str + "\n\n"
            "If the message clearly relates to an existing topic, return it. "
            "If it's a new distinct topic, return a short lowercase slug (e.g. 'fitness', 'home-renovation'). "
            "If the message is too generic or casual (greetings, small talk), return null.\n\n"
            "Respond ONLY with JSON: {\"topic\": \"name\"} or {\"topic\": null}"
        )
        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=user_message),
        ]
        try:
            response = self._llm.complete(messages)  # type: ignore[union-attr]
            data = json.loads(response)
            topic = data.get("topic")
            return topic if isinstance(topic, str) else None
        except Exception:
            return None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_memory_topic_detector.py -v`
Expected: All PASS

**Step 5: Type check and commit**

Run: `mypy rostro/memory/topic_detector.py`

```bash
git add rostro/memory/topic_detector.py tests/unit/test_memory_topic_detector.py
git commit -m "feat(memory): add TopicDetector for LLM-based topic classification"
```

---

### Task 4: Digester — extract conclusions from conversation history

Takes conversation history, asks GPT-4o-mini to extract conclusions.

**Files:**
- Create: `rostro/memory/digester.py`
- Create: `tests/unit/test_memory_digester.py`

**Step 1: Write the failing tests**

```python
"""Tests for conversation digester."""

from unittest.mock import MagicMock

import pytest

from rostro.memory.digester import Digester


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def digester(mock_llm):
    return Digester(llm=mock_llm)


class TestDigester:
    def test_extract_conclusions(self, digester, mock_llm):
        mock_llm.complete.return_value = (
            '["Prefers bomba rice for paella", "Allergic to tree nuts"]'
        )
        history = [
            {"role": "user", "content": "I always use bomba rice"},
            {"role": "assistant", "content": "Great choice!"},
            {"role": "user", "content": "I'm allergic to tree nuts"},
            {"role": "assistant", "content": "I'll remember that."},
        ]
        result = digester.extract(history)
        assert len(result) == 2
        assert "bomba rice" in result[0]

    def test_extract_empty_history(self, digester):
        result = digester.extract([])
        assert result == []

    def test_extract_no_conclusions(self, digester, mock_llm):
        mock_llm.complete.return_value = "[]"
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = digester.extract(history)
        assert result == []

    def test_extract_handles_malformed_json(self, digester, mock_llm):
        mock_llm.complete.return_value = "not json"
        history = [{"role": "user", "content": "test"}]
        result = digester.extract(history)
        assert result == []

    def test_extract_handles_llm_error(self, digester, mock_llm):
        mock_llm.complete.side_effect = Exception("API error")
        history = [{"role": "user", "content": "test"}]
        result = digester.extract(history)
        assert result == []

    def test_extract_includes_confirmed_assistant_observations(self, digester, mock_llm):
        mock_llm.complete.return_value = '["Enjoys cooking with fresh ingredients"]'
        history = [
            {"role": "assistant", "content": "It seems you enjoy cooking with fresh ingredients"},
            {"role": "user", "content": "Yes, exactly!"},
        ]
        result = digester.extract(history)
        assert len(result) == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_memory_digester.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Digester — extracts conclusions from conversation history via LLM."""

import json

from rostro.providers.base import Message


class Digester:
    """Extracts memorable conclusions from conversation history."""

    def __init__(self, llm: object) -> None:
        self._llm = llm

    def extract(self, history: list[dict[str, str]]) -> list[str]:
        if not history:
            return []

        conversation = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in history
        )
        prompt = (
            "Analyze this conversation and extract conclusions, decisions, or discoveries. "
            "Include observations made by the assistant that the user explicitly confirmed or approved. "
            "Ignore obvious facts, common sense, and small talk. "
            "Only extract things worth remembering long-term.\n\n"
            "Respond ONLY with a JSON array of short strings. "
            'Example: ["Prefers bomba rice for paella", "Allergic to tree nuts"]\n'
            "If nothing is worth remembering, respond with []"
        )
        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=conversation),
        ]
        try:
            response = self._llm.complete(messages)  # type: ignore[union-attr]
            conclusions = json.loads(response)
            if isinstance(conclusions, list):
                return [str(c) for c in conclusions if c]
            return []
        except Exception:
            return []
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_memory_digester.py -v`
Expected: All PASS

**Step 5: Type check and commit**

Run: `mypy rostro/memory/digester.py`

```bash
git add rostro/memory/digester.py tests/unit/test_memory_digester.py
git commit -m "feat(memory): add Digester for extracting conclusions from history"
```

---

### Task 5: SessionTimer — inactivity timer

Resets on each user message. Fires callback on timeout.

**Files:**
- Create: `rostro/memory/session_timer.py`
- Create: `tests/unit/test_memory_session_timer.py`

**Step 1: Write the failing tests**

```python
"""Tests for session timer."""

import time
from unittest.mock import MagicMock

import pytest

from rostro.memory.session_timer import SessionTimer


class TestSessionTimer:
    def test_callback_fires_after_timeout(self):
        callback = MagicMock()
        timer = SessionTimer(timeout_seconds=0.1, on_timeout=callback)
        timer.reset()
        time.sleep(0.3)
        callback.assert_called_once()
        timer.stop()

    def test_reset_postpones_callback(self):
        callback = MagicMock()
        timer = SessionTimer(timeout_seconds=0.2, on_timeout=callback)
        timer.reset()
        time.sleep(0.1)
        timer.reset()  # restart the timer
        time.sleep(0.1)
        callback.assert_not_called()  # hasn't fired yet
        time.sleep(0.2)
        callback.assert_called_once()
        timer.stop()

    def test_stop_cancels_timer(self):
        callback = MagicMock()
        timer = SessionTimer(timeout_seconds=0.1, on_timeout=callback)
        timer.reset()
        timer.stop()
        time.sleep(0.3)
        callback.assert_not_called()

    def test_is_active(self):
        callback = MagicMock()
        timer = SessionTimer(timeout_seconds=1.0, on_timeout=callback)
        assert not timer.is_active
        timer.reset()
        assert timer.is_active
        timer.stop()
        assert not timer.is_active

    def test_no_callback_without_reset(self):
        callback = MagicMock()
        timer = SessionTimer(timeout_seconds=0.1, on_timeout=callback)
        time.sleep(0.3)
        callback.assert_not_called()
        timer.stop()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_memory_session_timer.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Session timer — fires callback after inactivity timeout."""

import threading
from collections.abc import Callable


class SessionTimer:
    """Timer that fires a callback after a period of inactivity."""

    def __init__(self, timeout_seconds: float, on_timeout: Callable[[], None]) -> None:
        self._timeout = timeout_seconds
        self._on_timeout = on_timeout
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._timer is not None and self._timer.is_alive()

    def reset(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._timeout, self._on_timeout)
            self._timer.daemon = True
            self._timer.start()

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_memory_session_timer.py -v`
Expected: All PASS

**Step 5: Type check and commit**

Run: `mypy rostro/memory/session_timer.py`

```bash
git add rostro/memory/session_timer.py tests/unit/test_memory_session_timer.py
git commit -m "feat(memory): add SessionTimer for inactivity detection"
```

---

### Task 6: ContextBuilder — assemble system prompt layers

Builds the dynamic system prompt with topic index, memories, topic file, and memory instructions.

**Files:**
- Create: `rostro/memory/context_builder.py`
- Create: `tests/unit/test_memory_context_builder.py`

**Step 1: Write the failing tests**

```python
"""Tests for context builder."""

from datetime import datetime, timedelta, timezone

import pytest

from rostro.memory.context_builder import ContextBuilder
from rostro.memory.general_store import MemoryResult


class TestContextBuilder:
    def test_build_with_no_context(self):
        builder = ContextBuilder()
        result = builder.build()
        # Should still have memory instructions
        assert "persistent memory" in result.lower()

    def test_build_with_topic_index(self):
        builder = ContextBuilder()
        now = datetime.now(timezone.utc)
        one_hour_ago = (now - timedelta(hours=1)).isoformat()
        builder.set_topic_index([(one_hour_ago, "cooking", "rice recipes")])
        result = builder.build()
        assert "cooking" in result
        assert "rice recipes" in result

    def test_build_with_memories(self):
        builder = ContextBuilder()
        memories = [
            MemoryResult(text="Allergic to tree nuts", topic=None, score=0.9, created_at="2026-01-01"),
        ]
        builder.set_memories(memories)
        result = builder.build()
        assert "Allergic to tree nuts" in result

    def test_add_memories_accumulates_no_duplicates(self):
        builder = ContextBuilder()
        m1 = MemoryResult(text="Fact A", topic=None, score=0.9, created_at="2026-01-01")
        m2 = MemoryResult(text="Fact B", topic=None, score=0.8, created_at="2026-01-01")
        m3 = MemoryResult(text="Fact A", topic=None, score=0.95, created_at="2026-01-01")  # duplicate
        builder.add_memories([m1, m2])
        builder.add_memories([m3])
        result = builder.build()
        assert result.count("Fact A") == 1
        assert "Fact B" in result

    def test_build_with_topic_file(self):
        builder = ContextBuilder()
        builder.set_topic_content("cooking", "# Cooking\n\n- Likes bomba rice")
        result = builder.build()
        assert "Likes bomba rice" in result

    def test_topic_file_clears_memories_section(self):
        builder = ContextBuilder()
        m1 = MemoryResult(text="Old memory", topic=None, score=0.9, created_at="2026-01-01")
        builder.add_memories([m1])
        builder.set_topic_content("cooking", "# Cooking\n\n- Topic content here")
        result = builder.build()
        # Topic content should be present, memories section should not
        assert "Topic content here" in result
        assert "Old memory" not in result

    def test_timestamp_to_relative(self):
        now = datetime.now(timezone.utc)
        one_hour_ago = (now - timedelta(hours=1)).isoformat()
        relative = ContextBuilder.to_relative_time(one_hour_ago)
        assert "hour" in relative or "minute" in relative

    def test_reset_clears_all(self):
        builder = ContextBuilder()
        m1 = MemoryResult(text="Fact", topic=None, score=0.9, created_at="2026-01-01")
        builder.add_memories([m1])
        builder.set_topic_content("cooking", "content")
        builder.reset()
        result = builder.build()
        assert "Fact" not in result
        assert "cooking" not in result.lower() or "cooking" not in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_memory_context_builder.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
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
        self._topic_index: list[tuple[str, str, str]] = []  # (timestamp, topic, summary)
        self._memories: dict[str, MemoryResult] = {}  # text -> result (dedup key)
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
        self._memories.clear()  # topic replaces memories

    def reset(self) -> None:
        self._topic_index = []
        self._memories.clear()
        self._topic_name = None
        self._topic_content = None

    def build(self) -> str:
        parts: list[str] = []

        # Layer 2: Topic index
        if self._topic_index:
            lines = []
            for ts, topic, summary in self._topic_index:
                relative = self.to_relative_time(ts)
                lines.append(f"- {topic} ({relative}): {summary}")
            parts.append("Recent topics:\n" + "\n".join(lines))

        # Layer 3 or 4: Memories or Topic file
        if self._topic_content is not None:
            parts.append(f"Context about {self._topic_name}:\n{self._topic_content}")
        elif self._memories:
            lines = [f"- {m.text}" for m in self._memories.values()]
            parts.append("Things you know about the user:\n" + "\n".join(lines))

        # Layer 5: Memory instructions
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_memory_context_builder.py -v`
Expected: All PASS

**Step 5: Type check and commit**

Run: `mypy rostro/memory/context_builder.py`

```bash
git add rostro/memory/context_builder.py tests/unit/test_memory_context_builder.py
git commit -m "feat(memory): add ContextBuilder for dynamic system prompt assembly"
```

---

### Task 7: MemoryManager — orchestrator

Ties all memory components together. Single entry point for the controller.

**Files:**
- Create: `rostro/memory/manager.py`
- Create: `tests/unit/test_memory_manager.py`
- Modify: `rostro/memory/__init__.py`

**Step 1: Write the failing tests**

```python
"""Tests for memory manager."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rostro.memory.manager import MemoryConfig, MemoryManager


@pytest.fixture
def memory_config(tmp_path):
    return MemoryConfig(
        session_timeout_minutes=0.01,  # very short for testing
        topics_dir=tmp_path / "topics",
        db_path=tmp_path / "memory.db",
        topic_split_threshold_lines=50,
        min_conclusions_for_new_topic=3,
        embedding_similarity_threshold=0.7,
        max_memories_per_search=5,
        index_max_entries=20,
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.complete.return_value = "[]"
    return llm


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    vec = np.random.randn(1536).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    embedder.embed.return_value = vec.tolist()
    return embedder


@pytest.fixture
def manager(memory_config, mock_llm, mock_embedder):
    m = MemoryManager(config=memory_config, llm=mock_llm, embedder=mock_embedder)
    yield m
    m.stop()


class TestMemoryConfig:
    def test_from_dict(self):
        config = MemoryConfig.from_dict({"session_timeout_minutes": 10})
        assert config.session_timeout_minutes == 10

    def test_from_dict_defaults(self):
        config = MemoryConfig.from_dict({})
        assert config.session_timeout_minutes == 7
        assert config.topic_split_threshold_lines == 50


class TestMemoryManager:
    def test_get_context_returns_string(self, manager):
        result = manager.get_context()
        assert isinstance(result, str)
        assert "persistent memory" in result.lower()

    def test_on_user_message_resets_timer(self, manager):
        manager.on_user_message("Hello")
        assert manager._timer.is_active

    def test_on_user_message_triggers_topic_detection(self, manager, mock_llm):
        mock_llm.complete.return_value = '{"topic": null}'
        manager.on_user_message("Hello")
        # Topic detector should have been called
        assert mock_llm.complete.called

    def test_topic_detected_stops_searching(self, manager, mock_llm):
        mock_llm.complete.return_value = '{"topic": "cooking"}'
        manager.on_user_message("Let me tell you about paella")
        assert manager._active_topic == "cooking"
        # Second message should NOT trigger topic detection
        mock_llm.complete.reset_mock()
        manager.on_user_message("I use bomba rice")
        # Only the digester/other calls, not topic detection
        # (topic detection is skipped when _active_topic is set)

    def test_topic_index_loaded_on_init(self, manager, memory_config):
        # Write an index file
        topics_dir = memory_config.topics_dir
        topics_dir.mkdir(parents=True, exist_ok=True)
        index_path = topics_dir / "index.txt"
        index_path.write_text("2026-02-12T10:00:00|cooking|rice recipes\n")
        # Reload
        manager._load_topic_index()
        context = manager.get_context()
        assert "cooking" in context

    def test_set_conversation_history(self, manager):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        manager.set_history(history)
        assert manager._history == history
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_memory_manager.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Memory manager — orchestrates all memory components."""

import threading
from dataclasses import dataclass, field
from pathlib import Path

from rostro.memory.context_builder import ContextBuilder
from rostro.memory.digester import Digester
from rostro.memory.general_store import GeneralStore
from rostro.memory.session_timer import SessionTimer
from rostro.memory.topic_detector import TopicDetector
from rostro.memory.topic_store import TopicStore


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
    def from_dict(cls, data: dict) -> "MemoryConfig":
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

    def __init__(self, config: MemoryConfig, llm: object, embedder: object) -> None:
        self._config = config
        self._llm = llm

        # Sub-components
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

        # State
        self._active_topic: str | None = None
        self._history: list[dict[str, str]] = []
        self._lock = threading.Lock()

        # Load topic index on init
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

            # Extract conclusions
            conclusions = self._digester.extract(self._history)

            if conclusions:
                # Save to general memory (always)
                for conclusion in conclusions:
                    self._general.save(conclusion, topic=self._active_topic)

                # Save to topic file
                if self._active_topic is not None:
                    for conclusion in conclusions:
                        self._topics.append(self._active_topic, conclusion)
                    # Check if split needed
                    if self._topics.needs_split(self._active_topic):
                        self._trigger_split(self._active_topic)
                elif len(conclusions) >= self._config.min_conclusions_for_new_topic:
                    # Create new topic via detector
                    conversation_text = " ".join(
                        msg["content"] for msg in self._history if msg["role"] == "user"
                    )
                    topic = self._detector.detect(conversation_text, self._topics.list_topics())
                    if topic:
                        for conclusion in conclusions:
                            self._topics.append(topic, conclusion)

                # Update topic index
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
        from datetime import datetime, timezone

        topic = self._active_topic or "general"
        summary = conclusions[0] if conclusions else "conversation"
        if len(summary) > 60:
            summary = summary[:57] + "..."

        index_path = self._config.topics_dir / "index.txt"
        index_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        if index_path.exists():
            lines = index_path.read_text().strip().splitlines()

        now = datetime.now(timezone.utc).isoformat()
        new_line = f"{now}|{topic}|{summary}"
        lines.insert(0, new_line)
        lines = lines[: self._config.index_max_entries]
        index_path.write_text("\n".join(lines) + "\n")

    def _trigger_split(self, topic: str) -> None:
        """Trigger async topic file split. Placeholder for Task 8."""
        # TODO: implement async split with LLM
        pass
```

**Step 4: Update `__init__.py`**

```python
"""Memory system for persistent knowledge across conversations."""

from rostro.memory.manager import MemoryConfig, MemoryManager

__all__ = ["MemoryConfig", "MemoryManager"]
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_memory_manager.py -v`
Expected: All PASS

**Step 6: Run all memory tests**

Run: `pytest tests/unit/test_memory_*.py -v`
Expected: All PASS

**Step 7: Type check and commit**

Run: `mypy rostro/memory/`

```bash
git add rostro/memory/ tests/unit/test_memory_manager.py
git commit -m "feat(memory): add MemoryManager orchestrator and update module exports"
```

---

### Task 8: Topic split — async LLM-based splitting

When a topic file exceeds the threshold, ask LLM to split it into subtopics.

**Files:**
- Modify: `rostro/memory/manager.py` (replace `_trigger_split` placeholder)
- Create: `tests/unit/test_memory_topic_split.py`

**Step 1: Write the failing test**

```python
"""Tests for topic file splitting."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rostro.memory.manager import MemoryConfig, MemoryManager


@pytest.fixture
def memory_config(tmp_path):
    return MemoryConfig(
        session_timeout_minutes=100,
        topics_dir=tmp_path / "topics",
        db_path=tmp_path / "memory.db",
        topic_split_threshold_lines=5,
    )


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    vec = np.random.randn(1536).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    embedder.embed.return_value = vec.tolist()
    return embedder


class TestTopicSplit:
    def test_trigger_split_creates_subtopics(self, memory_config, mock_llm, mock_embedder):
        manager = MemoryManager(config=memory_config, llm=mock_llm, embedder=mock_embedder)
        topics_dir = memory_config.topics_dir

        # Create a topic file that exceeds threshold
        for i in range(8):
            manager._topics.append("cooking", f"Entry {i}")

        assert manager._topics.needs_split("cooking")

        # Mock LLM to return split plan
        split_response = json.dumps({
            "subtopics": {
                "cooking-baking": ["Entry 0", "Entry 1", "Entry 2"],
                "cooking-rice": ["Entry 3", "Entry 4", "Entry 5"],
                "cooking-general": ["Entry 6", "Entry 7"],
            }
        })
        mock_llm.complete.return_value = split_response

        manager._trigger_split("cooking")

        # Wait for async split (give thread time)
        import time
        time.sleep(0.5)

        assert not (topics_dir / "cooking.md").exists()
        assert (topics_dir / "cooking-baking.md").exists()
        assert (topics_dir / "cooking-rice.md").exists()
        manager.stop()

    def test_split_handles_llm_error(self, memory_config, mock_llm, mock_embedder):
        manager = MemoryManager(config=memory_config, llm=mock_llm, embedder=mock_embedder)

        for i in range(8):
            manager._topics.append("cooking", f"Entry {i}")

        mock_llm.complete.side_effect = Exception("API error")
        manager._trigger_split("cooking")

        import time
        time.sleep(0.5)

        # Original file should still exist
        assert (memory_config.topics_dir / "cooking.md").exists()
        manager.stop()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_memory_topic_split.py -v`
Expected: FAIL — `_trigger_split` is a no-op

**Step 3: Replace `_trigger_split` in `manager.py`**

Replace the placeholder `_trigger_split` method with:

```python
    def _trigger_split(self, topic: str) -> None:
        """Trigger async topic file split via LLM."""
        thread = threading.Thread(target=self._do_split, args=(topic,), daemon=True)
        thread.start()

    def _do_split(self, topic: str) -> None:
        """Perform topic split in background thread."""
        import json
        from rostro.providers.base import Message

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
            response = self._llm.complete(messages)  # type: ignore[union-attr]
            data = json.loads(response)
            subtopics_data = data.get("subtopics", {})
            if not subtopics_data:
                return

            from datetime import date
            today = date.today().isoformat()
            subtopics: dict[str, str] = {}
            for name, entries in subtopics_data.items():
                title = name.replace("-", " ").title()
                lines = [f"- {e} ({today})" for e in entries]
                subtopics[name] = f"# {title}\n\n" + "\n".join(lines) + "\n"

            with self._lock:
                self._topics.replace_with_subtopics(topic, subtopics)
        except Exception as e:
            print(f"[Memory] Topic split failed for '{topic}': {e}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_memory_topic_split.py -v`
Expected: All PASS

**Step 5: Type check and commit**

Run: `mypy rostro/memory/manager.py`

```bash
git add rostro/memory/manager.py tests/unit/test_memory_topic_split.py
git commit -m "feat(memory): add async topic splitting via LLM"
```

---

### Task 9: Integrate with controller and ConversationEngine

Wire MemoryManager into the runtime controller and update ConversationEngine to support memory context.

**Files:**
- Modify: `rostro/conversation/engine.py` — add `set_memory_context()` method
- Modify: `rostro/runtime/controller.py` — initialize and use MemoryManager
- Modify: `config/default.yaml` — add `memory:` section
- Modify: `tests/unit/test_conversation.py` — add tests for memory context

**Step 1: Write the failing test for ConversationEngine**

Add to `tests/unit/test_conversation.py`:

```python
    def test_set_memory_context(self):
        """Test injecting memory context into system prompt."""
        engine = ConversationEngine(system_prompt="You are helpful.")
        engine.set_memory_context("Things you know:\n- Likes rice")
        messages = engine.build_messages()
        assert len(messages) == 1
        assert "You are helpful." in messages[0].content
        assert "Likes rice" in messages[0].content

    def test_set_memory_context_empty(self):
        """Test that empty memory context doesn't break things."""
        engine = ConversationEngine(system_prompt="You are helpful.")
        engine.set_memory_context("")
        messages = engine.build_messages()
        assert messages[0].content == "You are helpful."

    def test_memory_context_updates(self):
        """Test that memory context can be updated."""
        engine = ConversationEngine(system_prompt="Base prompt.")
        engine.set_memory_context("Context v1")
        engine.set_memory_context("Context v2")
        messages = engine.build_messages()
        assert "Context v2" in messages[0].content
        assert "Context v1" not in messages[0].content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_conversation.py::TestConversationEngine::test_set_memory_context -v`
Expected: FAIL — `AttributeError: 'ConversationEngine' object has no attribute 'set_memory_context'`

**Step 3: Add `set_memory_context()` to ConversationEngine**

In `rostro/conversation/engine.py`, add a `_memory_context` field and modify `build_messages`:

```python
@dataclass
class ConversationEngine:
    """Manages conversation state and prompt building."""

    system_prompt: str = ""
    max_history: int = 10
    history: list[Message] = field(default_factory=list)
    _memory_context: str = field(default="", repr=False)

    def set_memory_context(self, context: str) -> None:
        """Set the memory context to inject into the system prompt."""
        self._memory_context = context

    def build_messages(self) -> list[Message]:
        """Build the full message list for the LLM."""
        messages: list[Message] = []

        # Build system prompt with optional memory context
        system = self.system_prompt
        if self._memory_context:
            system = system + "\n\n" + self._memory_context if system else self._memory_context

        if system:
            messages.append(Message(role="system", content=system))

        messages.extend(self.history)
        return messages
```

**Step 4: Run conversation tests**

Run: `pytest tests/unit/test_conversation.py -v`
Expected: All PASS

**Step 5: Add memory config to `config/default.yaml`**

Append to end of file:

```yaml

memory:
  session_timeout_minutes: 7
  topics_dir: data/topics
  topic_split_threshold_lines: 50
  min_conclusions_for_new_topic: 3
  db_path: data/memory.db
  embedding_similarity_threshold: 0.7
  max_memories_per_search: 5
  index_max_entries: 20
```

**Step 6: Integrate MemoryManager into controller.py**

Add imports at top of `rostro/runtime/controller.py`:

```python
from rostro.memory import MemoryConfig, MemoryManager
from rostro.providers.embeddings.openai import OpenAIEmbeddingProvider
```

Add to `__init__` (after `self._llm` line):

```python
        self._memory: MemoryManager | None = None
        self._embedder: OpenAIEmbeddingProvider | None = None
```

In `start()`, after initializing `self._llm` and before initializing `self._conversation`, add:

```python
        # Initialize memory system
        embeddings_config = providers_config.get("embeddings", {})
        self._embedder = OpenAIEmbeddingProvider(
            model=embeddings_config.get("model", "text-embedding-3-small"),
        )
        memory_config = MemoryConfig.from_dict(self.config.get("memory", {}))
        self._memory = MemoryManager(
            config=memory_config,
            llm=self._llm,
            embedder=self._embedder,
        )
```

In `_process_speech()`, after `self._conversation.add_user_message(user_text)` and before `self._stream_and_speak()`, add:

```python
            # Update memory system
            if self._memory and self._conversation:
                self._memory.on_user_message(user_text)
                context = self._memory.get_context()
                self._conversation.set_memory_context(context)
                # Keep history in sync for digestion
                self._memory.set_history(
                    [{"role": m.role, "content": m.content} for m in self._conversation.history]
                )
```

In `_stream_and_speak()`, after `self._conversation.add_assistant_message(full_response)`, add:

```python
                # Update memory history with assistant response
                if self._memory and self._conversation:
                    self._memory.set_history(
                        [{"role": m.role, "content": m.content} for m in self._conversation.history]
                    )
```

In `stop()`, add before `if self._vad`:

```python
        if self._memory:
            self._memory.stop()
```

**Step 7: Run all tests**

Run: `pytest tests/unit/ -v`
Expected: All PASS (existing + new)

**Step 8: Type check and commit**

Run: `mypy rostro/`

```bash
git add rostro/conversation/engine.py rostro/runtime/controller.py config/default.yaml tests/unit/test_conversation.py
git commit -m "feat(memory): integrate MemoryManager with controller and ConversationEngine"
```

---

### Task 10: Full integration test and quality checks

Verify everything works together, run full test suite, linting, and type checking.

**Files:**
- All `rostro/memory/*.py` files
- All `tests/unit/test_memory_*.py` files

**Step 1: Run full test suite**

Run: `pytest tests/unit/ -v`
Expected: All tests PASS

**Step 2: Type check entire project**

Run: `mypy rostro/`
Expected: Success, no errors

**Step 3: Lint**

Run: `ruff format . && ruff check .`
Expected: No errors

**Step 4: Verify data directory is gitignored**

Check `.gitignore` includes `data/`. If not, add it — the data directory contains user-specific memory and should not be committed.

Run: `grep -q "^data/" .gitignore || echo "data/" >> .gitignore`

**Step 5: Final commit**

```bash
git add .gitignore
git commit -m "chore: ensure data/ directory is gitignored"
```
