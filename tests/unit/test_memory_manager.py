"""Tests for memory manager."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rostro.memory.manager import MemoryConfig, MemoryManager


@pytest.fixture
def memory_config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(
        session_timeout_minutes=0.01,
        topics_dir=tmp_path / "topics",
        db_path=tmp_path / "memory.db",
        topic_split_threshold_lines=50,
        min_conclusions_for_new_topic=3,
        embedding_similarity_threshold=0.7,
        max_memories_per_search=5,
        index_max_entries=20,
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = "[]"
    return llm


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    vec = np.random.randn(1536).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    embedder.embed.return_value = vec.tolist()
    return embedder


@pytest.fixture
def manager(
    memory_config: MemoryConfig, mock_llm: MagicMock, mock_embedder: MagicMock
) -> MemoryManager:
    m = MemoryManager(config=memory_config, llm=mock_llm, embedder=mock_embedder)
    yield m  # type: ignore[misc]
    m.stop()


class TestMemoryConfig:
    def test_from_dict(self) -> None:
        config = MemoryConfig.from_dict({"session_timeout_minutes": 10})
        assert config.session_timeout_minutes == 10

    def test_from_dict_defaults(self) -> None:
        config = MemoryConfig.from_dict({})
        assert config.session_timeout_minutes == 7
        assert config.topic_split_threshold_lines == 50


class TestMemoryManager:
    def test_get_context_returns_string(self, manager: MemoryManager) -> None:
        result = manager.get_context()
        assert isinstance(result, str)
        assert "persistent memory" in result.lower()

    def test_on_user_message_resets_timer(self, manager: MemoryManager) -> None:
        manager.on_user_message("Hello")
        assert manager._timer.is_active

    def test_on_user_message_triggers_topic_detection(
        self, manager: MemoryManager, mock_llm: MagicMock
    ) -> None:
        mock_llm.complete.return_value = '{"topic": null}'
        manager.on_user_message("Hello")
        assert mock_llm.complete.called

    def test_topic_detected_stops_searching(
        self, manager: MemoryManager, mock_llm: MagicMock
    ) -> None:
        mock_llm.complete.return_value = '{"topic": "cooking"}'
        manager.on_user_message("Let me tell you about paella")
        assert manager._active_topic == "cooking"
        mock_llm.complete.reset_mock()
        manager.on_user_message("I use bomba rice")
        # Topic detection should NOT be called again (topic already set)

    def test_topic_index_loaded_on_init(
        self, memory_config: MemoryConfig, mock_llm: MagicMock, mock_embedder: MagicMock
    ) -> None:
        topics_dir = memory_config.topics_dir
        topics_dir.mkdir(parents=True, exist_ok=True)
        index_path = topics_dir / "index.txt"
        index_path.write_text("2026-02-12T10:00:00|cooking|rice recipes\n")
        mgr = MemoryManager(config=memory_config, llm=mock_llm, embedder=mock_embedder)
        context = mgr.get_context()
        assert "cooking" in context
        mgr.stop()

    def test_set_conversation_history(self, manager: MemoryManager) -> None:
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        manager.set_history(history)
        assert manager._history == history
