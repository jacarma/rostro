"""Tests for topic file splitting."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rostro.memory.manager import MemoryConfig, MemoryManager


@pytest.fixture
def memory_config(tmp_path: Path) -> MemoryConfig:
    return MemoryConfig(
        session_timeout_minutes=100,
        topics_dir=tmp_path / "topics",
        db_path=tmp_path / "memory.db",
        topic_split_threshold_lines=5,
    )


@pytest.fixture
def mock_llm() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    vec = np.random.randn(1536).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    embedder.embed.return_value = vec.tolist()
    return embedder


class TestTopicSplit:
    def test_trigger_split_creates_subtopics(
        self,
        memory_config: MemoryConfig,
        mock_llm: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        manager = MemoryManager(config=memory_config, llm=mock_llm, embedder=mock_embedder)
        topics_dir = memory_config.topics_dir

        for i in range(8):
            manager._topics.append("cooking", f"Entry {i}")

        assert manager._topics.needs_split("cooking")

        split_response = json.dumps(
            {
                "subtopics": {
                    "cooking-baking": ["Entry 0", "Entry 1", "Entry 2"],
                    "cooking-rice": ["Entry 3", "Entry 4", "Entry 5"],
                    "cooking-general": ["Entry 6", "Entry 7"],
                }
            }
        )
        mock_llm.complete.return_value = split_response

        manager._trigger_split("cooking")
        time.sleep(0.5)

        assert not (topics_dir / "cooking.md").exists()
        assert (topics_dir / "cooking-baking.md").exists()
        assert (topics_dir / "cooking-rice.md").exists()
        manager.stop()

    def test_split_handles_llm_error(
        self,
        memory_config: MemoryConfig,
        mock_llm: MagicMock,
        mock_embedder: MagicMock,
    ) -> None:
        manager = MemoryManager(config=memory_config, llm=mock_llm, embedder=mock_embedder)

        for i in range(8):
            manager._topics.append("cooking", f"Entry {i}")

        mock_llm.complete.side_effect = Exception("API error")
        manager._trigger_split("cooking")
        time.sleep(0.5)

        # Original file should still exist
        assert (memory_config.topics_dir / "cooking.md").exists()
        manager.stop()
