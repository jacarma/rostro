"""Tests for general memory store."""

import sqlite3
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
