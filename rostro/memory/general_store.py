"""General memory store with SQLite and embeddings."""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rostro.providers.base import EmbeddingProvider


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
        embedder: EmbeddingProvider,
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
        """Save a memory with its embedding.

        Args:
            text: The memory text to store.
            topic: Optional topic category.
        """
        embedding = self._embedder.embed(text)
        blob = np.array(embedding, dtype=np.float32).tobytes()
        now = datetime.now(UTC).isoformat()
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT INTO memories (text, embedding, topic, created_at) VALUES (?, ?, ?, ?)",
            (text, blob, topic, now),
        )
        conn.commit()
        conn.close()

    def search(self, query: str) -> list[MemoryResult]:
        """Search memories by semantic similarity.

        Args:
            query: The search query text.

        Returns:
            List of matching memories sorted by similarity score.
        """
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute("SELECT text, embedding, topic, created_at FROM memories").fetchall()
        conn.close()

        if not rows:
            return []

        query_vec = np.array(self._embedder.embed(query), dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm == 0:
            return []

        results: list[MemoryResult] = []
        for text, blob, topic, created_at in rows:
            mem_vec = np.frombuffer(blob, dtype=np.float32)
            mem_norm = float(np.linalg.norm(mem_vec))
            if mem_norm == 0:
                continue
            score = float(np.dot(query_vec, mem_vec) / (query_norm * mem_norm))
            if score >= self._threshold:
                results.append(
                    MemoryResult(text=text, topic=topic, score=score, created_at=created_at)
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: self._max_results]

    def count(self) -> int:
        """Return the number of stored memories."""
        conn = sqlite3.connect(self._db_path)
        count: int = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        return count
