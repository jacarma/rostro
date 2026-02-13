"""Tests for context builder."""

from datetime import UTC, datetime, timedelta

from rostro.memory.context_builder import ContextBuilder
from rostro.memory.general_store import MemoryResult


class TestContextBuilder:
    def test_build_with_no_context(self) -> None:
        builder = ContextBuilder()
        result = builder.build()
        assert "persistent memory" in result.lower()

    def test_build_with_topic_index(self) -> None:
        builder = ContextBuilder()
        now = datetime.now(UTC)
        one_hour_ago = (now - timedelta(hours=1)).isoformat()
        builder.set_topic_index([(one_hour_ago, "cooking", "rice recipes")])
        result = builder.build()
        assert "cooking" in result
        assert "rice recipes" in result

    def test_build_with_memories(self) -> None:
        builder = ContextBuilder()
        memories = [
            MemoryResult(
                text="Allergic to tree nuts",
                topic=None,
                score=0.9,
                created_at="2026-01-01",
            ),
        ]
        builder.set_memories(memories)
        result = builder.build()
        assert "Allergic to tree nuts" in result

    def test_add_memories_accumulates_no_duplicates(self) -> None:
        builder = ContextBuilder()
        m1 = MemoryResult(text="Fact A", topic=None, score=0.9, created_at="2026-01-01")
        m2 = MemoryResult(text="Fact B", topic=None, score=0.8, created_at="2026-01-01")
        m3 = MemoryResult(text="Fact A", topic=None, score=0.95, created_at="2026-01-01")
        builder.add_memories([m1, m2])
        builder.add_memories([m3])
        result = builder.build()
        assert result.count("Fact A") == 1
        assert "Fact B" in result

    def test_build_with_topic_file(self) -> None:
        builder = ContextBuilder()
        builder.set_topic_content("cooking", "# Cooking\n\n- Likes bomba rice")
        result = builder.build()
        assert "Likes bomba rice" in result

    def test_topic_file_clears_memories_section(self) -> None:
        builder = ContextBuilder()
        m1 = MemoryResult(text="Old memory", topic=None, score=0.9, created_at="2026-01-01")
        builder.add_memories([m1])
        builder.set_topic_content("cooking", "# Cooking\n\n- Topic content here")
        result = builder.build()
        assert "Topic content here" in result
        assert "Old memory" not in result

    def test_timestamp_to_relative(self) -> None:
        now = datetime.now(UTC)
        one_hour_ago = (now - timedelta(hours=1)).isoformat()
        relative = ContextBuilder.to_relative_time(one_hour_ago)
        assert "hour" in relative or "minute" in relative

    def test_reset_clears_all(self) -> None:
        builder = ContextBuilder()
        m1 = MemoryResult(text="Fact", topic=None, score=0.9, created_at="2026-01-01")
        builder.add_memories([m1])
        builder.set_topic_content("cooking", "content")
        builder.reset()
        result = builder.build()
        assert "Fact" not in result
