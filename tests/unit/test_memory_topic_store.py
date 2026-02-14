"""Tests for topic file store."""

from pathlib import Path

import pytest

from rostro.memory.topic_store import TopicStore


@pytest.fixture
def store(tmp_path: Path) -> TopicStore:
    return TopicStore(topics_dir=tmp_path, split_threshold=5)


class TestTopicStore:
    def test_list_topics_empty(self, store: TopicStore) -> None:
        assert store.list_topics() == []

    def test_append_creates_file(self, store: TopicStore, tmp_path: Path) -> None:
        store.append("cooking", "Likes bomba rice")
        path = tmp_path / "cooking.md"
        assert path.exists()
        content = path.read_text()
        assert "Likes bomba rice" in content
        assert "# Cooking" in content

    def test_append_multiple_entries(self, store: TopicStore, tmp_path: Path) -> None:
        store.append("cooking", "Likes bomba rice")
        store.append("cooking", "Uses homemade broth")
        content = (tmp_path / "cooking.md").read_text()
        assert "Likes bomba rice" in content
        assert "Uses homemade broth" in content

    def test_list_topics_returns_names(self, store: TopicStore) -> None:
        store.append("cooking", "entry")
        store.append("travel", "entry")
        topics = store.list_topics()
        assert sorted(topics) == ["cooking", "travel"]

    def test_read_topic(self, store: TopicStore) -> None:
        store.append("cooking", "Likes bomba rice")
        content = store.read("cooking")
        assert content is not None
        assert "Likes bomba rice" in content

    def test_read_nonexistent_topic(self, store: TopicStore) -> None:
        assert store.read("nonexistent") is None

    def test_needs_split_false_when_small(self, store: TopicStore) -> None:
        store.append("cooking", "One entry")
        assert not store.needs_split("cooking")

    def test_needs_split_true_when_over_threshold(self, store: TopicStore) -> None:
        for i in range(6):
            store.append("cooking", f"Entry {i}")
        assert store.needs_split("cooking")

    def test_line_count(self, store: TopicStore) -> None:
        store.append("cooking", "A")
        store.append("cooking", "B")
        # header + blank line + 2 entries = 4 lines
        assert store.line_count("cooking") == 4

    def test_replace_topic_with_subtopics(self, store: TopicStore, tmp_path: Path) -> None:
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
