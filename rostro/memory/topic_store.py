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
