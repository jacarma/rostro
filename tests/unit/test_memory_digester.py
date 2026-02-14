"""Tests for conversation digester."""

from unittest.mock import MagicMock

from rostro.memory.digester import Digester


def _make_digester(llm: MagicMock) -> Digester:
    return Digester(llm=llm)


class TestDigester:
    def test_extract_conclusions(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = '["Prefers bomba rice for paella", "Allergic to tree nuts"]'
        digester = _make_digester(llm)
        history = [
            {"role": "user", "content": "I always use bomba rice"},
            {"role": "assistant", "content": "Great choice!"},
            {"role": "user", "content": "I'm allergic to tree nuts"},
            {"role": "assistant", "content": "I'll remember that."},
        ]
        result = digester.extract(history)
        assert len(result) == 2
        assert "bomba rice" in result[0]

    def test_extract_empty_history(self) -> None:
        llm = MagicMock()
        digester = _make_digester(llm)
        result = digester.extract([])
        assert result == []

    def test_extract_no_conclusions(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = "[]"
        digester = _make_digester(llm)
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = digester.extract(history)
        assert result == []

    def test_extract_handles_malformed_json(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = "not json"
        digester = _make_digester(llm)
        history = [{"role": "user", "content": "test"}]
        result = digester.extract(history)
        assert result == []

    def test_extract_handles_llm_error(self) -> None:
        llm = MagicMock()
        llm.complete.side_effect = Exception("API error")
        digester = _make_digester(llm)
        history = [{"role": "user", "content": "test"}]
        result = digester.extract(history)
        assert result == []

    def test_extract_includes_confirmed_assistant_observations(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = '["Enjoys cooking with fresh ingredients"]'
        digester = _make_digester(llm)
        history = [
            {
                "role": "assistant",
                "content": "It seems you enjoy cooking with fresh ingredients",
            },
            {"role": "user", "content": "Yes, exactly!"},
        ]
        result = digester.extract(history)
        assert len(result) == 1
