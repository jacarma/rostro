"""Tests for topic detector."""

from unittest.mock import MagicMock

from rostro.memory.topic_detector import TopicDetector


def _make_detector(llm: MagicMock) -> TopicDetector:
    return TopicDetector(llm=llm)


class TestTopicDetector:
    def test_detect_existing_topic(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = '{"topic": "cooking"}'
        detector = _make_detector(llm)
        result = detector.detect("I want to make paella", ["cooking", "travel"])
        assert result == "cooking"

    def test_detect_no_topic(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = '{"topic": null}'
        detector = _make_detector(llm)
        result = detector.detect("Hello, how are you?", ["cooking", "travel"])
        assert result is None

    def test_detect_new_topic(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = '{"topic": "fitness"}'
        detector = _make_detector(llm)
        result = detector.detect("I started going to the gym", ["cooking", "travel"])
        assert result == "fitness"

    def test_detect_with_no_existing_topics(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = '{"topic": "cooking"}'
        detector = _make_detector(llm)
        result = detector.detect("Let me tell you about my recipe", [])
        assert result == "cooking"

    def test_detect_handles_malformed_json(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = "not json at all"
        detector = _make_detector(llm)
        result = detector.detect("test message", [])
        assert result is None

    def test_detect_handles_llm_error(self) -> None:
        llm = MagicMock()
        llm.complete.side_effect = Exception("API error")
        detector = _make_detector(llm)
        result = detector.detect("test message", [])
        assert result is None
