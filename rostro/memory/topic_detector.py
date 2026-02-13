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
            "If it's a new distinct topic, return a short lowercase slug "
            "(e.g. 'fitness', 'home-renovation'). "
            "If the message is too generic or casual (greetings, small talk), "
            "return null.\n\n"
            'Respond ONLY with JSON: {"topic": "name"} or {"topic": null}'
        )
        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=user_message),
        ]
        try:
            response = self._llm.complete(messages)  # type: ignore[attr-defined]
            data = json.loads(response)
            topic = data.get("topic")
            return topic if isinstance(topic, str) else None
        except Exception:
            return None
