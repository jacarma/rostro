"""Digester — extracts conclusions from conversation history via LLM."""

import json

from rostro.providers.base import Message


class Digester:
    """Extracts memorable conclusions from conversation history."""

    def __init__(self, llm: object) -> None:
        self._llm = llm

    def extract(self, history: list[dict[str, str]]) -> list[str]:
        if not history:
            return []

        conversation = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in history
        )
        prompt = (
            "Analyze this conversation and extract conclusions, decisions, "
            "or discoveries. Include observations made by the assistant that "
            "the user explicitly confirmed or approved. "
            "Ignore obvious facts, common sense, and small talk. "
            "Only extract things worth remembering long-term.\n\n"
            "Respond ONLY with a JSON array of short strings. "
            'Example: ["Prefers bomba rice for paella", "Allergic to tree nuts"]\n'
            "If nothing is worth remembering, respond with []"
        )
        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=conversation),
        ]
        try:
            response = self._llm.complete(messages)  # type: ignore[attr-defined]
            conclusions = json.loads(response)
            if isinstance(conclusions, list):
                return [str(c) for c in conclusions if c]
            return []
        except Exception:
            return []
