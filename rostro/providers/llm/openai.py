"""OpenAI LLM provider."""

from collections.abc import Iterator
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from rostro.providers.base import Message


class OpenAILLMProvider:
    """OpenAI Chat Completion API implementation."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        """Initialize the OpenAI LLM provider.

        Args:
            model: Model to use for completions.
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def complete(self, messages: list[Message], **kwargs: Any) -> str:
        """Generate a completion for the given messages.

        Args:
            messages: List of conversation messages.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            Generated response text.
        """
        openai_messages = cast(
            list[ChatCompletionMessageParam],
            [{"role": msg.role, "content": msg.content} for msg in messages],
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            **kwargs,
        )

        content = response.choices[0].message.content
        return content if content is not None else ""

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream a completion for the given messages.

        Args:
            messages: List of conversation messages.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Yields:
            Response text chunks.
        """
        openai_messages = cast(
            list[ChatCompletionMessageParam],
            [{"role": msg.role, "content": msg.content} for msg in messages],
        )

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            stream=True,
            **kwargs,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
