"""OpenAI Embedding provider."""

from openai import OpenAI


class OpenAIEmbeddingProvider:
    """OpenAI Embeddings API implementation."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        """Initialize the OpenAI Embedding provider.

        Args:
            model: Embedding model to use.
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for the given text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )

        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [item.embedding for item in response.data]
