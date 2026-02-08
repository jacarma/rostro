"""Tests for provider modules."""

from unittest.mock import MagicMock, patch

from rostro.providers.base import Message


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestOpenAISTTProvider:
    """Tests for OpenAI STT provider."""

    @patch("rostro.providers.stt.openai.OpenAI")
    def test_init(self, mock_openai):
        """Test initialization."""
        from rostro.providers.stt.openai import OpenAISTTProvider

        provider = OpenAISTTProvider(model="whisper-1")
        assert provider.model == "whisper-1"
        mock_openai.assert_called_once()

    @patch("rostro.providers.stt.openai.OpenAI")
    def test_transcribe(self, mock_openai):
        """Test transcription."""
        from rostro.providers.stt.openai import OpenAISTTProvider

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.audio.transcriptions.create.return_value = MagicMock(text="Hello world")

        provider = OpenAISTTProvider()
        result = provider.transcribe(b"fake audio", format="wav")

        assert result == "Hello world"
        mock_client.audio.transcriptions.create.assert_called_once()


class TestOpenAITTSProvider:
    """Tests for OpenAI TTS provider."""

    @patch("rostro.providers.tts.openai.OpenAI")
    def test_init(self, mock_openai):
        """Test initialization."""
        from rostro.providers.tts.openai import OpenAITTSProvider

        provider = OpenAITTSProvider(voice="nova")
        assert provider.default_voice == "nova"
        mock_openai.assert_called_once()

    @patch("rostro.providers.tts.openai.OpenAI")
    def test_synthesize(self, mock_openai):
        """Test synthesis."""
        from rostro.providers.tts.openai import OpenAITTSProvider

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.audio.speech.create.return_value = MagicMock(content=b"fake audio")

        provider = OpenAITTSProvider()
        result = provider.synthesize("Hello")

        assert result == b"fake audio"
        mock_client.audio.speech.create.assert_called_once()

    @patch("rostro.providers.tts.openai.OpenAI")
    def test_list_voices(self, mock_openai):
        """Test listing voices."""
        from rostro.providers.tts.openai import OpenAITTSProvider

        provider = OpenAITTSProvider()
        voices = provider.list_voices()

        assert "nova" in voices
        assert "alloy" in voices
        assert len(voices) == 11


class TestOpenAILLMProvider:
    """Tests for OpenAI LLM provider."""

    @patch("rostro.providers.llm.openai.OpenAI")
    def test_init(self, mock_openai):
        """Test initialization."""
        from rostro.providers.llm.openai import OpenAILLMProvider

        provider = OpenAILLMProvider(model="gpt-4o-mini")
        assert provider.model == "gpt-4o-mini"
        mock_openai.assert_called_once()

    @patch("rostro.providers.llm.openai.OpenAI")
    def test_complete(self, mock_openai):
        """Test completion."""
        from rostro.providers.llm.openai import OpenAILLMProvider

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAILLMProvider()
        messages = [Message(role="user", content="Hi")]
        result = provider.complete(messages)

        assert result == "Hello!"

    @patch("rostro.providers.llm.openai.OpenAI")
    def test_complete_with_none_content(self, mock_openai):
        """Test completion with None content."""
        from rostro.providers.llm.openai import OpenAILLMProvider

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=None))]
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAILLMProvider()
        messages = [Message(role="user", content="Hi")]
        result = provider.complete(messages)

        assert result == ""


class TestOpenAIEmbeddingProvider:
    """Tests for OpenAI Embedding provider."""

    @patch("rostro.providers.embeddings.openai.OpenAI")
    def test_init(self, mock_openai):
        """Test initialization."""
        from rostro.providers.embeddings.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
        assert provider.model == "text-embedding-3-small"
        mock_openai.assert_called_once()

    @patch("rostro.providers.embeddings.openai.OpenAI")
    def test_embed(self, mock_openai):
        """Test embedding single text."""
        from rostro.providers.embeddings.openai import OpenAIEmbeddingProvider

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        provider = OpenAIEmbeddingProvider()
        result = provider.embed("Hello")

        assert result == [0.1, 0.2, 0.3]

    @patch("rostro.providers.embeddings.openai.OpenAI")
    def test_embed_batch(self, mock_openai):
        """Test embedding multiple texts."""
        from rostro.providers.embeddings.openai import OpenAIEmbeddingProvider

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        provider = OpenAIEmbeddingProvider()
        result = provider.embed_batch(["Hello", "World"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]
