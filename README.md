# 🎭 Rostro

A lightweight voice-driven 2D avatar assistant using external AI APIs.

## Features

- **Voice-driven interaction** - Speak naturally, get spoken responses
- **2D avatar with lip-sync** - Animated face that moves with speech
- **Multiple AI providers** - OpenAI for STT, LLM, TTS, and embeddings
- **Low resource usage** - Designed for ARM devices and low-power systems
- **Extensible** - Easy to add new providers and face packs

## Requirements

- Python 3.11+
- OpenAI API key
- Audio input/output devices

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rostro.git
cd rostro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

```bash
# Run with default configuration
rostro

# Run with custom config
rostro --config path/to/config.yaml
```

## Configuration

Edit `config/default.yaml` to customize:

- AI providers and models
- Audio settings
- VAD sensitivity
- Avatar appearance

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check .

# Run formatting
ruff format .

# Run type checking
mypy rostro/

# Run tests
pytest tests/unit/ -v

# Run tests with coverage
pytest tests/unit/ --cov=rostro --cov-report=term-missing
```

## Project Structure

```
rostro/
├── activation/     # Voice activity detection
├── audio/          # Audio capture and playback
├── avatar/         # 2D rendering engine
├── conversation/   # Conversation management
├── memory/         # Semantic memory (Phase 2)
├── providers/      # AI provider adapters
├── runtime/        # Main controller
└── main.py         # Entry point
```

## License

MIT License - See [LICENSE](LICENSE) for details.
