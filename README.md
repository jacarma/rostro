# 🎭 Rostro

A lightweight voice-driven 2D avatar assistant using external AI APIs.

## Features

- **Voice-driven interaction** - Speak naturally, get spoken responses
- **Neural voice detection** - [Silero VAD](https://github.com/snakers4/silero-vad) for accurate speech detection (ONNX model via [Pipecat](https://github.com/pipecat-ai/pipecat))
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

## Character Packs

Each avatar is a **character pack** with its own face images, voice, and personality. Packs live in `assets/faces/<name>/`.

### Creating a pack

1. Create a directory under `assets/faces/` with 5 images (png, jpg, or jpeg):

```
assets/faces/mycharacter/
├── manifest.yaml
├── m0.jpg          # mouth closed
├── m1.jpg          # mouth slightly open
├── m2.jpg          # mouth medium
├── m3.jpg          # mouth wide open
└── blink.jpg       # eyes closed
```

2. Add a `manifest.yaml`:

```yaml
name: "My Character"
version: "1.0"
author: "Your Name"
type: photo
voice: sage                # OpenAI TTS voice (optional)
voice_instructions: "..."  # voice style instructions (optional)
system_prompt: |           # personality (optional)
  You are a friendly assistant.
```

The `voice`, `voice_instructions`, and `system_prompt` fields are optional. When present, they override the global values from `config/default.yaml`.

3. Activate it in `config/default.yaml`:

```yaml
avatar:
  face_pack: mycharacter
```

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
