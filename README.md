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

## Memory System

Rostro has persistent memory across conversations. It remembers conclusions, decisions, and discoveries — not obvious facts or small talk.

### How it works

1. **During conversation** — searches past memories by semantic similarity and tries to detect a topic (e.g. "cooking", "travel")
2. **Topic detected** — loads the relevant topic file into context for richer responses
3. **Inactivity timeout** (default 7 min) — extracts conclusions from the conversation via LLM, saves them, and clears history for a fresh start

### Storage

```
data/
├── memory.db              # SQLite with embeddings (semantic search)
└── topics/
    ├── index.txt          # Recent topics with timestamps
    ├── cooking.md         # Topic file with conclusions
    └── personal-intro.md  # Another topic file
```

- **Topic files** — markdown lists of conclusions per topic, human-readable. Auto-split when they grow past 50 lines.
- **General memory** — every conclusion is stored with an embedding vector for semantic search (text-embedding-3-small).
- **Topic index** — last 20 topics with timestamps, injected at conversation start.

### Configuration

In `config/default.yaml`:

```yaml
memory:
  session_timeout_minutes: 7       # Inactivity before digestion
  topics_dir: data/topics
  topic_split_threshold_lines: 50  # Auto-split trigger
  min_conclusions_for_new_topic: 3 # Min conclusions to create a topic
  db_path: data/memory.db
  embedding_similarity_threshold: 0.7
  max_memories_per_search: 5
  index_max_entries: 20
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
├── memory/         # Persistent memory system
├── providers/      # AI provider adapters
├── runtime/        # Main controller
└── main.py         # Entry point
```

## License

MIT License - See [LICENSE](LICENSE) for details.
