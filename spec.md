# 🎭 Rostro - A lightweight voice-driven 2D avatar assistant

A lightweight, open-source framework for building a **voice-driven 2D avatar assistant**
using **external AI APIs**, designed to run on **low-power Linux devices (ARM)** such as
single-board computers.

This project focuses on **clarity, portability, and simplicity**, avoiding heavy frameworks,
GPU dependencies, or local ML inference.

---

## 🎯 Goals

- Provide a **talking 2D avatar with a face** (eyes, mouth, expressions)
- Support **natural voice interaction**
- Use **external AI APIs** for intelligence
- Include **long-term semantic memory** via vector storage
- Run smoothly on **low-resource devices**
- Be easy to **extend, fork, and embed**
- Remain **framework-light and dependency-minimal**

---

## 🚫 Non-Goals

- No local LLM inference
- No 3D avatars
- No high-fidelity lip-sync (phoneme-perfect)
- No cloud backend or SaaS
- No mobile support (for now)

---

## 🧱 High-Level Architecture

```

┌──────────┐
│ Microphone │
└─────┬────┘
↓
┌───────────────┐
│ Speech-to-Text │ (External API)
└─────┬─────────┘
↓
┌───────────────┐
│ Conversation   │
│ + Memory       │
└─────┬─────────┘
↓
┌───────────────┐
│ LLM API        │
└─────┬─────────┘
↓
┌───────────────┐
│ Text-to-Speech │ (External API)
└─────┬─────────┘
↓
┌───────────────┐
│ Avatar Engine  │
│ (2D Rendering) │
└───────────────┘

```

---

## 🧠 Core Components

### 1. Conversation Engine

Responsible for:

- Managing conversation turns
- Building prompts
- Injecting memory context
- Handling system / persona instructions

**Key features**

- Stateless core + pluggable memory
- Supports multiple LLM providers
- Deterministic prompt assembly

---

### 2. Memory System

Provides semantic memory using vector embeddings.

**Memory types**

- Short-term (recent turns)
- Episodic (user facts, preferences)
- Summarized long-term memory

**Implementation**

- SQLite-based vector store
- Embeddings generated via external API
- k-NN similarity search

**Constraints**

- Optimized for 1k–50k memory entries
- Must work offline except for embedding generation

---

### 3. AI Integration Layer

Abstracts all external AI APIs behind swappable provider interfaces.

**Responsibilities**

- Speech-to-Text (STT)
- LLM chat completion
- Text-to-Speech (TTS)
- Embedding generation

**Provider Architecture**

Each AI capability is defined by a Protocol (interface) with concrete implementations per provider.

```
ai_providers/
├── base.py              # Protocol definitions
├── llm/
│   ├── base.py          # LLMProvider protocol
│   └── openai.py        # OpenAI implementation
├── stt/
│   ├── base.py          # STTProvider protocol
│   └── openai.py        # Whisper API implementation
├── tts/
│   ├── base.py          # TTSProvider protocol
│   └── openai.py        # OpenAI TTS implementation
└── embeddings/
    ├── base.py          # EmbeddingProvider protocol
    └── openai.py        # OpenAI embeddings implementation
```

**Protocol definitions**

```python
class STTProvider(Protocol):
    def transcribe(self, audio: bytes, format: str = "wav") -> str: ...

class TTSProvider(Protocol):
    def synthesize(self, text: str, voice: str | None = None) -> bytes: ...
    def list_voices(self) -> list[str]: ...

class LLMProvider(Protocol):
    def complete(self, messages: list[Message], **kwargs) -> str: ...
    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]: ...

class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

**Provider selection via config**

```yaml
providers:
  llm:
    provider: openai
    model: gpt-4o-mini
  stt:
    provider: openai
    model: whisper-1
  tts:
    provider: openai
    voice: nova
  embeddings:
    provider: openai
    model: text-embedding-3-small
```

**Phase 1:** OpenAI only (all four capabilities)
**Future:** Add providers (Anthropic, Groq, ElevenLabs, local Ollama, etc.)

**Error handling**

```yaml
error_handling:
  max_retries: 3
  retry_delay_ms: 1000
  timeout_ms: 30000
  fallback_message: "Sorry, I'm having connection issues. Please try again."
```

- On network failure: retry up to `max_retries` times
- On persistent failure: avatar shows error state, speaks `fallback_message` (if TTS works) or displays text
- On timeout: treat as failure

**Design principles**

- Provider-agnostic interfaces (Protocols)
- API keys via environment variables
- Network failures handled gracefully with retries and fallback messages
- No provider-specific logic leaks outside adapter modules

---

### 4. Audio I/O System

Handles real-time audio input/output.

**Features**

- Microphone capture
- Audio playback
- Volume analysis for lip-sync
- Device-agnostic selection

**Audio format**

```yaml
audio:
  sample_rate: 16000      # Hz (required by Whisper)
  channels: 1             # Mono
  format: int16           # 16-bit PCM
  chunk_duration_ms: 100  # Buffer size for processing
```

**Constraints**

- CPU-only
- No DSP-heavy processing
- Latency tolerant (1–3s acceptable)

---

### 5. Avatar Engine (2D)

Renders the on-screen avatar and synchronizes it with audio.

**Rendering**

- 2D sprites (PNG) or programmatic fallback
- Pygame-based
- Fixed resolution window

**States**

- Idle
- Listening
- Thinking
- Speaking
- Error

**Lip Sync**

- Volume-based mouth animation (4 levels)
- Frame-based timing tied to audio playback

**Face Packs**

The avatar system uses interchangeable "face packs" - directories containing all required assets for a face. This allows users to swap faces without code changes.

```
assets/
├── faces/
│   ├── default/           # Programmatic placeholder (Phase 1)
│   │   └── manifest.yaml
│   ├── cartoon_01/        # Example custom face pack
│   │   ├── manifest.yaml
│   │   ├── base.png
│   │   ├── eyes_open.png
│   │   ├── eyes_closed.png
│   │   ├── mouth_0.png    # Closed
│   │   ├── mouth_1.png    # Slightly open
│   │   ├── mouth_2.png    # Medium
│   │   ├── mouth_3.png    # Wide open
│   │   └── overlay_thinking.png  # Optional state indicator
│   └── realistic_01/      # Another face pack
│       └── ...
```

**Face Pack Manifest** (`manifest.yaml`)

```yaml
name: "Cartoon Assistant"
version: "1.0"
author: "..."
type: sprites  # sprites | programmatic

# Only for type: sprites
resolution: [512, 512]
composition: layered  # layered | single

layers:
  base: "base.png"
  eyes:
    open: "eyes_open.png"
    closed: "eyes_closed.png"
    blink_interval_ms: 4000
  mouth:
    - "mouth_0.png"  # Level 0: silence
    - "mouth_1.png"  # Level 1: low volume
    - "mouth_2.png"  # Level 2: medium volume
    - "mouth_3.png"  # Level 3: high volume
  overlays:  # Optional per-state overlays
    thinking: "overlay_thinking.png"
    listening: "overlay_listening.png"
    error: "overlay_error.png"
```

**Programmatic Fallback (default)**

Phase 1 includes a built-in programmatic face that requires no external assets:

```yaml
# assets/faces/default/manifest.yaml
name: "Default Programmatic"
type: programmatic
colors:
  face: "#FFE4C4"
  eyes: "#333333"
  mouth: "#CC6666"
  background: "#2D2D2D"
```

This draws a simple cartoon face using Pygame primitives (circles, ellipses).

**Config selection**

```yaml
avatar:
  face_pack: default      # or "cartoon_01", "realistic_01", etc.
  resolution: [800, 600]
  fps: 30
```

---

### 6. Runtime Controller

Orchestrates all components.

**Responsibilities**

- State transitions
- Event loop
- Error recovery
- Graceful shutdown

---

### 7. Activation System

Determines when the user wants to interact using Voice Activity Detection (VAD).

**How it works**

1. Continuously monitors microphone input (low CPU)
2. Detects voice activity when volume exceeds threshold
3. Starts recording
4. Detects end-of-speech after sustained silence
5. Triggers transcription pipeline

**VAD Implementation**

- Simple volume-based detection (RMS threshold)
- Optional: `webrtcvad` library for better accuracy (still CPU-light)
- Configurable parameters for different environments

**Configuration**

```yaml
activation:
  mode: vad                      # Options: vad, push_to_talk, proximity (future)
  vad:
    volume_threshold: 0.02       # RMS threshold (0.0-1.0)
    silence_duration_ms: 1500    # Silence before end-of-speech
    min_speech_duration_ms: 300  # Ignore very short sounds
    sample_window_ms: 50         # Analysis window size
```

**States**

- `dormant` → Monitoring, low CPU
- `listening` → Voice detected, recording
- `processing` → Silence detected, sending to STT

**Tuning considerations**

- `volume_threshold`: Lower = more sensitive, more false positives
- `silence_duration_ms`: Lower = faster response, risk cutting off speech
- Environment calibration may be needed (noisy vs quiet rooms)

**Future extensions**

- Proximity sensor trigger (hardware GPIO)
- Wake word detection
- Push-to-talk fallback

---

## 🎨 User Experience

### Interaction Flow

1. Avatar is idle on screen (VAD monitoring in background)
2. User starts speaking → VAD detects voice
3. Avatar switches to "listening" (recording)
4. User stops speaking → VAD detects silence (1.5s)
5. User speech is transcribed (STT API)
6. Avatar enters "thinking"
7. Response is generated (LLM API)
8. Avatar speaks with mouth animation (TTS API)
9. Avatar returns to idle (VAD resumes monitoring)

---

### Visual Style

- Simple, friendly, cartoon-like
- Minimal UI elements
- Focus on facial expression
- No menus required for basic operation

---

## ⚙️ Runtime Constraints

Target environment:

- Linux (ARM64 / x86_64)
- CPU-only
- 2–4 GB RAM
- No discrete GPU

**Performance targets**

- Idle RAM < 300 MB
- CPU idle < 10%
- Stable 24/7 operation

---

## 🧪 Development Principles

- Python-first
- Minimal dependencies
- Explicit over magical
- Easy to debug
- No background daemons
- No hidden threads

---

## 📜 License

MIT License - See LICENSE file.

---

## 🌐 Language

- Default system language: **English**
- Whisper auto-detects user's spoken language
- Responses follow user's language (configurable in persona)

---

## ✅ Quality Standards

### Code Style

- **Formatter:** `ruff format` (Black-compatible)
- **Linter:** `ruff check`
- **Type checking:** `mypy --strict`
- **Python version:** 3.11+

**Ruff configuration** (`pyproject.toml`)

```toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true
```

### Testing

- **Framework:** `pytest`
- **Coverage:** `pytest-cov` (minimum 80% for core modules)
- **Async testing:** `pytest-asyncio` (if needed)

**Test structure**

```
tests/
├── unit/
│   ├── test_activation.py
│   ├── test_audio.py
│   ├── test_avatar.py
│   ├── test_conversation.py
│   └── test_providers.py
├── integration/
│   └── test_pipeline.py
└── conftest.py          # Shared fixtures
```

**Test requirements**

- All public functions must have unit tests
- Providers must have mock-based tests (no real API calls in CI)
- Integration tests can be marked `@pytest.mark.integration` and skipped in CI

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML]
```

---

## 🤖 AI Agent Development Protocol

When an AI agent is implementing features, it MUST follow this checklist after each significant change:

### After Every Code Change

```bash
# 1. Format code
ruff format .

# 2. Lint and auto-fix
ruff check --fix .

# 3. Type check
mypy rostro/

# 4. Run tests
pytest tests/unit/ -v

# 5. Check coverage (for modified modules)
pytest tests/unit/ --cov=rostro --cov-report=term-missing
```

### Before Completing a Task

1. **All checks pass** - Format, lint, types, tests
2. **New code has tests** - Unit tests for new functions/classes
3. **No regressions** - Existing tests still pass
4. **Types are complete** - No `Any` types without justification
5. **Docstrings present** - Public functions have docstrings

### Commit Protocol

```bash
# Verify everything before commit
ruff format . && ruff check . && mypy rostro/ && pytest tests/unit/
```

Only commit if all commands succeed.

### When Adding a New Module

1. Create the module file in the appropriate directory
2. Create corresponding test file in `tests/unit/`
3. Add type hints to all functions
4. Add docstring to module and public functions
5. Run full check suite
6. Update `__init__.py` exports if needed

### When Modifying Existing Code

1. Read and understand existing tests first
2. Make changes
3. Run existing tests to check for regressions
4. Add new tests if behavior changed
5. Run full check suite

### Integration Test Protocol

Before marking a phase as complete:

```bash
# Run full test suite including integration
pytest tests/ -v

# Manual smoke test (if applicable)
python -m rostro.main --config config/default.yaml
```

### Error Resolution

If any check fails:

1. **Format error** → Run `ruff format .`
2. **Lint error** → Run `ruff check --fix .`, fix remaining manually
3. **Type error** → Fix type annotations, avoid `type: ignore` unless necessary
4. **Test failure** → Debug and fix, do not skip tests

Never proceed with failing checks. Fix issues before continuing.

---

## 📁 Suggested Project Structure

```
rostro/
├── activation/          # VAD and activation logic
├── audio/               # Audio I/O (capture, playback)
├── avatar/              # 2D rendering engine
│   ├── engine.py        # Main renderer
│   ├── face_pack.py     # Face pack loader
│   └── programmatic.py  # Built-in procedural face
├── conversation/        # Conversation state and prompt building
├── memory/              # Vector store and semantic memory
├── providers/           # AI provider adapters
│   ├── llm/
│   ├── stt/
│   ├── tts/
│   └── embeddings/
├── runtime/             # Main controller and state machine
├── assets/
│   └── faces/           # Face packs
│       ├── default/     # Built-in programmatic (no images needed)
│       │   └── manifest.yaml
│       └── .gitkeep
├── config/
│   └── default.yaml     # Default configuration
├── main.py
└── README.md
```

---

## 🔐 Configuration

- `.env` file for API keys
- YAML for runtime config
- No hardcoded secrets

**Environment variables**

```bash
OPENAI_API_KEY=sk-...
# Future providers
# ANTHROPIC_API_KEY=sk-ant-...
# ELEVENLABS_API_KEY=...
```

**Config file example** (`config/default.yaml`)

```yaml
providers:
  llm:
    provider: openai
    model: gpt-4o-mini
  stt:
    provider: openai
    model: whisper-1
  tts:
    provider: openai
    voice: nova
  embeddings:
    provider: openai
    model: text-embedding-3-small

audio:
  sample_rate: 16000
  channels: 1
  format: int16
  chunk_duration_ms: 100
  input_device: null   # null = system default
  output_device: null

activation:
  mode: vad
  vad:
    volume_threshold: 0.02
    silence_duration_ms: 1500
    min_speech_duration_ms: 300

avatar:
  face_pack: default
  resolution: [800, 600]
  fps: 30

error_handling:
  max_retries: 3
  retry_delay_ms: 1000
  timeout_ms: 30000
  fallback_message: "Sorry, I'm having connection issues. Please try again."

persona:
  name: "Assistant"
  system_prompt: |
    You are a friendly assistant. Keep responses concise and conversational.
```

---

## 📅 Implementation Phases

### Phase 0 - Project Setup

**Goal:** Repository ready for development with quality tooling

- [ ] Initialize git repository
- [ ] Create project structure (directories, `__init__.py` files)
- [ ] Setup `pyproject.toml` with dependencies and tool config
- [ ] Setup `ruff`, `mypy`, `pytest`
- [ ] Setup pre-commit hooks
- [ ] Create `.env.example` and `.gitignore`
- [ ] Create `config/default.yaml`
- [ ] Add LICENSE (MIT) and README.md
- [ ] Verify: `ruff check .`, `mypy rostro/`, `pytest` all pass (empty)

### Phase 1 - MVP

**Goal:** End-to-end voice conversation with basic avatar

- [ ] Audio capture and playback (sounddevice)
- [ ] VAD activation (volume-based)
- [ ] OpenAI provider adapters (STT, LLM, TTS, Embeddings)
- [ ] Basic conversation engine (no memory yet)
- [ ] Avatar engine with face pack system
- [ ] Programmatic face (default) - no external assets needed
- [ ] Volume-based lip sync (4 mouth levels)
- [ ] Basic states: idle, listening, speaking
- [ ] YAML configuration
- [ ] Unit tests for all modules (≥80% coverage)
- [ ] Integration smoke test

**Out of scope for Phase 1:**
- Semantic memory
- Multiple providers
- Thinking/error states with overlays
- Sprite-based face packs (architecture ready, implementation Phase 2)
- Blink animation

### Phase 2 - Memory & Polish

- [ ] SQLite vector store
- [ ] Semantic memory injection
- [ ] Sprite-based face pack loader (PNG assets)
- [ ] All 5 avatar states (idle, listening, thinking, speaking, error)
- [ ] State overlays support
- [ ] Blink animation
- [ ] Graceful error handling with avatar feedback
- [ ] Config hot-reload
- [ ] Example face pack with generated assets

### Phase 3 - Extensibility

- [ ] Additional providers (Anthropic, Groq, ElevenLabs)
- [ ] Proximity sensor support
- [ ] Plugin system for custom behaviors
- [ ] Documentation and examples

---
