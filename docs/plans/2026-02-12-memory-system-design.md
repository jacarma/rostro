# Memory System Design

## Overview

Persistent memory system for the Rostro voice assistant. Remembers conclusions, decisions, and discoveries across conversations. Designed to run on Le Potato (2GB RAM, ARM).

## Architecture

Three memory layers, all file and SQLite based:

```
┌─────────────────────────────────────────────────┐
│              Conversation History                │
│         (ConversationEngine, 10 turns)           │
│         Cleared on inactivity (5-10min)          │
└──────────────────┬──────────────────────────────┘
                   │ on session close: extract conclusions
                   ▼
┌──────────────────────────┐   ┌──────────────────────┐
│   Topic Files            │   │   General Memory     │
│   data/topics/*.md       │   │   data/memory.db     │
│                          │   │   (SQLite + embeddings)│
│ - One file per topic     │   │                      │
│ - Auto-split by size     │   │ - Every conclusion   │
│ - Detailed conclusions   │   │   goes here          │
│                          │   │ - Semantic search     │
└──────────────────────────┘   └──────────────────────┘

┌──────────────────────────┐
│   Topic Index            │
│   data/topics/index.txt  │
│   20 recent entries      │
│   timestamp + summary    │
└──────────────────────────┘
```

## Conversation Lifecycle

```
IDLE (waiting for voice)
  │
  │ user speaks
  ▼
ACTIVE CONVERSATION
  │
  │ on each user message:
  │   ├─ search top 3-5 relevant memories (cosine > 0.7)
  │   ├─ accumulate in context (no duplicates)
  │   ├─ attempt topic detection
  │   └─ if topic detected:
  │       ├─ load topic file into context
  │       └─ stop searching memories / detecting topic
  │
  │ inactivity 5-10 min (configurable, default 7)
  ▼
DIGESTION
  │
  │ prompt to GPT-4o-mini:
  │   "Extract conclusions, decisions, or discoveries.
  │    Include assistant observations confirmed by the user.
  │    Ignore obvious facts and common sense."
  │
  │ result:
  │   ├─ ALWAYS save each conclusion to general memory (with embedding)
  │   ├─ if topic loaded → also append to topic file
  │   │   └─ if file > 50 lines → trigger split (async)
  │   ├─ if no topic and 3+ conclusions → create new topic
  │   └─ if no topic and < 3 conclusions → general memory only
  │
  │ update recent topics index
  │ clear ConversationEngine.history
  ▼
IDLE (ready for new conversation)
```

## LLM Context Injection

The system prompt is built dynamically with layers:

1. **Base persona** - from config or face pack (always)
2. **Recent topics index** - at conversation start (always)
3. **Relevant memories** - cumulative, no duplicates, until topic is detected
4. **Topic file** - replaces layer 3 when topic is detected
5. **Memory instructions** - always present:
   "You have persistent memory. When the user shares decisions,
   discoveries, or conclusions, acknowledge them naturally. You may
   also propose your own observations - if the user confirms them,
   they will be saved as memory."

Layers 2-5 go in the system prompt, not as user messages.

## Topic Detection

- Lightweight prompt to GPT-4o-mini with the user message + list of files in `data/topics/`
- Structured response: `{"topic": "cooking"}` or `{"topic": null}`
- Runs in parallel with the normal LLM call (no added latency)
- Attempted on each message until a topic is found; then stops

## Topic Files

**Location:** `data/topics/`

**Format** (`data/topics/cooking.md`):

```markdown
# Cooking

- Prefers bomba rice for paella (2026-02-12)
- Uses homemade chicken broth, never stock cubes (2026-02-12)
- Allergic to tree nuts (2026-02-10)
```

Each entry is a single line with date. Plain format, human-readable, manually inspectable.

**Auto-split:**
- Trigger: file exceeds 50 lines (configurable)
- Checked when saving new conclusions
- GPT-4o-mini analyzes content and proposes subtopics
- Example: `cooking.md` → `cooking-baking.md` + `cooking-rice.md` + `cooking-general.md`
- Index updated with new names
- Runs async, does not block conversation

## Recent Topics Index

**Location:** `data/topics/index.txt`

**Format:**

```
2026-02-12T14:30:00|cooking|rice recipes and oven timing
2026-02-11T09:15:00|travel|planning trip to Portugal
```

- Maximum 20 lines, oldest discarded
- Absolute timestamps, converted to relative when injected into prompt
- Updated on each session close

## General Memory (Long-term)

**Storage:** `data/memory.db` (SQLite)

**Schema:**

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    topic TEXT,
    created_at TEXT NOT NULL
);
```

**Save:** embed with text-embedding-3-small, INSERT with embedding as blob (numpy → bytes).

**Search:** embed user message, load all embeddings, cosine similarity with numpy, filter > 0.7, return top 5.

**Expected volume:** 50-200 memories over months of use. Loading all embeddings (~1.2MB at 200) is instant on Le Potato.

## Modules

```
rostro/memory/
├── __init__.py
├── manager.py          # MemoryManager - orchestrates everything
├── session_timer.py    # Inactivity timer + digestion trigger
├── digester.py         # Extracts conclusions from history via LLM
├── topic_store.py      # CRUD for topic files + auto-split
├── topic_detector.py   # Detects topic via LLM
├── general_store.py    # SQLite + embeddings, semantic search
└── context_builder.py  # Builds system prompt layers 2-5
```

**MemoryManager** is the single entry point. The controller only talks to it:

```python
# In start()
self.memory = MemoryManager(config, llm_provider, embedding_provider)

# In _process_speech() before LLM call
context = self.memory.get_context()
self.conversation.set_memory_context(context)
self.memory.on_user_message(transcription)
```

## Configuration

```yaml
memory:
  session_timeout_minutes: 7
  topics_dir: data/topics
  topic_split_threshold_lines: 50
  min_conclusions_for_new_topic: 3
  db_path: data/memory.db
  embedding_similarity_threshold: 0.7
  max_memories_per_search: 5
  index_max_entries: 20
```
