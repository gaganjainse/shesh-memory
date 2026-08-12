# shesh-memory

**hierarchical memory + learning** — Episodic/semantic memory, habits, intentions, context assembly.

- Layer: Mind (Mind)
- License: GPL-3.0
- Part of: [Shesh ecosystem](https://github.com/gaganjainse/shesh-ecosystem)

---
**Hierarchical memory + habit learning + context-window management for Shesh.**

Solves two problems at once: the model can't remember across turns, and its
context window is finite. This component keeps memory in layers and assembles a
token-bounded prompt for each turn.

- License: GPL-3.0
- Layer: Mind (memory + learning)
- Part of: [Shesh ecosystem](https://github.com/gaganjainse/shesh-ecosystem)

## Memory layers

| Layer | Holds | Retention |
|---|---|---|
| **working** | current task/session | per-turn |
| **episodic** | timestamped events (jsonl + SQLite FTS) | append-only, retrieved by relevance |
| **semantic** | durable facts about you/preferences/intentions (markdown) | long-term |
| **procedural/habits** | learned patterns with evidence counts | promoted/decayed |
| **skills** | how to do things | loaded per task |

## Habit learning

Observations are normalized into signatures and counted with reliability. When a
pattern's confidence crosses a threshold, it becomes a **candidate habit**; habits
decay over time and archive when stale. Promotions are reviewable — Shesh never
silently changes behavior from one coincidence.

## Context assembly

`assemble_context(query, working, max_tokens)` returns a structured prompt that
fits the model's budget, prioritizing semantic facts → active habits → skills →
working task → relevant episodes → recent tail, trimming the lowest-priority
sections first.

## Tools (MCP, stdio)

- `remember(kind, content)` — record an episode
- `recall(query)` — search past episodes
- `note_fact(fact)` — durable semantic memory
- `learn_habit(signature, description, success)` — feed the habit learner
- `list_habits()`
- `decay_habits()` — daily decay/archive
- `assemble_context(...)` — build a token-bounded prompt

## Develop

```bash
uv sync --extra dev
uv run pytest -q          # 11 offline tests
uv run ruff check .
uv run shesh-memory-mcp
```

Storage is plain JSONL/Markdown/SQLite under
`~/.local/share/shesh/memory/` — human-readable, portable, no vector DB required
(embedding retrieval is an optional future provider).