# 🧠 shesh-memory

> **Hierarchical memory + habit learning for Shesh.** Keeps memory in layers and
> assembles a token-bounded context for every turn, so the agent remembers across
> turns without blowing its context window.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python) ![License](https://img.shields.io/badge/License-GPL--3.0--or--later-blue?style=for-the-badge) ![Tests](https://img.shields.io/badge/Tests-33-success?style=for-the-badge) ![CI](https://img.shields.io/github/actions/workflow/status/gaganjainse/shesh-memory/ci.yml?style=for-the-badge&label=CI)

- **License:** GPL-3.0-or-later
- **Owner:** Gagan Jain ([@gaganjainse](https://github.com/gaganjainse))
- **Layer:** Mind (memory + learning)
- **Part of:** [shesh-ecosystem](https://github.com/gaganjainse/shesh-ecosystem)

---

## Why this repo exists

A model can't remember across turns, and its context window is finite. This
component solves both at once: memories are stored in layers by retention, and a
token-bounded prompt is assembled from the relevant ones each turn.

---

## Quick start

```bash
uv sync --extra dev
uv run pytest -q        # 33 tests
uv run ruff check .
```

## Memory layers

| Layer | Holds | Retention |
|---|---|---|
| **working** | current task/session | per-turn |
| **episodic** | timestamped events (jsonl + SQLite FTS) | append-only, retrieved by relevance |
| **semantic** | durable facts about you/preferences/intentions (markdown) | long-term |
| **procedural / habits** | learned patterns with evidence counts | promoted / decayed |
| **skills** | how to do things | loaded per task |

Habit learning normalizes observations into signatures counted with reliability;
a pattern becomes a **candidate habit** only past a confidence threshold, and
habits decay/archive when stale — promotions are reviewable, never silent.

## Status

Component CI is green (reusable ecosystem pipeline). Security posture and
vulnerability reporting: [SECURITY.md](SECURITY.md).

## Documentation index

- **Part of:** [shesh-ecosystem](https://github.com/gaganjainse/shesh-ecosystem)
- **Compiled reading:** [shesh-docs](https://github.com/gaganjainse/shesh-docs)

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
