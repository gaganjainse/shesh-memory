"""sesha-memory: hierarchical memory + context-window management.

The model's context window is finite. This module solves retention by
organizing memory into layers and only surfacing what a turn needs:

  - WORKING:  the current task/session (small, always in context)
  - EPISODIC: timestamped events (what happened), retrieved by recency+relevance
  - SEMANTIC: durable facts about the user (habits, preferences, intentions)
  - PROCEDURAL: skills/patterns the agent has learned (how to do things)

Everything is local, plain-text/JSON/SQLite by default (no vector DB required
for small memory), with an optional embedding provider for semantic retrieval.
A bounded `ContextAssembler` builds the prompt, trimming by a token budget so
we never overflow the window. Habit learning turns repeated observations into
semantic memories with evidence counts (the "learn intentions/mannerisms" ask).
"""
from __future__ import annotations

__version__ = "0.1.0"
