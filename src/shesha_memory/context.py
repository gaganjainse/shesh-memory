"""Context-window management.

Given a finite token budget for a turn, assemble the most useful context from
the memory layers without overflowing:

  System/skills  (highest priority, compact)
  Semantic facts (durable, user-specific)
  Working memory (current task/session)
  Relevant episodes (retrieved by recency + keyword/embedding match)
  Recent episodes (tail, for continuity)

Each section is added in priority order and trimmed to a budget using an
approximate token count (4 chars ≈ 1 token, conservative). The result is a
structured system prompt the caller drops in before the conversation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .intentions import Intentions, Mannerisms
from .store import Episode, MemoryStore


def approx_tokens(text: str) -> int:
    # Conservative: ~4 chars/token for English; round up.
    return max(1, (len(text) + 3) // 4)


@dataclass
class Budget:
    total: int = 6000
    breakdown: dict[str, int] = field(default_factory=lambda: {
        "semantic": 1200,
        "working": 1200,
        "relevant": 1800,
        "recent": 800,
        "skills": 1000,
    })

    def remaining(self, used: int) -> int:
        return max(0, self.total - used)


@dataclass
class ContextSection:
    name: str
    content: str

    @property
    def tokens(self) -> int:
        return approx_tokens(self.content)


def _trim(text: str, budget_tokens: int) -> str:
    if approx_tokens(text) <= budget_tokens:
        return text
    # Keep the most recent lines (events accumulate chronologically).
    lines = text.splitlines()
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        t = approx_tokens(line)
        if used + t > budget_tokens:
            break
        kept.append(line)
        used += t
    return "[...earlier context trimmed...]\n" + "\n".join(reversed(kept))


class ContextAssembler:
    def __init__(
        self,
        store: MemoryStore,
        budget: Budget | None = None,
        retriever: Callable[[str, int], list[Episode]] | None = None,
    ) -> None:
        self.store = store
        self.budget = budget or Budget()
        # retriever(query, n) -> episodes; defaults to store.search
        self.retriever = retriever or (lambda q, n: store.search(q, n))

    def build(
        self,
        *,
        query: str = "",
        working: str = "",
        skills: str = "",
        n_relevant: int = 6,
        n_recent: int = 8,
    ) -> list[ContextSection]:
        b = self.budget
        sections: list[ContextSection] = []

        # 1. Mannerisms/style (sets tone, very compact)
        sections.append(ContextSection(
            "mannerisms", _trim(Mannerisms(self.store.root).text(), 600)))

        # 2. Semantic facts (durable, compact)
        sections.append(ContextSection(
            "semantic", _trim(self.store.read_semantic(), b.breakdown["semantic"])))

        # 3. Active intentions (what the user is working toward)
        intentions = Intentions(self.store.root).render()
        if intentions:
            sections.append(ContextSection("intentions", _trim(intentions, 600)))

        # 4. Active habits (very compact — one line each)
        habits = "\n".join(
            f"- {h.description}" for h in self._load_active_habits()
        )
        if habits:
            sections.append(ContextSection("habits", habits))

        # 5. Skills / safety (fixed compact block)
        if skills:
            sections.append(ContextSection("skills", _trim(skills, b.breakdown["skills"])))

        # 6. Working memory (current task)
        if working:
            sections.append(ContextSection("working", _trim(working, b.breakdown["working"])))

        # 7. Relevant episodes for this query
        if query:
            rel = self.retriever(query, n_relevant)
            rel_text = "\n".join(self._fmt(e) for e in rel)
            sections.append(ContextSection("relevant", _trim(rel_text, b.breakdown["relevant"])))

        # 8. Recent episodes (continuity)
        rec = self.store.recent(n_recent)
        rec_text = "\n".join(self._fmt(e) for e in rec[-n_recent:])
        sections.append(ContextSection("recent", _trim(rec_text, b.breakdown["recent"])))

        # Enforce total budget by trimming from the lowest priority up.
        return self._enforce_total(sections)

    def _load_active_habits(self):
        # Import here to avoid a hard cycle at module load.
        from .habits import HabitLearner
        return list(HabitLearner(self.store).active_habits())

    @staticmethod
    def _fmt(e: Episode) -> str:
        return f"[{e.kind}] {e.content}"

    def _enforce_total(self, sections: list[ContextSection]) -> list[ContextSection]:
        # Priority high→low: mannerisms, intentions, semantic, habits, skills,
        # working, relevant, recent.
        order = ["mannerisms", "intentions", "semantic", "habits", "skills",
                 "working", "relevant", "recent"]
        sections.sort(key=lambda s: order.index(s.name) if s.name in order else 99)
        used = 0
        kept: list[ContextSection] = []
        for s in sections:
            cap = self.budget.total - used
            if cap <= 0:
                break
            trimmed = _trim(s.content, cap)
            if trimmed.strip():
                kept.append(ContextSection(s.name, trimmed))
                used += approx_tokens(f"## {s.name}\n{trimmed}")
        return kept

    def render(self, sections: list[ContextSection]) -> str:
        out = []
        for s in sections:
            out.append(f"## {s.name}\n{s.content}")
        return "\n\n".join(out)
