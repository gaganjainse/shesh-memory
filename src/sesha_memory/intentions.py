"""Intentions and mannerisms — what the user is trying to do and how they like it.

Intentions are different from habits:
- HABITS are observed recurring patterns (frequentist).
- INTENTIONS are goals the user states or that are inferred from sequences of
  actions (e.g., "refactoring the kernel", "preparing a talk"). They have a
  lifecycle: active -> completed/abandoned, and a priority.
- MANNERISMS are communication/style preferences (terse, examples-first, etc.)

Both feed the ContextAssembler so Sesha adapts tone and suggestions without the
model having to re-learn them every turn. They're stored as editable Markdown so
the user stays in control — no opaque learned state.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Status(StrEnum):
    ACTIVE = "active"
    DONE = "done"
    ABANDONED = "abandoned"


@dataclass
class Intention:
    id: str
    title: str
    status: Status = Status.ACTIVE
    priority: int = 3          # 1 (high) .. 5 (low)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    notes: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "status": self.status.value,
                "priority": self.priority, "created": self.created,
                "updated": self.updated, "notes": self.notes}


class Intentions:
    """JSON-backed intention list, with a Markdown rendering for context."""

    def __init__(self, root: Path) -> None:
        self.path = root / "intentions.json"
        self.items: list[Intention] = []
        if self.path.exists():
            for d in json.loads(self.path.read_text(encoding="utf-8")):
                d["status"] = Status(d.get("status", "active"))
                self.items.append(Intention(**d))

    def add(self, title: str, priority: int = 3) -> Intention:
        it = Intention(id=uuid.uuid4().hex[:12], title=title, priority=priority)
        self.items.append(it)
        self._save()
        return it

    def complete(self, intention_id: str) -> None:
        for it in self.items:
            if it.id == intention_id:
                it.status = Status.DONE
                it.updated = time.time()
        self._save()

    def active(self) -> list[Intention]:
        return sorted((i for i in self.items if i.status == Status.ACTIVE),
                      key=lambda i: i.priority)

    def render(self) -> str:
        active = self.active()
        if not active:
            return ""
        lines = ["# Active intentions"]
        for it in active:
            lines.append(f"- [P{it.priority}] {it.title}")
        return "\n".join(lines)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([i.to_dict() for i in self.items], indent=2),
            encoding="utf-8",
        )


# Default mannerisms seed; edited over time via note_fact / direct editing.
DEFAULT_MANNERISMS = """# Communication style
- Be concise; lead with the answer, then details.
- Use tables for comparisons, bullet points otherwise.
- Cite sources; flag uncertainty.
- Prefer safe, reversible commands; confirm destructive ones.
"""


class Mannerisms:
    def __init__(self, root: Path) -> None:
        self.path = root / "mannerisms.md"
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(DEFAULT_MANNERISMS, encoding="utf-8")

    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def append(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"- {line.rstrip()}\n")
