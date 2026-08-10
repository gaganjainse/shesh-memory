"""Storage layer — append-only JSONL + SQLite index.

Design goals:
- local-first, no external service
- memory is human-readable/editable (markdown for semantic, jsonl for events)
- cheap to write; retrieval can be affordably expensive

Files under XDG_DATA_HOME/shesha/memory/:
  episodes.jsonl      one JSON event per line (append-only)
  semantic.md         durable facts as a markdown list
  habits.json         learned habits with evidence counts
  fts.sqlite          full-text index over episodes (optional)
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DATA_DIR = Path.home() / ".local" / "share" / "sesha" / "memory"


@dataclass
class Episode:
    ts: float
    kind: str           # observation | action | outcome | conversation
    content: str
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class MemoryStore:
    def __init__(self, root: Path | None = None, use_fts: bool = True) -> None:
        self.root = root or DATA_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.root / "episodes.jsonl"
        self.semantic_path = self.root / "semantic.md"
        self.habits_path = self.root / "habits.json"
        self.fts_path = self.root / "fts.sqlite"
        self.conn: sqlite3.Connection | None = None
        if use_fts:
            self._init_fts()

    # ── episodes ─────────────────────────────────────────────────────────
    def _init_fts(self) -> None:
        self.conn = sqlite3.connect(self.fts_path)
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS episodes USING fts5(ts, kind, content)"
        )
        self.conn.commit()

    def record(self, kind: str, content: str, **metadata) -> Episode:
        ep = Episode(ts=time.time(), kind=kind, content=content, metadata=metadata)
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(ep.to_json() + "\n")
        if self.conn is not None:
            self.conn.execute(
                "INSERT INTO episodes(ts, kind, content) VALUES (?, ?, ?)",
                (ep.ts, ep.kind, ep.content),
            )
            self.conn.commit()
        return ep

    def recent(self, n: int = 20) -> list[Episode]:
        if not self.episodes_path.exists():
            return []
        lines = self.episodes_path.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines[-n:]:
            try:
                d = json.loads(line)
                out.append(Episode(**d))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    def search(self, query: str, limit: int = 10) -> list[Episode]:
        if self.conn is None:
            return [e for e in self.recent(200) if query.lower() in e.content.lower()][:limit]
        cur = self.conn.execute(
            "SELECT ts, kind, content FROM episodes WHERE episodes MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        )
        return [Episode(ts=r[0], kind=r[1], content=r[2]) for r in cur.fetchall()]

    # ── semantic facts ───────────────────────────────────────────────────
    def read_semantic(self) -> str:
        return self.semantic_path.read_text(encoding="utf-8") if self.semantic_path.exists() else ""

    def write_semantic(self, text: str) -> None:
        self.semantic_path.write_text(text, encoding="utf-8")

    def append_semantic(self, line: str) -> None:
        with self.semantic_path.open("a", encoding="utf-8") as f:
            f.write(f"- {line.rstrip()}\n")

    # ── habits ───────────────────────────────────────────────────────────
    def read_habits(self) -> dict:
        if not self.habits_path.exists():
            return {}
        return json.loads(self.habits_path.read_text(encoding="utf-8"))

    def write_habits(self, habits: dict) -> None:
        self.habits_path.write_text(
            json.dumps(habits, indent=2, ensure_ascii=False), encoding="utf-8")

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
