"""SQLite-backed vector store for semantic retrieval.

Stores (text, embedding, metadata) and performs brute-force cosine search.
Brute force is fine for personal-scale memory (thousands of episodes, not
millions). The store can rebuild from episodes.jsonl and persists vectors.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .embeddings import Embedder, cosine


class VectorStore:
    def __init__(self, db_path: Path, embedder: Embedder, dim: int) -> None:
        self.db_path = db_path
        self.embedder = embedder
        self.dim = dim
        self.conn = sqlite3.connect(str(db_path))
        self._init()

    def _init(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "id TEXT PRIMARY KEY, text TEXT, embedding TEXT, metadata TEXT)"
        )
        self.conn.commit()

    def upsert(self, item_id: str, text: str, metadata: dict | None = None) -> None:
        vec = self.embedder(text)
        self.conn.execute(
            "INSERT OR REPLACE INTO vectors(id, text, embedding, metadata) VALUES (?,?,?,?)",
            (item_id, text, json.dumps(vec), json.dumps(metadata or {})),
        )
        self.conn.commit()

    def search(self, query: str, limit: int = 5) -> list[dict]:
        qv = self.embedder(query)
        rows = self.conn.execute(
            "SELECT id, text, embedding, metadata FROM vectors").fetchall()
        scored = []
        for _id, text, emb_json, meta_json in rows:
            emb = json.loads(emb_json)
            score = cosine(qv, emb)
            scored.append((score, text, json.loads(meta_json)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"text": t, "score": round(s, 3), "metadata": m}
            for s, t, m in scored[:limit] if s > 0
        ]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
