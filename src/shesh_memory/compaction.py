"""Episodic memory compaction and retention.

Over time the episode log grows without bound. This module summarizes old
episodes into the durable semantic memory (`semantic.md`) and trims the
episode JSONL to a retention window. The summarizer function is injected
so tests use a deterministic stub while production wires in the LLM.

Policy:
- Episodes older than `summarize_after_days` are batched and summarized.
- The summary is appended to semantic memory.
- Episodes older than `delete_after_days` are removed from the log.
- Recent episodes (within the retention window) are always kept verbatim.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A summarizer takes a batch of episode dicts and returns a markdown summary.
Summarizer = Callable[[list[dict[str, Any]]], str]

DAY = 86_400


def simple_summarizer(episodes: list[dict[str, Any]]) -> str:
    """Deterministic fallback: group by kind and list key facts.

    Production replaces this with an LLM call via shesh-mind.
    """
    if not episodes:
        return ""
    by_kind: dict[str, list[str]] = {}
    for e in episodes:
        by_kind.setdefault(e.get("kind", "unknown"), []).append(
            str(e.get("content", ""))[:200])
    lines = [f"- {kind}: {len(items)} event(s)" for kind, items in by_kind.items()]
    return "Compacted summary:\n" + "\n".join(lines)


@dataclass
class CompactionConfig:
    summarize_after_days: float = 14.0
    delete_after_days: float = 90.0
    batch_size: int = 200


@dataclass
class CompactionReport:
    summarized: int = 0
    deleted: int = 0
    summary_added: bool = False

    def to_dict(self) -> dict:
        return {
            "summarized": self.summarized,
            "deleted": self.deleted,
            "summary_added": self.summary_added,
        }


def _read_episodes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def compact(
    store_root: Path,
    config: CompactionConfig | None = None,
    summarizer: Summarizer | None = None,
    *,
    now: float | None = None,
) -> CompactionReport:
    """Summarize old episodes and delete very old ones.

    Reads episodes.jsonl, partitions by age, summarizes the middle band
    into semantic.md, rewrites the log keeping only recent episodes plus
    anything not yet old enough to delete.
    """
    cfg = config or CompactionConfig()
    summarize = summarizer or simple_summarizer
    t = now if now is not None else time.time()

    episodes_path = store_root / "episodes.jsonl"
    semantic_path = store_root / "semantic.md"
    episodes = _read_episodes(episodes_path)

    summarize_cutoff = t - cfg.summarize_after_days * DAY
    delete_cutoff = t - cfg.delete_after_days * DAY

    to_summarize: list[dict[str, Any]] = []
    to_delete: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []

    for ep in episodes:
        ts = float(ep.get("ts", t))
        if ts < delete_cutoff:
            to_delete.append(ep)
        elif ts < summarize_cutoff:
            to_summarize.append(ep)
        else:
            kept.append(ep)

    report = CompactionReport(deleted=len(to_delete))

    # Summarize in batches so prompts stay bounded.
    if to_summarize:
        chunks = [to_summarize[i:i + cfg.batch_size]
                  for i in range(0, len(to_summarize), cfg.batch_size)]
        summaries = [summarize(chunk) for chunk in chunks if chunk]
        summaries = [s for s in summaries if s]
        if summaries:
            header = f"\n## Compacted {time.strftime('%Y-%m-%d', time.gmtime(t))}\n"
            with semantic_path.open("a", encoding="utf-8") as f:
                f.write(header + "\n".join(summaries) + "\n")
            report.summary_added = True
        report.summarized = len(to_summarize)
        # Once summarized, these episodes are dropped from the live log
        # (their content lives in semantic memory).
        kept = kept  # no change; to_summarize is removed below

    # Rewrite the episode log: keep only recent episodes. The summarized band
    # is removed (its content now lives in semantic.md); the delete band too.
    if episodes_path.exists():
        out = "\n".join(json.dumps(e, ensure_ascii=False) for e in kept if e)
        episodes_path.write_text(out + ("\n" if out else ""), encoding="utf-8")

    return report
