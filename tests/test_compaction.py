"""Offline tests for episodic compaction/retention."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shesh_memory.compaction import (  # noqa: E402
    CompactionConfig, compact, simple_summarizer,
)
from shesh_memory.store import MemoryStore  # noqa: E402


DAY = 86_400


def _seed(store, now, ages_days, kind="observation"):
    # Write backdated episodes directly (record() would add a current-timestamp line).
    from shesh_memory.store import Episode
    for i, age in enumerate(ages_days):
        ep = Episode(ts=now - age * DAY, kind=kind, content=f"event {i} age {age}")
        with store.episodes_path.open("a", encoding="utf-8") as f:
            f.write(ep.to_json() + chr(10))


def test_summarizer_groups_by_kind():
    out = simple_summarizer([
        {"kind": "action", "content": "did x"},
        {"kind": "outcome", "content": "worked"},
    ])
    assert "action" in out and "outcome" in out


def test_compact_summarizes_old_and_trims(tmp_path):
    now = time.time()
    store = MemoryStore(root=tmp_path, use_fts=False)
    _seed(store, now, [0.1, 15, 100])  # recent, middle (summarize), old (delete)

    cfg = CompactionConfig(summarize_after_days=14, delete_after_days=90)
    report = compact(tmp_path, cfg, now=now)

    assert report.summarized == 1     # the 15-day event
    assert report.deleted == 1         # the 100-day event
    assert report.summary_added is True

    semantic = (tmp_path / "semantic.md").read_text()
    assert "Compacted" in semantic

    # episode log keeps only the recent one
    lines = [l for l in (tmp_path / "episodes.jsonl").read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert "age 0.1" in lines[0]


def test_compact_keeps_everything_when_all_recent(tmp_path):
    now = time.time()
    store = MemoryStore(root=tmp_path, use_fts=False)
    _seed(store, now, [1, 2, 3])
    report = compact(tmp_path, CompactionConfig(summarize_after_days=30), now=now)
    assert report.summarized == 0
    lines = (tmp_path / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 3


def test_compact_empty_store(tmp_path):
    report = compact(tmp_path)
    assert report.summarized == 0 and report.deleted == 0


def test_custom_summarizer_used(tmp_path):
    now = time.time()
    store = MemoryStore(root=tmp_path, use_fts=False)
    _seed(store, now, [20])
    seen = []

    def fake_sum(eps):
        seen.extend(eps)
        return "CUSTOM-SUMMARY"

    compact(tmp_path, CompactionConfig(summarize_after_days=14),
            summarizer=fake_sum, now=now)
    assert seen and "CUSTOM-SUMMARY" in (tmp_path / "semantic.md").read_text()
