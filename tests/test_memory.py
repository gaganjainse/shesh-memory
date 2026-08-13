"""Offline tests for memory store, habits, and context assembly."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shesh_memory.context import Budget, ContextAssembler, approx_tokens  # noqa: E402
from shesh_memory.habits import HabitLearner  # noqa: E402
from shesh_memory.store import MemoryStore  # noqa: E402


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(root=tmp_path, use_fts=False)
    yield s
    s.close()


def test_record_and_recent(store):
    store.record("observation", "user opened VS Code")
    store.record("action", "ran cargo test")
    rec = store.recent(5)
    assert len(rec) == 2
    assert rec[-1].content == "ran cargo test"


def test_search_falls_back_to_scan(store):
    store.record("observation", "loves Rust")
    store.record("observation", "prefers dark theme")
    hits = store.search("rust", limit=5)
    assert any("Rust" in h.content for h in hits)


def test_semantic_facts(store):
    store.append_semantic("User speaks English and Hindi.")
    store.append_semantic("Prefers bullet summaries.")
    text = store.read_semantic()
    assert "English and Hindi" in text and "bullet" in text


def test_habit_promotes_after_corroboration(store):
    learner = HabitLearner(store)
    # Need multiple corroborating observations to cross PROMOTE_AT (~log scaling)
    for _ in range(40):
        h = learner.observe("action:focus|hour:10", "enters focus mode ~10am", success=True)
    assert h.promoted
    assert any(x.signature == "action:focus|hour:10" for x in learner.active_habits())


def test_habit_not_promoted_on_one_observation(store):
    h = HabitLearner(store).observe("action:x", "did x once", success=True)
    assert not h.promoted


def test_habit_decay_archives_stale(store):
    learner = HabitLearner(store)
    for _ in range(40):
        learner.observe("sig", "a strong habit", success=True)
    # Force it old and decayed
    h = learner.habits["sig"]
    h.last_seen -= 100 * 24 * 3600  # 100 days
    h.last_decayed -= 100 * 24 * 3600  # 100 days since last decay run
    archived = learner.tick_decay()
    assert any(x.signature == "sig" for x in archived)
    assert not any(x.signature == "sig" for x in learner.active_habits())


def test_habit_persistence_roundtrip(store):
    learner = HabitLearner(store)
    learner.observe("sig", "persists", success=True)
    # New learner instance reads from disk
    learner2 = HabitLearner(store)
    assert "sig" in learner2.habits


def test_context_budget_respected(store):
    store.write_semantic("fact one\n" * 100)
    for i in range(50):
        store.record("episode", f"event number {i}")
    asm = ContextAssembler(store, Budget(total=500))
    sections = asm.build(query="event", n_relevant=20, n_recent=20)
    total = sum(s.tokens for s in sections)
    assert total <= 500
    # semantic (facts) should survive trimming over recent
    assert any(s.name == "semantic" for s in sections)


def test_context_includes_working_and_query(store):
    store.record("observation", "writing a parser in Rust")
    asm = ContextAssembler(store, Budget(total=2000))
    sections = asm.build(query="parser", working="Task: fix lexer bug")
    names = [s.name for s in sections]
    assert "working" in names
    # relevant episodes should mention parser
    rel = next(s for s in sections if s.name == "relevant")
    assert "parser" in rel.content.lower()


def test_approx_tokens_is_conservative():
    assert approx_tokens("hello") == 2
    assert approx_tokens("") >= 1


def test_mcp_server_tools_use_store(tmp_path, monkeypatch):
    # Import server and point it at a temp store
    from shesh_memory import server
    monkeypatch.setattr(server, "_store", MemoryStore(root=tmp_path, use_fts=False))
    monkeypatch.setattr(server, "_learner", None)
    assert server.remember("action", "did a thing")["ok"]
    ctx = server.assemble_context(query="thing", max_tokens=1000)
    assert "prompt" in ctx and ctx["tokens"] <= 1000
    server._store.close()
