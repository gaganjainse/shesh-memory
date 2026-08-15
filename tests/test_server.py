"""Smoke tests for the memory MCP server — every tool callable, right schema.

Isolation: the store's DATA_DIR is redirected to a temp dir and the server's
lazy singletons are reset, so no test touches the user's real memory dir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import shesh_memory.store as store_mod  # noqa: E402
from shesh_memory import server  # noqa: E402

EXPECTED_TOOLS = {
    "remember", "recall", "learn_habit", "list_habits", "note_fact",
    "set_mannerism", "add_intention", "complete_intention", "list_intentions",
    "assemble_context", "decay_habits", "compact_memory",
    "semantic_search", "index_memory",
}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "DATA_DIR", tmp_path)
    server._store = None
    server._learner = None
    server._intentions = None
    server._vstore = None
    yield
    if server._store is not None:
        server._store.close()
    if server._vstore is not None:
        server._vstore.close()


def test_all_tools_registered():
    import asyncio
    registered = {t.name for t in asyncio.run(server.mcp.list_tools())}
    missing = EXPECTED_TOOLS - registered
    assert not missing, f"unregistered tools: {sorted(missing)}"


def test_remember_recall_roundtrip():
    assert server.remember("action", "edited the token file")["ok"] is True
    hits = server.recall("token file")
    assert hits and any("token" in h["content"] for h in hits)
    for h in hits:
        assert set(h) == {"ts", "kind", "content"}


def test_habit_and_intention_schema():
    h = server.learn_habit("runs pytest before push", "verifies locally")
    assert set(h) == {"signature", "confidence", "promoted", "count"}
    listed = server.list_habits()
    assert any(x["signature"] == "runs pytest before push" for x in listed)

    it = server.add_intention("ship v0.1.0", priority=1)
    assert set(it) == {"id", "title", "priority"}
    assert server.complete_intention(it["id"]) == {"ok": True}
    assert all(x["id"] != it["id"] for x in server.list_intentions())


def test_fact_mannerism_context():
    assert server.note_fact("prefers flat badges") == {"ok": True}
    assert server.set_mannerism("be terse") == {"ok": True}
    ctx = server.assemble_context(query="badges", max_tokens=4000)
    assert set(ctx) == {"prompt", "tokens", "sections"}
    assert ctx["tokens"] <= 4000
    assert isinstance(ctx["sections"], list)


def test_compaction_and_decay_empty():
    report = server.compact_memory()
    assert isinstance(report, dict)
    assert server.decay_habits() == []


def test_vector_index_roundtrip():
    assert server.index_memory("shesh is the agent OS", memory_id="t1") == {
        "ok": True, "id": "t1"}
    # Exact-text query scores 1.0 against itself (local hash embeddings).
    results = server.semantic_search("shesh is the agent OS")
    assert isinstance(results, list)
    assert results, "indexed text should be self-similar"
    for r in results:
        assert set(r) == {"text", "score", "metadata"}
    assert any("agent OS" in r["text"] for r in results)
