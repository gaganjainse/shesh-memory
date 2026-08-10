"""Tests for intentions/mannerisms and memory wiring."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shesh_memory.context import Budget, ContextAssembler  # noqa: E402
from shesh_memory.intentions import Intentions, Mannerisms  # noqa: E402
from shesh_memory.server import (  # noqa: E402
    add_intention,
    complete_intention,
    list_intentions,
    set_mannerism,
)
from shesh_memory.store import MemoryStore  # noqa: E402


def test_mannerism_seed_exists(tmp_path):
    m = Mannerisms(tmp_path)
    assert "concise" in m.text().lower()
    m.append("Always include a one-line summary.")
    assert "one-line summary" in m.text()


def test_intention_lifecycle(tmp_path):
    it = Intentions(tmp_path)
    a = it.add("refactor kernel", priority=1)
    it.add("buy groceries", priority=4)
    assert [x.title for x in it.active()] == ["refactor kernel", "buy groceries"]
    it.complete(a.id)
    assert [x.title for x in it.active()] == ["buy groceries"]
    # persisted
    it2 = Intentions(tmp_path)
    assert [x.title for x in it2.active()] == ["buy groceries"]


def test_context_includes_mannerisms_and_intentions(tmp_path):
    store = MemoryStore(root=tmp_path, use_fts=False)
    store.write_semantic("User likes Rust.")
    intentions = Intentions(tmp_path)
    intentions.add("ship shesh-memory", priority=1)
    asm = ContextAssembler(store, Budget(total=3000))
    sections = asm.build(query="memory", working="fix tests")
    rendered = asm.render(sections)
    assert "## mannerisms" in rendered
    assert "## intentions" in rendered
    assert "ship shesh-memory" in rendered
    store.close()


def test_mcp_intention_tools(tmp_path, monkeypatch):
    from shesh_memory import server
    store = MemoryStore(root=tmp_path, use_fts=False)
    monkeypatch.setattr(server, "_store", store)
    monkeypatch.setattr(server, "_intentions", None)
    added = add_intention("write docs", priority=2)
    assert added["title"] == "write docs"
    active = list_intentions()
    assert any(i["title"] == "write docs" for i in active)
    complete_intention(added["id"])
    assert list_intentions() == []
    set_mannerism("Use examples.")
    assert "Use examples." in Mannerisms(tmp_path).text()
    store.close()
