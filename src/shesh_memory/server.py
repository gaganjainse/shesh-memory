"""MCP server exposing memory/learning tools to the agent."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

try:
    from shesh_audit.mcp_guard import GuardedMCP as _MCP
except ImportError:
    _MCP = FastMCP

from .context import Budget, ContextAssembler
from .habits import HabitLearner
from .intentions import Intentions, Mannerisms
from .store import MemoryStore

mcp = _MCP("shesh-memory")

# A process-local store; overridable for tests.
_store: MemoryStore | None = None
_learner: HabitLearner | None = None


def _s() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def _l() -> HabitLearner:
    global _learner
    if _learner is None:
        _learner = HabitLearner(_s())
    return _learner


@mcp.tool()
def remember(kind: str, content: str, **metadata) -> dict:
    """Record an episode (observation, action, outcome, conversation)."""
    ep = _s().record(kind, content, **metadata)
    return {"ok": True, "ts": ep.ts}


@mcp.tool()
def recall(query: str, limit: int = 8) -> list[dict]:
    """Retrieve relevant past episodes by keyword/search."""
    return [{"ts": e.ts, "kind": e.kind, "content": e.content}
            for e in _s().search(query, limit)]


@mcp.tool()
def learn_habit(signature: str, description: str, success: bool = True) -> dict:
    """Note a recurring pattern so it can become a learned habit."""
    h = _l().observe(signature, description, success=success)
    return {"signature": h.signature, "confidence": round(h.confidence, 2),
            "promoted": h.promoted, "count": h.count}


@mcp.tool()
def list_habits(include_archived: bool = False) -> list[dict]:
    """List learned habits (promoted patterns about the user)."""
    out = []
    for h in _l().habits.values():
        if h.archived and not include_archived:
            continue
        out.append({
            "signature": h.signature, "description": h.description,
            "count": h.count, "confidence": round(h.confidence, 2),
            "promoted": h.promoted, "archived": h.archived,
        })
    return out


@mcp.tool()
def note_fact(fact: str) -> dict:
    """Append a durable semantic fact about the user/preferences/intentions."""
    _s().append_semantic(fact)
    return {"ok": True}


@mcp.tool()
def set_mannerism(preference: str) -> dict:
    """Record a communication/style preference (how the user likes responses)."""
    Mannerisms(_s().root).append(preference)
    return {"ok": True}


_intentions: Intentions | None = None


def _i() -> Intentions:
    global _intentions
    if _intentions is None:
        _intentions = Intentions(_s().root)
    return _intentions


@mcp.tool()
def add_intention(title: str, priority: int = 3) -> dict:
    """Record something the user is trying to achieve (active goal)."""
    it = _i().add(title, priority)
    return {"id": it.id, "title": it.title, "priority": it.priority}


@mcp.tool()
def complete_intention(intention_id: str) -> dict:
    """Mark an intention/goal as completed."""
    _i().complete(intention_id)
    return {"ok": True}


@mcp.tool()
def list_intentions() -> list[dict]:
    """List active intentions/goals, highest priority first."""
    return [{"id": i.id, "title": i.title, "priority": i.priority}
            for i in _i().active()]


@mcp.tool()
def assemble_context(query: str = "", working: str = "", skills: str = "",
                     max_tokens: int = 6000) -> dict:
    """Build a token-bounded context block for the current turn."""
    asm = ContextAssembler(_s(), Budget(total=max_tokens))
    sections = asm.build(query=query, working=working, skills=skills)
    return {
        "prompt": asm.render(sections),
        "tokens": sum(s.tokens for s in sections),
        "sections": [s.name for s in sections],
    }


@mcp.tool()
def decay_habits() -> list[dict]:
    """Run daily decay; archives habits that stopped recurring."""
    return [{"signature": h.signature, "description": h.description}
            for h in _l().tick_decay()]


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()


@mcp.tool()
def compact_memory(summarize_after_days: float = 14.0,
                   delete_after_days: float = 90.0) -> dict:
    """Summarize old episodes into semantic memory and trim the episode log.

    Runs retention: episodes older than summarize_after_days are compacted
    (via the default summarizer; LLM-backed summarization can be injected),
    episodes older than delete_after_days are removed.
    """
    from pathlib import Path
    from .compaction import CompactionConfig, compact
    from .store import DATA_DIR
    cfg = CompactionConfig(summarize_after_days=summarize_after_days,
                           delete_after_days=delete_after_days)
    report = compact(Path(DATA_DIR), cfg)
    return report.to_dict()


# ── semantic/vector search (optional) ────────────────────────────────
_vstore = None

def _vector_store():
    global _vstore
    if _vstore is None:
        from pathlib import Path
        from .embeddings import local_embedder, LOCAL_DIM
        from .vectorstore import VectorStore
        from .store import DATA_DIR
        _vstore = VectorStore(DATA_DIR / "vectors.sqlite", local_embedder(), LOCAL_DIM)
    return _vstore


@mcp.tool()
def semantic_search(query: str, limit: int = 5) -> list[dict]:
    """Search memories by embedding similarity (local hash embeddings offline;
    swap in Ollama nomic-embed-text for real semantics)."""
    return _vector_store().search(query, limit)


@mcp.tool()
def index_memory(text: str, memory_id: str | None = None, **metadata) -> dict:
    """Add a text to the semantic vector index."""
    import uuid
    mid = memory_id or uuid.uuid4().hex[:12]
    _vector_store().upsert(mid, text, metadata)
    return {"ok": True, "id": mid}
