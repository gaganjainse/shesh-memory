"""MCP server exposing memory/learning tools to the agent."""
from __future__ import annotations

from shesh_audit.mcp_guard import GuardedMCP as _MCP

from .context import Budget, ContextAssembler
from .habits import HabitLearner
from .intentions import Intentions, Mannerisms
from .store import MemoryStore
from .vectorstore import VectorStore

mcp = _MCP("shesh-memory")

# A process-local store; overridable for tests.
_store: MemoryStore | None = None
_learner: HabitLearner | None = None


def _get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def _get_learner() -> HabitLearner:
    global _learner
    if _learner is None:
        _learner = HabitLearner(_get_store())
    return _learner


@mcp.tool()
def remember(kind: str, content: str, metadata: dict | None = None) -> dict:
    """Record an episode (observation, action, outcome, conversation).

    metadata carries optional structured fields (source, tags). A dict
    parameter, not **kwargs: FastMCP 3 schemas cannot express **kwargs.
    """
    ep = _get_store().record(kind, content, **(metadata or {}))
    return {"ok": True, "ts": ep.ts}


@mcp.tool()
def recall(query: str, limit: int = 8) -> list[dict]:
    """Retrieve relevant past episodes by keyword/search."""
    return [{"ts": e.ts, "kind": e.kind, "content": e.content}
            for e in _get_store().search(query, limit)]


@mcp.tool()
def learn_habit(signature: str, description: str, success: bool = True) -> dict:
    """Note a recurring pattern so it can become a learned habit."""
    h = _get_learner().observe(signature, description, success=success)
    return {"signature": h.signature, "confidence": round(h.confidence, 2),
            "promoted": h.promoted, "count": h.count}


@mcp.tool()
def list_habits(include_archived: bool = False) -> list[dict]:
    """List learned habits (promoted patterns about the user)."""
    out = []
    for h in _get_learner().habits.values():
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
    _get_store().append_semantic(fact)
    return {"ok": True}


@mcp.tool()
def set_mannerism(preference: str) -> dict:
    """Record a communication/style preference (how the user likes responses)."""
    Mannerisms(_get_store().root).append(preference)
    return {"ok": True}


_intentions: Intentions | None = None


def _get_intentions() -> Intentions:
    global _intentions
    if _intentions is None:
        _intentions = Intentions(_get_store().root)
    return _intentions


@mcp.tool()
def add_intention(title: str, priority: int = 3) -> dict:
    """Record something the user is trying to achieve (active goal)."""
    it = _get_intentions().add(title, priority)
    return {"id": it.id, "title": it.title, "priority": it.priority}


@mcp.tool()
def complete_intention(intention_id: str) -> dict:
    """Mark an intention/goal as completed."""
    _get_intentions().complete(intention_id)
    return {"ok": True}


@mcp.tool()
def list_intentions() -> list[dict]:
    """List active intentions/goals, highest priority first."""
    return [{"id": i.id, "title": i.title, "priority": i.priority}
            for i in _get_intentions().active()]


@mcp.tool()
def assemble_context(query: str = "", working: str = "", skills: str = "",
                     max_tokens: int = 6000) -> dict:
    """Build a token-bounded context block for the current turn."""
    asm = ContextAssembler(_get_store(), Budget(total=max_tokens))
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
            for h in _get_learner().tick_decay()]


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
_vstore: VectorStore | None = None


def _vector_store() -> VectorStore:
    global _vstore
    if _vstore is None:
        from .embeddings import LOCAL_DIM, local_embedder
        from .store import DATA_DIR
        _vstore = VectorStore(DATA_DIR / "vectors.sqlite", local_embedder(), LOCAL_DIM)
    return _vstore


@mcp.tool()
def semantic_search(query: str, limit: int = 5) -> list[dict]:
    """Search memories by embedding similarity (local hash embeddings offline;
    swap in Ollama nomic-embed-text for real semantics)."""
    return _vector_store().search(query, limit)


@mcp.tool()
def index_memory(text: str, memory_id: str | None = None,
                 metadata: dict | None = None) -> dict:
    """Add a text to the semantic vector index (metadata as explicit dict:
    **kwargs cannot be expressed in FastMCP 3 tool schemas)."""
    import uuid
    mid = memory_id or uuid.uuid4().hex[:12]
    _vector_store().upsert(mid, text, metadata or {})
    return {"ok": True, "id": mid}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
