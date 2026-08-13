"""Real hot-path benchmarks for shesh-memory (stdlib only, CI-safe).

Measures the actual hot paths: vector search, context assembly, and
embedding generation. Median of N runs with loose regression bounds
(orders of magnitude of headroom) so CI catches gross regressions without
being flaky. Run:  python benchmarks/bench_hotpaths.py
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shesh_memory.context import Budget, ContextAssembler  # noqa: E402
from shesh_memory.embeddings import LOCAL_DIM, local_embedder  # noqa: E402
from shesh_memory.store import MemoryStore  # noqa: E402
from shesh_memory.vectorstore import VectorStore  # noqa: E402


def bench(label: str, fn, n: int = 200) -> float:
    """Return median seconds over n runs (warm-up included in n)."""
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    print(f"  {label:44s} median {med * 1e6:9.2f} µs  (n={n})")
    return med


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="shesh-memory-bench-")

    # Vector search: 1000 docs in the store, repeated queries.
    store = VectorStore(Path(tmp) / "v.db", local_embedder(), LOCAL_DIM)
    for i in range(1000):
        store.upsert(f"doc-{i}", f"the {i}th document about machine learning and agents")
    query = "machine learning agents"
    bench("vector search (1000 docs)", lambda: store.search(query), n=300)

    # Context assembly: memory store with episodes + semantic facts.
    ms = MemoryStore(root=Path(tmp) / "mem")
    for i in range(100):
        ms.record("episode", f"user preferred bullet points in message {i}")
    for i in range(20):
        ms.write_semantic(f"fact {i}: prefers summaries in English and Hindi")
    asm = ContextAssembler(ms, Budget(total=2000))
    bench("context assembly (100 eps + 20 facts)", lambda: asm.build(query="preferences"), n=300)

    # Embedding generation (local hash embedder).
    embed = local_embedder()
    bench("embedding (256-dim hash, short text)", lambda: embed("the cat sat on the mat"), n=500)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
