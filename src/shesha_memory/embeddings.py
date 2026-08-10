"""Pluggable embeddings for semantic memory retrieval.

The default backend is a deterministic local hash embedding so the system
works fully offline. When an Ollama embedding model is available, it is
used for real semantic search. The provider returns a fixed-dimension
vector; dimensions mismatch is handled by the vector store.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Embedder = Callable[[str], list[float]]

LOCAL_DIM = 256


def local_embedder(dim: int = LOCAL_DIM) -> Embedder:
    """Deterministic bag-of-tokens hash embedding (offline).

    Not semantically meaningful, but stable and good enough to test the
    retrieval pipeline without a model. It hashes each token into a
    dimension and accumulates weighted counts, then L2-normalizes.
    """
    def _embed(text: str) -> list[float]:
        vec = [0.0] * dim
        for token in _tokenize(text):
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
    return _embed


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


def ollama_embedder(model: str = "nomic-embed-text",
                    base_url: str = "http://localhost:11434",
                    dim: int = 768, timeout: int = 30) -> Embedder:
    """Build an embedder backed by a local Ollama embedding model."""
    def _embed(text: str) -> list[float]:
        body = json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/embeddings",
            data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        vec = data.get("embedding", [])
        if not vec:
            return local_embedder(dim)(text)
        return [float(x) for x in vec]
    return _embed


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; handles dimension mismatch gracefully."""
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class EmbeddingResult:
    text: str
    score: float
    metadata: dict[str, Any]
