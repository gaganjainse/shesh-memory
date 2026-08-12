"""Offline tests for embeddings and the vector store."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shesh_memory.embeddings import (  # noqa: E402
    LOCAL_DIM,
    cosine,
    local_embedder,
)
from shesh_memory.vectorstore import VectorStore  # noqa: E402


def test_local_embedder_dimension_and_normalization():
    emb = local_embedder()
    v = emb("hello world")
    assert len(v) == LOCAL_DIM
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_similar_texts_have_higher_cosine_than_unrelated():
    emb = local_embedder()
    cat = emb("the cat sat on the mat")
    dog = emb("the dog sat on the rug")
    car = emb("quantum physics and calculus")
    assert cosine(cat, dog) > cosine(cat, car)


def test_cosine_range():
    emb = local_embedder()
    a = emb("apples and oranges")
    assert 0.0 <= cosine(a, a) <= 1.01


def test_vectorstore_upsert_and_search(tmp_path):
    emb = local_embedder()
    store = VectorStore(tmp_path / "vec.db", emb, LOCAL_DIM)
    store.upsert("1", "the cat sat on the mat", {"kind": "observation"})
    store.upsert("2", "quantum physics lecture tonight", {"kind": "event"})
    results = store.search("cat", limit=2)
    assert results
    assert results[0]["metadata"]["kind"] == "observation"
    assert store.count() == 2
    store.close()


def test_vectorstore_persists(tmp_path):
    db = tmp_path / "vec.db"
    emb = local_embedder()
    s1 = VectorStore(db, emb, LOCAL_DIM)
    s1.upsert("1", "remember the milk")
    s1.close()
    s2 = VectorStore(db, emb, LOCAL_DIM)
    assert s2.count() == 1
    s2.close()


def test_vectorstore_empty_search(tmp_path):
    emb = local_embedder()
    store = VectorStore(tmp_path / "v.db", emb, LOCAL_DIM)
    assert store.search("anything") == []
    store.close()
