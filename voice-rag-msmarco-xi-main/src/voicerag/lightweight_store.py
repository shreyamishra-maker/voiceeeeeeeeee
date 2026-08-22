"""Lightweight vector store — pure numpy cosine similarity + BM25, no FAISS.

Used for Vercel serverless deployment where faiss-cpu exceeds the 250MB limit.
API-compatible with HybridVectorStore so the harness/pipeline code works unchanged.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .schemas import Chunk, RetrievedChunk
from .embeddings import Embedder, _tokenize as _et_tokenize


def _tokenize(text: str) -> list[str]:
    return _et_tokenize(text)


class LightweightVectorStore:
    """Pure-numpy vector store: cosine similarity via matrix multiply on
    L2-normalized vectors, fused with BM25 via reciprocal rank fusion.
    Drop-in replacement for HybridVectorStore that needs no FAISS."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        vecs = self.embedder.encode(texts)
        # L2-normalize for cosine similarity via dot product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = vecs / norms

        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(str(p / "vectors.npy"), self._vectors)
        with open(p / "meta.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self._bm25}, f)

    def load(self, path: str) -> None:
        p = Path(path)
        self._vectors = np.load(str(p / "vectors.npy"))
        with open(p / "meta.pkl", "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self._bm25 = data["bm25"]

    def search(self, query: str, top_k_dense: int = 8, top_k_sparse: int = 8,
               top_k_final: int = 5, rrf_k: int = 60,
               strategy_filter: str | None = None,
               language_filter: str | None = None) -> tuple[list[RetrievedChunk], float]:
        t0 = time.perf_counter()

        # Dense retrieval: cosine similarity via dot product
        q_vec = self.embedder.encode([query])
        q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        q_vec = q_vec / q_norm

        dense_scores = (self._vectors @ q_vec.T).flatten()
        dense_idx = np.argsort(dense_scores)[::-1][:top_k_dense]

        # Sparse retrieval: BM25
        bm25_scores_all = self._bm25.get_scores(_tokenize(query))
        sparse_idx = np.argsort(bm25_scores_all)[::-1][:top_k_sparse]

        # Reciprocal Rank Fusion
        fused: dict[int, float] = {}
        for rank, idx in enumerate(dense_idx):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(sparse_idx):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)

        candidates = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        results: list[RetrievedChunk] = []
        for idx, fused_score in candidates:
            chunk = self.chunks[idx]
            if strategy_filter and chunk.strategy != strategy_filter:
                continue
            if language_filter and chunk.metadata.get("language") != language_filter:
                continue
            d_score = float(dense_scores[idx])
            s_score = float(bm25_scores_all[idx])
            results.append(RetrievedChunk(chunk=chunk, dense_score=d_score,
                                           sparse_score=s_score, fused_score=fused_score))
            if len(results) >= top_k_final:
                break

        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms
