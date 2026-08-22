"""Vector DB layer: FAISS (dense, cosine via inner-product on normalized
vectors) fused with BM25 (sparse, lexical) via Reciprocal Rank Fusion.

Why hybrid instead of "just FAISS": MSMARCO-style queries are often short and
keyword-heavy (names, numbers, entities) where lexical match beats dense
embeddings, especially with the offline fallback embedder. Dense retrieval
covers paraphrase/semantic queries. Fusing both is cheap (both indices are
in-memory, query-time cost is dominated by one matmul + one BM25 scan) and
keeps us inside the 200ms budget while materially improving recall over
either alone.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

try:
    import faiss
except ImportError:
    faiss = None

import numpy as np
from rank_bm25 import BM25Okapi

from .schemas import Chunk, RetrievedChunk
from .embeddings import Embedder, _tokenize as _et_tokenize
from .chunking import split_sentences  # noqa: F401  (kept for reuse by callers)


def _tokenize(text: str) -> list[str]:
    return _et_tokenize(text)


class HybridVectorStore:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._faiss_index: faiss.Index | None = None
        self._bm25: BM25Okapi | None = None

    # ---------- build ----------
    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]

        vecs = self.embedder.encode(texts)
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product == cosine on normalized vecs
        index.add(vecs)
        self._faiss_index = index

        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    # ---------- persist ----------
    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_index, str(p / "faiss.index"))
        with open(p / "meta.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self._bm25}, f)

    def load(self, path: str) -> None:
        p = Path(path)
        self._faiss_index = faiss.read_index(str(p / "faiss.index"))
        with open(p / "meta.pkl", "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self._bm25 = data["bm25"]

    # ---------- query ----------
    def search(self, query: str, top_k_dense: int = 8, top_k_sparse: int = 8,
               top_k_final: int = 5, rrf_k: int = 60,
               strategy_filter: str | None = None,
               language_filter: str | None = None) -> tuple[list[RetrievedChunk], float]:
        t0 = time.perf_counter()

        q_vec = self.embedder.encode([query])
        dense_scores, dense_idx = self._faiss_index.search(q_vec, top_k_dense)
        dense_idx = dense_idx[0]
        dense_scores = dense_scores[0]

        bm25_scores_all = self._bm25.get_scores(_tokenize(query))
        sparse_idx = np.argsort(bm25_scores_all)[::-1][:top_k_sparse]

        # Reciprocal Rank Fusion
        fused: dict[int, float] = {}
        for rank, idx in enumerate(dense_idx):
            if idx < 0:
                continue
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(sparse_idx):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        candidates = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        results: list[RetrievedChunk] = []
        for idx, fused_score in candidates:
            chunk = self.chunks[idx]
            if strategy_filter and chunk.strategy != strategy_filter:
                continue
            if language_filter and chunk.metadata.get("language") != language_filter:
                continue
            d_score = float(dense_scores[list(dense_idx).index(idx)]) if idx in dense_idx else None
            s_score = float(bm25_scores_all[idx])
            results.append(RetrievedChunk(chunk=chunk, dense_score=d_score,
                                           sparse_score=s_score, fused_score=fused_score))
            if len(results) >= top_k_final:
                break

        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms
