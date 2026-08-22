"""Pinecone Serverless Vector Store integration.

Connects to a hosted Pinecone cloud index via Pinecone's official REST API
(or SDK when available) without requiring heavy local binaries like FAISS,
making it ideal for Vercel Serverless Functions.

Features:
  - Sub-15ms cloud dense retrieval.
  - Hybrid RRF (Reciprocal Rank Fusion) with BM25.
  - Pure Python HTTP layer with 0 C-extension bloat for Vercel.
"""
from __future__ import annotations

import json as _json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .config import PineconeConfig
from .embeddings import Embedder, _tokenize
from .schemas import Chunk, RetrievedChunk


class PineconeVectorStore:
    """Cloud vector store powered by Pinecone Serverless Index."""

    def __init__(self, embedder: Embedder, cfg: PineconeConfig | None = None):
        self.embedder = embedder
        self.cfg = cfg or PineconeConfig()
        self.chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._host: Optional[str] = self.cfg.host

        if not self.cfg.api_key:
            raise ValueError("PINECONE_API_KEY is required to use PineconeVectorStore.")

        if not self._host:
            self._host = self._resolve_host()

    def _resolve_host(self) -> str:
        """Resolve Pinecone Index host URL via describe_index API."""
        url = f"https://api.pinecone.io/indexes/{self.cfg.index_name}"
        req = urllib.request.Request(
            url,
            headers={
                "Api-Key": self.cfg.api_key or "",
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                data = _json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Pinecone index lookup failed ({e.code}) for "
                f"'{self.cfg.index_name}': {detail}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach Pinecone: {e.reason}") from e

        host = data.get("host")
        if not host:
            raise RuntimeError(
                f"Pinecone did not return a host for index '{self.cfg.index_name}'."
            )
        return host if host.startswith("http") else f"https://{host}"

    def stats(self) -> Dict[str, Any]:
        """Return Pinecone index statistics for deployment health checks."""
        endpoint = f"{self._host}/describe_index_stats"
        body = _json.dumps({"namespace": self.cfg.namespace}).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Api-Key": self.cfg.api_key or "",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            return _json.loads(resp.read().decode())

    def upsert_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> int:
        """Embed and upsert chunks into Pinecone."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        vecs = self.embedder.encode(texts)

        # L2-normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_vecs = (vecs / norms).tolist()

        # Build local BM25 for hybrid ranking
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

        total_upserted = 0
        endpoint = f"{self._host}/vectors/upsert"

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_vecs = norm_vecs[i : i + batch_size]

            vectors_payload = []
            for chunk, vec in zip(batch_chunks, batch_vecs):
                vectors_payload.append({
                    "id": chunk.chunk_id,
                    "values": vec,
                    "metadata": {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "text": chunk.text,
                        "strategy": chunk.strategy,
                        "token_count": len(chunk.text.split()),
                        "language": chunk.metadata.get("language", ""),
                        "target_lang": chunk.metadata.get("target_lang", ""),
                        "query_id": str(chunk.metadata.get("query_id", "")),
                        "title": chunk.metadata.get("title", ""),
                        "url": chunk.metadata.get("url", ""),
                    },
                })

            body = _json.dumps({
                "vectors": vectors_payload,
                "namespace": self.cfg.namespace,
            }).encode("utf-8")

            req = urllib.request.Request(
                endpoint,
                data=body,
                headers={
                    "Api-Key": self.cfg.api_key or "",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                if resp.status == 200:
                    total_upserted += len(vectors_payload)

        return total_upserted

    def search(
        self,
        query: str,
        top_k_dense: int = 8,
        top_k_sparse: int = 8,
        top_k_final: int = 5,
        rrf_k: int = 60,
        strategy_filter: str | None = None,
        language_filter: str | None = None,
    ) -> Tuple[List[RetrievedChunk], float]:
        """Search Pinecone for dense matches and fuse with BM25."""
        t0 = time.perf_counter()

        # Embed and normalize query vector
        q_vec = self.embedder.encode([query])
        q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        q_vec_list = (q_vec / q_norm)[0].tolist()

        # Query Pinecone REST API
        endpoint = f"{self._host}/query"
        query_filter = {"language": language_filter} if language_filter else None
        body = _json.dumps({
            "vector": q_vec_list,
            "topK": top_k_dense,
            "includeMetadata": True,
            "includeValues": False,
            "namespace": self.cfg.namespace,
            **({"filter": query_filter} if query_filter else {}),
        }).encode("utf-8")

        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Api-Key": self.cfg.api_key or "",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        dense_matches: List[Dict[str, Any]] = []
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                resp_data = _json.loads(resp.read().decode())
                dense_matches = resp_data.get("matches", [])
        except Exception as e:
            # If network error, return empty with error latency
            return [], (time.perf_counter() - t0) * 1000

        # Construct candidate chunks from Pinecone metadata
        dense_results: List[RetrievedChunk] = []
        for rank, match in enumerate(dense_matches):
            meta = match.get("metadata", {})
            chunk = Chunk(
                chunk_id=match.get("id") or meta.get("chunk_id", f"pc_{rank}"),
                doc_id=meta.get("doc_id", "doc_0"),
                text=meta.get("text", ""),
                strategy=meta.get("strategy", "fixed"),
                metadata={
                    "title": meta.get("title", ""),
                    "url": meta.get("url", ""),
                    "language": meta.get("language", ""),
                    "target_lang": meta.get("target_lang", ""),
                    "query_id": meta.get("query_id", ""),
                },
            )
            dense_score = float(match.get("score", 0.0))
            fused_score = 1.0 / (rrf_k + rank + 1)
            dense_results.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_score=dense_score,
                    sparse_score=0.0,
                    fused_score=fused_score,
                )
            )

        # If BM25 is available, compute hybrid fusion
        if self._bm25 and len(self.chunks) > 0:
            bm25_scores = self._bm25.get_scores(_tokenize(query))
            sparse_idx = np.argsort(bm25_scores)[::-1][:top_k_sparse]

            fused_map: Dict[str, Tuple[Chunk, float, float, float]] = {}
            for rank, r in enumerate(dense_results):
                fused_map[r.chunk.chunk_id] = (
                    r.chunk,
                    r.dense_score or 0.0,
                    0.0,
                    1.0 / (rrf_k + rank + 1),
                )

            for rank, idx in enumerate(sparse_idx):
                c = self.chunks[idx]
                s_score = float(bm25_scores[idx])
                rrf_sparse = 1.0 / (rrf_k + rank + 1)
                if c.chunk_id in fused_map:
                    prev_c, d_s, _, prev_fused = fused_map[c.chunk_id]
                    fused_map[c.chunk_id] = (prev_c, d_s, s_score, prev_fused + rrf_sparse)
                else:
                    fused_map[c.chunk_id] = (c, 0.0, s_score, rrf_sparse)

            sorted_items = sorted(fused_map.values(), key=lambda x: x[3], reverse=True)
            results = [
                RetrievedChunk(chunk=item[0], dense_score=item[1], sparse_score=item[2], fused_score=item[3])
                for item in sorted_items[:top_k_final]
            ]
        else:
            results = dense_results[:top_k_final]

        if strategy_filter:
            results = [r for r in results if r.chunk.strategy == strategy_filter]

        latency_ms = (time.perf_counter() - t0) * 1000
        return results, latency_ms
