"""Chunking layer -- deliberately NOT a single naive fixed-size splitter.

Implements four strategies and a pipeline that runs all (or a chosen subset)
of them over every document, tagging each chunk with its `strategy` so the
retriever can compare / fuse across them at query time:

1. FixedSizeChunker      -- token-count windows with configurable overlap.
                             Cheap baseline, good recall on short factoid queries.
2. SentenceWindowChunker -- groups N sentences with overlap of M sentences,
                             preserves local syntactic coherence better than
                             raw token windows (good for MSMARCO-style short
                             passages that are already ~1-3 sentences).
3. SemanticChunker       -- greedily grows a chunk while consecutive sentences
                             stay embedding-similar to the running chunk
                             centroid, and cuts when similarity drops below a
                             threshold. Approximates topic-boundary splitting
                             without needing an LLM call.
4. MetadataAwareChunker  -- treats dataset-native boundaries (MSMARCO passage
                             id, query id, url/title fields, is_selected flag)
                             as first-class chunk boundaries and attaches them
                             as retrievable/filterable metadata instead of
                             throwing them away.

All chunkers share the `Chunk` schema so they're interchangeable in the index.
"""
from __future__ import annotations

import re
import uuid
from typing import Iterable

import numpy as np

from .schemas import Chunk
from .embeddings import Embedder

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\u0964])\s+")  # incl. Devanagari danda \u0964


def split_sentences(text: str) -> list[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents or ([text.strip()] if text.strip() else [])


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class FixedSizeChunker:
    strategy = "fixed"

    def __init__(self, size_tokens: int = 120, overlap_tokens: int = 24):
        self.size = size_tokens
        self.overlap = overlap_tokens

    def chunk(self, doc_id: str, text: str, metadata: dict) -> list[Chunk]:
        tokens = text.split()
        if not tokens:
            return []
        step = max(1, self.size - self.overlap)
        chunks = []
        for start in range(0, len(tokens), step):
            window = tokens[start:start + self.size]
            if not window:
                continue
            chunks.append(Chunk(
                chunk_id=_new_id(), doc_id=doc_id, text=" ".join(window),
                strategy=self.strategy,
                metadata={**metadata, "token_start": start, "token_end": start + len(window)},
            ))
            if start + self.size >= len(tokens):
                break
        return chunks


class SentenceWindowChunker:
    strategy = "sentence_window"

    def __init__(self, window: int = 3, overlap: int = 1):
        self.window = window
        self.overlap = overlap

    def chunk(self, doc_id: str, text: str, metadata: dict) -> list[Chunk]:
        sents = split_sentences(text)
        if not sents:
            return []
        step = max(1, self.window - self.overlap)
        chunks = []
        for start in range(0, len(sents), step):
            window = sents[start:start + self.window]
            if not window:
                continue
            chunks.append(Chunk(
                chunk_id=_new_id(), doc_id=doc_id, text=" ".join(window),
                strategy=self.strategy,
                metadata={**metadata, "sentence_start": start, "sentence_end": start + len(window)},
            ))
            if start + self.window >= len(sents):
                break
        return chunks


class SemanticChunker:
    strategy = "semantic"

    def __init__(self, embedder: Embedder, similarity_drop_threshold: float = 0.35):
        self.embedder = embedder
        self.threshold = similarity_drop_threshold

    def chunk(self, doc_id: str, text: str, metadata: dict) -> list[Chunk]:
        sents = split_sentences(text)
        if not sents:
            return []
        if len(sents) == 1:
            return [Chunk(chunk_id=_new_id(), doc_id=doc_id, text=sents[0],
                           strategy=self.strategy, metadata=metadata)]

        vecs = self.embedder.encode(sents)
        chunks: list[Chunk] = []
        current = [sents[0]]
        centroid = vecs[0].copy()
        count = 1
        for i in range(1, len(sents)):
            sim = float(np.dot(centroid / (np.linalg.norm(centroid) + 1e-9),
                                vecs[i] / (np.linalg.norm(vecs[i]) + 1e-9)))
            if sim < self.threshold and current:
                chunks.append(Chunk(chunk_id=_new_id(), doc_id=doc_id, text=" ".join(current),
                                     strategy=self.strategy, metadata=metadata))
                current = [sents[i]]
                centroid = vecs[i].copy()
                count = 1
            else:
                current.append(sents[i])
                centroid = centroid + vecs[i]
                count += 1
        if current:
            chunks.append(Chunk(chunk_id=_new_id(), doc_id=doc_id, text=" ".join(current),
                                 strategy=self.strategy, metadata=metadata))
        return chunks


class MetadataAwareChunker:
    """Respects native MSMARCO-XI boundaries: one chunk per passage, but keeps
    is_selected / query_id / language / url as structured, filterable metadata
    instead of flattening everything into plain text. Enables filtered
    retrieval later (e.g. "only search Hindi passages" or "only positive
    passages") which pure text chunking throws away.
    """
    strategy = "metadata_aware"

    def chunk(self, doc_id: str, text: str, metadata: dict) -> list[Chunk]:
        if not text.strip():
            return []
        return [Chunk(
            chunk_id=_new_id(), doc_id=doc_id, text=text.strip(),
            strategy=self.strategy,
            metadata={**metadata, "is_native_passage": True},
        )]


class ChunkingPipeline:
    """Runs the configured set of strategies over a corpus and returns the
    union of all resulting chunks, each tagged by strategy so the vector
    store can be queried per-strategy or across all of them.
    """

    def __init__(self, embedder: Embedder, strategies: Iterable[str],
                 fixed_size: int = 120, fixed_overlap: int = 24,
                 sw_window: int = 3, sw_overlap: int = 1,
                 semantic_threshold: float = 0.35):
        self._chunkers = {}
        if "fixed" in strategies:
            self._chunkers["fixed"] = FixedSizeChunker(fixed_size, fixed_overlap)
        if "sentence_window" in strategies:
            self._chunkers["sentence_window"] = SentenceWindowChunker(sw_window, sw_overlap)
        if "semantic" in strategies:
            self._chunkers["semantic"] = SemanticChunker(embedder, semantic_threshold)
        if "metadata_aware" in strategies:
            self._chunkers["metadata_aware"] = MetadataAwareChunker()

    def run(self, doc_id: str, text: str, metadata: dict) -> list[Chunk]:
        out: list[Chunk] = []
        for chunker in self._chunkers.values():
            out.extend(chunker.chunk(doc_id, text, metadata))
        return out

    def run_corpus(self, docs: list[tuple[str, str, dict]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc_id, text, metadata in docs:
            chunks.extend(self.run(doc_id, text, metadata))
        return chunks
