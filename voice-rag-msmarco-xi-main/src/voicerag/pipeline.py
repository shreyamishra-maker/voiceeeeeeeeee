"""Top-level convenience wrapper: build an index from the dataset, construct
the harness, and expose `.ask_text()` / `.ask_audio()`.
"""
from __future__ import annotations

from .config import PipelineConfig
from .data_loader import load_msmarco_xi, rows_to_documents
from .chunking import ChunkingPipeline
from .embeddings import get_embedder
from .vector_store import HybridVectorStore
from .stt import build_stt
from .generation import ExtractiveGenerator, LLMGenerator
from .harness import VoiceRAGHarness
from .schemas import PipelineResponse


class VoiceRAGPipeline:
    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.store: HybridVectorStore | None = None
        self.harness: VoiceRAGHarness | None = None

    def build_index(self, split: str = "train", limit: int | None = 200,
                    language: str = "hi") -> "VoiceRAGPipeline":
        rows = load_msmarco_xi(split=split, language=language, limit=limit,
                               streaming=True)
        docs = rows_to_documents(rows)
        corpus_texts = [d[1] for d in docs]

        embedder = get_embedder(self.cfg.retrieval.embedding_model, corpus_for_idf=corpus_texts)
        chunker = ChunkingPipeline(
            embedder=embedder,
            strategies=self.cfg.chunking.strategies,
            fixed_size=self.cfg.chunking.fixed_chunk_size_tokens,
            fixed_overlap=self.cfg.chunking.fixed_chunk_overlap_tokens,
            sw_window=self.cfg.chunking.sentence_window_size,
            sw_overlap=self.cfg.chunking.sentence_window_overlap,
            semantic_threshold=self.cfg.chunking.semantic_similarity_drop_threshold,
        )
        chunks = chunker.run_corpus(docs)

        store = HybridVectorStore(embedder)
        store.build(chunks)
        self.store = store

        if self.cfg.generation.provider == "llm" and self.cfg.stt.sarvam_api_key:
            generator = LLMGenerator(endpoint="https://api.anthropic.com/v1/messages",
                                      api_key="", model="claude-sonnet-4-6")
        else:
            generator = ExtractiveGenerator()

        stt = build_stt(self.cfg.stt)
        self.harness = VoiceRAGHarness(self.cfg, stt, store, generator)
        return self

    def ask_text(self, query: str, language: str | None = None) -> PipelineResponse:
        assert self.harness is not None, "call build_index() first"
        return self.harness.run_text(query, language=language)

    def ask_audio(self, audio_bytes: bytes) -> PipelineResponse:
        assert self.harness is not None, "call build_index() first"
        return self.harness.run_audio(audio_bytes)
