"""
Central configuration for the Voice-Enabled RAG pipeline.

All tunables live here so the harness, chunkers, retriever and guardrails
can be reconfigured from one place (or from environment variables) without
touching pipeline logic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class STTConfig:
    provider: str = os.environ.get("STT_PROVIDER", "sarvam")  # "sarvam" | "elevenlabs" | "mock"
    sarvam_api_key: str | None = os.environ.get("SARVAM_API_KEY")
    elevenlabs_api_key: str | None = os.environ.get("ELEVENLABS_API_KEY")
    language_hint: str = os.environ.get("STT_LANGUAGE_HINT", "hi-IN")  # MSMARCO-XI is Indic
    max_retries: int = 2
    timeout_s: float = 8.0


@dataclass
class ChunkingConfig:
    # Fixed-size chunker
    fixed_chunk_size_tokens: int = 120
    fixed_chunk_overlap_tokens: int = 24

    # Sentence-window chunker
    sentence_window_size: int = 3
    sentence_window_overlap: int = 1

    # Semantic chunker
    semantic_similarity_drop_threshold: float = 0.35  # split when similarity to running window drops below this

    # Which strategies to build (all of them get indexed, tagged by strategy,
    # and the retriever fuses results across them)
    strategies: tuple[str, ...] = ("fixed", "sentence_window", "semantic", "metadata_aware")


@dataclass
class RetrievalConfig:
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    top_k_dense: int = 8
    top_k_sparse: int = 8
    top_k_final: int = 5
    rrf_k: int = 60  # reciprocal rank fusion constant
    latency_budget_ms: float = 200.0  # hard requirement from task spec


@dataclass
class GuardrailConfig:
    off_topic_similarity_floor: float = 0.18   # below this max-sim to corpus -> off topic
    grounding_overlap_floor: float = 0.20      # below this token-overlap w/ context -> hallucination risk
    unsafe_categories: tuple[str, ...] = (
        "self_harm", "violence", "csam", "weapons", "illegal_drugs", "hate",
    )
    refusal_message_off_topic: str = (
        "I can only answer questions grounded in the provided document collection. "
        "That looks outside its scope, so I won't guess."
    )
    refusal_message_unsafe: str = (
        "I can't help with that request."
    )
    refusal_message_ungrounded: str = (
        "I found related passages but couldn't produce an answer that's clearly "
        "supported by them, so I'd rather say I don't know than guess."
    )


@dataclass
class GenerationConfig:
    provider: str = os.environ.get("GEN_PROVIDER", "extractive")  # "extractive" | "llm"
    max_answer_tokens: int = 120


@dataclass
class TTSConfig:
    provider: str = os.environ.get("TTS_PROVIDER", "elevenlabs")  # "elevenlabs" | "mock"
    elevenlabs_api_key: str | None = os.environ.get("ELEVENLABS_API_KEY")
    voice_id: str = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Sarah
    model_id: str = os.environ.get("ELEVENLABS_TTS_MODEL_ID", "eleven_multilingual_v2")
    stability: float = 0.5
    similarity_boost: float = 0.75
    timeout_s: float = 12.0



@dataclass
class HarnessConfig:
    max_retries_per_step: int = 2
    retry_backoff_base_s: float = 0.05
    step_timeout_s: float = 5.0


@dataclass
class PineconeConfig:
    api_key: str | None = os.environ.get("PINECONE_API_KEY")
    index_name: str = os.environ.get("PINECONE_INDEX_NAME", "voicerag-index")
    host: str | None = os.environ.get("PINECONE_HOST")  # e.g. "https://voicerag-index-xxxx.svc.pinecone.io"
    namespace: str = os.environ.get("PINECONE_NAMESPACE", "")
    top_k: int = 8
    timeout_s: float = 8.0


@dataclass
class PipelineConfig:
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    index_dir: str = os.environ.get("INDEX_DIR", "data/index")


