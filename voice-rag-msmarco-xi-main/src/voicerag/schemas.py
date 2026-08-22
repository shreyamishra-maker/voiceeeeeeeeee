"""Structured I/O contracts. Every stage of the harness reads/writes one of
these instead of passing raw strings/dicts around, so failures are typed and
retries/error-recovery have something concrete to inspect.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class TranscriptionResult(BaseModel):
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    provider: str
    latency_ms: float


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    strategy: str  # fixed | sentence_window | semantic | metadata_aware
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fused_score: float = 0.0


class GuardrailVerdict(BaseModel):
    passed: bool
    stage: Literal["input_safety", "off_topic", "grounding"]
    reason: Optional[str] = None


class GenerationResult(BaseModel):
    answer: str
    provider: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: float


class StageTiming(BaseModel):
    stage: str
    latency_ms: float
    retries: int = 0
    ok: bool = True
    error: Optional[str] = None


class PipelineResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    status: Literal["answered", "refused_unsafe", "refused_off_topic", "refused_ungrounded", "error"]
    guardrail_verdicts: list[GuardrailVerdict] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    total_latency_ms: float = 0.0
