"""Orchestration harness.

This is what satisfies requirement 5 ("your model/pipeline should be run
inside a proper harness"): the pipeline below is NOT a single prompt-in/
text-out call. It is a typed state machine over discrete steps, each with:

  - structured input/output (pydantic schemas, not raw strings)
  - bounded retries with backoff for steps that can transiently fail
    (STT call, embedding call, generation call)
  - per-step latency capture, fed straight into the P50/P70/P100 benchmark
  - explicit error recovery: a step that exhausts its retries degrades the
    pipeline to a safe refusal response (schemas.PipelineResponse) instead
    of raising and killing the request
  - three guardrail checkpoints wired in as first-class steps, not
    afterthoughts (see guardrails.py)
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

from .config import PipelineConfig
from .schemas import PipelineResponse, StageTiming, GuardrailVerdict, RetrievedChunk
from .stt import SpeechToText
from .vector_store import HybridVectorStore
from .lightweight_store import LightweightVectorStore
from .pinecone_store import PineconeVectorStore
from .guardrails import InputSafetyGuardrail, OffTopicGuardrail, GroundingGuardrail
from .generation import AnswerGenerator

T = TypeVar("T")


def _run_with_retries(fn: Callable[[], T], max_retries: int, backoff_base: float,
                       stage_name: str) -> tuple[T | None, StageTiming]:
    last_err: Exception | None = None
    t0 = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            result = fn()
            latency_ms = (time.perf_counter() - t0) * 1000
            return result, StageTiming(stage=stage_name, latency_ms=latency_ms,
                                        retries=attempt, ok=True)
        except Exception as e:  # noqa: BLE001 - intentional broad catch at harness boundary
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff_base * (2 ** attempt))
    latency_ms = (time.perf_counter() - t0) * 1000
    return None, StageTiming(stage=stage_name, latency_ms=latency_ms,
                              retries=max_retries, ok=False, error=str(last_err))


class VoiceRAGHarness:
    def __init__(self, cfg: PipelineConfig, stt: SpeechToText,
                 store: HybridVectorStore | LightweightVectorStore | PineconeVectorStore,
                 generator: AnswerGenerator):
        self.cfg = cfg
        self.stt = stt
        self.store = store
        self.generator = generator
        self.input_guardrail = InputSafetyGuardrail(cfg.guardrails)
        self.off_topic_guardrail = OffTopicGuardrail(cfg.guardrails)
        self.grounding_guardrail = GroundingGuardrail(cfg.guardrails)

    # ---- entry points ----
    def run_text(self, query: str, language: str | None = None) -> PipelineResponse:
        return self._run(query, language=language)

    def run_audio(self, audio_bytes: bytes) -> PipelineResponse:
        timings: list[StageTiming] = []
        h = self.cfg.harness

        transcription, t_timing = _run_with_retries(
            lambda: self.stt.transcribe(audio_bytes),
            h.max_retries_per_step, h.retry_backoff_base_s, "stt",
        )
        timings.append(t_timing)
        if transcription is None:
            return PipelineResponse(query="<transcription failed>", status="error",
                                     timings=timings,
                                     total_latency_ms=sum(t.latency_ms for t in timings))
        response = self._run(transcription.text, extra_timings=timings)
        return response

    # ---- core pipeline ----
    def _run(self, query: str, extra_timings: list[StageTiming] | None = None,
             language: str | None = None) -> PipelineResponse:
        timings: list[StageTiming] = list(extra_timings or [])
        verdicts: list[GuardrailVerdict] = []
        h = self.cfg.harness

        # 1. input safety guardrail
        t0 = time.perf_counter()
        safety_verdict = self.input_guardrail.check(query)
        timings.append(StageTiming(stage="guardrail_input_safety",
                                    latency_ms=(time.perf_counter() - t0) * 1000))
        verdicts.append(safety_verdict)
        if not safety_verdict.passed:
            return PipelineResponse(query=query, status="refused_unsafe",
                                     guardrail_verdicts=verdicts, timings=timings,
                                     total_latency_ms=sum(t.latency_ms for t in timings))

        # 2. retrieval (chunking already done at index-build time; this is
        #    the query-time chunk + vector DB retrieval the latency budget targets)
        retrieved, retrieval_err = self._retrieve_with_retries(query, h, language)
        timings.append(retrieval_err)
        if retrieved is None:
            return PipelineResponse(query=query, status="error",
                                     guardrail_verdicts=verdicts, timings=timings,
                                     total_latency_ms=sum(t.latency_ms for t in timings))

        # 3. off-topic guardrail
        t0 = time.perf_counter()
        best_dense = max((r.dense_score for r in retrieved if r.dense_score is not None), default=None)
        best_fused = max((r.fused_score for r in retrieved), default=0.0)
        off_topic_verdict = self.off_topic_guardrail.check(best_fused, best_dense)
        timings.append(StageTiming(stage="guardrail_off_topic",
                                    latency_ms=(time.perf_counter() - t0) * 1000))
        verdicts.append(off_topic_verdict)
        if not off_topic_verdict.passed:
            return PipelineResponse(query=query, status="refused_off_topic",
                                     guardrail_verdicts=verdicts, retrieved=retrieved,
                                     timings=timings,
                                     total_latency_ms=sum(t.latency_ms for t in timings))

        # 4. generation (with retries + error recovery)
        gen_result, gen_timing = _run_with_retries(
            lambda: self.generator.generate(query, retrieved),
            h.max_retries_per_step, h.retry_backoff_base_s, "generation",
        )
        timings.append(gen_timing)
        if gen_result is None:
            return PipelineResponse(query=query, status="error",
                                     guardrail_verdicts=verdicts, retrieved=retrieved,
                                     timings=timings,
                                     total_latency_ms=sum(t.latency_ms for t in timings))

        # 5. grounding / hallucination guardrail
        t0 = time.perf_counter()
        grounding_verdict = self.grounding_guardrail.check(gen_result.answer, retrieved)
        timings.append(StageTiming(stage="guardrail_grounding",
                                    latency_ms=(time.perf_counter() - t0) * 1000))
        verdicts.append(grounding_verdict)
        if not grounding_verdict.passed:
            return PipelineResponse(query=query, status="refused_ungrounded",
                                     guardrail_verdicts=verdicts, retrieved=retrieved,
                                     timings=timings,
                                     total_latency_ms=sum(t.latency_ms for t in timings))

        return PipelineResponse(query=query, answer=gen_result.answer, status="answered",
                                 guardrail_verdicts=verdicts, retrieved=retrieved,
                                 timings=timings,
                                 total_latency_ms=sum(t.latency_ms for t in timings))

    def _retrieve_with_retries(self, query: str, h, language: str | None = None) -> tuple[list[RetrievedChunk] | None, StageTiming]:
        r = self.cfg.retrieval
        box: dict = {}

        def _do():
            results, latency_ms = self.store.search(
                query, top_k_dense=r.top_k_dense, top_k_sparse=r.top_k_sparse,
                top_k_final=r.top_k_final, rrf_k=r.rrf_k,
                language_filter=language,
            )
            box["inner_latency_ms"] = latency_ms
            return results

        results, timing = _run_with_retries(_do, h.max_retries_per_step,
                                             h.retry_backoff_base_s, "retrieval")
        return results, timing
