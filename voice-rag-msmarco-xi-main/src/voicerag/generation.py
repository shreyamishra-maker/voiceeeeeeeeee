"""Answer generation layer, pluggable.

ExtractiveGenerator: builds an answer directly from the top retrieved
chunk(s) with a light templating pass (no external API call). This is the
default because it's the only option that can realistically stay inside a
200ms *end-to-end* budget alongside STT + retrieval -- a network round trip
to any hosted LLM (GPT/Claude/etc.) typically costs 300ms-2s on its own,
which the task's own latency target cannot accommodate if generation is a
network call. See README "Latency budget vs. LLM generation" for the honest
trade-off discussion.

LLMGenerator: pluggable call to a hosted chat-completion API for teams that
want higher-quality synthesized answers and are willing to report generation
latency as a separate, larger number from retrieval latency (which is what
we'd recommend doing transparently rather than silently missing the budget).
"""
from __future__ import annotations

import abc
import time
import urllib.request
import json as _json

from .schemas import RetrievedChunk, GenerationResult


class AnswerGenerator(abc.ABC):
    provider_name: str

    @abc.abstractmethod
    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> GenerationResult:
        ...


class ExtractiveGenerator(AnswerGenerator):
    provider_name = "extractive"

    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> GenerationResult:
        t0 = time.perf_counter()
        if not retrieved:
            latency_ms = (time.perf_counter() - t0) * 1000
            return GenerationResult(answer="", provider=self.provider_name,
                                     supporting_chunk_ids=[], latency_ms=latency_ms)

        from .chunking import split_sentences

        top = retrieved[0]
        supporting_ids = [r.chunk.chunk_id for r in retrieved[:3]]
        # Simple, fast synthesis: lead with the best-scoring passage, then
        # append only genuinely new *sentences* (not already covered) from
        # the next best passages, so overlapping windows from different
        # chunking strategies don't produce repeated text in the answer.
        seen_sentences: set[str] = set()
        answer_sentences: list[str] = []
        for r in [top] + list(retrieved[1:3]):
            for sent in split_sentences(r.chunk.text):
                key = sent.strip().lower()
                if key and key not in seen_sentences:
                    seen_sentences.add(key)
                    answer_sentences.append(sent.strip())
            if r is top:
                continue
        answer = " ".join(answer_sentences)
        latency_ms = (time.perf_counter() - t0) * 1000
        return GenerationResult(answer=answer, provider=self.provider_name,
                                 supporting_chunk_ids=supporting_ids, latency_ms=latency_ms)


class LLMGenerator(AnswerGenerator):
    """Generic hosted-LLM generator. Point `endpoint`/`headers_fn`/`payload_fn`
    at whatever chat-completions API you're using (Anthropic, OpenAI, etc.);
    kept provider-agnostic so swapping vendors doesn't touch the harness.
    """
    provider_name = "llm"

    def __init__(self, endpoint: str, api_key: str, model: str, timeout_s: float = 8.0):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def _build_prompt(self, query: str, retrieved: list[RetrievedChunk]) -> str:
        context = "\n\n".join(f"[{i+1}] {r.chunk.text}" for i, r in enumerate(retrieved))
        return (
            "Answer the question using ONLY the numbered context passages. "
            "If the context doesn't contain the answer, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )

    def generate(self, query: str, retrieved: list[RetrievedChunk]) -> GenerationResult:
        t0 = time.perf_counter()
        prompt = self._build_prompt(query, retrieved)
        body = _json.dumps({
            "model": self.model,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            self.endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = _json.loads(resp.read().decode())
        text = "".join(b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text")
        latency_ms = (time.perf_counter() - t0) * 1000
        return GenerationResult(answer=text, provider=self.provider_name,
                                 supporting_chunk_ids=[r.chunk.chunk_id for r in retrieved],
                                 latency_ms=latency_ms)
