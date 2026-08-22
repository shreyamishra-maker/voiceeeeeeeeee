"""Guardrail layer. Three checkpoints in the pipeline:

1. Input safety   -- runs on the transcribed query, before any retrieval or
                      generation spend. Blocks clearly unsafe/inappropriate
                      requests (self-harm, weapons, csam, hate, illegal
                      instructions) with a category-level pattern match. This
                      is a coarse, fast, defense-in-depth layer -- pair it
                      with your model provider's own safety filtering, don't
                      rely on it alone.
2. Off-topic      -- runs after embedding the query, before generation. If
                      the query's best similarity to anything in the corpus
                      is below a floor, we refuse rather than let the
                      generator hallucinate an answer with no real support.
3. Grounding      -- runs after generation. Scores lexical overlap between
                      the generated answer and the retrieved context it was
                      supposedly built from. Low overlap => likely
                      unsupported/hallucinated claim => refuse instead of
                      returning it.

Each check returns a GuardrailVerdict so the harness can log exactly which
stage rejected a request and why, for the guardrail failure analytics you'll
want to report alongside the latency numbers.
"""
from __future__ import annotations

import re

from .config import GuardrailConfig
from .embeddings import Embedder, _tokenize
from .schemas import GuardrailVerdict, RetrievedChunk

# Coarse category patterns for the input-safety layer. This is intentionally
# a blunt first line of defense (pattern/keyword match), not a substitute for
# a real safety classifier -- swap in a hosted moderation endpoint for
# production use.
_UNSAFE_PATTERNS: dict[str, list[str]] = {
    "self_harm": [r"\bkill myself\b", r"\bsuicide\b", r"\bself[- ]harm\b"],
    "weapons": [r"\bmake a bomb\b", r"\bbuild.{0,15}explosive\b", r"\bgun.{0,15}how to make\b"],
    "illegal_drugs": [r"\bsynthesi[sz]e.{0,20}(meth|drug)\b"],
    "hate": [r"\bracial slur\b"],
}


class InputSafetyGuardrail:
    def __init__(self, cfg: GuardrailConfig):
        self.cfg = cfg
        self._compiled = {
            cat: [re.compile(p, re.IGNORECASE) for p in pats]
            for cat, pats in _UNSAFE_PATTERNS.items()
            if cat in cfg.unsafe_categories
        }

    def check(self, query: str) -> GuardrailVerdict:
        for cat, patterns in self._compiled.items():
            for p in patterns:
                if p.search(query):
                    return GuardrailVerdict(passed=False, stage="input_safety",
                                             reason=f"matched unsafe category: {cat}")
        return GuardrailVerdict(passed=True, stage="input_safety")


class OffTopicGuardrail:
    def __init__(self, cfg: GuardrailConfig):
        self.cfg = cfg

    def check(self, best_fused_score: float, best_dense_score: float | None) -> GuardrailVerdict:
        score = best_dense_score if best_dense_score is not None else best_fused_score
        if score is None or score < self.cfg.off_topic_similarity_floor:
            return GuardrailVerdict(passed=False, stage="off_topic",
                                     reason=f"best similarity {score} below floor "
                                            f"{self.cfg.off_topic_similarity_floor}")
        return GuardrailVerdict(passed=True, stage="off_topic")


class GroundingGuardrail:
    """Lexical-overlap grounding check: what fraction of the answer's content
    tokens also appear in the retrieved context. Cheap proxy for "is this
    answer actually supported by what we retrieved" that needs no extra model
    call, so it doesn't blow the latency budget.
    """

    def __init__(self, cfg: GuardrailConfig):
        self.cfg = cfg

    def check(self, answer: str, retrieved: list[RetrievedChunk]) -> GuardrailVerdict:
        if not answer.strip():
            return GuardrailVerdict(passed=False, stage="grounding", reason="empty answer")
        context_tokens = set()
        for r in retrieved:
            context_tokens |= set(_tokenize(r.chunk.text))
        answer_tokens = set(_tokenize(answer))
        if not answer_tokens:
            return GuardrailVerdict(passed=False, stage="grounding", reason="no scorable tokens")
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        if overlap < self.cfg.grounding_overlap_floor:
            return GuardrailVerdict(passed=False, stage="grounding",
                                     reason=f"overlap {overlap:.2f} below floor "
                                            f"{self.cfg.grounding_overlap_floor}")
        return GuardrailVerdict(passed=True, stage="grounding", reason=f"overlap {overlap:.2f}")
