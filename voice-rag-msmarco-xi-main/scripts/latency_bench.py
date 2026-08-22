"""Measure P50/P70/P100 latency for the local RAG harness."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from voicerag.config import PipelineConfig
from voicerag.embeddings import get_lightweight_embedder
from voicerag.generation import ExtractiveGenerator
from voicerag.harness import VoiceRAGHarness
from voicerag.lightweight_store import LightweightVectorStore
from voicerag.stt import MockSTT


def percentile(values: list[float], percentage: int) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentage / 100) * (len(ordered) - 1)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--output", default="data/latency_report.json")
    args = parser.parse_args()
    if args.n < 3:
        raise SystemExit("--n must be at least 3")

    cfg = PipelineConfig()
    embedder = get_lightweight_embedder()
    store = LightweightVectorStore(embedder)
    store.load(str(_ROOT / "data" / "index"))
    harness = VoiceRAGHarness(cfg, MockSTT(cfg.stt), store, ExtractiveGenerator())
    queries = [
        "Where is the Taj Mahal located?",
        "What is the capital of India?",
        "Where does the Ganges river originate?",
        "Which is the national animal of India?",
        "When was Mahatma Gandhi born?",
    ]

    retrieval_ms: list[float] = []
    end_to_end_ms: list[float] = []
    for index in range(args.n):
        query = queries[index % len(queries)]
        start = time.perf_counter()
        result = harness.run_text(query)
        end_to_end_ms.append((time.perf_counter() - start) * 1000)
        retrieval_ms.append(next(t.latency_ms for t in result.timings if t.stage == "retrieval"))

    report = {
        "queries": args.n,
        "index_chunks": len(store.chunks),
        "retrieval_only_ms": {f"p{p}": round(percentile(retrieval_ms, p), 3) for p in (50, 70, 100)},
        "end_to_end_ms": {f"p{p}": round(percentile(end_to_end_ms, p), 3) for p in (50, 70, 100)},
        "mean_end_to_end_ms": round(statistics.fmean(end_to_end_ms), 3),
    }
    output = _ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
