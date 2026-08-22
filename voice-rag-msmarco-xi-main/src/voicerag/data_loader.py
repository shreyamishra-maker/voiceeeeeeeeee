"""Loads the task-mandated dataset: https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

Real path (needs internet + `datasets` installed):
    from datasets import load_dataset
    ds = load_dataset("ai4bharat/MSMARCO-XI", split="train")

This dev/sandbox environment has no route to huggingface.co, so
`load_msmarco_xi()` tries the real HF load first and transparently falls back
to `data/sample_msmarco_xi.jsonl` -- a small hand-built sample in the same
shape (query, passage, passage_id, query_id, language, is_selected) so the
whole pipeline (chunking -> index -> retrieval -> guardrails -> generation ->
latency bench) is runnable and testable end-to-end without the real dataset.
Point INDEX at the real dataset before final submission.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Resolve sample path relative to this file's location, works both locally
# and inside Vercel's serverless function file structure.
_THIS_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = _THIS_DIR.parent.parent / "data" / "sample_msmarco_xi.jsonl"

LANGUAGE_CONFIGS = {
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "sa": "san_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
}

# Fallback: check if sample is at project root (for Vercel where cwd differs)
if not SAMPLE_PATH.exists():
    _alt = Path(os.environ.get("VERCEL_PROJECT_ROOT", "")) / "data" / "sample_msmarco_xi.jsonl"
    if _alt.exists():
        SAMPLE_PATH = _alt


def _flatten_row(row: dict, language: str) -> list[dict]:
    passages = row.get("passages")
    if not isinstance(passages, dict):
        return [row]

    translated = passages.get("Translated_passages") or []
    english = passages.get("English_passages") or []
    selected = passages.get("is_selected") or []
    flattened = []
    for index, text in enumerate(translated):
        if not text:
            continue
        flattened.append({
            "passage_id": f"{row.get('query_id', 'query')}-{index}",
            "query_id": row.get("query_id"),
            "query": row.get("query", ""),
            "passage": text,
            "english_passage": english[index] if index < len(english) else "",
            "language": language,
            "is_selected": selected[index] if index < len(selected) else None,
            "query_type": row.get("query_type"),
            "answer": row.get("Answer", row.get("answer", "")),
            "source_lang": row.get("source_lang", "eng_Latn"),
            "target_lang": row.get("target_lang", LANGUAGE_CONFIGS.get(language, language)),
            "title": "",
            "url": "",
        })
    return flattened


def load_msmarco_xi(split: str = "train", limit: int | None = None,
                    language: str = "hi", streaming: bool = True) -> list[dict]:
    """Load translated MSMARCO-XI passages for one language configuration.

    The sample fallback is useful for local development, but production/index
    jobs can set ``MSMARCO_XI_STRICT=1`` to fail loudly when Hugging Face is
    unavailable instead of building an index from the sample corpus.
    """
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("ai4bharat/MSMARCO-XI", language, split=split,
                          streaming=streaming)
        flattened = []
        row_count = 0
        for row in ds:
            flattened.extend(_flatten_row(dict(row), language))
            row_count += 1
            if limit is not None and row_count >= limit:
                break
        if not flattened:
            raise RuntimeError(
                f"MSMARCO-XI/{language} returned no translated passages for split {split!r}"
            )
        return flattened
    except Exception as error:
        if os.environ.get("MSMARCO_XI_STRICT", "").lower() in {"1", "true", "yes"}:
            raise RuntimeError(
                f"Unable to load ai4bharat/MSMARCO-XI/{language} ({split})"
            ) from error
        rows = []
        # Try multiple possible locations for the sample file
        sample_paths = [
            SAMPLE_PATH,
            _THIS_DIR.parent.parent / "data" / "sample_msmarco_xi.jsonl",
            Path.cwd() / "data" / "sample_msmarco_xi.jsonl",
        ]
        sample_file = None
        for sp in sample_paths:
            if sp.exists():
                sample_file = sp
                break
        if sample_file is None:
            from .sample_data import EMBEDDED_SAMPLE_DATA
            return [dict(r) for r in EMBEDDED_SAMPLE_DATA[:limit]] if limit else [dict(r) for r in EMBEDDED_SAMPLE_DATA]
        with open(sample_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
        if not rows:
            from .sample_data import EMBEDDED_SAMPLE_DATA
            return [dict(r) for r in EMBEDDED_SAMPLE_DATA[:limit]] if limit else [dict(r) for r in EMBEDDED_SAMPLE_DATA]
        return rows


def stream_msmarco_xi(split: str = "train", language: str = "hi",
                      limit: int | None = None):
    """Stream translated passages from Hugging Face without downloading the dataset."""
    from datasets import load_dataset  # type: ignore

    try:
        dataset = load_dataset("ai4bharat/MSMARCO-XI", language, split=split,
                               streaming=True)
        language_filter = None
    except (ValueError, FileNotFoundError):
        dataset = load_dataset("ai4bharat/MSMARCO-XI", "default", split=split,
                               streaming=True)
        language_filter = {language, LANGUAGE_CONFIGS.get(language, language)}
    passage_count = 0
    for row in dataset:
        if language_filter and row.get("target_lang") not in language_filter:
            continue
        for passage in _flatten_row(dict(row), language):
            yield passage
            passage_count += 1
            if limit is not None and passage_count >= limit:
                return


def rows_to_documents(rows: list[dict]) -> list[tuple[str, str, dict]]:
    """Map dataset rows -> (doc_id, text, metadata) tuples the chunkers expect."""
    docs = []
    for r in rows:
        doc_id = str(r.get("passage_id") or r.get("id") or hash(r.get("passage", "")))
        text = r.get("passage") or r.get("text") or ""
        metadata = {
            "query_id": r.get("query_id"),
            "query": r.get("query"),
            "language": r.get("language", "hi"),
            "target_lang": r.get("target_lang"),
            "source_lang": r.get("source_lang"),
            "query_type": r.get("query_type"),
            "is_selected": r.get("is_selected"),
            "answer": r.get("answer"),
            "english_passage": r.get("english_passage"),
            "url": r.get("url"),
            "title": r.get("title"),
        }
        docs.append((doc_id, text, metadata))
    return docs
