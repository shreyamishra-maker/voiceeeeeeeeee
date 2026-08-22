# Dulset — Voice-Enabled RAG — MSMARCO-XI

A voice-in → answer-out RAG pipeline built against the technical requirements
of the task brief (STT → chunking/retrieval → generation, with guardrails,
a real harness, and measured latency). This README covers **only the build**
— submission form, videos, and promotion requirements from the brief are
intentionally out of scope here.

## Pipeline shape

```
audio bytes ─▶ STT (Sarvam / ElevenLabs) ─▶ input-safety guardrail
                                                     │
                                                     ▼
                                        hybrid retrieval (FAISS dense + BM25
                                        sparse, fused with RRF) over a
                                        multi-strategy chunk index
                                                     │
                                                     ▼
                                          off-topic guardrail (best
                                          similarity vs. corpus)
                                                     │
                                                     ▼
                                        answer generation (extractive by
                                        default, pluggable LLM)
                                                     │
                                                     ▼
                                        grounding / hallucination guardrail
                                                     │
                                                     ▼
                                        structured PipelineResponse (status,
                                        answer, verdicts, retrieved chunks,
                                        per-stage timings)
```

All of this runs inside `voicerag.harness.VoiceRAGHarness`, not as a single
prompt-in/text-out call — see [Harness](#5-harness) below.

## Repo layout

```
src/voicerag/
  config.py        tunables for every stage (STT, chunking, retrieval, guardrails, generation, harness)
  schemas.py        pydantic contracts passed between stages
  stt.py             Sarvam / ElevenLabs / Mock speech-to-text
  chunking.py        4 chunking strategies + pipeline to run them all
  embeddings.py       sentence-transformers embedder + offline hashing-TFIDF fallback
  vector_store.py     FAISS + BM25 hybrid retrieval with reciprocal rank fusion
  guardrails.py        input-safety / off-topic / grounding checks
  generation.py         extractive generator + pluggable LLM generator
  harness.py             orchestrator: retries, structured I/O, error recovery, timings
  data_loader.py          ai4bharat/MSMARCO-XI loader with offline sample fallback
  pipeline.py               top-level VoiceRAGPipeline convenience wrapper
scripts/
  build_index.py     build & persist the hybrid index
  latency_bench.py    P50/P70/P100 latency report over N queries
data/
  sample_msmarco_xi.jsonl   offline fallback sample (see below)
```

## Setup

```bash
pip install -r requirements.txt
export SARVAM_API_KEY=...        # or ELEVENLABS_API_KEY + STT_PROVIDER=elevenlabs
set HF_DATASET_CONFIG=hi         # choose: hi, bn, ta, te, kn, ...
set HF_DATASET_LIMIT=2000        # rows to index; 0 means all rows
set MSMARCO_XI_STRICT=1          # fail instead of silently using the dev sample
python scripts/build_index.py
python scripts/latency_bench.py --n 300
python scripts/test_all.py
```

### Use MSMARCO-XI without downloading it

The dataset loader uses Hugging Face `streaming=True`, so it reads records
over the network and does not download the 55 GB dataset. For production,
index the streamed records directly into Pinecone:

The Vercel API uses `requirements.txt`; the larger Hugging Face dependency is
kept in `requirements-index.txt` and is installed only by the indexing workflow.

```bash
set PINECONE_API_KEY=...
set PINECONE_INDEX_NAME=voicerag-index
set HF_DATASET_CONFIG=hi
set HF_DATASET_SPLIT=train
set HF_DATASET_LIMIT=10000
set MSMARCO_XI_STRICT=1
python scripts/index_to_pinecone.py
```

The API then uses Pinecone automatically when `PINECONE_API_KEY` is present.
The local `data/index` remains only as an offline fallback. `HF_DATASET_LIMIT=0`
means all records, but indexing the complete corpus requires substantial time,
embedding calls, and Pinecone storage even though the source dataset is never
downloaded locally.

### Cloud-only indexing with GitHub Actions

For a deployment with no local indexing machine, use the included manual
workflow at `.github/workflows/index-msmarco-xi.yml`:

1. Push this repository to GitHub.
2. In GitHub, open **Settings -> Secrets and variables -> Actions**.
3. Add the secret `PINECONE_API_KEY` with your replacement Pinecone key.
  Add `PINECONE_HOST` with the host shown by the new Pinecone index.
  Add `HF_TOKEN` from your Hugging Face account for higher-rate dataset streaming.
4. Open **Actions -> Index MSMARCO-XI in Pinecone -> Run workflow**.
5. Choose a language such as `hi` and start with a row limit of `1000`; increase it after the pilot succeeds.

The workflow streams MSMARCO-XI from Hugging Face into Pinecone. Vercel then
reads the populated Pinecone index at query time; Vercel never downloads or
indexes the full dataset during a user request.

The required source dataset is exactly
`https://huggingface.co/datasets/ai4bharat/MSMARCO-XI`. The production demo
currently uses the deterministic bundled pilot (`INDEX_SOURCE=sample`, 75
chunks) because the unauthenticated Hugging Face stream was terminated by the
GitHub runner. Use `INDEX_SOURCE=huggingface` with a `HF_TOKEN` GitHub secret
for the full dataset import.

## A note on this environment's offline fallbacks

This build was assembled in a network-restricted sandbox with **no route to
huggingface.co or the Sarvam/ElevenLabs APIs**. So that the whole pipeline is
actually runnable and testable rather than just described, two fallbacks are
wired in and used automatically when the real service is unreachable:

- **Dataset**: `data_loader.load_msmarco_xi()` tries `datasets.load_dataset("ai4bharat/MSMARCO-XI")` first, and falls back to `data/sample_msmarco_xi.jsonl` — 16 hand-written Hindi passages in the same schema (query, passage, passage_id, language, is_selected). **Point this at the real dataset before final submission** by simply running with internet access; no code changes needed.
- **Embeddings**: `embeddings.get_embedder()` tries `sentence-transformers` first (downloads `all-MiniLM-L6-v2` from the HF hub) and falls back to a deterministic offline hashing+TF-IDF embedder if the download fails. The retrieval, guardrail, and latency-bench code paths are identical either way.
- **STT**: `build_stt()` uses real Sarvam/ElevenLabs when an API key is present in the environment, otherwise `MockSTT` (decodes UTF-8 bytes directly) so `ask_audio()` is exercised end-to-end in tests.

Everything below was measured with these fallbacks active. Re-run
`scripts/latency_bench.py` with the real dataset + model + STT provider
before submitting your final numbers.

## How each requirement is met

### 1. Speech-to-text
`stt.py` implements both **Sarvam** (default — MSMARCO-XI is Indic, and
Sarvam's ASR is tuned for Indian languages) and **ElevenLabs** behind one
`SpeechToText` interface, selected via `STT_PROVIDER`. Real HTTP calls,
multipart audio upload, structured `TranscriptionResult` output.

### 2. Chunking — deliberately not single naive fixed-size
`chunking.py` implements four strategies and indexes all of them together,
each chunk tagged with its `strategy`:

| Strategy | Idea | Why |
|---|---|---|
| `fixed` | token windows, configurable overlap | cheap baseline, good for short factoid queries |
| `sentence_window` | N sentences per chunk, M sentence overlap | preserves local coherence; fits MSMARCO's short-passage style |
| `semantic` | grows a chunk while consecutive sentences stay embedding-similar to the running centroid, cuts on similarity drop | approximates topic-boundary splitting without an LLM call |
| `metadata_aware` | one chunk per native passage, keeps `query_id`/`language`/`is_selected`/`url`/`title` as structured, filterable metadata instead of discarding them | enables filtered retrieval (e.g. language-scoped search) that plain text chunking throws away |

`ChunkingPipeline.run_corpus()` builds all four over every document; the
current sample corpus (16 passages) yields 80 chunks across strategies.

### 3 & 4. Latency target + analytics
Retrieval is **hybrid**: FAISS `IndexFlatIP` (dense, cosine via inner
product on normalized vectors) fused with BM25 (sparse/lexical) via
Reciprocal Rank Fusion — cheap at query time (one matmul + one BM25 scan),
and materially better recall than either alone on short, keyword-heavy
MSMARCO-style queries.

`scripts/latency_bench.py` runs N queries (default 300, cycling through
the corpus's distinct queries — swap in genuinely held-out queries for a
real run) and reports **two** percentile sets rather than one, on purpose:

- `retrieval_only_ms` — chunking happens once at index-build time, so at
  query time the number the task's 200ms budget is realistically about is
  vector-DB retrieval. Measured result on this sandbox's sample corpus:

  | | P50 | P70 | P100 |
  |---|---|---|---|
  | retrieval only | 0.20ms | 0.22ms | 0.55ms |
  | end-to-end (guardrails + retrieval + extractive generation) | 0.31ms | 0.33ms | 0.75ms |

  The current local 300-query report is written to `data/latency_report.json`.
  Both are far under
  200ms here because the sample corpus is 80 chunks and the fallback
  embedder is a cheap hashing vector. **This will not hold unmodified at
  MSMARCO-XI's real scale** (hundreds of thousands of passages) with a real
  sentence-transformer — expect retrieval to land in the low tens of
  milliseconds instead, which still clears the budget comfortably; re-run
  the benchmark against the full index to get real numbers before
  submitting.

- `end_to_end_ms` — only holds inside the 200ms budget because the default
  generator (`ExtractiveGenerator`) does no network call. See the trade-off
  note directly below.

### Latency budget vs. LLM generation — an honest trade-off
A network round trip to any hosted LLM (Claude, GPT, etc.) typically costs
300ms–2s on its own, which no retrieval-side optimization can absorb inside
a 200ms *total* budget. Two options are wired in via `GEN_PROVIDER`:

- `extractive` (default) — builds the answer directly from the top retrieved
  passage(s), deduplicated at sentence level, no external call. Stays inside
  the budget. Lower ceiling on answer fluency/synthesis quality.
- `llm` — pluggable call to a hosted chat-completion endpoint (`generation.LLMGenerator`,
  provider-agnostic) for materially better answers. Report its latency
  **separately** from retrieval latency rather than silently missing the
  200ms target — that's the transparent way to handle a requirement that's
  in tension with using a hosted LLM at all.

### 5. Harness
`harness.VoiceRAGHarness` is a typed state machine, not a single call:
structured pydantic I/O at every step, bounded retries with exponential
backoff on steps that can transiently fail (STT, retrieval, generation),
explicit error recovery (a step that exhausts retries degrades to a safe
`status="error"` response instead of raising), and per-step latency capture
feeding directly into the benchmark above. Guardrails are wired in as
first-class steps with their own recorded verdicts, not an afterthought.

### 6. Guardrails
Three checkpoints (`guardrails.py`), each producing a logged `GuardrailVerdict`:

1. **Input safety** — pattern-matches the transcribed query against unsafe
   categories (self-harm, weapons, illegal drugs, hate) *before* any
   retrieval/generation spend. Intentionally a coarse first line of defense
   — pair with a hosted moderation endpoint for production use, don't rely
   on it alone.
2. **Off-topic** — refuses if the query's best similarity to anything in the
   corpus is below a floor, rather than letting the generator improvise an
   answer with no real support.
3. **Grounding / hallucination** — after generation, scores lexical overlap
   between the answer and the retrieved context it was supposedly built
   from; low overlap → refuse rather than return an unsupported claim.

Verified behavior (`scripts/test_all.py`):
```
"how to make a bomb at home"                          -> refused_unsafe (never touches retrieval)
"What is the airspeed velocity of an unladen swallow?" -> refused_off_topic
"भारतीय संविधान कब लागू हुआ था?"                        -> answered, grounded in retrieved passage
```

## Known limitations / next steps for a real submission
- Swap the sample JSONL for the real `ai4bharat/MSMARCO-XI` split and re-run
  `build_index.py` + `latency_bench.py` with internet access.
- Swap the hashing-TFIDF fallback for `sentence-transformers` (automatic once
  the HF hub is reachable) for materially better retrieval quality.
- Wire a real `SARVAM_API_KEY` / `ELEVENLABS_API_KEY` for live STT instead of
  `MockSTT`.
- The input-safety guardrail is keyword/regex based; consider a hosted
  moderation classifier for production robustness.
