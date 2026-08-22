"""FastAPI backend for Dulset — Vercel serverless function & local dev.

Exposes:
  POST /api/query   — {text, language} -> PipelineResponse JSON
  GET  /api/health  — health check
  POST /api/tts     — ElevenLabs text-to-speech
  GET  /api/tts/voices — Available ElevenLabs voices
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add all possible source locations to sys.path so imports work in every Vercel directory structure
_FILE_DIR = Path(__file__).resolve().parent
_POSSIBLE_ROOTS = [
    _FILE_DIR,
    _FILE_DIR.parent,
    _FILE_DIR / "src",
    _FILE_DIR.parent / "src",
    Path.cwd(),
    Path.cwd() / "src",
    Path("/var/task"),
    Path("/var/task/src"),
]
for _p in _POSSIBLE_ROOTS:
    _p_str = str(_p)
    if _p.exists() and _p_str not in sys.path:
        sys.path.insert(0, _p_str)

# Load .env file into os.environ if present
for _r in [_FILE_DIR.parent, _FILE_DIR, Path.cwd()]:
    _env_f = _r / ".env"
    if _env_f.exists():
        try:
            with open(_env_f, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Dulset", version="1.0.0")


@app.middleware("http")
async def normalize_vercel_function_path(request, call_next):
    """Remove the serverless function filename added by Vercel rewrites."""
    path = request.scope.get("path", "")
    for prefix in ("/backend/index.py", "/api/index.py"):
        if path == prefix or path.startswith(prefix + "/"):
            normalized_path = path[len(prefix):] or "/"
            request.scope["path"] = normalized_path
            request.scope["raw_path"] = normalized_path.encode("ascii")
            break
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Lazy pipeline singleton with thread lock ----------
_pipeline = None
_pipeline_init_lock = threading.Lock()
_pipeline_error: Optional[str] = None


def _get_pipeline():
    global _pipeline, _pipeline_error
    if _pipeline is not None:
        return _pipeline

    with _pipeline_init_lock:
        if _pipeline is not None:
            return _pipeline

        try:
            from voicerag.config import PipelineConfig
            from voicerag.embeddings import get_lightweight_embedder
            from voicerag.generation import ExtractiveGenerator
            from voicerag.stt import build_stt
            from voicerag.harness import VoiceRAGHarness
            from voicerag.lightweight_store import LightweightVectorStore
            from voicerag.sample_data import EMBEDDED_SAMPLE_DATA
            from voicerag.hindi_sample_data import HINDI_SAMPLE_DATA
            from voicerag.data_loader import rows_to_documents
            from voicerag.chunking import ChunkingPipeline

            cfg = PipelineConfig()

            if os.environ.get("VERCEL") and not os.environ.get("PINECONE_API_KEY", "").strip():
                raise RuntimeError(
                    "PINECONE_API_KEY is required on Vercel. Add it in Vercel "
                    "Project Settings > Environment Variables and redeploy."
                )

            # 1. Try Pinecone Vector Store if PINECONE_API_KEY is present
            store = None
            if os.environ.get("PINECONE_API_KEY"):
                try:
                    from voicerag.pinecone_store import PineconeVectorStore
                    embedder = get_lightweight_embedder(corpus_for_idf=["dummy"])
                    store = PineconeVectorStore(embedder, cfg=cfg.pinecone)
                except Exception as pe:
                    if os.environ.get("VERCEL"):
                        raise RuntimeError(
                            "Pinecone initialization failed. Check PINECONE_API_KEY, "
                            "PINECONE_INDEX_NAME, PINECONE_HOST, and the index dimension "
                            f"(expected 384): {pe}"
                        ) from pe
                    print(f"[WARN] Pinecone init failed ({pe}), falling back to lightweight store.")
                    store = None

            # 2. Try loading pre-built lightweight index from disk
            if store is None:
                index_candidates = [
                    _FILE_DIR.parent / "data" / "index",
                    _FILE_DIR / "data" / "index",
                    Path.cwd() / "data" / "index",
                    Path("/var/task/data/index"),
                ]
                index_path = None
                for cand in index_candidates:
                    if cand.exists() and (cand / "vectors.npy").exists():
                        index_path = cand
                        break

                if index_path is not None:
                    try:
                        embedder = get_lightweight_embedder(corpus_for_idf=["dummy"])
                        store = LightweightVectorStore(embedder)
                        store.load(str(index_path))
                    except Exception as le:
                        print(f"[WARN] Failed to load index from disk ({le}), building in-memory.")
                        store = None

            # 3. Fallback: Build index in-memory from embedded sample data
            if store is None:
                docs = rows_to_documents(EMBEDDED_SAMPLE_DATA)
                corpus_texts = [d[1] for d in docs]
                embedder = get_lightweight_embedder(corpus_for_idf=corpus_texts)
                chunker = ChunkingPipeline(
                    embedder=embedder,
                    strategies=cfg.chunking.strategies,
                    fixed_size=cfg.chunking.fixed_chunk_size_tokens,
                    fixed_overlap=cfg.chunking.fixed_chunk_overlap_tokens,
                    sw_window=cfg.chunking.sentence_window_size,
                    sw_overlap=cfg.chunking.sentence_window_overlap,
                    semantic_threshold=cfg.chunking.semantic_similarity_drop_threshold,
                )
                chunks = chunker.run_corpus(docs)
                store = LightweightVectorStore(embedder)
                store.build(chunks)

            generator = ExtractiveGenerator()
            stt = build_stt(cfg.stt)
            harness = VoiceRAGHarness(cfg, stt, store, generator)

            # Keep the demo bilingual while a language-specific Pinecone
            # namespace is being populated. Production Hindi vectors win
            # whenever they are available; this is only a small local pilot.
            hindi_docs = rows_to_documents(HINDI_SAMPLE_DATA)
            hindi_embedder = get_lightweight_embedder(
                corpus_for_idf=[document[1] for document in hindi_docs]
            )
            hindi_chunker = ChunkingPipeline(
                embedder=hindi_embedder,
                strategies=cfg.chunking.strategies,
                fixed_size=cfg.chunking.fixed_chunk_size_tokens,
                fixed_overlap=cfg.chunking.fixed_chunk_overlap_tokens,
                sw_window=cfg.chunking.sentence_window_size,
                sw_overlap=cfg.chunking.sentence_window_overlap,
                semantic_threshold=cfg.chunking.semantic_similarity_drop_threshold,
            )
            hindi_store = LightweightVectorStore(hindi_embedder)
            hindi_store.build(hindi_chunker.run_corpus(hindi_docs))
            hindi_harness = VoiceRAGHarness(cfg, stt, hindi_store, generator)

            class Pipeline:
                def __init__(self, harness, hindi_harness, store, cfg):
                    self.harness = harness
                    self.hindi_harness = hindi_harness
                    self.store = store
                    self.cfg = cfg

                def ask_text(self, query, language=None):
                    response = self.harness.run_text(query, language=language)
                    if language == "hi" and response.status == "refused_off_topic":
                        return self.hindi_harness.run_text(query, language="hi")
                    return response

            _pipeline = Pipeline(harness, hindi_harness, store, cfg)
            _pipeline_error = None
            return _pipeline
        except Exception as e:
            import traceback
            _pipeline_error = f"{type(e).__name__}: {str(e)}"
            traceback.print_exc()
            raise e


# ---------- Request / Response models ----------
class QueryRequest(BaseModel):
    text: str
    language: Optional[str] = "en"


def _dataset_language(language: str | None) -> str | None:
    """Convert browser locale values to MSMARCO-XI language configs."""
    if not language:
        return None
    return language.split("-", 1)[0].lower()


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int = 0
    vectors_indexed: Optional[int] = None
    embedder: str = ""
    error: Optional[str] = None


class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    model_id: Optional[str] = None
    api_key: Optional[str] = None


# ---------- API routes ----------
@app.get("/api/health")
@app.get("/health")
@app.get("/api/index.py")
async def health():
    try:
        p = _get_pipeline()
        if p:
            vectors_indexed = None
            if hasattr(p.store, "stats"):
                try:
                    stats = p.store.stats()
                    vectors_indexed = int(stats.get("totalVectorCount", 0))
                except Exception as stats_error:
                    return HealthResponse(
                        status="degraded",
                        chunks_indexed=len(p.store.chunks),
                        embedder=getattr(p.store.embedder, "name", "TFIDFEmbedder"),
                        error=f"Pinecone stats check failed: {stats_error}",
                    )
            return HealthResponse(
                status="ok",
                chunks_indexed=len(p.store.chunks),
                vectors_indexed=vectors_indexed,
                embedder=getattr(p.store.embedder, "name", "TFIDFEmbedder"),
            )
        return HealthResponse(status="initializing", error=_pipeline_error)
    except Exception as e:
        return HealthResponse(status="error", error=str(e))


@app.post("/api/query")
@app.post("/query")
@app.post("/")
@app.post("/api/index.py")
async def query(req: QueryRequest):
    try:
        if not req.text or not req.text.strip():
            return {
                "query": req.text,
                "answer": "Please provide a non-empty question.",
                "status": "error",
                "guardrail_verdicts": [],
                "retrieved": [],
                "timings": [],
                "total_latency_ms": 0,
            }

        p = _get_pipeline()
        if p is None:
            return {
                "query": req.text,
                "answer": f"Pipeline initialization error: {_pipeline_error or 'Unknown'}",
                "status": "error",
                "guardrail_verdicts": [],
                "retrieved": [],
                "timings": [],
                "total_latency_ms": 0,
            }

        language = _dataset_language(req.language)
        # The bundled pilot predates language metadata; leave English queries
        # compatible with it while Hindi queries require Hindi-indexed vectors.
        language_filter = language if language != "en" else None
        resp = p.ask_text(req.text.strip(), language=language_filter)
        return resp.model_dump()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "query": req.text,
            "answer": f"Server processing notice: {str(e)}",
            "status": "error",
            "error": str(e),
            "guardrail_verdicts": [],
            "retrieved": [],
            "timings": [],
            "total_latency_ms": 0,
        }


@app.get("/api/tts/voices")
@app.get("/tts/voices")
async def get_tts_voices():
    """Returns available ElevenLabs voices and configuration status."""
    try:
        from voicerag.tts import PRESET_VOICES
        has_server_key = bool(os.environ.get("ELEVENLABS_API_KEY", "").strip())
        default_voice = os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
        return {
            "status": "ok",
            "has_server_api_key": has_server_key,
            "default_voice_id": default_voice,
            "voices": PRESET_VOICES,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "voices": []}


@app.post("/api/tts")
@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    """Synthesize text into speech using ElevenLabs API."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    api_key = (req.api_key or os.environ.get("ELEVENLABS_API_KEY", "")).strip()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="NO_API_KEY: ElevenLabs API key is not configured. Please enter your API key in Settings or set ELEVENLABS_API_KEY in your environment."
        )

    try:
        from voicerag.config import TTSConfig
        from voicerag.tts import ElevenLabsTTS

        cfg = TTSConfig(elevenlabs_api_key=api_key)
        tts_engine = ElevenLabsTTS(cfg, api_key_override=api_key)
        audio_bytes = tts_engine.synthesize(
            text=req.text.strip(),
            voice_id=req.voice_id or os.environ.get("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
            model_id=req.model_id or os.environ.get("ELEVENLABS_TTS_MODEL_ID", "eleven_multilingual_v2"),
        )

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=answer.mp3",
                "Cache-Control": "public, max-age=3600",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Static files (served by the same backend process) ----------
_frontend_dir = _FILE_DIR.parent / "frontend"
if _frontend_dir.exists():
    @app.get("/")
    async def serve_index():
        return FileResponse(str(_frontend_dir / "index.html"))

    app.mount("/", StaticFiles(directory=str(_frontend_dir)), name="static")

# Vercel ASGI handler alias
handler = app
