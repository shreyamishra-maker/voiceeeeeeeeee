"""Upload Dulset dataset to a hosted Pinecone Serverless index."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

# Load .env
_env_file = _ROOT / ".env"
if _env_file.exists():
    with open(_env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

from voicerag.config import PipelineConfig, PineconeConfig
from voicerag.data_loader import rows_to_documents, stream_msmarco_xi
from voicerag.sample_data import EMBEDDED_SAMPLE_DATA
from voicerag.chunking import ChunkingPipeline
from voicerag.embeddings import get_lightweight_embedder
from voicerag.pinecone_store import PineconeVectorStore


def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("[ERROR] PINECONE_API_KEY is not set. Please set it in .env or your environment.")
        sys.exit(1)

    index_name = os.environ.get("PINECONE_INDEX_NAME", "voicerag-index")
    print(f"[*] Connecting to Pinecone index: {index_name}")

    cfg = PipelineConfig()
    p_cfg = PineconeConfig(api_key=api_key, index_name=index_name)

    language = os.environ.get("HF_DATASET_CONFIG", "hi")
    split = os.environ.get("HF_DATASET_SPLIT", "train")
    row_limit = int(os.environ.get("HF_DATASET_LIMIT", "0")) or None
    source = os.environ.get("INDEX_SOURCE", "huggingface").lower()
    batch_rows = int(os.environ.get("HF_DATASET_BATCH_ROWS", "100"))
    if batch_rows < 1:
        print("[ERROR] HF_DATASET_BATCH_ROWS must be at least 1.")
        sys.exit(1)
    if source == "sample":
        print("[*] Using bundled offline sample corpus for a deterministic pilot...")
        rows = iter(EMBEDDED_SAMPLE_DATA[:row_limit] if row_limit else EMBEDDED_SAMPLE_DATA)
    elif source == "huggingface":
        print(f"[*] Streaming MSMARCO-XI/{language} ({split}) from Hugging Face...")
        rows = stream_msmarco_xi(split=split, language=language, limit=row_limit)
    else:
        print("[ERROR] INDEX_SOURCE must be 'sample' or 'huggingface'.")
        sys.exit(1)

    print("[*] Initializing lightweight embedder...")
    embedder = get_lightweight_embedder()

    chunker = ChunkingPipeline(
        embedder=embedder,
        strategies=cfg.chunking.strategies,
        fixed_size=cfg.chunking.fixed_chunk_size_tokens,
        fixed_overlap=cfg.chunking.fixed_chunk_overlap_tokens,
        sw_window=cfg.chunking.sentence_window_size,
        sw_overlap=cfg.chunking.sentence_window_overlap,
        semantic_threshold=cfg.chunking.semantic_similarity_drop_threshold,
    )
    store = PineconeVectorStore(embedder, cfg=p_cfg)
    total_upserted = 0
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) < batch_rows:
            continue
        chunks = chunker.run_corpus(rows_to_documents(batch))
        total_upserted += store.upsert_chunks(chunks)
        print(f"[*] Upserted {total_upserted} chunks...")
        batch = []

    if batch:
        chunks = chunker.run_corpus(rows_to_documents(batch))
        total_upserted += store.upsert_chunks(chunks)

    print(f"[SUCCESS] Upserted {total_upserted} chunks into Pinecone index '{index_name}' successfully!")


if __name__ == "__main__":
    main()
