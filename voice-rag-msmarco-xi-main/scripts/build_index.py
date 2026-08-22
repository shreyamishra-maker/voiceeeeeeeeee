import sys
import os
from pathlib import Path

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from voicerag.config import PipelineConfig
from voicerag.data_loader import load_msmarco_xi, rows_to_documents
from voicerag.chunking import ChunkingPipeline
from voicerag.embeddings import get_lightweight_embedder
from voicerag.lightweight_store import LightweightVectorStore

def main():
    print("Building vector index for Dulset...")
    cfg = PipelineConfig()

    language = os.environ.get("HF_DATASET_CONFIG", "hi")
    split = os.environ.get("HF_DATASET_SPLIT", "train")
    row_limit = int(os.environ.get("HF_DATASET_LIMIT", "200")) or None
    strict_dataset = os.environ.get("MSMARCO_XI_STRICT", "").lower() in {"1", "true", "yes"}
    print(f"Dataset: MSMARCO-XI/{language} ({split}), rows: {row_limit or 'all'}")
    print(f"Dataset fallback: {'disabled' if strict_dataset else 'enabled'}")

    print("Loading documents...")
    rows = load_msmarco_xi(split=split, language=language, limit=row_limit,
                            streaming=True)
    print(f"Loaded {len(rows)} dataset rows ({sum(bool(row.get('passage')) for row in rows)} passages)")
    docs = rows_to_documents(rows)
    corpus_texts = [d[1] for d in docs]
    
    print("Initializing embedder...")
    embedder = get_lightweight_embedder(corpus_for_idf=corpus_texts)
    
    print("Chunking documents...")
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
    
    print("Building vector store...")
    store = LightweightVectorStore(embedder)
    store.build(chunks)
    
    index_path = _ROOT / "data" / "index"
    print(f"Saving vector store to {index_path}...")
    store.save(str(index_path))
    print("Done! You can now deploy to Vercel.")

if __name__ == "__main__":
    main()
