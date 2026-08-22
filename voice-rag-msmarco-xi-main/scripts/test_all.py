import sys
import json
from pathlib import Path

# Setup paths
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

def run_checks():
    print("=" * 60)
    print("[SCAN] Comprehensive Code & Integrity Scan")
    print("=" * 60)

    # 1. JSON Configuration validation
    json_files = [
        _ROOT / "vercel.json",
        _ROOT / "pyrightconfig.json",
        _ROOT / ".vscode" / "settings.json",
    ]
    for jf in json_files:
        if jf.exists():
            with open(jf, "r", encoding="utf-8") as f:
                json.load(f)
            print(f"[OK] {jf.name}: Valid JSON")

    # 2. Data file validation
    sample_data = _ROOT / "data" / "sample_msmarco_xi.jsonl"
    if sample_data.exists():
        with open(sample_data, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        print(f"[OK] sample_msmarco_xi.jsonl: Valid JSONL ({len(lines)} records)")

    # 3. Import & Schema validation
    from voicerag.config import PipelineConfig
    from voicerag.schemas import PipelineResponse, Chunk, RetrievedChunk, GuardrailVerdict
    from voicerag.embeddings import get_lightweight_embedder
    from voicerag.chunking import ChunkingPipeline, split_sentences
    from voicerag.lightweight_store import LightweightVectorStore
    from voicerag.guardrails import InputSafetyGuardrail, OffTopicGuardrail, GroundingGuardrail
    from voicerag.generation import ExtractiveGenerator
    from voicerag.stt import MockSTT, build_stt
    from voicerag.harness import VoiceRAGHarness
    from backend.index import _get_pipeline, app
    print("[OK] All modules imported successfully without errors")

    # 4. Config check
    cfg = PipelineConfig()
    print("[OK] PipelineConfig initialized")

    # 5. Guardrails check
    safety = InputSafetyGuardrail(cfg.guardrails)
    v_safe = safety.check("Where is the Taj Mahal located?")
    assert v_safe.passed, "Safe query should pass input safety"
    print("[OK] InputSafetyGuardrail check passed")

    # 6. Embedder and Store check
    embedder = get_lightweight_embedder(corpus_for_idf=["Taj Mahal is in Agra"])
    store = LightweightVectorStore(embedder)
    index_path = _ROOT / "data" / "index"
    assert (index_path / "vectors.npy").exists(), "Pre-built vectors.npy should exist"
    assert (index_path / "meta.pkl").exists(), "Pre-built meta.pkl should exist"
    store.load(str(index_path))
    assert len(store.chunks) > 0, "Store should contain indexed chunks"
    print(f"[OK] LightweightVectorStore loaded ({len(store.chunks)} chunks)")

    # 7. Search check
    results, latency = store.search("Taj Mahal", top_k_final=3)
    assert len(results) > 0, "Retrieval should return results"
    print(f"[OK] Store search passed ({len(results)} chunks in {latency:.2f}ms)")

    # 8. Generation check
    gen = ExtractiveGenerator()
    gen_res = gen.generate("Taj Mahal", results)
    assert len(gen_res.answer) > 0, "Generator should produce an answer"
    print(f"[OK] ExtractiveGenerator produced answer: '{gen_res.answer[:45]}...'")

    # 9. Full Harness check
    stt = MockSTT(cfg.stt)
    harness = VoiceRAGHarness(cfg, stt, store, gen)
    resp = harness.run_text("Where is the Taj Mahal located?")
    assert resp.status == "answered", f"Expected status 'answered', got '{resp.status}'"
    assert "Agra" in (resp.answer or ""), "Answer should mention Agra"
    print(f"[OK] Harness run_text passed ({resp.total_latency_ms:.2f}ms)")

    # 10. Voice / Audio STT Harness check
    audio_resp = harness.run_audio(b"What is the capital of India?")
    assert audio_resp.status == "answered", f"Expected status 'answered', got '{audio_resp.status}'"
    print(f"[OK] Harness run_audio passed ({audio_resp.total_latency_ms:.2f}ms)")

    # 11. API Pipeline singleton check
    p = _get_pipeline()
    assert p is not None, "API _get_pipeline() should return a valid Pipeline"
    api_resp = p.ask_text("Which is the national animal of India?")
    assert api_resp.status == "answered", "API ask_text should succeed"
    assert "Tiger" in (api_resp.answer or ""), "Answer should mention Tiger"
    print("[OK] API index.py singleton query passed")

    print("=" * 60)
    print("[SUCCESS] ALL 11 TEST CATEGORIES PASSED WITH ZERO ERRORS!")
    print("=" * 60)

if __name__ == "__main__":
    run_checks()
