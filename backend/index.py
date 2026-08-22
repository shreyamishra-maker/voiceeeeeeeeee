from pathlib import Path
import runpy


_BACKEND = Path(__file__).resolve().parent.parent / "voice-rag-msmarco-xi-main" / "backend" / "index.py"
globals().update(runpy.run_path(str(_BACKEND)))