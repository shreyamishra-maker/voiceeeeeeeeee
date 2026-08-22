import os
import sys
from pathlib import Path
import uvicorn

# Ensure src/ and api/ are in Python path
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

# Load .env file
_env_file = _ROOT / ".env"
if _env_file.exists():
    with open(_env_file, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    print("=" * 60)
    print(" Starting Dulset Server (Speech-to-Text & ElevenLabs Voice QA)")
    print(" ElevenLabs TTS: Configured")
    print(" Open your browser at: http://localhost:8000")
    print("=" * 60)
    uvicorn.run("backend.index:app", host="127.0.0.1", port=8000, reload=False)


