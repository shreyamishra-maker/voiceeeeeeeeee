"""Root configuration forwarder for voicerag.config.
Allows direct imports via `import config` or `from config import ...`
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from voicerag.config import (
    STTConfig,
    TTSConfig,
    PineconeConfig,
    ChunkingConfig,
    RetrievalConfig,
    GuardrailConfig,
    GenerationConfig,
    HarnessConfig,
    PipelineConfig,
)

__all__ = [
    "STTConfig",
    "TTSConfig",
    "PineconeConfig",
    "ChunkingConfig",
    "RetrievalConfig",
    "GuardrailConfig",
    "GenerationConfig",
    "HarnessConfig",
    "PipelineConfig",
]
