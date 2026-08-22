"""Text-to-Speech (TTS) layer with ElevenLabs integration.

Provides:
  - ElevenLabsTTS: Generates ultra-realistic voice audio for RAG answers
    using ElevenLabs Text-to-Speech REST API.
  - MockTTS: Offline fallback stand-in.
  - Preset voice catalog for easy voice selection.
"""
from __future__ import annotations

import abc
import json as _json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List

from .config import TTSConfig


# Common high-quality ElevenLabs premade voices (compatible across all tiers)
PRESET_VOICES: List[Dict[str, str]] = [
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Sarah", "description": "Mature, reassuring, confident female"},
    {"id": "JBFqnCBsd6RMkjVDRZzb", "name": "George", "description": "Warm, captivating storyteller male"},
    {"id": "Xb7hH8MSUJpSbSDYk0k2", "name": "Alice", "description": "Clear, engaging educator female"},
    {"id": "IKne3meq5aSn9XLyUdCD", "name": "Charlie", "description": "Deep, confident, energetic Australian male"},
    {"id": "FGY2WhTYpPnrIDTdsKH5", "name": "Laura", "description": "Enthusiast, quirky, friendly female"},
    {"id": "CwhRBWXzGAHq8TQ4Fs17", "name": "Roger", "description": "Laid-back, casual, resonant male"},
    {"id": "SAz9YHcvj6GT2YYXdXww", "name": "River", "description": "Relaxed, neutral, informative"},
    {"id": "TX3LPaxmHKxFdv7VOQHJ", "name": "Liam", "description": "Energetic, expressive social media voice"},
]



class TextToSpeech(abc.ABC):
    provider_name: str

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        model_id: str | None = None,
    ) -> bytes:
        """Synthesize text into MP3 audio bytes."""
        ...


class ElevenLabsTTS(TextToSpeech):
    """ElevenLabs Text-to-Speech integration.

    Docs: https://elevenlabs.io/docs/api-reference/text-to-speech
    """
    provider_name = "elevenlabs"
    BASE_ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech"

    def __init__(self, cfg: TTSConfig, api_key_override: str | None = None):
        self.cfg = cfg
        resolved_key = api_key_override or cfg.elevenlabs_api_key
        if not resolved_key:
            raise ValueError("ELEVENLABS_API_KEY is not set. Please provide an API key.")
        self.api_key: str = resolved_key

    def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        model_id: str | None = None,
    ) -> bytes:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty for speech synthesis.")

        target_voice_id = voice_id or self.cfg.voice_id or "EXAVITQu4vr4xnSDxMaL"
        target_model_id = model_id or self.cfg.model_id or "eleven_multilingual_v2"
        url = f"{self.BASE_ENDPOINT}/{target_voice_id}"

        payload = {
            "text": text.strip(),
            "model_id": target_model_id,
            "voice_settings": {
                "stability": self.cfg.stability,
                "similarity_boost": self.cfg.similarity_boost,
                "use_speaker_boost": True,
            },
        }

        data_bytes = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"ElevenLabs TTS returned status {resp.status}")
                return resp.read()
        except urllib.error.HTTPError as e:
            try:
                err_detail = e.read().decode("utf-8", errors="ignore")
                err_json = _json.loads(err_detail)
                msg = err_json.get("detail", {}).get("message") or err_json.get("message") or str(e)
            except Exception:
                msg = f"HTTP {e.code}: {e.reason}"
            raise RuntimeError(f"ElevenLabs API error: {msg}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to ElevenLabs: {e.reason}") from e


class MockTTS(TextToSpeech):
    """Offline placeholder TTS that returns an empty audio stream or placeholder."""
    provider_name = "mock"

    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg

    def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        model_id: str | None = None,
    ) -> bytes:
        # Minimal silent/stub MP3 header
        return b""


def build_tts(cfg: TTSConfig, api_key_override: str | None = None) -> TextToSpeech:
    key = api_key_override or cfg.elevenlabs_api_key
    if cfg.provider == "elevenlabs" and key:
        return ElevenLabsTTS(cfg, api_key_override=key)
    return MockTTS(cfg)
