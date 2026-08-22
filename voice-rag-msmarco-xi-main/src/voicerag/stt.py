"""Speech-to-text layer.

Task requirement: "Use either Sarvam or ElevenLabs for voice-to-text. Pick one."
We pick **Sarvam** as the default because the dataset (ai4bharat/MSMARCO-XI) is
Indic-language, and Sarvam's ASR is tuned for Indian languages. ElevenLabs is kept
as a drop-in alternative behind the same interface in case a team wants to switch.

A MockSTT is included so the rest of the pipeline is runnable/testable without
network access or API keys (this sandbox has neither) -- swap STT_PROVIDER=sarvam
and set SARVAM_API_KEY to go live.
"""
from __future__ import annotations

import abc
import io
import time
import urllib.request
import json as _json

from .config import STTConfig
from .schemas import TranscriptionResult


class SpeechToText(abc.ABC):
    provider_name: str

    @abc.abstractmethod
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        ...


class SarvamSTT(SpeechToText):
    """Real Sarvam AI Speech-to-Text (saarika) integration.

    Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe
    """
    provider_name = "sarvam"
    ENDPOINT = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        if not cfg.sarvam_api_key:
            raise RuntimeError("SARVAM_API_KEY not set")

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        t0 = time.perf_counter()
        boundary = "----voicerag"
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n')
        body.write(b"Content-Type: audio/wav\r\n\r\n")
        body.write(audio_bytes)
        body.write(f"\r\n--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="language_code"\r\n\r\n')
        body.write(self.cfg.language_hint.encode())
        body.write(f"\r\n--{boundary}--\r\n".encode())

        req = urllib.request.Request(
            self.ENDPOINT,
            data=body.getvalue(),
            headers={
                "api-subscription-key": self.cfg.sarvam_api_key,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            payload = _json.loads(resp.read().decode())
        latency_ms = (time.perf_counter() - t0) * 1000
        return TranscriptionResult(
            text=payload.get("transcript", ""),
            language=payload.get("language_code", self.cfg.language_hint),
            confidence=payload.get("confidence"),
            provider=self.provider_name,
            latency_ms=latency_ms,
        )


class ElevenLabsSTT(SpeechToText):
    """Real ElevenLabs Speech-to-Text integration (alternative provider)."""
    provider_name = "elevenlabs"
    ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        if not cfg.elevenlabs_api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set")

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        t0 = time.perf_counter()
        boundary = "----voicerag"
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n')
        body.write(b"Content-Type: audio/wav\r\n\r\n")
        body.write(audio_bytes)
        body.write(f"\r\n--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="model_id"\r\n\r\nscribe_v1\r\n')
        body.write(f"--{boundary}--\r\n".encode())

        req = urllib.request.Request(
            self.ENDPOINT,
            data=body.getvalue(),
            headers={
                "xi-api-key": self.cfg.elevenlabs_api_key,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            payload = _json.loads(resp.read().decode())
        latency_ms = (time.perf_counter() - t0) * 1000
        return TranscriptionResult(
            text=payload.get("text", ""),
            language=payload.get("language_code"),
            confidence=None,
            provider=self.provider_name,
            latency_ms=latency_ms,
        )


class MockSTT(SpeechToText):
    """Offline stand-in: 'transcribes' by decoding UTF-8 bytes directly.
    Lets the rest of the harness (retries, guardrails, latency bench) run
    end-to-end without network access. Never used when a real provider key
    is configured.
    """
    provider_name = "mock"

    def __init__(self, cfg: STTConfig):
        self.cfg = cfg

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        t0 = time.perf_counter()
        text = audio_bytes.decode("utf-8", errors="ignore")
        latency_ms = (time.perf_counter() - t0) * 1000
        return TranscriptionResult(
            text=text, language=self.cfg.language_hint, confidence=1.0,
            provider=self.provider_name, latency_ms=latency_ms,
        )


def build_stt(cfg: STTConfig) -> SpeechToText:
    if cfg.provider == "sarvam" and cfg.sarvam_api_key:
        return SarvamSTT(cfg)
    if cfg.provider == "elevenlabs" and cfg.elevenlabs_api_key:
        return ElevenLabsSTT(cfg)
    return MockSTT(cfg)
