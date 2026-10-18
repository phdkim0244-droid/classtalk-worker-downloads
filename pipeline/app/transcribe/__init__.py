"""전사 백엔드 선택기 (어댑터 패턴)."""
from __future__ import annotations

from .base import Transcriber


def get_transcriber(backend: str) -> Transcriber:
    if backend == "mock":
        from .mock import MockTranscriber
        return MockTranscriber()
    if backend == "faster-whisper":
        from .faster_whisper_backend import FasterWhisperTranscriber
        return FasterWhisperTranscriber()
    if backend == "faster-whisper-diar":
        from .faster_whisper_diar_backend import FasterWhisperDiarTranscriber
        return FasterWhisperDiarTranscriber()
    if backend == "whisperx":
        from .whisperx_backend import WhisperXTranscriber
        return WhisperXTranscriber()
    raise SystemExit(
        f"알 수 없는 STT_BACKEND: {backend} (whisperx|faster-whisper|mock)"
    )
