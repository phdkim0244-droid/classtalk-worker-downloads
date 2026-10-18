"""faster-whisper 전사 백엔드 (경량, torch·pyannote 불필요).

화자분리(diarization)는 하지 않는다 → 모든 발화가 단일 화자로 라벨된다.
(학생별 구분/참여 균등도가 필요하면 whisperx 백엔드 + HF 토큰 사용.)
CPU 에서도 동작(int8). 온프레미스에서 GPU/토큰 없이 빠르게 전사만 뽑을 때.
"""
from __future__ import annotations

import os

from .base import Segment, Transcriber
from .speech_timing import transcribe_speech


class FasterWhisperTranscriber(Transcriber):
    name = "faster-whisper"

    def __init__(self) -> None:
        self.model_size = os.environ.get("WHISPER_MODEL", "large-v3")
        self.device = os.environ.get("DEVICE", "cpu")
        self.compute_type = os.environ.get(
            "COMPUTE_TYPE", "int8" if self.device == "cpu" else "float16"
        )
        self.language = os.environ.get("WHISPER_LANG", "ko")
        self.beam_size = max(1, int(os.environ.get("WHISPER_BEAM_SIZE", "5")))
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, audio_path: str, *, is_teacher: bool) -> list[Segment]:
        model = self._load()
        from faster_whisper.audio import decode_audio
        wav = decode_audio(audio_path, sampling_rate=16000)
        segments, _, _ = transcribe_speech(
            model, wav, language=self.language, beam_size=self.beam_size,
        )
        return segments
