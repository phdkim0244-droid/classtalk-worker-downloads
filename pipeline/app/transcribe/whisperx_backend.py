"""WhisperX 전사 백엔드 (온프레미스 기본).

무거운 의존성(torch, whisperx, pyannote)은 이 백엔드를 선택할 때만 import 된다.
설치: pip install -r requirements-whisperx.txt
화자분리(pyannote)에는 HuggingFace 토큰(HF_TOKEN, 무료)이 필요.

GPU 권장. CPU 에서도 동작하나 매우 느림(DEVICE=cpu, COMPUTE_TYPE=int8).
"""
from __future__ import annotations

import os

from .base import Segment, Transcriber


class WhisperXTranscriber(Transcriber):
    name = "whisperx"

    def __init__(self) -> None:
        self.model_size = os.environ.get("WHISPER_MODEL", "large-v3")
        self.device = os.environ.get("DEVICE", "cuda")
        self.compute_type = os.environ.get(
            "COMPUTE_TYPE", "float16" if self.device == "cuda" else "int8"
        )
        self.language = os.environ.get("WHISPER_LANG", "ko")
        self.hf_token = os.environ.get("HF_TOKEN", "")

    def transcribe(self, audio_path: str, *, is_teacher: bool) -> list[Segment]:
        import whisperx  # 지연 import

        audio = whisperx.load_audio(audio_path)

        # 1) 전사
        model = whisperx.load_model(
            self.model_size, self.device, compute_type=self.compute_type,
            language=self.language,
        )
        result = model.transcribe(audio, language=self.language)

        # 2) 단어 단위 정렬
        align_model, meta = whisperx.load_align_model(
            language_code=self.language, device=self.device
        )
        result = whisperx.align(
            result["segments"], align_model, meta, audio, self.device,
            return_char_alignments=False,
        )

        # 3) 교사 채널: 화자분리 없이 단일 화자
        if is_teacher:
            return [
                Segment(
                    speaker="SPEAKER_00",
                    start_ms=int(s["start"] * 1000),
                    end_ms=int(s["end"] * 1000),
                    text=(s.get("text") or "").strip(),
                )
                for s in result["segments"]
                if (s.get("text") or "").strip()
            ]

        # 3') 모둠 채널: pyannote 화자분리 후 발화별 화자 배정
        diarize = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_token, device=self.device
        )
        diarize_segments = diarize(audio, min_speakers=2, max_speakers=6)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        segs: list[Segment] = []
        for s in result["segments"]:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            segs.append(Segment(
                speaker=s.get("speaker", "SPEAKER_00"),
                start_ms=int(s["start"] * 1000),
                end_ms=int(s["end"] * 1000),
                text=text,
            ))
        return segs
