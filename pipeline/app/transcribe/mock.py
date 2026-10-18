"""모의 전사기 — WhisperX 없이 데이터 흐름을 검증하기 위한 백엔드.

실제 음성을 전사하지 않고, 오디오 길이에 비례해 가짜 발화를 생성한다.
파이프라인 배선(다운로드→전사→utterances 저장→대시보드 열람) 테스트용.
"""
from __future__ import annotations

import os

from .base import Segment, Transcriber


class MockTranscriber(Transcriber):
    name = "mock"

    def transcribe(self, audio_path: str, *, is_teacher: bool) -> list[Segment]:
        size = os.path.getsize(audio_path)
        # 파일 크기로 대략적 길이 추정 (검증용): Opus ≈ 16KB/s
        est_ms = max(3000, int(size / 16000 * 1000))
        n = min(6, max(2, est_ms // 5000))  # 5초당 1발화, 2~6개
        step = est_ms // n

        segs: list[Segment] = []
        for i in range(n):
            if is_teacher:
                speaker = "SPEAKER_00"
                text = f"[모의] 교사 발화 {i + 1}"
            else:
                speaker = f"SPEAKER_{i % 2:02d}"  # 두 명 번갈아
                text = f"[모의] 학생 발화 {i + 1}"
            segs.append(Segment(
                speaker=speaker,
                start_ms=i * step,
                end_ms=(i + 1) * step,
                text=text,
            ))
        return segs
