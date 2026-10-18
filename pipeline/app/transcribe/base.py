"""전사 백엔드 공통 인터페이스."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Segment:
    """전사 발화 1개 (화자분리 결과 포함)."""
    speaker: str      # 원시 화자 id (예: SPEAKER_00). 교사 채널은 무시됨.
    start_ms: int
    end_ms: int
    text: str


class Transcriber(ABC):
    """오디오 → 발화 세그먼트 리스트. 화자 라벨 매핑은 상위(speakers.py)에서 처리."""

    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_path: str, *, is_teacher: bool) -> list[Segment]:
        """audio_path 파일을 전사. is_teacher 면 화자분리 없이 단일 화자로 취급."""
        raise NotImplementedError
