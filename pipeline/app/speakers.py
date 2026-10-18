"""원시 화자 id(SPEAKER_00…) → 가명 라벨(학생1/학생2… 또는 교사) 매핑.

개인정보 보호 원칙: 실명 금지, 가명만 저장.
"""
from __future__ import annotations

from .transcribe.base import Segment


def label_segments(
    segments: list[Segment], *, is_teacher: bool, speaker_prefix: str = "학생"
) -> list[dict]:
    """세그먼트에 가명 화자 라벨을 붙여 utterances insert 용 dict 리스트로 변환.
    speaker_prefix: 화자분리된 채널의 화자 접두어(수업 모둠='학생', 협의회='교사')."""
    if is_teacher:
        mapping = {}
        for s in segments:
            mapping.setdefault(s.speaker, "교사")
    else:
        # 화자가 처음 등장하는 순서대로 <접두어>1, <접두어>2 …
        mapping: dict[str, str] = {}
        next_number = 1
        for s in segments:
            if s.speaker == "SPEAKER_UNKNOWN":
                mapping[s.speaker] = "미상"
                continue
            if s.speaker not in mapping:
                mapping[s.speaker] = f"{speaker_prefix}{next_number}"
                next_number += 1

    default = "교사" if is_teacher else f"{speaker_prefix}1"
    return [
        {
            "speaker_label": mapping.get(s.speaker, default),
            "start_ms": s.start_ms,
            "end_ms": s.end_ms,
            "text": s.text,
        }
        for s in segments
    ]
