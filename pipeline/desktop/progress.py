"""Local display events. No transcript, URLs or credentials are included."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from collections import deque


def clock_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


@dataclass(frozen=True)
class Progress:
    stage: str
    done: float | None = None
    total: float | None = None
    title: str = ""
    file_index: int = 0
    file_count: int = 0
    started_at: float = 0
    updated_at: float = 0
    remaining_seconds: float | None = None
    speed: float | None = None
    engine: str = ""

    @property
    def percent(self) -> int | None:
        if self.stage == "complete":
            return 100
        if self.done is None or self.total is None or self.total <= 0:
            return None
        if not math.isfinite(self.done) or not math.isfinite(self.total):
            return None
        # A rounded 100 must not appear before this stage has really finished.
        return min(100, max(0, math.floor(100 * self.done / self.total)))

    @property
    def heading(self) -> str:
        labels = {
            "importing_model_libraries": "전사 도구 불러오는 중",
            "downloading_public_model": "전사 모델 확인·다운로드 중",
            "downloading_gpu_model": "GPU 전사 모델 확인·다운로드 중",
            "initializing_gpu_model": "Mac GPU 전사 모델 준비 중",
            "initializing_whisper_model": "음성 인식 모델 준비 중",
            "initializing_speaker_model": "화자 구분 모델 준비 중",
            "initializing_short_speaker_model": "짧은 발화 확인 모델 준비 중",
            "verify_speakers": "짧은 발화 화자 교차 확인 중",
            "refine_speakers": "구간 안 화자 전환 확인 중",
            "models_ready": "모델 준비 완료",
            "download": "음성 파일 받는 중",
            "decode": "음성 파일 읽는 중",
            "detect": "음성 구간 찾는 중",
            "transcribe": "전사 중",
            "speakers": "화자 특징 추출 중",
            "cluster": "화자 구분 중",
            "label": "전사 결과 정리 중",
            "save": "전사 결과 저장 중",
            "finish": "수업 완료 확인 중",
            "complete": "전사 완료",
        }
        label = labels.get(self.stage, "처리 중")
        if self.stage in ("transcribe", "speakers", "verify_speakers", "refine_speakers") and self.percent is not None:
            label += f" · {self.percent}%"
        return label

    @property
    def detail(self) -> str:
        if self.stage == "transcribe" and self.total:
            return (f"감지된 음성 {clock_text(self.done / 1000)} / {clock_text(self.total / 1000)} 처리"
                    " · 무음 제외, 이 단계의 진행률")
        if self.stage == "transcribe":
            return "감지된 음성이 없습니다. 결과를 정리합니다."
        if self.stage == "refine_speakers" and self.total:
            return f"음성 검사 {int(self.done)} / {int(self.total)}개 · 이 단계의 진행률"
        if self.stage in ("speakers", "verify_speakers") and self.total:
            return f"음성 구간 {int(self.done)} / {int(self.total)}개 · 이 단계의 진행률"
        if self.stage in ("downloading_public_model", "downloading_gpu_model"):
            return "처음에는 약 3.1GB를 받습니다. 다음부터는 저장된 모델을 사용합니다."
        if self.stage == "complete":
            return "모든 파일의 저장을 확인했습니다. 웹에서 전사와 AI 분석을 확인하세요."
        return "처리 중입니다. 이 단계는 진행률을 계산할 수 없습니다."

    def remaining_text(self, now: float) -> str:
        if self.stage not in ("transcribe", "speakers", "verify_speakers", "refine_speakers"):
            return ""
        if self.percent == 100:
            return "이 단계 처리 완료 · 다음 단계 준비 중"
        if self.remaining_seconds is None:
            return "남은 시간 계산 중 · 처리 속도를 측정하고 있습니다."
        age = max(0, now - self.updated_at)
        # A blocked decoder must not turn a wall-clock countdown into success.
        if age >= self.remaining_seconds or age > max(60, self.remaining_seconds / 2):
            return "현재 구간 처리 중 · 남은 시간 추정 갱신 대기"
        remaining = max(1, math.ceil((self.remaining_seconds - age) / 10) * 10)
        label = {"transcribe": "전사 단계", "speakers": "화자 특징 추출", "verify_speakers": "짧은 발화 화자 확인", "refine_speakers": "구간 안 화자 전환 확인"}[self.stage]
        rate = f" · 음성 길이 대비 {self.speed:.1f}배속" if self.speed else ""
        return f"{label} 남음 약 {clock_text(remaining)}{rate}"


class ProgressEstimator:
    """Estimate only measured work in the current file/stage; never invent a total ETA.

    Use at least three completed chunks and ten seconds of observation. A rolling
    window adapts to harder speech; entering another file/stage clears samples.
    Estimates exclude unmeasured speaker processing, saving and following files.
    """
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.key = None
        self.samples = deque(maxlen=20)
        self.first_at = None

    def update(self, stage, done, total, *, file_index=0):
        now = self.clock()
        key = (file_index, stage, total)
        valid = (stage in ("transcribe", "speakers", "verify_speakers", "refine_speakers") and done is not None and total is not None
                 and math.isfinite(done) and math.isfinite(total) and total > 0 and done >= 0)
        if key != self.key or not valid or (self.samples and done < self.samples[-1][1]):
            self.key = key
            self.samples.clear()
            self.first_at = None
        if not valid:
            return now, None, None
        if self.first_at is None:
            self.first_at = now
        if not self.samples or done > self.samples[-1][1]:
            self.samples.append((now, done))
        if len(self.samples) < 4:
            return now, None, None
        elapsed = now - self.samples[0][0]
        processed = done - self.samples[0][1]
        if now - self.first_at < 10 or elapsed <= 0 or processed <= 0:
            return now, None, None
        rate = processed / elapsed
        return now, max(0, total - done) / rate, rate / 1000 if stage == "transcribe" else None
