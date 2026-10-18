"""A second, local voice encoder checks unresolved short replies.

Never infer a teaching role, borrow a neighbour's identity, or alter ASR text/time.
The original encoder establishes clusters. CAM++ may fill an unresolved turn only
when both encoders prefer the same cluster and CAM++ separates it from alternatives.
Scores are similarity heuristics, not calibrated identity probabilities.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import numpy as np

MODEL_NAME = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/" + MODEL_NAME
MODEL_SHA256 = "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2"
MODEL_BYTES = 28_281_164


def valid_model(path: Path) -> bool:
    return path.is_file() and path.stat().st_size == MODEL_BYTES and hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256


def bundled_model() -> Path | None:
    root = getattr(sys, "_MEIPASS", None)
    return Path(root) / "speaker-models" / MODEL_NAME if root else None


def prepare_model(cache: Path) -> Path:
    bundled = bundled_model()
    if bundled is not None:
        if not valid_model(bundled):
            raise RuntimeError("화자 확인 모델 파일을 확인할 수 없습니다. 워커를 다시 설치하세요.")
        return bundled  # Installed applications never download this model at runtime.
    target = cache / MODEL_NAME
    if valid_model(target):
        return target
    import requests
    cache.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with requests.get(MODEL_URL, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=cache, suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                size = 0
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > MODEL_BYTES:
                        raise RuntimeError("화자 모델 파일 크기가 올바르지 않습니다.")
                    output.write(chunk)
        if not valid_model(temporary):
            raise RuntimeError("화자 모델 파일 검증에 실패했습니다.")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


class ShortVoiceEncoder:
    def __init__(self, model: Path):
        import sherpa_onnx
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=2, debug=False, provider="cpu"))

    def embed(self, wav, start_ms: int, end_ms: int):
        # Sub-400ms interjections remain unknown: less padding is not evidence of identity.
        audio = wav[max(0, start_ms * 16):min(len(wav), end_ms * 16)]
        if len(audio) < 6400 or not np.all(np.isfinite(audio)) or not np.any(audio):
            return None
        stream = self.extractor.create_stream()
        stream.accept_waveform(16000, audio)
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            return None
        vector = np.asarray(self.extractor.compute(stream), dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if np.isfinite(norm) and norm > 0 else None


def verified_short_matches(primary, secondary, anchor_labels, candidates):
    """Return only new supported labels; never relabel existing/long turns.

    Each cluster needs at least two independent long turns in both encoders.
    Missing reference clusters disable recovery so an unseen competitor cannot
    be silently omitted from the decision. No recovered turn becomes an anchor.
    """
    cluster_ids = sorted(set(anchor_labels.values()))
    if len(cluster_ids) < 2:
        return {}
    primary_centers, secondary_centers = [], []
    for label in cluster_ids:
        ids = [i for i, k in anchor_labels.items() if k == label and i in primary and i in secondary]
        if len(ids) < 2:
            return {}
        for source, centers in ((primary, primary_centers), (secondary, secondary_centers)):
            mean = np.mean([source[i] for i in ids], axis=0)
            norm = np.linalg.norm(mean)
            if not np.isfinite(norm) or norm <= 0:
                return {}
            centers.append(mean / norm)
    result = {}
    for i in candidates:
        if i in anchor_labels or i not in primary or i not in secondary:
            continue
        first = np.asarray(primary_centers) @ primary[i]
        second = np.asarray(secondary_centers) @ secondary[i]
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            continue
        ranking = np.argsort(second)[::-1]
        best, runner_up = ranking[:2]
        if (int(np.argmax(first)) == best and second[best] >= 0.50
                and second[best] - second[runner_up] >= 0.10):
            result[i] = cluster_ids[best]
    return result
