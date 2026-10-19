"""faster-whisper 전사 + 토큰 없는 화자분리 백엔드.

pyannote(HF 토큰·게이트 모델) 대신, resemblyzer 음성 임베딩 + 응집 클러스터링으로
모둠 내 학생을 자동 구분한다. 계정/토큰/외부 다운로드 불필요(모델 패키지 내장).

한계: 목소리가 비슷하거나 겹쳐 말하면 오차. 완벽한 화자분리는 아님.
옵션(env): WHISPER_MODEL(large-v3), SPEAKERS(고정 화자수), DIAR_THRESHOLD(0.65),
          WHISPER_LANG(ko), DEVICE(cpu)
"""
from __future__ import annotations

import os

import numpy as np

from .base import Segment, Transcriber
from .speech_timing import transcribe_speech

SR = 16000


def _load_audio_16k_mono(path: str) -> np.ndarray:
    """webm/opus 등 → 16kHz mono float32 [-1,1] (av/PyAV 사용, ffmpeg 불필요)."""
    import av

    container = av.open(path)
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=SR
    )
    chunks: list[np.ndarray] = []
    for frame in container.decode(stream):
        out = resampler.resample(frame)
        frames = out if isinstance(out, list) else [out]
        for rf in frames:
            chunks.append(rf.to_ndarray().reshape(-1))
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


class FasterWhisperDiarTranscriber(Transcriber):
    name = "faster-whisper-diar"

    def __init__(self) -> None:
        self.model_size = os.environ.get("WHISPER_MODEL", "large-v3")
        self.device = os.environ.get("DEVICE", "cpu")
        self.compute_type = os.environ.get(
            "COMPUTE_TYPE", "int8" if self.device == "cpu" else "float16"
        )
        self.language = os.environ.get("WHISPER_LANG", "ko")
        self.beam_size = max(1, int(os.environ.get("WHISPER_BEAM_SIZE", "5")))
        self.n_speakers = int(os.environ["SPEAKERS"]) if os.environ.get("SPEAKERS") else None
        self.threshold = float(os.environ.get("DIAR_THRESHOLD", "0.65"))
        self._model = None
        self._encoder = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def _load_encoder(self):
        if self._encoder is None:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder(device="cpu", verbose=False)
        return self._encoder

    def _embed(self, wav: np.ndarray, start_ms: int, end_ms: int):
        a = max(0, int(start_ms / 1000 * SR))
        b = min(len(wav), int(end_ms / 1000 * SR))
        seg = wav[a:b]
        if len(seg) < int(0.4 * SR):  # 0.4초 미만은 임베딩 신뢰도 낮음
            return None
        try:
            embedding = self._load_encoder().embed_utterance(seg)
            norm = np.linalg.norm(embedding)
            return embedding / norm if np.isfinite(norm) and norm > 0 else None
        except ValueError:
            return None

    def _cluster(self, embeddings: list[np.ndarray]) -> list[int]:
        from sklearn.cluster import AgglomerativeClustering

        X = np.vstack(embeddings)
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        if len(X) == 1:
            return [0]
        if self.n_speakers and self.n_speakers >= 1:
            n = min(self.n_speakers, len(X))
            # Ward on unit vectors accounts for cluster variance. Average
            # cosine linkage can spend a fixed speaker slot on one outlier and
            # merge two frequently recurring voices into the remaining slot.
            model = AgglomerativeClustering(n_clusters=n, linkage="ward")
        else:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.threshold,
                metric="cosine",
                linkage="average",
            )
        return list(model.fit_predict(X))

    def transcribe(self, audio_path: str, *, is_teacher: bool, progress=None) -> list[Segment]:
        progress = progress or (lambda *_: None)
        progress("decode", None, None)
        model = self._load_model()
        wav = _load_audio_16k_mono(audio_path)
        segs, turn_ids, turns = transcribe_speech(
            model, wav, language=self.language, beam_size=self.beam_size, progress=progress,
            window_seconds=getattr(self, "speech_window_seconds", 0),
        )
        if is_teacher or not segs:
            return segs

        # Cluster acoustic turns, not decoder sentences: decoder sentences can
        # bridge silent gaps or two speakers, and brief ASR fragments are poor
        # embedding samples. Every fragment of one acoustic turn shares its ID.
        embs: list[np.ndarray] = []
        idx_with_emb: list[int] = []
        progress("speakers", 0, len(turns))
        for i, s in enumerate(turns):
            e = self._embed(wav, s.start_ms, s.end_ms)
            if e is not None:
                embs.append(e)
                idx_with_emb.append(i)
            progress("speakers", i + 1, len(turns))

        # Resemblyzer's native partial window is 1.6 seconds. Short responses
        # should not dictate cluster formation, even if an embedding is possible.
        progress("cluster", None, None)
        reliable = [j for j, i in enumerate(idx_with_emb)
                    if turns[i].end_ms - turns[i].start_ms >= 1600]
        labels = self._cluster([embs[j] for j in reliable]) if reliable else []
        speakers = {idx_with_emb[j]: f"SPEAKER_{label:02d}" for j, label in zip(reliable, labels)}
        if labels:
            cluster_ids = sorted(set(labels))
            centers = np.vstack([np.mean([embs[j] for j, label in zip(reliable, labels) if label == k], axis=0)
                                 for k in cluster_ids])
            centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
            for j, i in enumerate(idx_with_emb):
                if j in reliable:
                    continue
                similarities = centers @ embs[j]
                ranking = np.argsort(similarities)[::-1]
                best = ranking[0]
                margin = similarities[best] - similarities[ranking[1]] if len(ranking) > 1 else 1.0
                if similarities[best] >= 0.65 and margin >= 0.05:
                    speakers[i] = f"SPEAKER_{cluster_ids[best]:02d}"
        if getattr(self, "verify_short_speakers", False):
            from .short_voice_verifier import verified_short_matches
            primary = dict(zip(idx_with_emb, embs))
            anchors = {idx_with_emb[j]: label for j, label in zip(reliable, labels)}
            candidates = [i for i in idx_with_emb if i not in speakers and i not in anchors]
            if (candidates or getattr(self, "refine_speaker_turns", False)) and len(set(anchors.values())) >= 2:
                # A separate model checks the same bounded audio; no neighbour
                # concatenation, padding with another voice, or role guessing.
                targets = list(anchors) + candidates
                secondary = {}
                progress("verify_speakers", 0, len(targets))
                for completed, i in enumerate(targets, 1):
                    turn = turns[i]
                    vector = self._short_encoder.embed(wav, turn.start_ms, turn.end_ms)
                    if vector is not None:
                        secondary[i] = vector
                    progress("verify_speakers", completed, len(targets))
                matches = verified_short_matches(primary, secondary, anchors, candidates)
                speakers.update({i: f"SPEAKER_{label:02d}" for i, label in matches.items()})
                if getattr(self, "refine_speaker_turns", False):
                    from .turn_refinement import refine_turns
                    return refine_turns(segs, turn_ids, turns, primary, secondary, anchors, speakers,
                                        lambda a, b: self._embed(wav, a, b),
                                        lambda a, b: self._short_encoder.embed(wav, a, b), progress)
        for segment, turn_id in zip(segs, turn_ids):
            segment.speaker = speakers.get(turn_id, "SPEAKER_UNKNOWN")

        return segs
