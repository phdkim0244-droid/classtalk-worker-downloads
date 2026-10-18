"""Local-only STT and pseudonymous speakers; no operator configuration imports."""
from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL = "large-v3"
MODEL_REVISIONS = {
    "large-v3": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
    "large-v3-turbo": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    "medium": "08e178d48790749d25932bbc082711ddcfdfbc4f",
    "small": "536b0662742c02347bc0e980a01041f333bce120",
    "base": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
}
MODEL_REPOSITORIES = {
    "large-v3-turbo": "dropbox-dash/faster-whisper-large-v3-turbo",
    **{name: f"Systran/faster-whisper-{name}" for name in ("large-v3", "medium", "small", "base")},
}
MODEL_FILES = {
    # v3 uses 128 mel bands; its preprocessor config must accompany the weights.
    **{name: ("config.json", "tokenizer.json", "vocabulary.json",
              "preprocessor_config.json", "model.bin")
       for name in ("large-v3", "large-v3-turbo")},
    **{name: ("config.json", "tokenizer.json", "vocabulary.txt", "model.bin")
       for name in ("medium", "small", "base")},
}


def download_model(cache: Path, model_size: str = DEFAULT_MODEL) -> str:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    # A GUI has no terminal. snapshot_download's threaded tqdm locks deadlock
    # in frozen Windows builds; download this fixed model's files sequentially.
    disable_progress_bars()
    model_path = None
    for filename in MODEL_FILES[model_size]:
        downloaded = hf_hub_download(
            MODEL_REPOSITORIES[model_size], filename,
            revision=MODEL_REVISIONS[model_size],
            cache_dir=str(cache), endpoint="https://huggingface.co", token=False,
        )
        model_path = str(Path(downloaded).parent)
    return model_path


class Engine:
    def __init__(self, cache: Path, *, model_size: str = DEFAULT_MODEL, acceleration: str = "auto"):
        if model_size not in MODEL_REVISIONS:
            raise ValueError("지원하지 않는 로컬 전사 모델입니다.")
        self.cache = cache
        self.model_size = model_size
        if acceleration not in ("auto", "cpu"):
            raise ValueError("지원하지 않는 가속 설정입니다.")
        self.acceleration = acceleration
        self.backend = "cpu"
        self.transcriber = None

    def prepare(self, progress=None) -> None:
        progress = progress or (lambda _: None)
        progress("importing_model_libraries")
        # Release the previous CPU instance before preparing another large model.
        self.transcriber = None
        if self.backend == "mlx":
            from desktop.acceleration import release_mlx_model
            release_mlx_model()
        from faster_whisper import WhisperModel
        from app.transcribe.faster_whisper_diar_backend import FasterWhisperDiarTranscriber
        self.cache.mkdir(parents=True, exist_ok=True)
        # Construct without the operator backend's environment-based __init__.
        # The desktop build must not inherit an operator's model/server settings.
        transcriber = FasterWhisperDiarTranscriber.__new__(FasterWhisperDiarTranscriber)
        transcriber.model_size = self.model_size
        transcriber.device = "cpu"
        transcriber.compute_type = "int8"
        transcriber.language = "ko"
        transcriber.beam_size = 5
        transcriber.speech_window_seconds = 0
        transcriber.n_speakers = None
        transcriber.threshold = 0.65
        transcriber._encoder = None
        from desktop.acceleration import supports_mlx, download_mlx_model, MlxWhisperModel
        self.backend = "cpu"
        if self.acceleration == "auto" and self.model_size == DEFAULT_MODEL and supports_mlx():
            progress("downloading_gpu_model")
            model_path = download_mlx_model(self.cache)
            progress("initializing_gpu_model")
            try:
                transcriber._model = MlxWhisperModel(model_path)
            except Exception:
                from desktop.acceleration import release_mlx_model
                release_mlx_model()
                raise
            transcriber.speech_window_seconds = 30
            self.backend = "mlx"
        else:
            transcriber.speech_window_seconds = 28 if self.acceleration == "auto" else 0
            progress("downloading_public_model")
            model_path = download_model(self.cache, self.model_size)
            progress("initializing_whisper_model")
            transcriber._model = WhisperModel(
                model_path, device="cpu", compute_type="int8", local_files_only=True,
                cpu_threads=4, num_workers=1,
            )
        # Encoder weights are shipped inside the application, with no HF account.
        progress("initializing_speaker_model")
        transcriber._load_encoder()
        self.transcriber = transcriber
        progress("models_ready")

    @property
    def description(self) -> str:
        if self.backend == "mlx":
            return f"Mac GPU 가속 · {self.model_size}"
        label = "CPU 묶음 처리" if self.acceleration == "auto" else "CPU 호환"
        return f"{label} · {self.model_size}"

    def transcribe(self, path: Path, recording: dict, job: dict, *, progress=None) -> list[dict]:
        from app.speakers import label_segments
        if self.transcriber is None:
            raise RuntimeError("전사 모델이 준비되지 않았습니다.")
        teacher = recording.get("groupNo") is None
        speakers = recording.get("students") or job.get("studentsPerGroup")
        self.transcriber.n_speakers = int(speakers) if speakers else None
        segments = self.transcriber.transcribe(str(path), is_teacher=teacher, progress=progress)
        if progress:
            progress("label", None, None)
        labels = label_segments(
            segments, is_teacher=teacher,
            speaker_prefix="교사" if job.get("sessionType") == "council" else "학생",
        )
        return [{"speaker": row["speaker_label"], "startMs": row["start_ms"],
                 "endMs": row["end_ms"], "text": row["text"]} for row in labels]
