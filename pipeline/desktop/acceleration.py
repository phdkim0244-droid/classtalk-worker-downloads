"""Apple Silicon acceleration. Audio stays in memory and never leaves the device."""
from __future__ import annotations

import platform
from pathlib import Path
from types import SimpleNamespace

MLX_REPOSITORY = "mlx-community/whisper-large-v3-mlx"
MLX_REVISION = "49e6aa286ad60c14352c404340ded53710378a11"
MLX_FILES = ("config.json", "weights.npz")


def supports_mlx() -> bool:
    # Pinned MLX wheels require macOS 15. Older Macs keep the CPU engine.
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        if int(platform.mac_ver()[0].split(".")[0]) < 15:
            return False
        import mlx.core as mx
        return mx.metal.is_available()
    except (ImportError, OSError, RuntimeError, ValueError):
        return False


def download_mlx_model(cache: Path) -> str:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    downloaded = None
    for filename in MLX_FILES:
        downloaded = hf_hub_download(MLX_REPOSITORY, filename, revision=MLX_REVISION,
                                     cache_dir=str(cache), token=False,
                                     endpoint="https://huggingface.co")
    return str(Path(downloaded).parent)


def release_mlx_model() -> None:
    """Release GPU weights before switching engines, including a failed prepare."""
    from mlx_whisper.transcribe import ModelHolder
    import mlx.core as mx
    ModelHolder.model = None
    ModelHolder.model_path = None
    mx.clear_cache()


class MlxWhisperModel:
    """Adapt local MLX results to the existing speech timing/speaker pipeline.

    MLX 0.4.3 supports deterministic greedy decoding, not beam search. Keep this
    distinction explicit; never silently claim CPU beam-5 output equivalence.
    Passing an ndarray reuses PyAV decoding and does not require system FFmpeg.
    """
    def __init__(self, path: str):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder
        self.path = str(Path(path).resolve(strict=True))
        # A local path prevents mlx-whisper from resolving a mutable hub model.
        if not all((Path(self.path) / name).is_file() for name in MLX_FILES):
            raise ValueError("GPU 모델 파일이 준비되지 않았습니다.")
        from desktop.metal_attention import accelerate_encoder
        model = ModelHolder.get_model(self.path, mx.float16)
        accelerate_encoder(model)
        if not getattr(model, "_classtalk_quantized", False):
            # Compress linear/embedding weights in memory, retaining large-v3's
            # architecture. No quantized model is uploaded or written to disk.
            import mlx.nn as nn
            nn.quantize(model, group_size=64, bits=8)
            mx.eval(model.parameters())
            model._classtalk_quantized = True
            mx.clear_cache()

    def transcribe(self, audio, *, language, vad_filter, word_timestamps,
                   beam_size, condition_on_previous_text, temperature):
        import mlx_whisper
        if vad_filter or condition_on_previous_text or temperature != 0:
            raise ValueError("GPU 전사는 독립된 음성 구간과 결정적 디코딩만 지원합니다.")
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=self.path, language=language, verbose=None,
            word_timestamps=word_timestamps, condition_on_previous_text=False,
            temperature=0,
        )
        rows = [SimpleNamespace(start=row["start"], end=row["end"], text=row["text"],
                                words=[SimpleNamespace(**word) for word in row.get("words", [])])
                for row in result["segments"]]
        return iter(rows), None
