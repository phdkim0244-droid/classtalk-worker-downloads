"""No real GPU, model download, credentials or microphone used in unit checks."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as Row
from unittest.mock import patch, Mock

from desktop.acceleration import supports_mlx, MlxWhisperModel, download_mlx_model, MLX_REVISION


class AccelerationTests(unittest.TestCase):
    def test_windows_intel_and_older_macos_never_import_mlx(self):
        for system, machine, version in [("Windows", "AMD64", ""), ("Darwin", "x86_64", "15.1"),
                                         ("Darwin", "arm64", "14.7")]:
            with self.subTest(system=system, machine=machine, version=version), \
                    patch("platform.system", return_value=system), \
                    patch("platform.machine", return_value=machine), \
                    patch("platform.mac_ver", return_value=(version, (), machine)):
                self.assertFalse(supports_mlx())

    def test_download_pins_every_file_without_credentials(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("huggingface_hub.hf_hub_download", return_value=directory + "/weights.npz") as get:
            self.assertEqual(download_mlx_model(Path(directory)), directory)
            self.assertEqual(get.call_count, 2)
            for call in get.call_args_list:
                self.assertEqual(call.kwargs["revision"], MLX_REVISION)
                self.assertIs(call.kwargs["token"], False)

    def test_adapter_preserves_word_timing_without_prompts_or_network_paths(self):
        model = MlxWhisperModel.__new__(MlxWhisperModel)
        model.path = "/existing/local/model"
        decode = Mock(return_value={"segments": [{"start": 0.1, "end": 1.5, "text": "2ml",
                          "words": [{"start": 0.1, "end": 1.5, "word": "2ml", "probability": .9}]}]})
        audio = object()
        with patch.dict("sys.modules", {"mlx_whisper": Row(transcribe=decode)}):
            rows, _ = model.transcribe(audio, language="ko", vad_filter=False, word_timestamps=True,
                                      beam_size=5, condition_on_previous_text=False, temperature=0)
        self.assertEqual(list(rows)[0].words[0].word, "2ml")
        self.assertIs(decode.call_args.args[0], audio)
        self.assertEqual(decode.call_args.kwargs["path_or_hf_repo"], model.path)
        self.assertIsNone(decode.call_args.kwargs["verbose"])
        self.assertNotIn("initial_prompt", decode.call_args.kwargs)
        self.assertNotIn("beam_size", decode.call_args.kwargs)

    def test_cpu_override_does_not_probe_or_download_gpu(self):
        from desktop.engine import Engine
        with tempfile.TemporaryDirectory() as directory, \
                patch("desktop.acceleration.supports_mlx") as probe, \
                patch("desktop.acceleration.download_mlx_model") as download, \
                patch("desktop.engine.download_model", return_value=directory), \
                patch("faster_whisper.WhisperModel"), \
                patch("app.transcribe.faster_whisper_diar_backend.FasterWhisperDiarTranscriber._load_encoder"), \
                patch("app.transcribe.short_voice_verifier.prepare_model", return_value=Path(directory)), \
                patch("app.transcribe.short_voice_verifier.ShortVoiceEncoder"):
            engine = Engine(Path(directory), acceleration="cpu")
            engine.prepare()
            probe.assert_not_called()
            download.assert_not_called()
            self.assertEqual(engine.backend, "cpu")
            self.assertEqual(engine.transcriber.speech_window_seconds, 0)

    def test_auto_on_cpu_retains_beam5_and_uses_bounded_original_audio(self):
        from desktop.engine import Engine
        with tempfile.TemporaryDirectory() as directory, \
                patch("desktop.acceleration.supports_mlx", return_value=False), \
                patch("desktop.engine.download_model", return_value=directory), \
                patch("faster_whisper.WhisperModel"), \
                patch("app.transcribe.faster_whisper_diar_backend.FasterWhisperDiarTranscriber._load_encoder"), \
                patch("app.transcribe.short_voice_verifier.prepare_model", return_value=Path(directory)), \
                patch("app.transcribe.short_voice_verifier.ShortVoiceEncoder"):
            engine = Engine(Path(directory))
            engine.prepare()
            self.assertEqual(engine.transcriber.speech_window_seconds, 28)
            self.assertEqual(engine.transcriber.beam_size, 5)
            self.assertIn("CPU 묶음", engine.description)

    def test_failed_gpu_prepare_releases_weights_before_cpu_retry(self):
        from desktop.engine import Engine
        with tempfile.TemporaryDirectory() as directory, \
                patch("desktop.acceleration.supports_mlx", return_value=True), \
                patch("desktop.acceleration.download_mlx_model", return_value=directory), \
                patch("desktop.acceleration.MlxWhisperModel", side_effect=RuntimeError("GPU memory")), \
                patch("desktop.acceleration.release_mlx_model") as release:
            engine = Engine(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "GPU memory"):
                engine.prepare()
            release.assert_called_once_with()
            self.assertIsNone(engine.transcriber)
            self.assertEqual(engine.backend, "cpu")


if __name__ == "__main__":
    unittest.main()
