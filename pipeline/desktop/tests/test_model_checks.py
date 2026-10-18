"""Reference assertions are isolated from the production decoder's input."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from desktop.model_checks import check_model_regressions


class ModelRegressionTests(unittest.TestCase):
    def test_expected_numbers_only_validate_output_and_never_hint_the_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "synthetic.wav"
            audio.write_bytes(b"test audio")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"cases": [{
                "id": "contrast", "audio": audio.name,
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "expectedPattern": r"^[^\d]*7\s*ml[^\d]*3\s*ml[^\d]*$",
            }]}))
            engine = Mock()
            engine.transcribe.return_value = [{"text": "첫 컵은 7 ml, 둘째 컵은 3 ml"}]
            result = check_model_regressions(engine, manifest)
            self.assertTrue(result[0]["ok"])
            engine.transcribe.assert_called_once_with(
                audio, {"groupNo": 1, "students": 1},
                {"sessionType": "class", "studentsPerGroup": 1})
            for wrong in ("첫 컵은 3 ml, 둘째 컵은 3 ml", "첫 컵은 3 ml, 둘째 컵은 7 ml",
                          "첫 컵은 7 mm, 둘째 컵은 3 ml", "첫 컵은 7 ml, 둘째 컵은 3 mm"):
                engine.transcribe.return_value = [{"text": wrong}]
                with self.assertRaises(AssertionError):
                    check_model_regressions(engine, manifest)

    def test_modified_audio_is_rejected_before_transcription(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "synthetic.wav").write_bytes(b"different audio")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"cases": [{
                "id": "contrast", "audio": "synthetic.wav", "sha256": "0" * 64,
                "expectedPattern": "unrelated reference",
            }]}))
            engine = Mock()
            with self.assertRaises(ValueError):
                check_model_regressions(engine, manifest)
            engine.transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
