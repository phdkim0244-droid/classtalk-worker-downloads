import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import numpy as np
from app.transcribe.short_voice_verifier import verified_short_matches, ShortVoiceEncoder, prepare_model
from desktop.progress import Progress, ProgressEstimator

class ShortVoiceTests(unittest.TestCase):
    def profiles(self):
        p = {0: np.array([1., 0.]), 1: np.array([1., 0.]), 2: np.array([0., 1.]), 3: np.array([0., 1.])}
        return p, dict(p), {0: 7, 1: 7, 2: 9, 3: 9}

    def test_independent_encoders_must_agree_with_clear_second_match(self):
        first, second, labels = self.profiles()
        first[4] = np.array([.72, .69]); second[4] = np.array([1., 0.])
        self.assertEqual(verified_short_matches(first, second, labels, [4]), {4: 7})
        for vector in (np.array([0., 1.]), np.array([.72, .69]), np.array([.4, .1]), np.array([np.nan, 0.])):
            second[4] = vector
            self.assertEqual(verified_short_matches(first, second, labels, [4]), {})
        first[4] = np.array([np.nan, 0.]); second[4] = np.array([1., 0.])
        self.assertEqual(verified_short_matches(first, second, labels, [4]), {})

    def test_missing_competitor_reference_does_not_force_match(self):
        first, second, labels = self.profiles()
        first[4] = second[4] = np.array([1., 0.]); del second[3]
        self.assertEqual(verified_short_matches(first, second, labels, [4]), {})
        self.assertEqual(verified_short_matches(first, first, {0: 7, 1: 7}, [4]), {})

    def test_recovered_candidates_never_become_references(self):
        first, second, labels = self.profiles()
        first[4] = second[4] = np.array([1., 0.]); first[5] = second[5] = np.array([.71, .70])
        self.assertEqual(verified_short_matches(first, second, labels, [4, 5, 0]), {4: 7})
        self.assertEqual(set(labels), {0, 1, 2, 3})

    def test_encoder_never_borrows_neighbour_audio_or_encodes_silence(self):
        encoder = ShortVoiceEncoder.__new__(ShortVoiceEncoder); extractor = encoder.extractor = Mock()
        extractor.compute.return_value = [1., 0.]
        audio = np.full(16000 * 3, .1, dtype=np.float32)
        self.assertIsNone(encoder.embed(audio, 1000, 1399))
        self.assertIsNone(encoder.embed(np.zeros(16000), 0, 1000))
        self.assertIsNone(encoder.embed(np.full(16000, np.nan), 0, 1000))
        extractor.create_stream.assert_not_called()
        np.testing.assert_array_equal(encoder.embed(audio, 1000, 1500), [1., 0.])
        np.testing.assert_array_equal(extractor.create_stream.return_value.accept_waveform.call_args.args[1], audio[16000:24000])

    def test_bundled_model_checked_without_network(self):
        with patch('app.transcribe.short_voice_verifier.bundled_model', return_value=Path('/bundle/model.onnx')), patch('app.transcribe.short_voice_verifier.valid_model', return_value=True), patch('requests.get') as get:
            self.assertEqual(prepare_model(Path('/unused')), Path('/bundle/model.onnx')); get.assert_not_called()
        with patch('app.transcribe.short_voice_verifier.bundled_model', return_value=Path('/bad')), patch('app.transcribe.short_voice_verifier.valid_model', return_value=False), patch('requests.get') as get:
            with self.assertRaises(RuntimeError): prepare_model(Path('/unused'))
            get.assert_not_called()

    def test_bad_download_preserves_cache_and_cleans_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'model.onnx'; target.write_bytes(b'old-invalid')
            response = Mock(); response.__enter__ = Mock(return_value=response); response.__exit__ = Mock(return_value=False)
            response.iter_content.return_value = [b'bad bytes']
            with patch('app.transcribe.short_voice_verifier.bundled_model', return_value=None), patch('app.transcribe.short_voice_verifier.MODEL_NAME', 'model.onnx'), patch('app.transcribe.short_voice_verifier.MODEL_BYTES', 9), patch('app.transcribe.short_voice_verifier.MODEL_SHA256', hashlib.sha256(b'goodbytes').hexdigest()), patch('requests.get', return_value=response):
                with self.assertRaises(RuntimeError): prepare_model(Path(directory))
            self.assertEqual(target.read_bytes(), b'old-invalid')
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_second_encoder_stage_percentage_and_eta(self):
        now = [0.]; estimator = ProgressEstimator(clock=lambda: now[0])
        for value in (0, 10, 20, 30):
            now[0] = value / 2
            stamp, remaining, speed = estimator.update('verify_speakers', value, 100)
        p = Progress('verify_speakers',30,100,updated_at=stamp,remaining_seconds=remaining,speed=speed)
        self.assertIn('30%',p.heading); self.assertIn('30 / 100',p.detail)
        self.assertIn('짧은 발화 화자 확인',p.remaining_text(stamp)); self.assertIsNone(speed)

if __name__ == '__main__': unittest.main()
