"""Regression cases for silence inflation and incorrect speaker carry-forward.

No account, audio device or network/model download is used by these tests.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as Row
import unittest
from unittest.mock import patch

from app.speakers import label_segments
from app.transcribe.base import Segment
from app.transcribe.speech_timing import SpeechTurn, align_words, transcribe_speech
from desktop.engine import Engine, MODEL_REVISIONS, download_model


class SpeechTimingTests(unittest.TestCase):
    def test_decoder_sentence_cannot_charge_twenty_seconds_of_silence(self):
        decoded = [Row(start=0, end=26, text="첫 질문 다음 질문", words=[
            Row(start=1.1, end=2.0, word=" 첫 질문"),
            Row(start=24.1, end=25.9, word=" 다음 질문"),
        ])]
        segments, turns = align_words(decoded, [SpeechTurn(1000, 2200), SpeechTurn(24000, 26000)])
        self.assertEqual(turns, [0, 1])
        self.assertEqual([(w.start_ms, w.end_ms) for s in segments for w in s.words], [(1100, 2000), (24100, 25900)])
        self.assertEqual([(s.start_ms, s.end_ms) for s in segments], [(1100, 2000), (24100, 25900)])
        self.assertEqual([s.text for s in segments], ["첫 질문", "다음 질문"])
        self.assertEqual(sum(s.end_ms - s.start_ms for s in segments), 2700)

    def test_one_decoder_sentence_can_belong_to_two_acoustic_speakers(self):
        decoded = [Row(words=[Row(start=1, end=2, word=" 질문"),
                              Row(start=2.5, end=3.2, word=" 답변")])]
        segments, turn_ids = align_words(decoded, [SpeechTurn(900, 2100), SpeechTurn(2400, 3400)])
        self.assertEqual(turn_ids, [0, 1])
        self.assertEqual(len(segments), 2)

    def test_word_boundaries_are_clipped_to_recorded_speech(self):
        decoded = [Row(words=[Row(start=0, end=22, word=" 발화")])]
        segments, _ = align_words(decoded, [SpeechTurn(20000, 21000)])
        self.assertEqual((segments[0].start_ms, segments[0].end_ms), (20000, 21000))

    def test_no_words_fallback_never_spans_a_silent_gap(self):
        segments, _ = align_words([Row(start=0, end=25, text="불완전한 타임스탬프", words=[])],
                                  [SpeechTurn(1000, 2000), SpeechTurn(24000, 25000)])
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].words, [])
        self.assertLessEqual(segments[0].end_ms - segments[0].start_ms, 1000)

    def test_unanchored_words_do_not_invent_a_whole_speech_turn(self):
        decoded = [Row(words=[Row(start=20, end=20, word="시간 근거 없음"),
                              Row(start=25, end=27, word="무음 안의 추측")])]
        self.assertEqual(align_words(decoded, [SpeechTurn(1000, 2500), SpeechTurn(40000, 42000)]),
                         ([], []))

    def test_silent_file_does_not_invoke_decoder_or_hallucinate(self):
        with patch("app.transcribe.speech_timing.detect_speech", return_value=[]):
            class Model:
                def transcribe(self, *_args, **_kwargs):
                    raise AssertionError("Silence must not be decoded")
            self.assertEqual(transcribe_speech(Model(), [], language="ko"), ([], [], []))

    def test_next_turn_leading_word_cannot_be_decoded_into_previous_turn(self):
        import numpy as np
        turns = [SpeechTurn(1000, 2000), SpeechTurn(24000, 25000)]
        calls = []
        class Model:
            def transcribe(self, audio, **options):
                calls.append((len(audio), options))
                word = " 첫 질문" if len(calls) == 1 else " 두 설명"
                return iter([Row(words=[Row(start=0, end=0.8, word=word)])]), None
        with patch("app.transcribe.speech_timing.detect_speech", return_value=turns):
            result, ids, _ = transcribe_speech(Model(), np.zeros(30 * 16000), language="ko")
        self.assertEqual([length for length, _ in calls], [16000, 16000])
        self.assertTrue(all(not options['vad_filter'] and options['word_timestamps']
                            and options['beam_size'] == 5 and options['temperature'] == 0
                            and 'initial_prompt' not in options and 'hotwords' not in options
                            for _, options in calls))
        self.assertEqual(ids, [0, 1])
        self.assertEqual([(w.start_ms, w.end_ms) for s in result for w in s.words], [(1000, 1800), (24000, 24800)])
        self.assertEqual([(s.start_ms, s.end_ms, s.text) for s in result],
                         [(1000, 2000, "첫 질문"), (24000, 25000, "두 설명")])

    def test_unknown_speaker_does_not_borrow_prior_identity_or_consume_number(self):
        raw = [Segment("SPEAKER_02", 0, 2000, "한 명"),
               Segment("SPEAKER_UNKNOWN", 2300, 2500, "네"),
               Segment("SPEAKER_01", 3000, 5000, "다른 사람")]
        self.assertEqual([s["speaker_label"] for s in label_segments(raw, is_teacher=False)],
                         ["학생1", "미상", "학생2"])
        self.assertEqual([s["speaker_label"] for s in label_segments(raw, is_teacher=True)],
                         ["교사", "교사", "교사"])

    def test_desktop_model_ignores_operator_environment(self):
        with patch.dict("os.environ", {"WHISPER_MODEL": "some-remote-model", "SPEAKERS": "99"}):
            self.assertEqual(Engine(Path("unused")).model_size, "large-v3")
        with self.assertRaises(ValueError):
            Engine(Path("unused"), model_size="some-remote-model")

    def test_v3_downloads_keep_128_mel_preprocessor_and_pinned_public_files(self):
        # A missing preprocessor config silently selects the 80-band default and
        # makes v3 inference fail. These files must all belong to one revision.
        for name, repository in (("large-v3", "Systran/faster-whisper-large-v3"),
                                 ("large-v3-turbo", "dropbox-dash/faster-whisper-large-v3-turbo")):
            with self.subTest(model=name), patch("huggingface_hub.hf_hub_download", side_effect=lambda _repo, filename, **_kw: f"/cache/snapshot/{filename}") as download:
                self.assertEqual(download_model(Path("cache"), name), str(Path("/cache/snapshot")))
            self.assertEqual([call.args[1] for call in download.call_args_list],
                             ["config.json", "tokenizer.json", "vocabulary.json", "preprocessor_config.json", "model.bin"])
            for call in download.call_args_list:
                self.assertEqual(call.args[0], repository)
                self.assertEqual(call.kwargs["revision"], MODEL_REVISIONS[name])
                self.assertIs(call.kwargs["token"], False)
                self.assertEqual(call.kwargs["endpoint"], "https://huggingface.co")

    def test_previous_models_keep_their_existing_file_contract(self):
        for name in ("base", "small", "medium"):
            with self.subTest(model=name), patch("huggingface_hub.hf_hub_download", side_effect=lambda _repo, filename, **_kw: f"/old-cache/{filename}") as download:
                download_model(Path("cache"), name)
                self.assertEqual([call.args[1] for call in download.call_args_list],
                                 ["config.json", "tokenizer.json", "vocabulary.txt", "model.bin"])
                self.assertTrue(all(call.args[0] == f"Systran/faster-whisper-{name}"
                                    for call in download.call_args_list))

    def test_diarization_uses_acoustic_turn_not_decoder_fragment(self):
        import numpy as np
        from app.transcribe.faster_whisper_diar_backend import FasterWhisperDiarTranscriber
        transcriber = FasterWhisperDiarTranscriber()
        segments = [Segment("SPEAKER_00", 110, 600, "긴 발화의 일부"),
                    Segment("SPEAKER_00", 2200, 2350, "네"),
                    Segment("SPEAKER_00", 3020, 3200, "다른 발화 일부")]
        turns = [SpeechTurn(100, 2100), SpeechTurn(2200, 2350), SpeechTurn(3000, 5000)]
        calls = []
        def embedding(_wav, start, end):
            calls.append((start, end))
            return None if end - start < 400 else np.ones(256)
        with patch.object(transcriber, "_load_model", return_value=object()), \
                patch("app.transcribe.faster_whisper_diar_backend._load_audio_16k_mono", return_value=np.zeros(64000)), \
                patch("app.transcribe.faster_whisper_diar_backend.transcribe_speech", return_value=(segments, [0, 1, 2], turns)), \
                patch.object(transcriber, "_embed", side_effect=embedding), \
                patch.object(transcriber, "_cluster", return_value=[0, 1]):
            result = transcriber.transcribe("unused.wav", is_teacher=False)
        self.assertEqual(calls, [(100, 2100), (2200, 2350), (3000, 5000)])
        self.assertEqual([s.speaker for s in result], ["SPEAKER_00", "SPEAKER_UNKNOWN", "SPEAKER_01"])

    def test_short_response_needs_its_own_clear_centroid_match(self):
        import numpy as np
        from app.transcribe.faster_whisper_diar_backend import FasterWhisperDiarTranscriber
        transcriber = FasterWhisperDiarTranscriber()
        turns = [SpeechTurn(0, 2000), SpeechTurn(2500, 3000),
                 SpeechTurn(3500, 4000), SpeechTurn(4500, 6500)]
        segments = [Segment("SPEAKER_00", t.start_ms, t.end_ms, "음성") for t in turns]
        embeddings = [np.array([1., 0.]), np.array([2 ** -0.5, 2 ** -0.5]),
                      np.array([1., 0.]), np.array([0., 1.])]
        with patch.object(transcriber, "_load_model", return_value=object()), \
                patch("app.transcribe.faster_whisper_diar_backend._load_audio_16k_mono", return_value=np.zeros(104000)), \
                patch("app.transcribe.faster_whisper_diar_backend.transcribe_speech", return_value=(segments, list(range(4)), turns)), \
                patch.object(transcriber, "_embed", side_effect=embeddings), \
                patch.object(transcriber, "_cluster", return_value=[0, 1]) as cluster:
            result = transcriber.transcribe("unused.wav", is_teacher=False)
        self.assertEqual(len(cluster.call_args[0][0]), 2)  # only full voice windows establish identities
        self.assertEqual([s.speaker for s in result],
                         ["SPEAKER_00", "SPEAKER_UNKNOWN", "SPEAKER_00", "SPEAKER_01"])


if __name__ == "__main__":
    unittest.main()
