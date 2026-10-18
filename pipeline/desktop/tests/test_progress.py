"""Progress reflects completed work, never a timer or unconsumed ASR output."""
import threading
import unittest
from types import SimpleNamespace as Row
from unittest.mock import patch

from app.transcribe.speech_timing import SpeechTurn, transcribe_speech, speech_windows
from desktop.client import Cancelled, WorkerError
from desktop.controller import process_job
from desktop.progress import Progress, ProgressEstimator, clock_text


class ProgressTests(unittest.TestCase):
    def test_windows_keep_silence_and_map_each_speaker_to_original_time(self):
        import numpy as np
        turns = [SpeechTurn(1000, 2000), SpeechTurn(5000, 7000), SpeechTurn(40000, 41000)]
        events, chunks = [], []
        class Model:
            def transcribe(self, chunk, **kwargs):
                chunks.append(chunk)
                if len(chunks) == 1:
                    return iter([Row(words=[Row(word=" 첫째", start=.1, end=.9),
                                              Row(word=" 둘째", start=4.1, end=5.8)])]), None
                return iter([Row(words=[Row(word=" 셋째", start=.1, end=.9)])]), None
        wav = np.arange(42 * 16000, dtype=np.float32)
        with patch("app.transcribe.speech_timing.detect_speech", return_value=turns):
            rows, ids, actual_turns = transcribe_speech(Model(), wav, language="ko", window_seconds=28,
                                                       progress=lambda *args: events.append(args))
        self.assertEqual(len(chunks[0]), 6 * 16000)  # includes 3-second gap, no concatenation
        np.testing.assert_array_equal(chunks[0], wav[16000:112000])
        self.assertEqual(ids, [0, 1, 2])
        self.assertEqual([(r.start_ms, r.end_ms) for r in rows], [(1000, 2000), (5000, 7000), (40000, 41000)])
        self.assertEqual([r.text for r in rows], ["첫째", "둘째", "셋째"])
        self.assertEqual(events[-2:], [("transcribe", 3000, 4000), ("transcribe", 4000, 4000)])
        self.assertEqual(actual_turns, turns)
        self.assertEqual(list(speech_windows([], 28)), [])

    def test_eta_requires_measurements_and_resets_at_file_or_stage_boundary(self):
        now = [0.0]
        estimator = ProgressEstimator(lambda: now[0])
        for done, seconds in [(0, 0), (1000, 4), (2000, 8)]:
            now[0] = seconds
            self.assertIsNone(estimator.update("transcribe", done, 10000, file_index=1)[1])
        now[0] = 12
        updated, eta, speed = estimator.update("transcribe", 3000, 10000, file_index=1)
        self.assertEqual(eta, 28)
        self.assertEqual(speed, .25)
        event = Progress("transcribe", 3000, 10000, updated_at=updated,
                         remaining_seconds=eta, speed=speed)
        self.assertIn("전사 단계 남음 약", event.remaining_text(13))
        self.assertIn("갱신 대기", event.remaining_text(100))
        self.assertIsNone(estimator.update("transcribe", 1000, 10000, file_index=2)[1])
        self.assertIsNone(estimator.update("speakers", 1, 10, file_index=2)[1])
        self.assertIsNone(estimator.update("transcribe", float("nan"), 10)[1])

    def test_eta_adapts_to_slowdown_and_never_reports_job_complete(self):
        now = [0.0]
        estimator = ProgressEstimator(lambda: now[0])
        for i in range(4):
            now[0] = i * 10
            early = estimator.update("transcribe", i * 1000, 100000)[1]
        for i in range(4, 26):
            now[0] += 30
            late = estimator.update("transcribe", i * 1000, 100000)[1]
        self.assertGreater(late, early)
        event = Progress("transcribe", 100000, 100000, remaining_seconds=0)
        self.assertIn("다음 단계", event.remaining_text(1000))
        self.assertNotIn("전사 완료", event.remaining_text(1000))

    def test_fast_speaker_updates_still_finish_the_eta_warmup(self):
        now = [0.0]
        estimator = ProgressEstimator(lambda: now[0])
        for count in range(45):
            now[0] = count / 4
            _, eta, speed = estimator.update("speakers", count, 100, file_index=1)
        # Recent 20 samples span <10s, but total stage observation exceeds 10s.
        self.assertEqual(eta, 14)
        self.assertIsNone(speed)

    def test_progress_advances_after_lazy_decode_and_uses_speech_duration(self):
        import numpy as np
        events = []
        class Model:
            def transcribe(self, *_args, **_kwargs):
                def decode():
                    # The first turn is one second, the second three seconds.
                    self_test.assertEqual(events[-1][1], 0 if len(events) == 2 else 1000)
                    yield Row(start=0, end=0.5, text="테스트", words=[])
                return decode(), None
        self_test = self
        with patch("app.transcribe.speech_timing.detect_speech",
                   return_value=[SpeechTurn(0, 1000), SpeechTurn(30000, 33000)]):
            segments, _, _ = transcribe_speech(Model(), np.zeros(33 * 16000), language="ko",
                                               progress=lambda *args: events.append(args))
        self.assertEqual(events, [("detect", None, None), ("transcribe", 0, 4000),
                                  ("transcribe", 1000, 4000), ("transcribe", 4000, 4000)])
        self.assertEqual([s.text for s in segments], ["테스트", "테스트"])

    def test_silence_and_unknown_length_do_not_divide_by_zero_or_invent_percent(self):
        events = []
        with patch("app.transcribe.speech_timing.detect_speech", return_value=[]):
            self.assertEqual(transcribe_speech(None, [], language="ko",
                                               progress=lambda *args: events.append(args)), ([], [], []))
        self.assertEqual(events[-1], ("transcribe", 0, 0))
        for stage, done, total in [("transcribe", 0, 0), ("download", None, None),
                                   ("transcribe", float("nan"), 1)]:
            self.assertIsNone(Progress(stage, done, total).percent)
        self.assertEqual(Progress("transcribe", 9999, 10000).percent, 99)
        self.assertEqual(clock_text(3661), "1:01:01")

    def run_job(self, *, fail_save=False, cancel_during_transcribe=False):
        events, requests = [], []
        stop = threading.Event()
        class Api:
            def post(self, path, body, **kwargs):
                requests.append(path)
                if fail_save and path == "jobs/recording":
                    raise WorkerError("테스트 저장 실패")
                return {"downloadUrl": "unused"}
            def download(self, *_): pass
        class Engine:
            def transcribe(self, *_, progress):
                progress("transcribe", 0, 1000)
                if cancel_during_transcribe:
                    stop.set()
                progress("transcribe", 1000, 1000)
                return []
        job = {"sessionId": "fixture", "leaseId": "lease", "title": "테스트",
               "recordings": [{"id": "first"}, {"id": "second"}]}
        try:
            process_job(Api(), Engine(), job, stop, lambda *_: None, events.append)
        except (WorkerError, Cancelled):
            if not (fail_save or cancel_during_transcribe):
                raise
        return events, requests

    def test_two_files_have_separate_stage_percent_and_complete_only_after_saving(self):
        events, requests = self.run_job()
        self.assertEqual([e.stage for e in events], [
            "download", "transcribe", "transcribe", "save",
            "download", "transcribe", "transcribe", "save", "finish", "complete"])
        self.assertEqual([e.file_index for e in events if e.stage == "download"], [1, 2])
        self.assertEqual(requests[-1], "jobs/finish")
        self.assertTrue(all(e.file_count == 2 and e.started_at == events[0].started_at for e in events))

    def test_save_failure_never_announces_complete(self):
        events, _ = self.run_job(fail_save=True)
        self.assertNotIn("complete", [e.stage for e in events])

    def test_cancel_at_speech_boundary_prevents_later_progress_and_submission(self):
        events, requests = self.run_job(cancel_during_transcribe=True)
        self.assertNotIn("jobs/recording", requests)
        self.assertNotIn("jobs/finish", requests)
        self.assertEqual(events[-1].percent, 0)


if __name__ == "__main__":
    unittest.main()
