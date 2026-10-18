import unittest
from pathlib import Path
from unittest.mock import Mock
from desktop.engine import Engine
from app.transcribe.base import Segment

class MixedTeacherTests(unittest.TestCase):
    def engine(self):
        e=Engine(Path('/unused'))
        e.transcriber=Mock()
        e.transcriber.transcribe.return_value=[Segment('SPEAKER_00',0,2000,'교사 목소리'),Segment('SPEAKER_01',2200,4000,'학생 목소리'),Segment('SPEAKER_UNKNOWN',4200,4400,'네')]
        return e
    def test_teacher_lesson_diarizes_without_imposing_group_student_count(self):
        e=self.engine();r=e.transcribe(Path('/unused.wav'),{'groupNo':None},{'sessionType':'lesson','studentsPerGroup':4})
        self.assertFalse(e.transcriber.transcribe.call_args.kwargs['is_teacher'])
        self.assertIsNone(e.transcriber.n_speakers)
        self.assertTrue(e.transcriber.verify_short_speakers)
        self.assertEqual(e.transcriber.threshold,0.30)
        self.assertEqual([x['speaker'] for x in r],['화자1','화자2','미상'])
    def test_group_recording_retains_fixed_student_count_and_student_labels(self):
        e=self.engine();r=e.transcribe(Path('/unused.wav'),{'groupNo':1,'students':3},{'sessionType':'lesson','studentsPerGroup':4})
        self.assertEqual(e.transcriber.n_speakers,3)
        self.assertFalse(e.transcriber.verify_short_speakers)
        self.assertEqual(e.transcriber.threshold,0.65)
        self.assertEqual([x['speaker'] for x in r],['학생1','학생2','미상'])
    def test_council_teacher_channel_preserves_existing_single_teacher_contract(self):
        e=self.engine();r=e.transcribe(Path('/unused.wav'),{'groupNo':None},{'sessionType':'council'})
        self.assertTrue(e.transcriber.transcribe.call_args.kwargs['is_teacher'])
        self.assertFalse(e.transcriber.verify_short_speakers)
        self.assertTrue(all(x['speaker']=='교사' for x in r))

    def test_explicit_mixed_count_is_independent_of_group_enrolment(self):
        e=self.engine();e.transcribe(Path('/unused.wav'),{'groupNo':None,'speakerCount':4},{'sessionType':'lesson','studentsPerGroup':30})
        self.assertEqual(e.transcriber.n_speakers,4)
        self.assertFalse(e.transcriber.transcribe.call_args.kwargs['is_teacher'])
    def test_invalid_mixed_count_fails_before_transcription(self):
        for count in [0,101,True,'4']:
            e=self.engine()
            with self.assertRaises(ValueError):
                e.transcribe(Path('/unused.wav'),{'groupNo':None,'speakerCount':count},{'sessionType':'lesson'})
            e.transcriber.transcribe.assert_not_called()
