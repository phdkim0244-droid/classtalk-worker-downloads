import unittest
from unittest.mock import Mock
import numpy as np
from app.transcribe.base import Segment, WordSpan
from app.transcribe.turn_refinement import VoiceReferences, windows, split_on_voice_changes, acoustic_boundaries, split_at_boundaries
from desktop.progress import Progress

class TurnRefinementTests(unittest.TestCase):
    def test_voice_windows_stay_within_turn_and_do_not_use_silence(self):
        self.assertEqual(windows(1000, 3000), [])
        spans = windows(1000, 4600)
        self.assertEqual(spans[0], (1000,2600))
        self.assertEqual(spans[-1], (3000,4600))
        self.assertTrue(all(b-a==1600 and a>=1000 and b<=4600 for a,b in spans))

    def test_two_sustained_voices_split_words_once_and_conserve_time(self):
        words=[WordSpan(i*1000,i*1000+800,t) for i,t in enumerate([' 무슨',' 뜻이야?',' 바다예요.',' 네.'])]
        original=Segment('SPEAKER_00',0,4000,''.join(w.text for w in words).strip(),words)
        result=split_on_voice_changes(original,[(0,1600,0),(400,2000,0),(2000,3600,1),(2400,4000,1)])
        self.assertEqual([s.speaker for s in result],['SPEAKER_00','SPEAKER_01'])
        self.assertEqual(' '.join(s.text for s in result),original.text)
        self.assertEqual(sum(s.end_ms-s.start_ms for s in result),4000)
        self.assertEqual(result[0].end_ms,result[1].start_ms)

    def test_overlap_boundary_is_unknown_not_majority_identity(self):
        words=[WordSpan(i*500,i*500+400,' 음성') for i in range(8)]
        s=Segment('SPEAKER_00',0,4000,''.join(w.text for w in words).strip(),words)
        result=split_on_voice_changes(s,[(0,2000,0),(300,2300,0),(1700,3700,1),(2000,4000,1)])
        self.assertIn('SPEAKER_UNKNOWN',[r.speaker for r in result])
        self.assertEqual(sum(len(r.words) for r in result),8)

    def test_one_spurious_window_does_not_split_stable_speaker(self):
        s=Segment('SPEAKER_00',0,5000,'질문과 답을 설명하는 한 사람')
        self.assertEqual(split_on_voice_changes(s,[(0,1600,0),(400,2000,0),(800,2400,1),(1200,2800,0)]),[s])

    def test_detected_mixture_without_word_timing_is_not_attributed_to_one_voice(self):
        s=Segment('SPEAKER_00',0,4000,'질문 답변')
        r=split_on_voice_changes(s,[(0,1600,0),(400,2000,0),(2000,3600,1),(2400,4000,1)])
        self.assertEqual(r[0].speaker,'SPEAKER_UNKNOWN')
        self.assertEqual(r[0].text,s.text)

    def test_two_models_and_multiple_independent_references_are_required(self):
        p={0:np.array([1.,0.]),1:np.array([1.,0.]),2:np.array([0.,1.]),3:np.array([0.,1.])}
        refs=VoiceReferences(p,p,{0:0,1:0,2:1,3:1})
        self.assertEqual(refs.match_short(p[0],p[0]),0)
        self.assertIsNone(refs.match_short(p[0],p[2]))
        self.assertIsNone(refs.match(p[0],p[0],exclude=0))
        self.assertIsNone(refs.match_short(np.array([float('nan'),0]),p[0]))
        self.assertIsNone(VoiceReferences(p,p,{0:0,1:0,2:1}).match_short(p[0],p[0]))

    def test_acoustic_change_requires_both_models_and_stable_sides(self):
        x=np.array([1.,0.]);y=np.array([0.,1.])
        # Touching windows at 2000 with independent stable neighbours.
        items=[(0,1600,x,x),(400,2000,x,x),(800,2400,x,x),(1200,2800,x,x),
               (1600,3200,y,y),(2000,3600,y,y),(2400,4000,y,y)]
        self.assertIn(2000,acoustic_boundaries(items))
        self.assertEqual(acoustic_boundaries([(a,b,x,s) for a,b,p,s in items]),[])
        self.assertEqual(acoustic_boundaries([(a,b,x,x) for a,b,p,s in items]),[])

    def test_acoustic_boundary_words_stay_unknown_and_text_time_are_preserved(self):
        words = [WordSpan(0, 1500, " 질문"), WordSpan(1500, 2400, " 마무리"),
                 WordSpan(2600, 2850, " 네"), WordSpan(3000, 5000, " 답변입니다")]
        source = Segment("SPEAKER_00", 0, 5000, "질문 마무리 네 답변입니다", words)
        refs = Mock()
        refs.match.side_effect = [0, 1]
        result = split_at_boundaries(source, [2800], refs, lambda a,b: (a,b), lambda a,b: (a,b), 7)
        self.assertEqual(sum(r.end_ms-r.start_ms for r in result), 5000)
        self.assertEqual(" ".join(r.text for r in result), source.text)
        self.assertEqual(next(r.speaker for r in result if any(w.text == " 네" for w in r.words)), "SPEAKER_UNKNOWN")
        self.assertTrue(all(c.kwargs["exclude"] == 7 for c in refs.match.call_args_list))
        self.assertEqual(result[0].speaker, "SPEAKER_00")

    def test_progress_has_stage_specific_percent_and_eta(self):
        p=Progress('refine_speakers',10,20,updated_at=10,remaining_seconds=40)
        self.assertIn('50%',p.heading)
        self.assertIn('10 / 20',p.detail)
        self.assertIn('화자 전환',p.remaining_text(10))

if __name__=='__main__':unittest.main()
