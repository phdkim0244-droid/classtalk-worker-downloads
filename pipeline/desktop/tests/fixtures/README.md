# Synthetic speech fixture

`synthetic-ko.wav` is 12 seconds of the synthetic Korean speech generated for
ClassTalk's functional check on 2026-10-18 using macOS Yuna text-to-speech.
It contains no microphone recording and no actual teacher/student voices.
Format: mono PCM16, 16 kHz. This fixture is for CI only and is not bundled in the installer.

Source text (the fixture contains the opening portion):

> 이 음성은 클래스톡 기능 점검을 위한 가상 수업 자료입니다. 실제 학생의 목소리가 아닙니다.
> 오늘은 물이 증발하는 현상을 살펴봅니다.

The opt-in `--model-smoke-audio` check downloads the public Whisper large-v3 model,
transcribes this file locally, and checks Korean text, speaker labels and
timestamps. It does not connect to ClassTalk or submit any content.

`synthetic-numeric-ko.wav` is a 5.496-second excerpt (56.644–62.140 seconds)
from the fixed `students-only.wav` classroom-quality fixture. It was generated
with macOS text-to-speech, contains no microphone audio, and has the same PCM
format as above. Original source SHA-256:
`c26c01db315f8e2b9ae2f09e0ba07ad61f2bf3a16bdb9c5d8198f7ba46d07a19`.
Excerpt SHA-256:
`1c0936f3e87926d923e6115fb12144ea47e4a80fc2dcfa31e42ad4501d97f6d6`.

The spoken amounts are two milliliters and one milliliter, in that order.
`synthetic-numeric-remaining-ko.wav` is the 9.560-second excerpt at
43.236–52.796 seconds from the same source, containing eight milliliters and
nine milliliters, in that order. Its SHA-256 is
`53949bb6234161de551e0a708c47781c25db284f7cd6f1f1bcdcbb34cc7a8ddd`.
Both excerpts are checked so a corrected subtraction cannot hide a new error
in the original measurements or their units (for example, mm instead of mL).
`model-regressions.json` stores the expected result separately from the audio.
The opt-in `--model-smoke-regressions` check verifies the audio checksum, runs
the ordinary decoder with only audio/channel settings, then asserts the result.
Expected words and numbers never become a prompt, hotwords, or correction rules.
Native Windows/macOS builds run this regression after the existing smoke test,
using the same already-loaded model. The fixture and manifest are CI inputs and
are not included in the user installer.
