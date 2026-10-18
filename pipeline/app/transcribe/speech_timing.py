"""Speech-bounded word groups on the original recording timeline.

Whisper segment boundaries may include many seconds of silence after VAD audio
is concatenated. They are not a safe source for participation durations or
speaker embeddings. Keep VAD turns and align words to those acoustic regions.
"""
from __future__ import annotations

from dataclasses import dataclass

from .base import Segment

SR = 16000
VAD_PARAMETERS = {
    "min_speech_duration_ms": 100,
    "min_silence_duration_ms": 300,
    "speech_pad_ms": 80,
    "max_speech_duration_s": 15,
}


@dataclass
class SpeechTurn:
    start_ms: int
    end_ms: int


def detect_speech(wav) -> list[SpeechTurn]:
    from faster_whisper.vad import get_speech_timestamps, VadOptions

    if len(wav) == 0:
        return []
    chunks = get_speech_timestamps(wav, VadOptions(**VAD_PARAMETERS))
    return [SpeechTurn(round(c["start"] * 1000 / SR), round(c["end"] * 1000 / SR))
            for c in chunks if c["end"] > c["start"]]


def _closest_turn(start_ms: int, end_ms: int, turns: list[SpeechTurn]) -> int:
    """Maximum acoustic overlap, then closest word midpoint; deterministic ties."""
    midpoint = (start_ms + end_ms) / 2
    return max(range(len(turns)), key=lambda i: (
        max(0, min(end_ms, turns[i].end_ms) - max(start_ms, turns[i].start_ms)),
        -abs(midpoint - (turns[i].start_ms + turns[i].end_ms) / 2),
    ))


def align_words(whisper_segments, turns: list[SpeechTurn]) -> tuple[list[Segment], list[int]]:
    """Split decoder sentences at acoustic turns without duplicating their text.

    Words without a positive overlap with detected speech cannot establish
    timing and are omitted rather than being assigned an invented whole turn.
    """
    if not turns:
        return [], []
    groups: list[tuple[int, int, int, str]] = []
    for sentence in whisper_segments:
        words = list(getattr(sentence, "words", None) or [])
        pieces = words or [sentence]
        for word in pieces:
            text = getattr(word, "word", None) if words else getattr(word, "text", "")
            if not text or not text.strip():
                continue
            start = round(float(word.start) * 1000)
            end = max(start, round(float(word.end) * 1000))
            turn_id = _closest_turn(start, end, turns)
            turn = turns[turn_id]
            # Degenerate/outside timestamps cannot justify any speech duration.
            clipped_start = max(start, turn.start_ms)
            clipped_end = min(end, turn.end_ms)
            if clipped_end <= clipped_start:
                continue
            if groups and groups[-1][0] == turn_id:
                prior = groups[-1]
                groups[-1] = (turn_id, min(prior[1], clipped_start),
                              max(prior[2], clipped_end), prior[3] + text)
            else:
                groups.append((turn_id, clipped_start, clipped_end, text))
    out, ids = [], []
    for turn_id, start, end, text in groups:
        out.append(Segment("SPEAKER_00", start, end, text.strip()))
        ids.append(turn_id)
    return out, ids


def speech_windows(turns: list[SpeechTurn], max_seconds: float):
    """Group complete turns on the original timeline; never concatenate silence away."""
    start = 0
    for end in range(1, len(turns) + 1):
        if end == len(turns) or turns[end].end_ms - turns[start].start_ms > max_seconds * 1000:
            yield start, turns[start:end]
            start = end


def transcribe_speech(model, wav, *, language: str, beam_size: int = 5, progress=None,
                      window_seconds: float = 0):
    """Decode bounded original audio, then restore each acoustic turn's clock.

    The compatibility path decodes one turn at a time. Optimized windows group
    complete turns but retain the original intervening silence, avoiding the
    timestamp shifts caused by concatenating VAD crops. No script or previous
    window text is supplied to the decoder.
    """
    progress = progress or (lambda *_: None)
    progress("detect", None, None)
    turns = detect_speech(wav)
    total_ms = sum(turn.end_ms - turn.start_ms for turn in turns)
    completed_ms = 0
    progress("transcribe", 0, total_ms)
    if not turns:
        return [], [], []
    segments, turn_ids = [], []
    windows = speech_windows(turns, window_seconds) if window_seconds > 0 else (
        (i, [turn]) for i, turn in enumerate(turns))
    for first_turn, window in windows:
        start_ms, end_ms = window[0].start_ms, window[-1].end_ms
        start_sample = round(start_ms * SR / 1000)
        end_sample = round(end_ms * SR / 1000)
        chunk = wav[start_sample:end_sample]
        decoded, _ = model.transcribe(
            chunk, language=language, vad_filter=False,
            word_timestamps=True, beam_size=beam_size,
            condition_on_previous_text=False,
            # The default retry temperatures sample different guesses on weak
            # speech. Keep beam decoding reproducible instead of adding random
            # alternatives to numeric or short utterances. This is not a
            # confidence guarantee; the original audio remains the evidence.
            temperature=0,
        )
        local, ids = align_words(decoded, [SpeechTurn(turn.start_ms - start_ms, turn.end_ms - start_ms)
                                          for turn in window])
        for segment, local_id in zip(local, ids):
            turn_id = first_turn + local_id
            turn = turns[turn_id]
            # Words establish that recognized text overlaps speech. Acoustic
            # boundaries (including 80ms safety padding) determine duration:
            # decoder word alignment often clips a trailing Korean syllable.
            # This remains an estimate, not forced alignment or exact phonation.
            segment.start_ms = turn.start_ms
            segment.end_ms = turn.end_ms
            segments.append(segment)
            turn_ids.append(turn_id)
        # Only advance after the lazy decoder has actually consumed this turn.
        completed_ms += sum(turn.end_ms - turn.start_ms for turn in window)
        progress("transcribe", completed_ms, total_ms)
    return segments, turn_ids, turns
