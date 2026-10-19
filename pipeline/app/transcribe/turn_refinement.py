"""Conservative, local sub-turn voice matching on word-aligned audio.

VAD is a speech/silence detector, not a speaker-change detector. Keep ASR word
spans so two confident voices inside a continuous VAD turn need not share an
identity. No transcript semantics, teaching roles, neighbouring turns or recovered
short replies are used as speaker references. Similarities are not probabilities.
"""
from __future__ import annotations

import numpy as np
from .base import Segment

UNKNOWN = "SPEAKER_UNKNOWN"


class VoiceReferences:
    """Match two encoders against multiple independent original long turns.

    Local exemplars preserve variation that a single mean can erase. Requiring
    support from multiple examples avoids promoting one noisy nearest neighbour.
    A source turn is excluded when inspecting its own subwindows.
    """
    def __init__(self, primary, secondary, anchors):
        self.labels = sorted(set(anchors.values()))
        self.references = {}
        for label in self.labels:
            ids = [i for i, k in anchors.items() if k == label and i in primary and i in secondary]
            self.references[label] = (ids, np.array([primary[i] for i in ids]),
                                     np.array([secondary[i] for i in ids]))

    def match(self, first, second, *, exclude=None):
        if first is None or second is None or len(self.labels) < 2:
            return None
        scores = [[], []]
        for label in self.labels:
            ids, p, s = self.references[label]
            keep = [j for j, i in enumerate(ids) if i != exclude]
            if len(keep) < 2:
                return None  # Never omit a competing voice from the decision.
            for model, (matrix, vector) in enumerate(((p, first), (s, second))):
                values = np.sort(matrix[keep] @ vector)
                scores[model].append(float(np.mean(values[-min(3, len(values)):])))
        winners = []
        for values, floor, margin in zip(scores, (0.65, 0.55), (0.06, 0.10)):
            if not np.all(np.isfinite(values)):
                return None
            order = np.argsort(values)[::-1]
            if values[order[0]] < floor or values[order[0]] - values[order[1]] < margin:
                return None
            winners.append(int(order[0]))
        return self.labels[winners[0]] if winners[0] == winners[1] else None

    def match_short(self, first, second):
        if first is None or second is None or len(self.labels) < 2:
            return None
        primary_scores, secondary_scores = [], []
        for label in self.labels:
            ids, p, s = self.references[label]
            if len(ids) < 2:
                return None
            mean = p.mean(axis=0)
            norm = np.linalg.norm(mean)
            if not np.isfinite(norm) or norm <= 0:
                return None
            primary_scores.append(float(mean @ first / norm))
            nearest = np.sort(s @ second)
            secondary_scores.append(float(nearest[-min(3, len(nearest)):].mean()))
        if not np.all(np.isfinite(primary_scores + secondary_scores)):
            return None
        order = np.argsort(secondary_scores)[::-1]
        if (np.argmax(primary_scores) == order[0] and secondary_scores[order[0]] >= 0.55
                and secondary_scores[order[0]] - secondary_scores[order[1]] >= 0.10):
            return self.labels[order[0]]
        return None


def windows(start: int, end: int):
    """At most 1.6 s of original contiguous audio, never outside this VAD turn."""
    width, step = 1600, 400
    if end - start < 2400:
        return []
    starts = list(range(start, end - width + 1, step))
    if starts[-1] != end - width:
        starts.append(end - width)
    return [(a, a + width) for a in starts]


def split_on_voice_changes(segment, votes):
    """Split only with sustained independent voice evidence and usable word times.

    Ambiguous transition words remain unknown. Text is partitioned exactly once;
    no word is invented, repeated, discarded or assigned from punctuation alone.
    Without reliable word timing, retain text and flag a detected mixture unknown.
    """
    sustained = []
    for i, (a, b, label) in enumerate(votes):
        neighbours = votes[max(0, i - 1):i] + votes[i + 1:i + 2]
        if label is not None and any(k == label for _, _, k in neighbours):
            sustained.append((a, b, label))
    if len({k for _, _, k in sustained}) < 2:
        return [segment]
    words = segment.words
    if (not words or ''.join(w.text for w in words).strip() != segment.text.strip()
            or any(w.end_ms <= w.start_ms for w in words)
            or any(a.start_ms > b.start_ms for a, b in zip(words, words[1:]))):
        return [Segment(UNKNOWN, segment.start_ms, segment.end_ms, segment.text, words)]
    labels = []
    for word in words:
        midpoint = (word.start_ms + word.end_ms) / 2
        covering = [k for a, b, k in sustained if a <= midpoint <= b]
        # Contradictory windows indicate a boundary or overlap, not a vote to guess.
        labels.append(f"SPEAKER_{covering[0]:02d}" if covering and len(set(covering)) == 1 else UNKNOWN)
    if len(set(labels) - {UNKNOWN}) < 2:
        return [Segment(UNKNOWN, segment.start_ms, segment.end_ms, segment.text, words)]
    return _partition_words(segment, labels)


def _partition_words(segment, labels, forced_breaks=()):
    words = segment.words
    groups = []
    for i, (word, label) in enumerate(zip(words, labels)):
        if groups and groups[-1][0] == label and i not in forced_breaks:
            groups[-1][1].append(word)
        else:
            groups.append((label, [word]))
    # Midpoint boundaries conserve the parent's speech duration exactly, without
    # double-counting overlapping decoder words or filling gaps between VAD turns.
    boundaries = [segment.start_ms]
    for (_, left), (_, right) in zip(groups, groups[1:]):
        boundaries.append(max(boundaries[-1], min(segment.end_ms,
                              round((left[-1].end_ms + right[0].start_ms) / 2))))
    boundaries.append(segment.end_ms)
    if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        return [Segment(UNKNOWN, segment.start_ms, segment.end_ms, segment.text, words)]
    return [Segment(label, boundaries[i], boundaries[i + 1], ''.join(w.text for w in group).strip(), group)
            for i, (label, group) in enumerate(groups)]


def refine_turns(segments, turn_ids, turns, primary, secondary, anchors, speakers,
                 embed_primary, embed_secondary, progress):
    refs = VoiceReferences(primary, secondary, anchors)
    # Only original long turns establish identities. Newly recovered replies do
    # not feed back into reference training.
    for i in primary:
        if i not in speakers and i not in anchors:
            label = refs.match_short(primary[i], secondary.get(i))
            if label is not None:
                speakers[i] = f"SPEAKER_{label:02d}"
    plans = {i: windows(turns[i].start_ms, turns[i].end_ms) for i in set(turn_ids)}
    total = sum(map(len, plans.values())) + len(segments)
    done = 0
    votes = {}
    acoustic = {}
    progress("refine_speakers", done, total)
    for i, spans in plans.items():
        votes[i] = []
        acoustic[i] = []
        for a, b in spans:
            first, second = embed_primary(a, b), embed_secondary(a, b)
            label = refs.match(first, second, exclude=i)
            acoustic[i].append((a, b, first, second))
            votes[i].append((a, b, label))
            done += 1
            progress("refine_speakers", done, total)
    result = []
    for s, i in zip(segments, turn_ids):
        s.speaker = speakers.get(i, UNKNOWN)
        cuts = acoustic_boundaries(acoustic[i])
        if cuts:
            result.extend(split_at_boundaries(s, cuts, refs, embed_primary, embed_secondary, i))
        else:
            result.extend(split_on_voice_changes(s, votes[i]))
        done += 1
        progress("refine_speakers", done, total)
    return result


def acoustic_boundaries(windows_with_vectors):
    """Two models must detect contrast plus stability on both sides of a cut.

    A change in pitch/word/noise alone is not sufficient. Overlap is left
    unresolved; this is conservative single-active-speaker segmentation.
    """
    candidates = []
    def similar(x, y, floors):
        return all(a is not None and b is not None and np.isfinite(a @ b) and a @ b >= floor
                   for a, b, floor in zip(x[2:], y[2:], floors))
    for i, left in enumerate(windows_with_vectors):
        if i == 0:
            continue
        right_index = next((j for j in range(i + 1, len(windows_with_vectors) - 1)
                            if windows_with_vectors[j][0] >= left[1]), None)
        if right_index is None:
            continue
        right = windows_with_vectors[right_index]
        if right[0] != left[1] or any(v is None for v in (*left[2:], *right[2:])):
            continue
        p, c = float(left[2] @ right[2]), float(left[3] @ right[3])
        if (np.isfinite(p) and np.isfinite(c) and p < 0.65 and c < 0.45
                and similar(left, windows_with_vectors[i - 1], (0.75, 0.65))
                and similar(right, windows_with_vectors[right_index + 1], (0.75, 0.65))):
            candidates.append((2 - p - c, left[1]))
    selected = []
    for _, cut in sorted(candidates, reverse=True):
        if all(abs(cut - other) >= 1600 for other in selected):
            selected.append(cut)
    return sorted(selected)


def split_at_boundaries(segment, cuts, refs, embed_primary, embed_secondary, turn_id):
    words = segment.words
    if not words or ''.join(w.text for w in words).strip() != segment.text.strip():
        return [Segment(UNKNOWN, segment.start_ms, segment.end_ms, segment.text, words)]
    if any(w.end_ms <= w.start_ms for w in words) or any(a.start_ms > b.start_ms for a, b in zip(words, words[1:])):
        return [Segment(UNKNOWN, segment.start_ms, segment.end_ms, segment.text, words)]
    boundaries = []
    for cut in cuts:
        choices = [(abs((words[i - 1].end_ms + words[i].start_ms) / 2 - cut), i)
                   for i in range(1, len(words))]
        if choices:
            distance, i = min(choices)
            if distance <= 600:
                boundaries.append(i)
    boundaries = sorted(set(boundaries))
    if not boundaries:
        return [Segment(UNKNOWN, segment.start_ms, segment.end_ms, segment.text, words)]
    indices = [0] + boundaries + [len(words)]
    times = [segment.start_ms] + [round((words[i - 1].end_ms + words[i].start_ms) / 2) for i in boundaries] + [segment.end_ms]
    if any(b <= a for a, b in zip(times, times[1:])):
        return [Segment(UNKNOWN, segment.start_ms, segment.end_ms, segment.text, words)]
    # The acoustic detector has 400 ms steps and word snapping allows 600 ms.
    # Words near a candidate cut cannot establish either side's identity. Keep
    # this uncertainty explicit instead of assigning a short reply to its neighbour.
    labels = []
    uncertainty_ms = 600
    for j, (a, b) in enumerate(zip(indices, indices[1:])):
        start, end = times[j:j + 2]
        stable_start = max(start, max((c + uncertainty_ms for c in cuts if c <= start + 600), default=start))
        stable_end = min(end, min((c - uncertainty_ms for c in cuts if c >= end - 600), default=end))
        label = (refs.match(embed_primary(stable_start, stable_end), embed_secondary(stable_start, stable_end), exclude=turn_id)
                 if stable_end - stable_start >= 400 else None)
        for word in words[a:b]:
            near_cut = any(word.start_ms < cut + uncertainty_ms and word.end_ms > cut - uncertainty_ms for cut in cuts)
            labels.append(f"SPEAKER_{label:02d}" if label is not None and not near_cut else UNKNOWN)
    return _partition_words(segment, labels, boundaries)
