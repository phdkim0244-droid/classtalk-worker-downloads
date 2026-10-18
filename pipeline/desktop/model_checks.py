"""Opt-in synthetic regression checks; expected text never enters ASR options."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def check_model_regressions(engine, manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        raise ValueError("Invalid synthetic regression manifest")
    results = []
    for case in manifest["cases"]:
        audio = manifest_path.parent / case["audio"]
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        if digest != case["sha256"]:
            raise ValueError("Synthetic regression audio checksum differs")
        # Only audio and the ordinary single-speaker channel contract enter ASR.
        # The reference regex is an assertion on the result, never a prompt,
        # initial text, hotword list, correction rule, or decoding preference.
        segments = engine.transcribe(audio, {"groupNo": 1, "students": 1},
                                     {"sessionType": "class", "studentsPerGroup": 1})
        text = " ".join(segment["text"] for segment in segments)
        if not re.search(case["expectedPattern"], text, flags=re.IGNORECASE):
            raise AssertionError(f"Synthetic audio regression failed: {case['id']}")
        results.append({"id": case["id"], "audioSha256": digest,
                        "ok": True, "segmentCount": len(segments),
                        "transcript": text})
    if not results:
        raise ValueError("Synthetic regression manifest has no cases")
    return results
