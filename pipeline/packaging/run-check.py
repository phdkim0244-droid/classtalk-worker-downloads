"""Bound packaged GUI checks so native errors cannot leave CI waiting on a dialog."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--audio")
    parser.add_argument("--regressions")
    args = parser.parse_args()
    report = Path(args.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    command = [str(Path(args.executable).resolve()), "--self-test-report", str(report)]
    command += (["--model-smoke-audio", str(Path(args.audio).resolve())]
                if args.audio else ["--self-test"])
    if args.regressions:
        if not args.audio:
            parser.error("--regressions requires --audio")
        command += ["--model-smoke-regressions", str(Path(args.regressions).resolve())]
    failure = None
    try:
        timeout = 900 if args.audio else 180
        deadline = time.monotonic() + timeout
        last_stage = None
        with subprocess.Popen(command) as process:
            while True:
                if report.exists():
                    try:
                        progress = json.loads(report.read_text(encoding="utf-8"))
                        stage = progress.get("stage")
                        if stage and stage != last_stage:
                            print(json.dumps({"stage": stage,
                                              "elapsedSeconds": progress.get("elapsedSeconds")}), flush=True)
                            last_stage = stage
                    except (ValueError, OSError):
                        pass  # The app may be writing its next checkpoint.
                try:
                    process.wait(timeout=min(1, max(0, deadline - time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        process.kill()
                        process.wait(timeout=15)
                        raise
            if process.returncode:
                failure = f"Packaged check exited with code {process.returncode}"
    except subprocess.TimeoutExpired:
        failure = "Packaged check timed out (model: 900 seconds; offline: 180 seconds)"
    except OSError as exc:
        failure = f"Cannot launch packaged application: {exc}"
    if report.exists():
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            payload = {"ok": False, "error": "Invalid packaged check report"}
    else:
        payload = {"ok": False, "error": failure or "Packaged application did not write a report"}
    if failure:
        payload.update(ok=False, processError=failure)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows runner consoles may use cp1252. Keep the saved report UTF-8,
    # while escaping its console copy so Korean diagnostics cannot fail CI.
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
