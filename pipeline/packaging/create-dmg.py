"""Bound transient hdiutil failures without replacing a complete DMG on failure."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
import time


def create_dmg(source: Path, destination: Path) -> None:
    # Stay on the destination filesystem so the final replacement is atomic.
    # TemporaryDirectory owns every file removed after a failed attempt.
    with tempfile.TemporaryDirectory(prefix="classtalk-dmg-", dir=destination.parent) as directory:
        image = Path(directory) / "package.dmg"
        for attempt in range(1, 4):
            result = subprocess.run([
                "hdiutil", "create", "-volname", "ClassTalk Worker",
                "-srcfolder", str(source), "-ov", "-format", "UDZO", str(image),
            ], check=False)
            if result.returncode == 0:
                image.replace(destination)
                return
            image.unlink(missing_ok=True)
            if attempt < 3:
                print(f"DMG creation attempt {attempt}/3 failed; retrying in 3 seconds.", flush=True)
                time.sleep(3)
    raise RuntimeError("DMG creation failed after 3 attempts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    create_dmg(args.source.resolve(), args.output.resolve())
