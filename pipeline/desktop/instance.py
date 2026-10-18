"""OS-owned per-user lock: released automatically after exit or a process crash."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


class InstanceLock:
    def __init__(self, directory: Path, origin: str):
        key = hashlib.sha256(origin.encode("utf-8")).hexdigest()[:20]
        self.path = directory / f"instance-{key}.lock"
        self.file = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        stream = os.fdopen(descriptor, "r+b")
        try:
            if os.fstat(descriptor).st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):
            stream.close()
            return False
        self.file = stream
        return True

    def release(self) -> None:
        if self.file is None:
            return
        try:
            self.file.seek(0)
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        finally:
            self.file.close()
            self.file = None
