"""Background worker lifecycle; each claimed job has an independently renewed lease."""
from __future__ import annotations

import socket
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

from desktop.client import Api, Cancelled, LeaseLost, Unauthorized, WorkerError
from desktop.progress import Progress, ProgressEstimator


class Heartbeat:
    def __init__(self, api: Api, job: dict, cancel: threading.Event, *, interval: float = 20):
        self.api = api
        self.body = {"sessionId": job["sessionId"], "leaseId": job["leaseId"]}
        self.cancel = cancel
        self.interval = interval
        self.stopped = threading.Event()
        self.lost = threading.Event()
        self.error: WorkerError | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stopped.wait(self.interval):
            if self.cancel.is_set():
                return
            try:
                self.api.post("jobs/heartbeat", self.body)
            except WorkerError as exc:
                self.error = exc
                self.lost.set()
                return

    def check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled("작업이 중지되었습니다.")
        if self.lost.is_set():
            raise self.error or LeaseLost("작업 연결이 끊겼습니다.")

    def is_set(self) -> bool:
        """Cancellation interface also used between network retries."""
        return self.cancel.is_set() or self.lost.is_set()

    def wait(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self.cancel.wait(min(remaining, 0.1))
        return True

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join(timeout=36)


def process_job(api: Api, engine, job: dict, cancel: threading.Event, report, progress=None) -> None:
    body = {"sessionId": job["sessionId"], "leaseId": job["leaseId"]}
    started_at = time.monotonic()
    estimator = ProgressEstimator()
    with Heartbeat(api, job, cancel) as lease:
        try:
            recordings = job["recordings"]
            index = 0

            def update(stage, done=None, total=None):
                # Successful finish has already released the lease server-side.
                # A simultaneous heartbeat conflict must not undo that success.
                if stage != "complete":
                    lease.check()
                if progress:
                    updated, remaining, speed = estimator.update(stage, done, total, file_index=index)
                    progress(Progress(stage, done, total, job["title"], index,
                                      len(recordings), started_at, updated, remaining, speed,
                                      getattr(engine, "description", "")))

            for index, recording in enumerate(recordings, 1):
                lease.check()
                report(f"전사 중 · {job['title']} · {index}/{len(recordings)} 파일")
                update("download")
                # TemporaryDirectory closes files before PyAV opens them (Windows safe).
                with tempfile.TemporaryDirectory(prefix="classtalk-audio-") as temp:
                    audio = Path(temp) / "recording.audio"
                    # Earlier files may take hours on CPU. Sign immediately before
                    # each download, under the current lease, instead of reusing
                    # URLs returned when the entire lesson was first claimed.
                    source = api.post("jobs/audio", {**body, "recordingId": recording["id"]},
                                      retry=True, cancel=lease)
                    lease.check()
                    api.download(source["downloadUrl"], audio, lease)
                    lease.check()
                    segments = engine.transcribe(audio, recording, job, progress=update)
                    lease.check()
                    update("save")
                    # Server atomically replaces this recording's segments under the lease.
                    api.post("jobs/recording", {**body, "recordingId": recording["id"],
                             "segments": segments}, retry=True, cancel=lease)
            lease.check()
            update("finish")
            api.post("jobs/finish", body, retry=True, cancel=lease)
            report(f"완료 · {job['title']} · 웹에서 결과를 확인하세요.")
            update("complete")
        except (Cancelled, LeaseLost, Unauthorized):
            lease.check()
            raise
        except Exception as exc:
            lease.check()
            # A generic message avoids transmitting local paths, audio URLs or secrets.
            try:
                api.post("jobs/finish", {**body, "error": "음성 전사에 실패했습니다. 워커 상태를 확인하고 다시 요청해 주세요."},
                         retry=True, cancel=lease)
            except WorkerError:
                pass
            if isinstance(exc, WorkerError):
                raise
            raise WorkerError("음성 처리에 실패했습니다. 파일 형식과 여유 공간을 확인하고 다시 요청해 주세요.") from exc


class Controller:
    def __init__(self, api: Api, store, engine, emit):
        self.api, self.store, self.engine, self.emit = api, store, engine, emit
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.cancel.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _pair(self) -> str:
        pair = self.api.post("pair", {"deviceName": socket.gethostname()[:80]}, auth=False)
        url = self.api.verification_url(pair["verificationUrl"])
        self.emit("pair", {"code": pair["userCode"], "url": url})
        webbrowser.open(url)
        deadline = time.monotonic() + min(int(pair.get("expiresIn", 600)), 600)
        interval = max(5, min(int(pair.get("interval", 5)), 30))
        while time.monotonic() < deadline:
            if self.cancel.wait(interval):
                raise Cancelled("계정 연결이 취소되었습니다.")
            result = self.api.post("pair/status", {"deviceCode": pair["deviceCode"]}, auth=False)
            if result.get("status") == "approved":
                token = result.get("workerToken")
                self.store.save(token)
                return token
            if result.get("status") == "expired":
                break
        raise WorkerError("계정 연결 시간이 만료되었습니다. 다시 연결해 주세요.")

    def _run(self) -> None:
        try:
            token = self.store.load()
            if not token:
                token = self._pair()
            self.api.token = token
            self.emit("connected", None)
            self.emit("status", "전사 모델 준비 중… 처음에는 약 3.1GB를 다운로드합니다. 잠시 기다려 주세요.")
            prepared_at = time.monotonic()
            self.engine.prepare(lambda stage: self.emit("progress", Progress(stage, started_at=prepared_at)))
            while not self.cancel.is_set():
                self.emit("status", f"대기 중 · {self.engine.description} · 연결한 계정의 전사 요청만 확인합니다.")
                try:
                    # Claims are not blindly retried: an unknown result might already hold a lease.
                    result = self.api.post("jobs/claim", {}, cancel=self.cancel)
                    job = result.get("job")
                    if job:
                        process_job(self.api, self.engine, job, self.cancel,
                                    lambda message: self.emit("status", message),
                                    lambda value: self.emit("progress", value))
                        if self.cancel.wait(2):
                            break
                    elif self.cancel.wait(5):
                        break
                except (LeaseLost, WorkerError) as exc:
                    if isinstance(exc, (Cancelled, Unauthorized)):
                        raise
                    self.emit("status", str(exc))
                    if self.cancel.wait(15):
                        break
        except Unauthorized as exc:
            self.api.token = None
            try:
                self.store.clear()
            except WorkerError:
                pass
            self.emit("error", str(exc))
        except Cancelled:
            self.emit("status", "중지됨 · 다시 시작하면 남은 요청을 확인합니다.")
        except WorkerError as exc:
            self.emit("error", str(exc))
        except Exception:
            self.emit("error", "워커를 시작하지 못했습니다. 인터넷·저장 공간을 확인하세요. GPU 준비 실패 시 ‘CPU 호환’으로 다시 시작할 수 있습니다.")
        finally:
            self.emit("stopped", None)

    def stop(self) -> None:
        self.cancel.set()

    def disconnect(self) -> None:
        self.stop()
        # The GUI only exposes disconnect once the processing thread has stopped.
        if self.thread and self.thread.is_alive():
            raise WorkerError("현재 작업을 중지하는 중입니다. 잠시 후 연결을 해제해 주세요.")
        token = self.api.token or self.store.load()
        if token:
            self.api.token = token
            try:
                self.api.post("disconnect", {}, retry=True)
            except Unauthorized:
                pass
        self.store.clear()
        self.api.token = None
        self.emit("disconnected", None)
