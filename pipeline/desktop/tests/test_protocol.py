"""No live accounts, keys or model downloads are used in protocol tests."""
from __future__ import annotations

import json
import sys
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from desktop.client import Api, Cancelled, LeaseLost, Unauthorized, WorkerError, checked_download, checked_origin
from desktop.controller import Controller, Heartbeat, process_job
from desktop.instance import InstanceLock


class Handler(BaseHTTPRequestHandler):
    events = []
    results = {}
    uncertain_result = True

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.events.append((self.path, self.headers.get("Authorization"), body))
        status, result = 200, {}
        if self.path.endswith("pair"):
            result = {"deviceCode": "opaque-device-code", "userCode": "ABCD-EFGH", "expiresIn": 600}
        elif self.path.endswith("jobs/audio"):
            result = {"downloadUrl": f"https://abc.supabase.co/storage/v1/object/sign/recordings/{body['recordingId']}?token=fresh"}
        elif self.path.endswith("jobs/recording"):
            # Simulate a response lost after the server atomically saved results.
            Handler.results[body["recordingId"]] = body["segments"]
            if Handler.uncertain_result:
                Handler.uncertain_result = False
                status = 503
        elif self.path.endswith("jobs/heartbeat"):
            status = 409
        elif self.path.endswith("disconnect"):
            status = 401
        elif self.path.endswith("jobs/claim"):
            status = 302
        payload = json.dumps(result).encode()
        self.send_response(status)
        if status == 302:
            self.send_header("Location", "https://attacker.example/capture")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        Handler.events = []
        Handler.results = {}
        Handler.uncertain_result = True
        self.api = Api(self.origin, allow_local=True)
        self.api.token = "test-token-not-a-real-credential"

    def tearDown(self):
        self.api.close()

    def test_pair_never_sends_existing_credential(self):
        self.api.post("pair", {"deviceName": "Test"}, auth=False)
        self.assertIsNone(Handler.events[0][1])

    def test_no_credentials_to_redirect(self):
        with self.assertRaises(WorkerError):
            self.api.post("jobs/claim", {})
        self.assertEqual(len(Handler.events), 1)

    def test_atomic_recording_retry_does_not_duplicate_segments(self):
        body = {"recordingId": "one", "sessionId": "own", "leaseId": "lease",
                "segments": [{"speaker": "학생1", "startMs": 0, "endMs": 10, "text": "테스트"}]}
        self.api.post("jobs/recording", body, retry=True)
        self.assertEqual(len(Handler.events), 2)
        self.assertEqual(Handler.results["one"], body["segments"])
        self.assertEqual(Handler.events[0][2], Handler.events[1][2])

    def test_revocation_and_lost_lease_are_not_retried(self):
        with self.assertRaises(Unauthorized):
            self.api.post("disconnect", {}, retry=True)
        with self.assertRaises(LeaseLost):
            self.api.post("jobs/heartbeat", {}, retry=True)
        self.assertEqual(len(Handler.events), 2)

    def test_origin_and_download_reject_untrusted_destinations(self):
        for origin in ("http://remote.example", "https://remote.example", "http://localhost@evil.example",
                       "http://127.0.0.1:3000/path", "https://classtalk-web-three.vercel.app@evil.example"):
            with self.assertRaises(ValueError):
                checked_origin(origin, allow_local=True)
        with self.assertRaises(ValueError):
            checked_origin(self.origin)
        for url in ("http://abc.supabase.co/storage/v1/object/sign/audio/a", "https://abc.supabase.co.evil/a",
                    "https://user:pass@abc.supabase.co/storage/v1/object/sign/a", "https://abc.supabase.co:8080/storage/v1/object/sign/a"):
            with self.assertRaises(WorkerError):
                checked_download(url)
        self.assertEqual(checked_origin(self.origin, allow_local=True), self.origin)
        with self.assertRaises(WorkerError):
            self.api.verification_url("https://other.example/worker/connect")

    def test_custom_production_domain_preserves_local_identity_only(self):
        from desktop import PRODUCTION_ORIGIN, storage_identity
        self.assertEqual(PRODUCTION_ORIGIN, "https://talk.ydssam.kr")
        api = Api()
        self.assertEqual(api.origin, PRODUCTION_ORIGIN)
        self.assertEqual(api.verification_url(PRODUCTION_ORIGIN + "/worker/connect?code=ABC123"),
                         PRODUCTION_ORIGIN + "/worker/connect?code=ABC123")
        self.assertEqual(storage_identity(PRODUCTION_ORIGIN), "https://classtalk-web-three.vercel.app")
        self.assertEqual(storage_identity(self.origin), self.origin)
        for origin in ["https://classtalk-web-three.vercel.app", "https://talk.ydssam.kr.evil.example",
                       "https://talk.ydssam.kr@evil.example", "http://talk.ydssam.kr"]:
            with self.assertRaises(ValueError):
                checked_origin(origin)
        with tempfile.TemporaryDirectory() as directory:
            old = InstanceLock(Path(directory), "https://classtalk-web-three.vercel.app")
            new = InstanceLock(Path(directory), storage_identity(PRODUCTION_ORIGIN))
            self.assertTrue(old.acquire())
            try:
                self.assertFalse(new.acquire())
            finally:
                new.release()
                old.release()

    def test_cancellation_prevents_submission(self):
        stop = threading.Event()
        stop.set()
        with self.assertRaises(Cancelled):
            self.api.post("jobs/recording", {}, retry=True, cancel=stop)
        self.assertEqual(Handler.events, [])

    def test_heartbeat_failure_prevents_result_submission(self):
        job = {"sessionId": "own", "leaseId": "lease", "title": "Test",
               "recordings": [{"id": "recording", "downloadUrl": "unused", "groupNo": 1}]}
        actual_heartbeat = Heartbeat
        class Engine:
            def transcribe(self, *_, **kwargs):
                time.sleep(0.1)
                return [{"speaker": "학생1", "startMs": 0, "endMs": 1, "text": "테스트"}]
        with patch.object(self.api, "download"), patch("desktop.controller.Heartbeat",
                side_effect=lambda *args: actual_heartbeat(*args, interval=0.01)):
            with self.assertRaises(LeaseLost):
                process_job(self.api, Engine(), job, threading.Event(), lambda *_: None)
        self.assertTrue(Handler.events)
        self.assertTrue(all(event[0].endswith(("jobs/audio", "jobs/heartbeat")) for event in Handler.events))

    def test_later_recordings_get_fresh_urls_only_when_ready_to_download(self):
        Handler.uncertain_result = False
        job = {"sessionId": "own", "leaseId": "lease", "title": "Long lesson",
               "recordings": [{"id": name, "downloadUrl": "expired-initial-claim-url", "groupNo": 1}
                              for name in ("first", "second")]}
        class Engine:
            def transcribe(self, *_, **kwargs):
                return [{"speaker": "학생1", "startMs": 0, "endMs": 1, "text": "테스트"}]
        with patch.object(self.api, "download") as download:
            process_job(self.api, Engine(), job, threading.Event(), lambda *_: None)
        self.assertEqual([row[0] for row in Handler.events], [
            "/api/worker/jobs/audio", "/api/worker/jobs/recording",
            "/api/worker/jobs/audio", "/api/worker/jobs/recording", "/api/worker/jobs/finish"])
        self.assertEqual([call.args[0] for call in download.call_args_list], [
            "https://abc.supabase.co/storage/v1/object/sign/recordings/first?token=fresh",
            "https://abc.supabase.co/storage/v1/object/sign/recordings/second?token=fresh"])
        audio_requests = [row for row in Handler.events if row[0].endswith("jobs/audio")]
        self.assertTrue(all(row[2]["sessionId"] == "own" and row[2]["leaseId"] == "lease"
                            and row[1] == f"Bearer {self.api.token}" for row in audio_requests))

    def test_audio_download_uses_a_separate_session_without_auth(self):
        class Response:
            status_code = 200
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def iter_content(self, _): return iter([b"audio"])
        class AudioSession:
            def __init__(self): self.trust_env = True
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def get(self, url, **kwargs):
                self.arguments = kwargs
                return Response()
        audio = AudioSession()
        with tempfile.TemporaryDirectory() as tmp, patch("desktop.client.requests.Session", return_value=audio):
            dest = Path(tmp) / "audio"
            self.api.download("https://abc.supabase.co/storage/v1/object/sign/recordings/test?token=audio-only", dest, threading.Event())
            self.assertEqual(dest.read_bytes(), b"audio")
        self.assertFalse(audio.trust_env)
        self.assertFalse(audio.arguments["allow_redirects"])
        self.assertNotIn("headers", audio.arguments)
        self.assertNotIn("auth", audio.arguments)

    def test_heartbeat_failure_also_cancels_an_in_progress_result_retry(self):
        job = {"sessionId": "own", "leaseId": "lease", "title": "Test",
               "recordings": [{"id": "recording", "downloadUrl": "unused", "groupNo": 1}]}
        actual_heartbeat = Heartbeat
        class Engine:
            def transcribe(self, *_, **kwargs):
                return [{"speaker": "학생1", "startMs": 0, "endMs": 1, "text": "테스트"}]
        with patch.object(self.api, "download"), patch("desktop.controller.Heartbeat",
                side_effect=lambda *args: actual_heartbeat(*args, interval=0.05)):
            with self.assertRaises(LeaseLost):
                process_job(self.api, Engine(), job, threading.Event(), lambda *_: None)
        paths = [event[0] for event in Handler.events]
        self.assertEqual(paths.count("/api/worker/jobs/recording"), 1)
        self.assertEqual(paths.count("/api/worker/jobs/finish"), 0)

    def test_desktop_does_not_import_operator_configuration(self):
        import desktop.engine
        import desktop.secure_store
        self.assertNotIn("app.config", sys.modules)
        self.assertNotIn("app.supabase_io", sys.modules)
        self.assertNotIn("app.pipeline_core", sys.modules)

    def test_second_process_cannot_use_same_device_until_first_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            first = InstanceLock(Path(directory), self.origin)
            self.assertTrue(first.acquire())
            command = [sys.executable, "-c", (
                "from desktop.instance import InstanceLock; from pathlib import Path; import sys; "
                "lock=InstanceLock(Path(sys.argv[1]),sys.argv[2]); "
                "print('acquired' if lock.acquire() else 'locked'); lock.release()"
            ), directory, self.origin]
            try:
                self.assertEqual(subprocess.check_output(command, text=True).strip(), "locked")
                # Another explicitly configured origin is independent.
                other = InstanceLock(Path(directory), "http://localhost:9999")
                self.assertTrue(other.acquire())
                other.release()
            finally:
                first.release()
            self.assertEqual(subprocess.check_output(command, text=True).strip(), "acquired")


if __name__ == "__main__":
    unittest.main()
