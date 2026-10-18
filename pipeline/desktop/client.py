"""Small, bounded HTTP protocol. API credentials never accompany audio downloads."""
from __future__ import annotations

import re
import threading
from pathlib import Path
from urllib.parse import urlsplit

import requests

from desktop import PRODUCTION_ORIGIN, VERSION


class WorkerError(Exception):
    pass


class Unauthorized(WorkerError):
    pass


class LeaseLost(WorkerError):
    pass


class Cancelled(WorkerError):
    pass


def checked_origin(value: str, *, allow_local: bool = False) -> str:
    parsed = urlsplit(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("서버 주소에 인증 정보나 쿼리를 넣을 수 없습니다.")
    if parsed.path not in ("", "/"):
        raise ValueError("서버 주소에는 경로를 넣을 수 없습니다.")
    origin = value.rstrip("/")
    if origin == PRODUCTION_ORIGIN:
        return origin
    if (allow_local and parsed.hostname in ("localhost", "127.0.0.1", "::1")
            and parsed.scheme in ("http", "https")):
        return origin
    raise ValueError("운영 서버 또는 명시한 로컬 개발 서버만 사용할 수 있습니다.")


def checked_download(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or parsed.username or parsed.password
            or parsed.port not in (None, 443) or parsed.fragment
            or not re.fullmatch(r"[a-z0-9-]+\.supabase\.co", parsed.hostname or "")
            or not parsed.path.startswith("/storage/v1/object/sign/")):
        raise WorkerError("허용되지 않은 음성 다운로드 주소입니다.")
    return value


class Api:
    def __init__(self, origin: str = PRODUCTION_ORIGIN, *, allow_local: bool = False):
        self.origin = checked_origin(origin, allow_local=allow_local)
        self.token: str | None = None
        self.session = requests.Session()
        # Do not consume .netrc, HTTP proxy credentials or environment CA overrides.
        self.session.trust_env = False
        self.session.headers["User-Agent"] = f"ClassTalk-Worker/{VERSION}"

    def post(self, path: str, body: dict, *, auth: bool = True,
             retry: bool = False, cancel: threading.Event | None = None) -> dict:
        allowed = {"pair", "pair/status", "jobs/claim", "jobs/heartbeat", "jobs/audio",
                   "jobs/recording", "jobs/finish", "disconnect"}
        if path not in allowed:
            raise ValueError("Unknown worker endpoint")
        if auth and not self.token:
            raise Unauthorized("계정을 연결해 주세요.")
        headers = {"Authorization": f"Bearer {self.token}"} if auth else {}
        for attempt in range(3 if retry else 1):
            if cancel and cancel.is_set():
                raise Cancelled("작업이 중지되었습니다.")
            try:
                response = self.session.post(
                    f"{self.origin}/api/worker/{path}", json=body, headers=headers,
                    timeout=(5, 30), allow_redirects=False,
                )
                try:
                    if response.status_code == 401:
                        raise Unauthorized("계정 연결이 만료되었거나 해제되었습니다. 다시 연결해 주세요.")
                    if response.status_code == 409:
                        raise LeaseLost("작업 권한이 만료되었습니다. 요청을 다시 확인합니다.")
                    if response.status_code >= 500 and retry and attempt < 2:
                        pass
                    elif not 200 <= response.status_code < 300:
                        # Never surface backend response bodies: they may include URLs/tokens.
                        raise WorkerError(f"서버 요청 실패 ({response.status_code}). 잠시 후 다시 시도해 주세요.")
                    else:
                        data = response.json()
                        if not isinstance(data, dict):
                            raise WorkerError("서버 응답 형식이 올바르지 않습니다.")
                        return data
                finally:
                    response.close()
            except (requests.RequestException, ValueError) as exc:
                if not retry or attempt == 2:
                    raise WorkerError("서버에 연결하지 못했습니다. 인터넷 연결을 확인해 주세요.") from exc
            wait = 2 ** attempt
            if cancel:
                if cancel.wait(wait):
                    raise Cancelled("작업이 중지되었습니다.")
            else:
                threading.Event().wait(wait)
        raise WorkerError("서버 요청이 반복해서 실패했습니다. 잠시 후 다시 시도해 주세요.")

    def verification_url(self, url: str) -> str:
        parsed = urlsplit(url)
        target_origin = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.username or parsed.password or target_origin != self.origin:
            raise WorkerError("안전하지 않은 계정 연결 주소입니다.")
        return url

    def download(self, url: str, destination: Path, cancel: threading.Event) -> None:
        checked_download(url)
        # Separate session, no bearer header, cookies, netrc, redirects or credentials.
        with requests.Session() as audio:
            audio.trust_env = False
            try:
                with audio.get(url, stream=True, timeout=(5, 30), allow_redirects=False) as response:
                    if response.status_code != 200:
                        raise WorkerError("음성 다운로드가 만료되었거나 실패했습니다. 다시 요청해 주세요.")
                    size = 0
                    with destination.open("wb") as output:
                        for chunk in response.iter_content(1024 * 256):
                            if cancel.is_set():
                                raise Cancelled("작업이 중지되었습니다.")
                            size += len(chunk)
                            if size > 1024 * 1024 * 1024:
                                raise WorkerError("음성 파일이 1GB 제한을 초과했습니다.")
                            output.write(chunk)
                    if not size:
                        raise WorkerError("빈 음성 파일입니다.")
            except requests.RequestException as exc:
                raise WorkerError("음성 다운로드에 실패했습니다. 다시 요청해 주세요.") from exc

    def close(self) -> None:
        self.session.close()
