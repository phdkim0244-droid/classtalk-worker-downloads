"""Only native encrypted credential stores are supported. No file fallback."""
from __future__ import annotations

import sys

from desktop.client import WorkerError


class TokenStore:
    def __init__(self, origin: str):
        # Construct a specific backend, rather than keyring's plugin discovery.
        # This prevents an installed plaintext/chained backend being selected.
        if sys.platform == "darwin":
            from keyring.backends.macOS import Keyring
        elif sys.platform == "win32":
            from keyring.backends.Windows import WinVaultKeyring as Keyring
        else:
            raise WorkerError("이 앱은 Windows와 macOS를 지원합니다.")
        self.backend = Keyring()
        self.service = "kr.ydaasam.classtalk.worker"
        self.account = origin

    def load(self) -> str | None:
        try:
            return self.backend.get_password(self.service, self.account)
        except Exception as exc:
            raise WorkerError("보안 저장소를 열 수 없습니다. 키체인/자격 증명 관리자 접근을 허용해 주세요.") from exc

    def save(self, token: str) -> None:
        if not isinstance(token, str) or not 32 <= len(token) <= 512:
            raise WorkerError("유효하지 않은 계정 연결 응답입니다.")
        try:
            self.backend.set_password(self.service, self.account, token)
        except Exception as exc:
            raise WorkerError("계정 연결을 보안 저장소에 저장하지 못했습니다.") from exc

    def clear(self) -> None:
        from keyring.errors import PasswordDeleteError
        try:
            self.backend.delete_password(self.service, self.account)
        except PasswordDeleteError:
            pass
        except Exception as exc:
            raise WorkerError("보안 저장소의 연결 정보를 지우지 못했습니다.") from exc
