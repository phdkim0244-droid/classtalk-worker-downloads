"""ClassTalk's account-scoped desktop worker. Never imports operator credentials."""

VERSION = "0.2.4"
PRODUCTION_ORIGIN = "https://talk.ydssam.kr"


def storage_identity(origin: str) -> str:
    """Keep this same project's existing keychain account and instance lock.

    This is a local storage identifier only; all production requests use the
    verified custom domain. Development origins keep independent credentials.
    """
    return "https://classtalk-web-three.vercel.app" if origin == PRODUCTION_ORIGIN else origin
