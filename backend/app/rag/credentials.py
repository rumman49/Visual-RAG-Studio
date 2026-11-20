"""Process-local provider credentials. Secrets are never persisted or returned."""

from threading import RLock


class RuntimeCredentials:
    def __init__(self) -> None:
        self._lock = RLock()
        self._keys: dict[str, str] = {}

    def set_key(self, provider: str, value: str) -> None:
        with self._lock:
            self._keys[provider.lower()] = value

    def get_key(self, provider: str) -> str | None:
        with self._lock:
            return self._keys.get(provider.lower())

    def clear_key(self, provider: str) -> None:
        with self._lock:
            self._keys.pop(provider.lower(), None)

    def set_groq_api_key(self, value: str) -> None:
        self.set_key("groq", value)

    def get_groq_api_key(self) -> str | None:
        return self.get_key("groq")

    def clear_groq_api_key(self) -> None:
        self.clear_key("groq")


runtime_credentials = RuntimeCredentials()
