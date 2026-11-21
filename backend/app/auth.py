"""API-key boundary for `/api/v1` routes (AD-12). The key is never logged or returned."""

import hmac

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings

API_KEY_HEADER = "X-API-Key"


def _unauthorized(code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": {"code": code, "message": message}})


def install_api_key_auth(app: FastAPI, settings: Settings) -> None:
    """Require `X-API-Key` on `/api/v1/*` when a key is configured.

    Register this before CORS so 401 responses still carry CORS headers. Preflight
    (OPTIONS) requests carry no credentials and are left to CORS.
    """
    if settings.api_key is None:
        return
    expected = settings.api_key.get_secret_value().encode()

    @app.middleware("http")
    async def api_key_guard(request: Request, call_next):
        if request.method != "OPTIONS" and request.url.path.startswith("/api/v1"):
            supplied = request.headers.get(API_KEY_HEADER)
            if not supplied:
                return _unauthorized("api_key_required", f"{API_KEY_HEADER} header is required.")
            if not hmac.compare_digest(supplied.encode(), expected):
                return _unauthorized("api_key_invalid", f"{API_KEY_HEADER} is invalid.")
        return await call_next(request)
