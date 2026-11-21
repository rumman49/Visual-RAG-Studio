import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings

VERCEL = "https://my-app.vercel.app"


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def _client(settings: Settings) -> TestClient:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/ping")
    def ping():
        return {"ok": True}

    return TestClient(app)


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/api/v1/ping",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )


def test_configured_origin_is_allowed_and_others_are_rejected():
    client = _client(_settings(cors_origins=f"{VERCEL}/, https://other.example"))
    ok = _preflight(client, VERCEL)
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == VERCEL
    bad = _preflight(client, "https://evil.example")
    assert bad.status_code == 400
    assert "access-control-allow-origin" not in bad.headers


def test_default_allows_local_vite_origins_only():
    client = _client(_settings())
    for origin in ("http://127.0.0.1:5173", "http://localhost:5173"):
        assert _preflight(client, origin).status_code == 200
    assert _preflight(client, "http://localhost:3000").status_code == 400
    assert _preflight(client, VERCEL).status_code == 400


@pytest.mark.parametrize("value", ["*", "https://*.vercel.app", "my-app.vercel.app"])
def test_wildcards_and_schemeless_origins_are_rejected(value):
    with pytest.raises(ValidationError, match="VRS_CORS_ORIGINS"):
        _settings(cors_origins=value)


def test_real_app_no_longer_reflects_arbitrary_origins():
    from app.main import app

    res = TestClient(app).options(
        "/api/v1/registry",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert res.status_code == 400
