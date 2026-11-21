import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.auth import install_api_key_auth
from app.config import Settings

KEY = "s3cret-test-key"


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def _client(api_key: str | None) -> TestClient:
    app = FastAPI()
    install_api_key_auth(app, _settings(api_key=api_key))
    app.add_middleware(CORSMiddleware, allow_origins=["http://x.test"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.get("/api/v1/ping")
    def ping():
        return {"ok": True}

    @app.get("/api/health/live")
    def live():
        return {"status": "alive"}

    return TestClient(app)


def test_missing_key_is_rejected_with_typed_401():
    res = _client(KEY).get("/api/v1/ping")
    assert res.status_code == 401
    assert res.json()["detail"]["code"] == "api_key_required"
    assert KEY not in res.text


def test_wrong_key_is_rejected_with_typed_401():
    res = _client(KEY).get("/api/v1/ping", headers={"X-API-Key": "nope"})
    assert res.status_code == 401
    assert res.json()["detail"]["code"] == "api_key_invalid"
    assert set(res.json()["detail"]) == {"code", "message"}


def test_correct_key_is_accepted():
    res = _client(KEY).get("/api/v1/ping", headers={"X-API-Key": KEY})
    assert res.status_code == 200


def test_health_stays_open_without_key():
    assert _client(KEY).get("/api/health/live").status_code == 200


def test_401_carries_cors_headers_and_preflight_passes():
    client = _client(KEY)
    res = client.get("/api/v1/ping", headers={"Origin": "http://x.test"})
    assert res.status_code == 401
    assert res.headers["access-control-allow-origin"] == "http://x.test"
    pre = client.options(
        "/api/v1/ping",
        headers={"Origin": "http://x.test", "Access-Control-Request-Method": "GET",
                 "Access-Control-Request-Headers": "x-api-key"},
    )
    assert pre.status_code == 200


def test_no_key_configured_leaves_loopback_dev_open():
    assert _client(None).get("/api/v1/ping").status_code == 200


def test_production_without_key_refuses_to_start_naming_setting():
    with pytest.raises(ValidationError, match="VRS_API_KEY"):
        _settings(VRS_ENV="production")


def test_production_with_key_starts():
    assert _settings(VRS_ENV="production", api_key=KEY).api_key is not None


def test_non_loopback_host_requires_key():
    with pytest.raises(ValidationError, match="VRS_API_KEY"):
        _settings(host="0.0.0.0")
    assert _settings(host="0.0.0.0", api_key=KEY).host == "0.0.0.0"


def test_readiness_reports_auth_mode():
    from app.main import app

    body = TestClient(app).get("/api/health/ready").json()
    assert body["auth"] in {"required", "none"}
