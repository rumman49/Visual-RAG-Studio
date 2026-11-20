from fastapi.testclient import TestClient

from app.main import app


def test_liveness_is_truthful() -> None:
    response = TestClient(app).get("/api/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_does_not_claim_unchecked_services_are_connected() -> None:
    response = TestClient(app).get("/api/health/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert set(payload["checks"].values()) == {"valid", "not_checked"}
    assert "not initialized" in payload["message"]


def test_cors_allows_frontend_origins() -> None:
    client = TestClient(app)
    response = client.options(
        "/api/health/live",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"

