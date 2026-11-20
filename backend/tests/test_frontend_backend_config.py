"""Verify frontend .env configuration and backend CORS/routes."""

from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent.parent


def test_frontend_env_and_example_files_exist() -> None:
    frontend_env = ROOT / "frontend" / ".env"
    frontend_example = ROOT / "frontend" / ".env.example"
    assert frontend_env.exists(), "frontend/.env must exist"
    assert frontend_example.exists(), "frontend/.env.example must exist"

    env_text = frontend_env.read_text(encoding="utf-8")
    example_text = frontend_example.read_text(encoding="utf-8")

    assert "VITE_API_BASE_URL" in env_text
    assert "VITE_API_BASE_URL" in example_text
    assert "http://127.0.0.1:8001" in env_text


def test_cors_and_routes_accessible_with_frontend_origin() -> None:
    client = TestClient(app)
    # Liveness check
    live_res = client.get("/api/health/live", headers={"Origin": "http://127.0.0.1:5173"})
    assert live_res.status_code == 200
    assert live_res.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"

    # Registry check
    reg_res = client.get("/api/v1/registry", headers={"Origin": "http://127.0.0.1:5173"})
    assert reg_res.status_code == 200
    assert reg_res.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"

