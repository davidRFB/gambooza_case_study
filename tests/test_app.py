"""Tests for the FastAPI app."""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_docs_available():
    resp = client.get("/docs")
    assert resp.status_code == 200
