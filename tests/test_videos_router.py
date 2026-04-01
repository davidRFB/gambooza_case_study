"""Tests for the videos router."""

import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.models import Base
from backend.database.connection import get_db
from backend.main import app


@pytest.fixture
def client():
    """Test client with a fresh in-memory DB."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _make_fake_mp4():
    return ("test_video.mp4", io.BytesIO(b"fake video content"), "video/mp4")


def test_upload_video(client):
    name, file, ct = _make_fake_mp4()
    resp = client.post("/api/videos/upload", files={"file": (name, file, ct)})
    assert resp.status_code == 201
    data = resp.json()
    assert data["original_name"] == "test_video.mp4"
    assert data["status"] == "pending"
    assert "test_video.mp4" in data["filename"]


def test_upload_rejects_bad_extension(client):
    resp = client.post(
        "/api/videos/upload",
        files={"file": ("notes.txt", io.BytesIO(b"text"), "text/plain")},
    )
    assert resp.status_code == 422


def test_list_videos(client):
    name, file, ct = _make_fake_mp4()
    client.post("/api/videos/upload", files={"file": (name, file, ct)})

    resp = client.get("/api/videos/")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_get_status(client):
    name, file, ct = _make_fake_mp4()
    upload = client.post(
        "/api/videos/upload", files={"file": (name, file, ct)},
    ).json()

    resp = client.get(f"/api/videos/{upload['id']}/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    assert data["tap_a_count"] == 0
    assert data["tap_b_count"] == 0
    assert data["total"] == 0
    assert data["events"] == []


def test_get_status_not_found(client):
    resp = client.get("/api/videos/9999/status")
    assert resp.status_code == 404


def test_process_returns_202(client):
    name, file, ct = _make_fake_mp4()
    upload = client.post(
        "/api/videos/upload", files={"file": (name, file, ct)},
    ).json()

    resp = client.post(f"/api/videos/{upload['id']}/process")
    assert resp.status_code == 202
    data = resp.json()
    assert data["video_id"] == upload["id"]


def test_process_not_found(client):
    resp = client.post("/api/videos/9999/process")
    assert resp.status_code == 404


def test_delete_video(client):
    name, file, ct = _make_fake_mp4()
    upload = client.post(
        "/api/videos/upload", files={"file": (name, file, ct)},
    ).json()

    resp = client.delete(f"/api/videos/{upload['id']}")
    assert resp.status_code == 204

    resp = client.get(f"/api/videos/{upload['id']}/status")
    assert resp.status_code == 404


def test_delete_not_found(client):
    resp = client.delete("/api/videos/9999")
    assert resp.status_code == 404
