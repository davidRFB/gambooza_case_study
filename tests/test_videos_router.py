"""Tests for the videos router."""

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.connection import get_db
from backend.database.models import Base
from backend.main import app


@pytest.fixture
def client(tmp_path):
    """Test client with a fresh in-memory DB and isolated results dir."""
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
    with patch("backend.routers.videos.RESULTS_DIR", tmp_path / "results"):
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
        "/api/videos/upload",
        files={"file": (name, file, ct)},
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
        "/api/videos/upload",
        files={"file": (name, file, ct)},
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
        "/api/videos/upload",
        files={"file": (name, file, ct)},
    ).json()

    resp = client.delete(f"/api/videos/{upload['id']}")
    assert resp.status_code == 204

    resp = client.get(f"/api/videos/{upload['id']}/status")
    assert resp.status_code == 404


def test_delete_not_found(client):
    resp = client.delete("/api/videos/9999")
    assert resp.status_code == 404


def test_upload_with_restaurant_camera(client):
    name, file, ct = _make_fake_mp4()
    resp = client.post(
        "/api/videos/upload",
        files={"file": (name, file, ct)},
        params={"restaurant_name": "mikes_pub", "camera_id": "cam1"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["restaurant_name"] == "mikes_pub"
    assert data["camera_id"] == "cam1"


def test_upload_without_restaurant_camera(client):
    name, file, ct = _make_fake_mp4()
    resp = client.post("/api/videos/upload", files={"file": (name, file, ct)})
    assert resp.status_code == 201
    data = resp.json()
    assert data["restaurant_name"] is None
    assert data["camera_id"] is None


def test_roi_config_exists_false(client):
    resp = client.get(
        "/api/videos/roi-config-exists",
        params={"restaurant_name": "nonexistent", "camera_id": "cam99"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["exists"] is False
    assert data["config_name"] == "nonexistent_cam99"


def test_restaurants_list(client):
    # Upload a video with restaurant+camera
    name, file, ct = _make_fake_mp4()
    client.post(
        "/api/videos/upload",
        files={"file": (name, file, ct)},
        params={"restaurant_name": "test_rest", "camera_id": "cam1"},
    )

    resp = client.get("/api/videos/restaurants")
    assert resp.status_code == 200
    data = resp.json()
    assert "test_rest" in data["restaurants"]
    assert "cam1" in data["cameras"]["test_rest"]


def test_save_roi_config(client, tmp_path):
    with patch("backend.config.ROI_CONFIGS_DIR", tmp_path):
        resp = client.post(
            "/api/videos/roi-config",
            json={
                "restaurant_name": "testrest",
                "camera_id": "cam1",
                "roi_data": {
                    "yolo": {
                        "tap_roi": [0.1, 0.2, 0.3, 0.4],
                        "sam3_tap_bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
                    }
                },
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["config_name"] == "testrest_cam1"


def test_save_roi_config_bad_name(client):
    resp = client.post(
        "/api/videos/roi-config",
        json={
            "restaurant_name": "bad/name",
            "camera_id": "cam1",
            "roi_data": {"yolo": {"tap_roi": [], "sam3_tap_bboxes": []}},
        },
    )
    assert resp.status_code == 422


def test_save_roi_config_missing_yolo(client):
    resp = client.post(
        "/api/videos/roi-config",
        json={
            "restaurant_name": "test",
            "camera_id": "cam1",
            "roi_data": {"simple": {}},
        },
    )
    assert resp.status_code == 422


def test_save_roi_config_with_simple_section(client, tmp_path):
    """ROI config with both 'simple' and 'yolo' sections saves correctly."""
    with patch("backend.routers.videos.ROI_CONFIGS_DIR", tmp_path):
        resp = client.post(
            "/api/videos/roi-config",
            json={
                "restaurant_name": "testrest",
                "camera_id": "cam2",
                "roi_data": {
                    "simple": {
                        "roi_1": [0.48, 0.44, 0.52, 0.47],
                        "roi_2": [0.56, 0.41, 0.58, 0.53],
                    },
                    "yolo": {
                        "tap_roi": [0.1, 0.2, 0.3, 0.4],
                        "sam3_tap_bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
                    },
                },
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["config_name"] == "testrest_cam2"

    # Verify saved file has both sections
    import json

    saved = json.loads((tmp_path / "testrest_cam2.json").read_text())
    assert "simple" in saved
    assert "yolo" in saved
    assert saved["simple"]["roi_1"] == [0.48, 0.44, 0.52, 0.47]


def test_save_roi_config_simple_bad_roi(client):
    """Simple section with invalid roi_1 (wrong length) should fail."""
    resp = client.post(
        "/api/videos/roi-config",
        json={
            "restaurant_name": "test",
            "camera_id": "cam1",
            "roi_data": {
                "simple": {"roi_1": [0.1, 0.2]},
                "yolo": {
                    "tap_roi": [0.1, 0.2, 0.3, 0.4],
                    "sam3_tap_bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
                },
            },
        },
    )
    assert resp.status_code == 422
