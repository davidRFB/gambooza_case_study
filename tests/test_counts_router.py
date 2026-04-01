"""Tests for the counts router."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.models import Base, Video, TapEvent
from backend.database.connection import get_db
from backend.main import app


@pytest.fixture
def client():
    """Test client with a seeded in-memory DB."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    # Seed: one completed video with tap events
    db = Session()
    v = Video(filename="test.mp4", original_name="test.mp4", status="completed")
    db.add(v)
    db.commit()
    db.add_all([
        TapEvent(video_id=v.id, tap="A", frame_start=0, frame_end=100,
                 timestamp_start=0.0, timestamp_end=5.0, count=1),
        TapEvent(video_id=v.id, tap="A", frame_start=200, frame_end=300,
                 timestamp_start=10.0, timestamp_end=15.0, count=1),
        TapEvent(video_id=v.id, tap="B", frame_start=400, frame_end=500,
                 timestamp_start=20.0, timestamp_end=25.0, count=2),
    ])
    db.commit()
    db.close()

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


def test_get_counts(client):
    resp = client.get("/api/counts/")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["tap_a"] == 2   # two events, count=1 each
    assert data[0]["tap_b"] == 2   # one event, count=2
    assert data[0]["total"] == 4


def test_get_counts_filter_tap(client):
    resp = client.get("/api/counts/?tap=A")
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["tap_a"] == 2
    assert data[0]["tap_b"] == 0


def test_get_counts_empty_when_no_completed(client):
    # Query a nonexistent video
    resp = client.get("/api/counts/?video_id=999")
    assert resp.status_code == 200
    assert resp.json() == []


def test_summary(client):
    resp = client.get("/api/counts/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tap_a_total"] == 2
    assert data["tap_b_total"] == 2
    assert data["grand_total"] == 4
    assert data["video_count"] == 1
