"""Tests for database models and connection."""

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, TapEvent, Video


@pytest.fixture
def db():
    """In-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_tables_exist(db):
    insp = inspect(db.bind)
    tables = insp.get_table_names()
    assert "videos" in tables
    assert "tap_events" in tables


def test_videos_columns(db):
    insp = inspect(db.bind)
    cols = {c["name"] for c in insp.get_columns("videos")}
    expected = {
        "id",
        "filename",
        "original_name",
        "upload_date",
        "status",
        "duration_sec",
        "error_message",
        "ml_approach",
        "processing_started_at",
        "processing_finished_at",
        "output_dir",
        "restaurant_name",
        "camera_id",
    }
    assert expected == cols


def test_tap_events_columns(db):
    insp = inspect(db.bind)
    cols = {c["name"] for c in insp.get_columns("tap_events")}
    expected = {
        "id",
        "video_id",
        "tap",
        "frame_start",
        "frame_end",
        "timestamp_start",
        "timestamp_end",
        "confidence",
        "count",
    }
    assert expected == cols


def test_insert_video(db):
    v = Video(filename="abc123_test.mp4", original_name="test.mp4")
    db.add(v)
    db.commit()
    db.refresh(v)
    assert v.id is not None
    assert v.status == "pending"


def test_insert_tap_event(db):
    v = Video(filename="abc_test.mp4", original_name="test.mp4")
    db.add(v)
    db.commit()

    ev = TapEvent(
        video_id=v.id,
        tap="A",
        frame_start=0,
        frame_end=100,
        timestamp_start=0.0,
        timestamp_end=5.0,
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    assert ev.count == 1
    assert ev.confidence is None


def test_cascade_delete(db):
    v = Video(filename="abc_test.mp4", original_name="test.mp4")
    db.add(v)
    db.commit()

    ev = TapEvent(
        video_id=v.id,
        tap="B",
        frame_start=10,
        frame_end=50,
        timestamp_start=0.5,
        timestamp_end=2.5,
    )
    db.add(ev)
    db.commit()

    db.delete(v)
    db.commit()

    assert db.query(TapEvent).count() == 0
