"""Tests for Pydantic schemas."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from backend.database.schemas import (
    VideoUploadResponse,
    VideoListItem,
    VideoStatusResponse,
    TapEventResponse,
    CountResult,
    CountSummary,
)


def test_video_upload_response():
    v = VideoUploadResponse(
        id=1, filename="abc_test.mp4",
        original_name="test.mp4", status="pending",
    )
    d = v.model_dump()
    assert d["id"] == 1
    assert d["status"] == "pending"


def test_video_status_response():
    v = VideoStatusResponse(
        id=1, filename="abc_test.mp4", original_name="test.mp4",
        upload_date=datetime(2025, 1, 1), status="completed",
        duration_sec=90.0, error_message=None, ml_approach="simple",
        processing_started_at=datetime(2025, 1, 1, 12, 0, 0),
        processing_finished_at=datetime(2025, 1, 1, 12, 0, 30),
        output_dir="results/web_1",
        tap_a_count=3, tap_b_count=2, total=5, events=[],
    )
    d = v.model_dump()
    assert d["tap_a_count"] == 3
    assert d["total"] == 5
    assert d["events"] == []


def test_tap_event_response():
    ev = TapEventResponse(
        id=1, tap="A", frame_start=0, frame_end=100,
        timestamp_start=0.0, timestamp_end=5.0,
        confidence=0.85, count=1,
    )
    assert ev.count == 1
    assert ev.confidence == 0.85


def test_count_result():
    c = CountResult(
        video_id=1, original_name="test.mp4",
        upload_date=datetime(2025, 1, 1),
        tap_a=3, tap_b=2, total=5,
    )
    assert c.total == 5


def test_count_summary():
    s = CountSummary(
        tap_a_total=10, tap_b_total=8,
        grand_total=18, video_count=3,
    )
    assert s.grand_total == 18
