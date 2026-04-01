"""Pydantic schemas — shapes for API request/response JSON."""

from datetime import datetime

from pydantic import BaseModel

# ── Video schemas ──────────────────────────────────────────────────


class VideoUploadResponse(BaseModel):
    id: int
    filename: str
    original_name: str
    status: str

    model_config = {"from_attributes": True}


class VideoListItem(BaseModel):
    id: int
    original_name: str
    upload_date: datetime
    status: str
    ml_approach: str | None

    model_config = {"from_attributes": True}


class TapEventResponse(BaseModel):
    id: int
    tap: str
    frame_start: int
    frame_end: int
    timestamp_start: float
    timestamp_end: float
    confidence: float | None
    count: int

    model_config = {"from_attributes": True}


class VideoStatusResponse(BaseModel):
    id: int
    filename: str
    original_name: str
    upload_date: datetime
    status: str
    duration_sec: float | None
    error_message: str | None
    ml_approach: str | None
    processing_started_at: datetime | None
    processing_finished_at: datetime | None
    output_dir: str | None
    tap_a_count: int
    tap_b_count: int
    total: int
    events: list[TapEventResponse]


# ── Count schemas ──────────────────────────────────────────────────


class CountResult(BaseModel):
    video_id: int
    original_name: str
    upload_date: datetime
    tap_a: int
    tap_b: int
    total: int


class CountSummary(BaseModel):
    tap_a_total: int
    tap_b_total: int
    grand_total: int
    video_count: int
