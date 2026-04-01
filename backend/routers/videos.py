"""Video endpoints — upload, list, status, delete."""

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.config import UPLOADS_DIR
from backend.database.connection import get_db
from backend.database.models import Video, TapEvent
from backend.database.schemas import (
    VideoUploadResponse,
    VideoListItem,
    VideoStatusResponse,
    TapEventResponse,
)

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp4", ".mov"}


@router.post("/upload", response_model=VideoUploadResponse, status_code=201)
def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(422, f"Only {ALLOWED_EXTENSIONS} files are allowed")

    unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    dest = UPLOADS_DIR / unique_name
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    video = Video(filename=unique_name, original_name=file.filename)
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


@router.get("/", response_model=list[VideoListItem])
def list_videos(db: Session = Depends(get_db)):
    return db.query(Video).order_by(Video.upload_date.desc()).all()


@router.get("/{video_id}/status", response_model=VideoStatusResponse)
def get_video_status(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    tap_a_count = (
        db.query(func.coalesce(func.sum(TapEvent.count), 0))
        .filter(TapEvent.video_id == video_id, TapEvent.tap == "A")
        .scalar()
    )
    tap_b_count = (
        db.query(func.coalesce(func.sum(TapEvent.count), 0))
        .filter(TapEvent.video_id == video_id, TapEvent.tap == "B")
        .scalar()
    )

    events = db.query(TapEvent).filter(TapEvent.video_id == video_id).all()

    return VideoStatusResponse(
        id=video.id,
        filename=video.filename,
        original_name=video.original_name,
        upload_date=video.upload_date,
        status=video.status,
        duration_sec=video.duration_sec,
        error_message=video.error_message,
        ml_approach=video.ml_approach,
        processing_started_at=video.processing_started_at,
        processing_finished_at=video.processing_finished_at,
        tap_a_count=tap_a_count,
        tap_b_count=tap_b_count,
        total=tap_a_count + tap_b_count,
        events=[TapEventResponse.model_validate(e) for e in events],
    )


@router.delete("/{video_id}", status_code=204)
def delete_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    # Remove file from disk
    file_path = UPLOADS_DIR / video.filename
    if file_path.exists():
        file_path.unlink()

    db.delete(video)  # cascade deletes tap_events
    db.commit()
