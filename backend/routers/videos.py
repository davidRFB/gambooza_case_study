"""Video endpoints — upload, list, status, delete."""

import json
import logging
import re
import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.config import RESULTS_DIR, ROI_CONFIGS_DIR
from backend.database.connection import SessionLocal, get_db
from backend.database.models import TapEvent, Video
from backend.database.schemas import (
    TapEventResponse,
    VideoListItem,
    VideoStatusResponse,
    VideoUploadResponse,
)
from backend.services.processor import process_video

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".mp4", ".mov"}


@router.post("/upload", response_model=VideoUploadResponse, status_code=201)
def upload_video(
    file: UploadFile = File(...),
    restaurant_name: str | None = Query(None),
    camera_id: str | None = Query(None),
    db: Session = Depends(get_db),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(422, f"Only {ALLOWED_EXTENSIONS} files are allowed")

    unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"

    # Create DB row first to get the ID
    video = Video(
        filename=unique_name,
        original_name=file.filename,
        restaurant_name=restaurant_name,
        camera_id=camera_id,
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    # Save video into data/results/web_{id}/
    video_dir = RESULTS_DIR / f"web_{video.id}"
    video_dir.mkdir(parents=True, exist_ok=True)
    dest = video_dir / unique_name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info("Uploaded video: id=%d, name=%s, dir=%s", video.id, video.original_name, video_dir)
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
        restaurant_name=video.restaurant_name,
        camera_id=video.camera_id,
        duration_sec=video.duration_sec,
        error_message=video.error_message,
        ml_approach=video.ml_approach,
        processing_started_at=video.processing_started_at,
        processing_finished_at=video.processing_finished_at,
        output_dir=video.output_dir,
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

    # Remove the entire results directory for this video
    video_dir = RESULTS_DIR / f"web_{video_id}"
    if video_dir.exists():
        shutil.rmtree(video_dir)

    db.delete(video)  # cascade deletes tap_events
    db.commit()
    logger.info("Deleted video: id=%d, name=%s", video_id, video.original_name)


def _run_processing(video_id: int):
    """Background task wrapper — creates its own DB session."""

    db = SessionLocal()
    try:
        process_video(video_id, db)
    finally:
        db.close()


@router.get("/restaurants")
def list_restaurants(db: Session = Depends(get_db)):

    # Get distinct restaurant+camera combos from DB
    restaurants: dict[str, set[str]] = {}
    rows = (
        db.query(Video.restaurant_name, Video.camera_id)
        .filter(Video.restaurant_name.isnot(None), Video.camera_id.isnot(None))
        .distinct()
        .all()
    )
    for r_name, c_id in rows:
        restaurants.setdefault(r_name, set()).add(c_id)

    # Also scan ROI config files for {restaurant}_{camera}.json
    if ROI_CONFIGS_DIR.exists():
        for f in ROI_CONFIGS_DIR.glob("*.json"):
            name = f.stem  # e.g. "mikes_pub_cam1"
            if name == "default":
                continue
            # Split on last underscore to get restaurant_camera
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                r_name, c_id = parts
                restaurants.setdefault(r_name, set()).add(c_id)

    # Convert sets to sorted lists
    return {
        "restaurants": sorted(restaurants.keys()),
        "cameras": {k: sorted(v) for k, v in restaurants.items()},
    }


@router.get("/roi-config-exists")
def check_roi_config_exists(
    restaurant_name: str = Query(...),
    camera_id: str = Query(...),
):
    config_name = f"{restaurant_name}_{camera_id}"
    path = ROI_CONFIGS_DIR / f"{config_name}.json"
    result = {"exists": path.exists(), "config_name": config_name, "roi_data": None}
    if path.exists():
        with open(path) as f:
            result["roi_data"] = json.load(f)
    return result


@router.post("/roi-config")
def save_roi_config(
    data: dict,
):
    restaurant_name = data.get("restaurant_name")
    camera_id = data.get("camera_id")
    if not restaurant_name or not camera_id:
        raise HTTPException(422, "restaurant_name and camera_id are required")

    # Sanitize for filename safety
    pattern = re.compile(r"^[a-zA-Z0-9_-]+$")
    if not pattern.match(restaurant_name) or not pattern.match(camera_id):
        raise HTTPException(
            422,
            "restaurant_name and camera_id must contain only letters, numbers, hyphens, underscores",
        )

    roi_data = data.get("roi_data")
    if not roi_data or "yolo" not in roi_data:
        raise HTTPException(422, "roi_data with 'yolo' section is required")
    yolo = roi_data["yolo"]
    if "tap_roi" not in yolo or "sam3_tap_bboxes" not in yolo:
        raise HTTPException(422, "yolo section must contain tap_roi and sam3_tap_bboxes")

    # Validate optional "simple" section for pre-filtering ROIs
    simple = roi_data.get("simple")
    if simple is not None:
        if "roi_1" not in simple:
            raise HTTPException(422, "simple section must contain roi_1")
        for key in ("roi_1", "roi_2"):
            if key in simple and (not isinstance(simple[key], list) or len(simple[key]) != 4):
                raise HTTPException(422, f"simple.{key} must be a list of 4 floats")

    config_name = f"{restaurant_name}_{camera_id}"
    path = ROI_CONFIGS_DIR / f"{config_name}.json"
    ROI_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(roi_data, f, indent=2)

    logger.info("Saved ROI config: %s", path)
    return {"config_name": config_name, "path": str(path)}


@router.get("/{video_id}/frame")
def get_video_frame(video_id: int, db: Session = Depends(get_db)):

    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")

    video_path = RESULTS_DIR / f"web_{video_id}" / video.filename
    if not video_path.exists():
        raise HTTPException(404, "Video file not found on disk")

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(500, "Could not read frame from video")

    _, jpeg = cv2.imencode(".jpg", frame)
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@router.post("/{video_id}/process", status_code=202)
def start_processing(
    video_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")
    if video.status == "processing":
        raise HTTPException(409, "Video is already being processed")

    background_tasks.add_task(_run_processing, video_id)
    logger.info("Processing queued: video_id=%d", video_id)
    return {"message": "Processing started", "video_id": video_id}
