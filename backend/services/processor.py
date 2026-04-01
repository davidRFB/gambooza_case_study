"""Background video processing — runs ML pipeline and saves results to DB."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from backend.config import UPLOADS_DIR, ROI_CONFIGS_DIR, YOLO_BASE_CONFIG
from backend.database.models import Video, TapEvent as TapEventModel


def process_video(video_id: int, db: Session, roi_config: str = "default"):
    """Run the YOLO pipeline on a video and save tap events to the DB.

    Parameters
    ----------
    video_id : DB id of the video to process
    db : SQLAlchemy session
    roi_config : name of ROI config file in data/roi_configs/ (without .json)
    """
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        return

    video.status = "processing"
    video.processing_started_at = datetime.utcnow()
    db.commit()

    try:
        video_path = UPLOADS_DIR / video.filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Load ROI config
        roi = _load_roi_config(roi_config)

        # Build temp YOLO config and run pipeline
        pour_events = _run_yolo_pipeline(video_path, video_id, roi)

        # Map results to DB
        tap_events_db = _map_pour_events(video_id, pour_events)
        db.add_all(tap_events_db)

        video.status = "completed"
        video.ml_approach = "yolo"
        video.output_dir = str(Path("results") / f"web_{video_id}")
        video.processing_finished_at = datetime.utcnow()
        db.commit()

    except Exception as e:
        video.status = "error"
        video.error_message = str(e)[:500]
        video.processing_finished_at = datetime.utcnow()
        db.commit()


def _run_yolo_pipeline(video_path: Path, video_id: int, roi: dict) -> list[dict]:
    """Create a temp YOLO config and run the pipeline.

    Returns the list of pour_event dicts from YOLODetectorResult.
    """
    # Load base config
    with open(YOLO_BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    # Override video path and output dir
    cfg["video_path"] = str(video_path)
    output_dir = str(Path("results") / f"web_{video_id}")
    cfg["output_dir"] = output_dir

    # Override ROI and SAM3 bboxes from roi config
    yolo_roi = roi["yolo"]
    cfg.setdefault("roi", {})["tap_roi"] = yolo_roi["tap_roi"]
    cfg.setdefault("sam3", {})["tap_bboxes"] = yolo_roi["sam3_tap_bboxes"]

    # Write temp config
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="pipeline_web_", delete=False,
    )
    yaml.dump(cfg, tmp)
    tmp.close()

    try:
        YOLODetector = _import_yolo_detector()
        detector = YOLODetector(config_path=tmp.name)
        result = detector.run(interactive=False, force=False)
        return result.pour_events
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _import_yolo_detector():
    """Lazy import — avoids loading heavy ML deps at module level."""
    from backend.ml.approach_yolo.detector import YOLODetector
    return YOLODetector


def _map_pour_events(video_id: int, pour_events: list[dict]) -> list[TapEventModel]:
    """Convert YOLO pour_event dicts to DB TapEvent rows.

    Mapping:
    - "TAP_A" / "TAP_B" → "A" / "B"
    - Events without a tap field or with "UNKNOWN" are skipped
    - "time_start" / "time_end" → timestamp_start / timestamp_end
    """
    TAP_MAP = {"TAP_A": "A", "TAP_B": "B"}
    rows = []

    for pe in pour_events:
        tap_raw = pe.get("tap")
        tap = TAP_MAP.get(tap_raw)
        if tap is None:
            continue  # skip UNKNOWN or unassigned

        rows.append(TapEventModel(
            video_id=video_id,
            tap=tap,
            frame_start=pe["frame_start"],
            frame_end=pe["frame_end"],
            timestamp_start=pe["time_start"],
            timestamp_end=pe["time_end"],
            confidence=None,
            count=1,
        ))

    return rows


def _load_roi_config(name: str) -> dict:
    """Load a named ROI config from data/roi_configs/{name}.json.

    Returns dict with 'simple' and 'yolo' sections.
    - simple: tap_roi, tap_a_roi, tap_b_roi (normalized 0-1)
    - yolo: tap_roi (normalized crop), sam3_tap_bboxes (pixel-space)
    """
    path = ROI_CONFIGS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"ROI config not found: {path}")
    with open(path) as f:
        roi = json.load(f)
    if "yolo" not in roi:
        raise ValueError("ROI config missing 'yolo' section")
    yolo = roi["yolo"]
    for key in ("tap_roi", "sam3_tap_bboxes"):
        if key not in yolo:
            raise ValueError(f"ROI config yolo section missing key: {key}")
    return roi
