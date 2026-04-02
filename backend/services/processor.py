"""Background video processing — runs ML pipeline and saves results to DB."""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from backend.config import PROJECT_ROOT, RESULTS_DIR, ROI_CONFIGS_DIR, YOLO_BASE_CONFIG
from backend.database.models import TapEvent as TapEventModel
from backend.database.models import Video

logger = logging.getLogger(__name__)


def _resolve_roi_config_name(restaurant_name: str | None, camera_id: str | None) -> str | None:
    """Derive ROI config filename from restaurant+camera. Returns None if not found."""
    if restaurant_name and camera_id:
        candidate = f"{restaurant_name}_{camera_id}"
        path = ROI_CONFIGS_DIR / f"{candidate}.json"
        if path.exists():
            return candidate
    return None


def process_video(video_id: int, db: Session):
    """Run the YOLO pipeline on a video and save tap events to the DB.

    Parameters
    ----------
    video_id : DB id of the video to process
    db : SQLAlchemy session
    """
    logger.info("process_video started: video_id=%d", video_id)

    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        logger.warning("Video not found in DB: video_id=%d", video_id)
        return

    video.status = "processing"
    video.processing_started_at = datetime.utcnow()
    db.commit()
    logger.info("Video %d status -> processing", video_id)

    try:
        video_dir = RESULTS_DIR / f"web_{video_id}"
        video_path = video_dir / video.filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Resolve ROI config from restaurant + camera (mandatory)
        roi_config = _resolve_roi_config_name(video.restaurant_name, video.camera_id)
        if roi_config is None:
            raise ValueError(
                f"No ROI config found for restaurant='{video.restaurant_name}', "
                f"camera='{video.camera_id}'. Create one before processing."
            )
        roi = _load_roi_config(roi_config)
        logger.info("ROI config '%s' loaded", roi_config)

        # Build temp YOLO config and run pipeline
        pour_events = _run_yolo_pipeline(video_path, video_id, roi)
        logger.info("Pipeline returned %d pour events", len(pour_events))

        # Clear any existing tap events from previous runs
        old_count = db.query(TapEventModel).filter(TapEventModel.video_id == video_id).delete()
        if old_count:
            logger.info("Cleared %d old tap events for video %d", old_count, video_id)

        # Map results to DB
        tap_events_db = _map_pour_events(video_id, pour_events)
        db.add_all(tap_events_db)
        logger.info("Saved %d tap events to DB", len(tap_events_db))

        video.status = "completed"
        video.ml_approach = "yolo"
        video.output_dir = str(RESULTS_DIR / f"web_{video_id}")
        video.processing_finished_at = datetime.utcnow()
        db.commit()
        logger.info("Video %d status -> completed", video_id)

    except Exception as e:
        logger.exception("Video %d processing failed: %s", video_id, e)
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
    output_dir = str(RESULTS_DIR / f"web_{video_id}")
    cfg["output_dir"] = output_dir

    # Override ROI and SAM3 bboxes from roi config
    yolo_roi = roi["yolo"]
    cfg.setdefault("roi", {})["tap_roi"] = yolo_roi["tap_roi"]
    cfg.setdefault("sam3", {})["tap_bboxes"] = yolo_roi["sam3_tap_bboxes"]

    # Set preview_second to 0 to avoid seeking past end of short videos
    cfg.setdefault("yolo", {})["preview_second"] = 0

    # Resolve all relative paths to absolute (load_config resolves relative
    # to config file's grandparent, which breaks for temp files in /tmp/)
    yolo_cfg = cfg.get("yolo", {})
    if "tracker" in yolo_cfg and not Path(yolo_cfg["tracker"]).is_absolute():
        yolo_cfg["tracker"] = str(PROJECT_ROOT / yolo_cfg["tracker"])
    if "model" in yolo_cfg and not Path(yolo_cfg["model"]).is_absolute():
        yolo_cfg["model"] = str(PROJECT_ROOT / yolo_cfg["model"])
    sam3_cfg = cfg.get("sam3", {})
    if "model" in sam3_cfg and not Path(sam3_cfg["model"]).is_absolute():
        sam3_cfg["model"] = str(PROJECT_ROOT / sam3_cfg["model"])

    # Pre-create tap_roi.json in output dir so ROI stage skips interactive mode
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tap_roi_json = output_path / "tap_roi.json"
    tap_roi_json.write_text(
        json.dumps(
            {
                "tap_roi": yolo_roi["tap_roi"],
                "sam3_tap_bboxes": yolo_roi["sam3_tap_bboxes"],
            },
            indent=2,
        )
    )

    # Write temp config
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="pipeline_web_",
        delete=False,
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

        rows.append(
            TapEventModel(
                video_id=video_id,
                tap=tap,
                frame_start=pe["frame_start"],
                frame_end=pe["frame_end"],
                timestamp_start=pe["time_start"],
                timestamp_end=pe["time_end"],
                confidence=None,
                count=1,
            )
        )

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
