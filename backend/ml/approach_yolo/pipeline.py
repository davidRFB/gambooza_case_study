"""
Unified beer tap counting pipeline.

Orchestrates four stages:
  1. ROI selection — crop region + TAP_A / TAP_B bboxes (no divider line)
  2. YOLO tracking — detect and track cups with YOLO-World + BoT-SORT
  3. Relink — merge fragmented tracks + classify pours (movement + frame count)
  4. SAM3 tap tracking — segment tap handles, determine which tap via centroid Y

Usage
-----
# Full pipeline (interactive ROI + bbox selection on first run):
python scripts/run_yolo_pipeline.py --config config/pipeline.yaml --interactive

# Re-run with saved coordinates:
python scripts/run_yolo_pipeline.py --config config/pipeline.yaml

# Run a single stage:
python scripts/run_yolo_pipeline.py --config config/pipeline.yaml --stage relink

# Force re-run even if outputs exist:
python scripts/run_yolo_pipeline.py --config config/pipeline.yaml --force
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import yaml

from backend.ml.common import (
    crop_normalized,
    load_roi_config,
    resolve_roi,
    select_tap_bboxes_interactive,
)

logger = logging.getLogger(__name__)

STAGES = ["roi_selection", "yolo_tracking", "relink", "sam3_tap_tracking"]


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Resolve paths relative to project root (parent of config/)
    project_root = config_path.resolve().parent.parent
    cfg["video_path"] = str(project_root / cfg["video_path"])
    cfg["output_dir"] = str(project_root / cfg["output_dir"])
    # Resolve tracker path
    yolo_cfg = cfg.get("yolo", {})
    if "tracker" in yolo_cfg and not Path(yolo_cfg["tracker"]).is_absolute():
        yolo_cfg["tracker"] = str(project_root / yolo_cfg["tracker"])
    # Resolve roi_json path
    roi_cfg = cfg.get("roi", {})
    if roi_cfg.get("roi_json") and not Path(roi_cfg["roi_json"]).is_absolute():
        roi_cfg["roi_json"] = str(project_root / roi_cfg["roi_json"])
    return cfg


# ── Stage 1+2: ROI Selection ─────────────────────────────────────────────────


def stage_roi_selection(cfg: dict, interactive: bool = False, force: bool = False):
    """Resolve crop ROI and SAM3 tap bboxes. Saves to tap_roi.json.

    When *interactive* is True the user gets one window per step:
      1. Drag-rectangle for the crop region (full-frame normalised coords)
      2. One drag-rectangle per label on the cropped frame (TAP_A, TAP_B)

    No A|B divider line — tap side comes from SAM centroid Y.
    """

    video_path = Path(cfg["video_path"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_json = output_dir / "tap_roi.json"
    roi_cfg = cfg.get("roi", {})
    sam3_cfg = cfg.get("sam3", {})

    config_roi = roi_cfg.get("tap_roi")

    # If roi_json points to an external file, load from there
    external_roi = roi_cfg.get("roi_json")
    if external_roi and not interactive:
        ext_path = Path(external_roi)
        if ext_path.exists():
            data = load_roi_config(ext_path)
            if "tap_roi" in data:
                cfg["_tap_roi"] = tuple(data["tap_roi"])
                cfg["_tap_divider"] = tuple(data["tap_divider"]) if "tap_divider" in data else None
                if "sam3_tap_bboxes" in data:
                    cfg.setdefault("sam3", {})["tap_bboxes"] = data["sam3_tap_bboxes"]
                # Copy into current output dir for traceability
                if ext_path.resolve() != roi_json.resolve():
                    shutil.copy2(ext_path, roi_json)
                logger.info("ROI loaded from external file: %s", ext_path)
                return
        else:
            logger.warning("roi_json path not found: %s, falling back to defaults", ext_path)

    # Skip if already done and not forced (divider is optional)
    if roi_json.exists() and not force and not interactive:
        data = load_roi_config(roi_json)
        if "tap_roi" in data:
            cfg["_tap_roi"] = tuple(data["tap_roi"])
            cfg["_tap_divider"] = tuple(data["tap_divider"]) if "tap_divider" in data else None
            if "sam3_tap_bboxes" in data:
                cfg.setdefault("sam3", {})["tap_bboxes"] = data["sam3_tap_bboxes"]
            logger.info("ROI already resolved in %s, skipping", roi_json)
            return

    # Read first frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame_0 = cap.read()
    cap.release()
    if not ret:
        logger.error("Could not read %s", video_path)
        sys.exit(1)

    # -- Window 1: crop region (full frame, normalised) ---------------------
    tap_roi = resolve_roi(config_roi, roi_json, frame_0, interactive=interactive)

    # Optional divider from YAML only (not interactive; unused by YOLO/relink)
    tap_divider = None
    if roi_cfg.get("tap_divider") is not None:
        tap_divider = tuple(roi_cfg["tap_divider"])

    # -- Window 2+: TAP_A / TAP_B bboxes on cropped frame ------------------
    object_labels = sam3_cfg.get("object_labels", ["TAP_A", "TAP_B"])
    tap_bboxes = sam3_cfg.get("tap_bboxes")

    if interactive or tap_bboxes is None:
        existing = load_roi_config(roi_json)
        if "sam3_tap_bboxes" in existing and not interactive:
            tap_bboxes = existing["sam3_tap_bboxes"]
            logger.info("SAM3 tap bboxes loaded from %s", roi_json)
        else:
            logger.info("Opening TAP_A / TAP_B bbox selectors on cropped frame")
            crop = crop_normalized(frame_0, tap_roi)
            tap_bboxes = select_tap_bboxes_interactive(crop, object_labels)

    # -- Persist everything -------------------------------------------------
    save_data = {"tap_roi": list(tap_roi)}
    if tap_divider is not None:
        save_data["tap_divider"] = list(tap_divider)
    if tap_bboxes:
        save_data["sam3_tap_bboxes"] = tap_bboxes
    roi_json.write_text(json.dumps(save_data, indent=2))
    logger.info("ROI config saved to %s", roi_json)

    cfg["_tap_roi"] = tap_roi
    cfg["_tap_divider"] = tap_divider
    if tap_bboxes:
        cfg.setdefault("sam3", {})["tap_bboxes"] = tap_bboxes


# ── Stage 3a: YOLO Tracking ──────────────────────────────────────────────────


def stage_yolo_tracking(cfg: dict, force: bool = False):
    """Run YOLO-World tracking on the cropped video."""
    from backend.ml.approach_yolo import yolo_track

    video_path = Path(cfg["video_path"])
    output_dir = Path(cfg["output_dir"])
    yolo_cfg = cfg.get("yolo", {})

    raw_csv = output_dir / "raw_detections.csv"
    if raw_csv.exists() and not force:
        logger.info("raw_detections.csv already exists at %s, skipping", raw_csv)
        return

    tap_roi, _ = _get_roi(cfg)

    record_range = yolo_cfg.get("record_range")
    if record_range:
        record_range = tuple(record_range)

    yolo_track.run_yolo_tracking(
        video_path=video_path,
        output_dir=output_dir,
        tap_roi=tap_roi,
        model_name=yolo_cfg.get("model", "data/models/yolov8x-worldv2.pt"),
        classes=yolo_cfg.get("classes", ["cup", "person"]),
        sample_every=yolo_cfg.get("sample_every", 1),
        conf_threshold=yolo_cfg.get("conf_threshold", 0.25),
        tracker=yolo_cfg.get("tracker", "config/botsort.yaml"),
        preview_second=yolo_cfg.get("preview_second", 60.0),
        record_range=record_range,
        save_video=yolo_cfg.get("save_video", False),
    )


# ── Stage 3b: Relink ─────────────────────────────────────────────────────────


def stage_relink(cfg: dict, force: bool = False):
    """Relink fragmented cup tracks and classify pour events."""
    from backend.ml.approach_yolo import relink as relink_mod

    video_path = Path(cfg["video_path"])
    output_dir = Path(cfg["output_dir"])
    relink_cfg = cfg.get("relink", {})

    relinked_csv = output_dir / "relinked_detections.csv"
    pour_json = output_dir / "pour_events.json"
    if relinked_csv.exists() and pour_json.exists() and not force:
        import json

        logger.info("relinked_detections.csv already exists at %s, skipping", relinked_csv)
        cfg["_pour_events"] = json.loads(pour_json.read_text())
        return

    raw_csv = output_dir / "raw_detections.csv"
    if not raw_csv.exists():
        logger.error("%s not found. Run yolo_tracking stage first.", raw_csv)
        sys.exit(1)

    record_range = relink_cfg.get("record_range")
    if record_range:
        record_range = tuple(record_range)

    _csv, pour_events = relink_mod.run_relink(
        input_csv=raw_csv,
        output_dir=output_dir,
        overlap_threshold=relink_cfg.get("overlap_threshold", 15),
        min_track_dets=relink_cfg.get("min_track_dets", 2),
        max_interp_gap=relink_cfg.get("max_interp_gap", 10),
        min_pour_frames=relink_cfg.get("min_pour_frames", 30),
        movement_threshold=relink_cfg.get("movement_threshold", 5.0),
        stationary_ratio=relink_cfg.get("stationary_ratio", 0.8),
        stationary_px=relink_cfg.get("stationary_px", 10.0),
        video_padding=relink_cfg.get("video_padding", 2.0),
        video_path=video_path,
        record_range=record_range,
        save_video=relink_cfg.get("save_video", False),
    )
    cfg["_pour_events"] = pour_events


# ── Stage 4: SAM3 Tap Handle Tracking ────────────────────────────────────────


def stage_sam3_tap_tracking(cfg: dict, interactive: bool = False, force: bool = False):
    """Run SAM3VideoPredictor on cropped video to track tap handles.

    Uses pour frame ranges from the relink stage (if available) so the SAM
    output video only covers the movement periods, matching the YOLO video.
    """
    import json as _json

    from backend.ml.approach_yolo.sam3_tracking import run_sam3_video_tracking

    video_path = Path(cfg["video_path"])
    output_dir = Path(cfg["output_dir"])
    sam3_cfg = cfg.get("sam3", {})

    centroids_csv = output_dir / "sam3_centroids.csv"
    if centroids_csv.exists() and not force:
        logger.info("sam3_centroids.csv already exists at %s, skipping", centroids_csv)
        return

    tap_roi, _ = _get_roi(cfg)

    # Resolve tap bboxes (normally set by stage_roi_selection already)
    tap_bboxes = sam3_cfg.get("tap_bboxes")
    object_labels = sam3_cfg.get("object_labels", ["TAP_A", "TAP_B"])

    if tap_bboxes is None:
        roi_json = output_dir / "tap_roi.json"
        data = load_roi_config(roi_json)
        if "sam3_tap_bboxes" in data:
            tap_bboxes = data["sam3_tap_bboxes"]
            logger.info("SAM3 tap bboxes loaded from %s", roi_json)
        elif interactive:
            cap = cv2.VideoCapture(str(video_path))
            ret, frame_0 = cap.read()
            cap.release()
            crop = crop_normalized(frame_0, tap_roi)
            tap_bboxes = select_tap_bboxes_interactive(crop, object_labels)
            data["sam3_tap_bboxes"] = tap_bboxes
            roi_json.write_text(_json.dumps(data, indent=2))
            logger.info("SAM3 tap bboxes saved to %s", roi_json)
        else:
            logger.error("No SAM3 tap bboxes found. Run with --interactive or set in config.")
            sys.exit(1)

    # Load pour frame ranges from relink stage (for selective video output)
    frame_ranges = None
    ranges_path = output_dir / "pour_frame_ranges.json"
    if ranges_path.exists():
        ranges_data = _json.loads(ranges_path.read_text())
        frame_ranges = [(r["start_frame"], r["end_frame"]) for r in ranges_data]
        logger.info("SAM3 will output video for %d pour segment(s)", len(frame_ranges))

    run_sam3_video_tracking(
        video_path=video_path,
        output_dir=output_dir,
        tap_roi=tap_roi,
        tap_bboxes=tap_bboxes,
        object_labels=object_labels,
        colors=sam3_cfg.get("colors", [[0, 255, 0], [0, 128, 255]]),
        model_path=sam3_cfg.get("model", "data/models/sam3.pt"),
        max_frames=sam3_cfg.get("max_frames"),
        frame_skip=sam3_cfg.get("frame_skip", 5),
        save_snapshot_every=sam3_cfg.get("save_snapshot_every", 50),
        half=sam3_cfg.get("half", True),
        frame_ranges=frame_ranges,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_roi(cfg: dict) -> tuple[tuple, tuple | None]:
    """Get resolved ROI from in-memory config or tap_roi.json."""
    if "_tap_roi" in cfg:
        return cfg["_tap_roi"], cfg.get("_tap_divider")

    output_dir = Path(cfg["output_dir"])
    roi_json = output_dir / "tap_roi.json"
    data = load_roi_config(roi_json)
    if "tap_roi" not in data:
        logger.error("No ROI found. Run roi_selection stage first.")
        sys.exit(1)
    tap_roi = tuple(data["tap_roi"])
    tap_divider = tuple(data["tap_divider"]) if "tap_divider" in data else None
    cfg["_tap_roi"] = tap_roi
    cfg["_tap_divider"] = tap_divider
    return tap_roi, tap_divider


# ── Tap Assignment ────────────────────────────────────────────────────────────


def _assign_pours_to_taps(pour_json: Path, centroids_csv: Path) -> list[dict] | None:
    """Assign each pour event to TAP_A or TAP_B based on SAM3 centroid movement.

    During a pour, the corresponding tap handle moves (centroid Y changes).
    For each pour's frame range, we compute the centroid-Y standard deviation
    for each tap — the tap with more movement is the one that poured.
    """
    import numpy as np
    import pandas as pd

    if not pour_json.exists() or not centroids_csv.exists():
        return None

    pours = json.loads(pour_json.read_text())
    if not pours:
        return []

    df = pd.read_csv(centroids_csv)
    if df.empty:
        return [{**p, "tap": "UNKNOWN", "tap_movement": {}} for p in pours]

    labels = sorted(df["label"].unique())
    assigned = []

    for p in pours:
        f_start, f_end = p["frame_start"], p["frame_end"]
        window = df[(df["frame"] >= f_start) & (df["frame"] <= f_end)]

        movements = {}
        for label in labels:
            sub = window[window["label"] == label]
            if len(sub) >= 2:
                movements[label] = round(float(np.std(sub["centroid_y"])), 2)
            else:
                movements[label] = 0.0

        if movements:
            best_tap = max(movements, key=movements.get)
            # Only assign if there's meaningful movement difference
            if movements[best_tap] > 0:
                tap = best_tap
            else:
                tap = "UNKNOWN"
        else:
            tap = "UNKNOWN"

        assigned.append(
            {
                **p,
                "tap": tap,
                "tap_movement": movements,
            }
        )

    return assigned


# ── Main ──────────────────────────────────────────────────────────────────────

STAGE_FUNCS = {
    "roi_selection": stage_roi_selection,
    "yolo_tracking": stage_yolo_tracking,
    "relink": stage_relink,
    "sam3_tap_tracking": stage_sam3_tap_tracking,
}


def main():
    p = argparse.ArgumentParser(
        description="Unified beer tap counting pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path, help="Path to pipeline.yaml config file.")
    p.add_argument("--stage", choices=STAGES, help="Run only this stage.")
    p.add_argument(
        "--interactive", action="store_true", help="Force interactive ROI/bbox selection."
    )
    p.add_argument(
        "--force", action="store_true", help="Re-run stages even if outputs already exist."
    )
    args = p.parse_args()

    cfg = load_config(args.config)

    # Override interactive flag
    if args.interactive:
        roi_cfg = cfg.setdefault("roi", {})
        roi_cfg["tap_roi"] = None
        roi_cfg["tap_divider"] = None
        sam3_cfg = cfg.setdefault("sam3", {})
        sam3_cfg["tap_bboxes"] = None

    # Video metadata
    video_path = Path(cfg["video_path"])
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = video_total_frames / video_fps if video_fps else 0
    cap.release()

    print(f"\n{'=' * 60}")
    print(f"  VIDEO: {video_path.name}")
    print(
        f"  {video_fps:.0f} fps | {video_total_frames} frames | "
        f"{video_duration:.1f}s ({video_duration / 60:.1f} min)"
    )
    print(f"{'=' * 60}")

    stages_enabled = cfg.get("stages", {})
    stage_times: dict[str, float] = {}
    pipeline_t0 = time.time()

    for stage_name in STAGES:
        if args.stage and args.stage != stage_name:
            continue
        if not stages_enabled.get(stage_name, True):
            print(f"\n{'=' * 60}")
            print(f"  SKIP: {stage_name} (disabled in config)")
            print(f"{'=' * 60}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  STAGE: {stage_name}")
        print(f"{'=' * 60}")

        func = STAGE_FUNCS[stage_name]
        t0 = time.time()

        if stage_name in ("roi_selection", "sam3_tap_tracking"):
            func(cfg, interactive=args.interactive, force=args.force)
        else:
            func(cfg, force=args.force)

        elapsed = time.time() - t0
        stage_times[stage_name] = elapsed
        print(f"  [{stage_name}] completed in {elapsed:.1f}s")

    total_elapsed = time.time() - pipeline_t0

    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE")
    print(f"  Outputs in: {cfg['output_dir']}")
    print(f"{'=' * 60}")
    print(f"\n  Video: {video_path.name} ({video_duration:.1f}s / {video_duration / 60:.1f} min)")
    print("  Stage timings:")
    for name, elapsed in stage_times.items():
        print(f"    {name:25s} {elapsed:8.1f}s")
    print(f"    {'─' * 35}")
    print(f"    {'TOTAL':25s} {total_elapsed:8.1f}s")
    print()

    # ── Assign pours to taps & build final summary ────────────────────────
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"

    merged: dict = {}
    if summary_path.exists():
        try:
            merged = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            merged = {}

    # Pipeline execution metadata
    merged["pipeline_execution"] = {
        "video_path": str(video_path),
        "video_fps": float(video_fps) if video_fps else 0.0,
        "video_total_frames": video_total_frames,
        "video_duration_s": round(video_duration, 3),
        "stage_seconds": {k: round(v, 3) for k, v in stage_times.items()},
        "total_elapsed_s": round(total_elapsed, 3),
        "single_stage": args.stage,
    }

    # Tap assignment: correlate pour events with SAM3 centroid movement
    pour_json = output_dir / "pour_events.json"
    centroids_csv = output_dir / "sam3_centroids.csv"
    assigned_pours = _assign_pours_to_taps(pour_json, centroids_csv)

    if assigned_pours is not None:
        tap_a = sum(1 for p in assigned_pours if p["tap"] == "TAP_A")
        tap_b = sum(1 for p in assigned_pours if p["tap"] == "TAP_B")
        unknown = sum(1 for p in assigned_pours if p["tap"] == "UNKNOWN")
        total_beers = tap_a + tap_b + unknown

        merged["results"] = {
            "tap_a_beers": tap_a,
            "tap_b_beers": tap_b,
            "unknown_tap": unknown,
            "total_beers": total_beers,
            "pour_events": assigned_pours,
        }

        # Save enriched pour events back
        enriched_path = output_dir / "pour_events_assigned.json"
        enriched_path.write_text(json.dumps(assigned_pours, indent=2))

        print(f"\n{'=' * 60}")
        print("  FINAL RESULTS")
        print(f"{'=' * 60}")
        print(f"    TAP A:   {tap_a} beer(s)")
        print(f"    TAP B:   {tap_b} beer(s)")
        if unknown:
            print(f"    UNKNOWN: {unknown} (no SAM3 data for that range)")
        print(f"    TOTAL:   {total_beers} beer(s)")
        print(f"{'=' * 60}")
    else:
        print("\n  (No pour events or SAM3 data — skipping tap assignment)")

    summary_path.write_text(json.dumps(merged, indent=2))
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
