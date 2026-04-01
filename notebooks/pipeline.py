"""
Unified beer tap counting pipeline.

Orchestrates four stages:
  1. ROI selection — crop area + A/B divider
  2. YOLO tracking — detect and track cups with YOLO-World + BoT-SORT
  3. Relink — merge fragmented tracks via temporal co-existence constraints
  4. SAM3 tap tracking — segment tap handles and track centroid positions

Usage
-----
# Full pipeline (interactive ROI + bbox selection on first run):
python notebooks/pipeline.py --config config/pipeline.yaml --interactive

# Re-run with saved coordinates:
python notebooks/pipeline.py --config config/pipeline.yaml

# Run a single stage:
python notebooks/pipeline.py --config config/pipeline.yaml --stage relink

# Force re-run even if outputs exist:
python notebooks/pipeline.py --config config/pipeline.yaml --force
"""

import argparse
import sys
from pathlib import Path

import yaml

# Ensure notebooks/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from common import (
    resolve_roi, resolve_divider, save_roi_config, load_roi_config,
    crop_normalized, select_tap_bboxes_interactive,
)


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
    """Resolve ROI, tap divider, and SAM3 tap bboxes. Saves to tap_roi.json.

    When *interactive* is True the user gets three windows in sequence:
      1. Drag-rectangle for the overall TAP area
      2. Two-click divider line (shown for verification)
      3. One drag-rectangle per SAM3 tap label (TAP_A, TAP_B)
    """
    import json
    import shutil

    video_path = Path(cfg["video_path"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_json = output_dir / "tap_roi.json"
    roi_cfg = cfg.get("roi", {})
    sam3_cfg = cfg.get("sam3", {})

    config_roi = roi_cfg.get("tap_roi")
    config_divider = roi_cfg.get("tap_divider")

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
                print(f"ROI loaded from external file: {ext_path}")
                return
        else:
            print(f"WARNING: roi_json path not found: {ext_path}, falling back to defaults.")

    # Skip if already done and not forced
    if roi_json.exists() and not force and not interactive:
        data = load_roi_config(roi_json)
        if "tap_roi" in data and "tap_divider" in data:
            cfg["_tap_roi"] = tuple(data["tap_roi"])
            cfg["_tap_divider"] = tuple(data["tap_divider"])
            if "sam3_tap_bboxes" in data:
                cfg.setdefault("sam3", {})["tap_bboxes"] = data["sam3_tap_bboxes"]
            print(f"ROI already resolved in {roi_json}, skipping.")
            return

    # Read first frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame_0 = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: Could not read {video_path}")
        sys.exit(1)

    # -- Window 1: TAP area ROI ---------------------------------------------
    tap_roi = resolve_roi(config_roi, roi_json, frame_0, interactive=interactive)

    # -- Window 2: A|B divider line -----------------------------------------
    tap_divider = resolve_divider(config_divider, roi_json, tap_roi, frame_0,
                                  interactive=interactive)

    # -- Window 3: SAM3 tap bboxes (one drag-rectangle per label) -----------
    object_labels = sam3_cfg.get("object_labels", ["TAP_A", "TAP_B"])
    tap_bboxes = sam3_cfg.get("tap_bboxes")

    if interactive or tap_bboxes is None:
        existing = load_roi_config(roi_json)
        if "sam3_tap_bboxes" in existing and not interactive:
            tap_bboxes = existing["sam3_tap_bboxes"]
            print(f"SAM3 tap bboxes loaded from {roi_json}")
        else:
            print("Opening SAM3 tap bbox selectors on cropped frame ...")
            crop = crop_normalized(frame_0, tap_roi)
            tap_bboxes = select_tap_bboxes_interactive(crop, object_labels)

    # -- Persist everything -------------------------------------------------
    save_data = {"tap_roi": list(tap_roi)}
    if tap_divider:
        save_data["tap_divider"] = list(tap_divider)
    if tap_bboxes:
        save_data["sam3_tap_bboxes"] = tap_bboxes
    roi_json.write_text(json.dumps(save_data, indent=2))
    print(f"ROI config saved to {roi_json}")

    cfg["_tap_roi"] = tap_roi
    cfg["_tap_divider"] = tap_divider
    if tap_bboxes:
        cfg.setdefault("sam3", {})["tap_bboxes"] = tap_bboxes


# ── Stage 3a: YOLO Tracking ──────────────────────────────────────────────────

def stage_yolo_tracking(cfg: dict, force: bool = False):
    """Run YOLO-World tracking on the cropped video."""
    from importlib import import_module
    yolo_track = import_module("03_YOLO_track")

    video_path = Path(cfg["video_path"])
    output_dir = Path(cfg["output_dir"])
    yolo_cfg = cfg.get("yolo", {})

    raw_csv = output_dir / "raw_detections.csv"
    if raw_csv.exists() and not force:
        print(f"raw_detections.csv already exists at {raw_csv}, skipping.")
        return

    tap_roi, tap_divider = _get_roi(cfg)

    record_range = yolo_cfg.get("record_range")
    if record_range:
        record_range = tuple(record_range)

    yolo_track.run_yolo_tracking(
        video_path=video_path,
        output_dir=output_dir,
        tap_roi=tap_roi,
        tap_divider=tap_divider,
        model_name=yolo_cfg.get("model", "yolov8x-worldv2.pt"),
        classes=yolo_cfg.get("classes", ["cup", "person"]),
        sample_every=yolo_cfg.get("sample_every", 1),
        conf_threshold=yolo_cfg.get("conf_threshold", 0.25),
        tracker=yolo_cfg.get("tracker", "config/botsort.yaml"),
        movement_threshold=yolo_cfg.get("movement_threshold", 5.0),
        merge_gap=yolo_cfg.get("merge_gap", 5.0),
        preview_second=yolo_cfg.get("preview_second", 60.0),
        record_range=record_range,
    )


# ── Stage 3b: Relink ─────────────────────────────────────────────────────────

def stage_relink(cfg: dict, force: bool = False):
    """Relink fragmented cup tracks."""
    from importlib import import_module
    relink_mod = import_module("05_relink_coexistence")

    video_path = Path(cfg["video_path"])
    output_dir = Path(cfg["output_dir"])
    relink_cfg = cfg.get("relink", {})

    relinked_csv = output_dir / "relinked_detections.csv"
    if relinked_csv.exists() and not force:
        print(f"relinked_detections.csv already exists at {relinked_csv}, skipping.")
        return

    raw_csv = output_dir / "raw_detections.csv"
    if not raw_csv.exists():
        print(f"ERROR: {raw_csv} not found. Run yolo_tracking stage first.")
        sys.exit(1)

    record_range = relink_cfg.get("record_range")
    if record_range:
        record_range = tuple(record_range)

    relink_mod.run_relink(
        input_csv=raw_csv,
        output_dir=output_dir,
        overlap_threshold=relink_cfg.get("overlap_threshold", 15),
        min_track_dets=relink_cfg.get("min_track_dets", 2),
        max_interp_gap=relink_cfg.get("max_interp_gap", 10),
        video_path=video_path if record_range else None,
        record_range=record_range,
    )


# ── Stage 4: SAM3 Tap Handle Tracking ────────────────────────────────────────

def stage_sam3_tap_tracking(cfg: dict, interactive: bool = False, force: bool = False):
    """Run SAM3VideoPredictor on cropped video to track tap handles."""
    from sam3_tracking import run_sam3_video_tracking

    video_path = Path(cfg["video_path"])
    output_dir = Path(cfg["output_dir"])
    sam3_cfg = cfg.get("sam3", {})

    centroids_csv = output_dir / "sam3_centroids.csv"
    if centroids_csv.exists() and not force:
        print(f"sam3_centroids.csv already exists at {centroids_csv}, skipping.")
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
            print(f"SAM3 tap bboxes loaded from {roi_json}")
        elif interactive:
            cap = cv2.VideoCapture(str(video_path))
            ret, frame_0 = cap.read()
            cap.release()
            crop = crop_normalized(frame_0, tap_roi)
            tap_bboxes = select_tap_bboxes_interactive(crop, object_labels)
            data["sam3_tap_bboxes"] = tap_bboxes
            roi_json.write_text(__import__("json").dumps(data, indent=2))
            print(f"SAM3 tap bboxes saved to {roi_json}")
        else:
            print("ERROR: No SAM3 tap bboxes found. Run with --interactive or set in config.")
            sys.exit(1)

    run_sam3_video_tracking(
        video_path=video_path,
        output_dir=output_dir,
        tap_roi=tap_roi,
        tap_bboxes=tap_bboxes,
        object_labels=object_labels,
        colors=sam3_cfg.get("colors", [[0, 255, 0], [0, 128, 255]]),
        model_path=sam3_cfg.get("model", "sam3.pt"),
        max_frames=sam3_cfg.get("max_frames"),
        frame_skip=sam3_cfg.get("frame_skip", 5),
        save_snapshot_every=sam3_cfg.get("save_snapshot_every", 50),
        half=sam3_cfg.get("half", True),
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
        print(f"ERROR: No ROI found. Run roi_selection stage first.")
        sys.exit(1)
    tap_roi = tuple(data["tap_roi"])
    tap_divider = tuple(data["tap_divider"]) if "tap_divider" in data else None
    cfg["_tap_roi"] = tap_roi
    cfg["_tap_divider"] = tap_divider
    return tap_roi, tap_divider


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
    p.add_argument("--config", required=True, type=Path,
                   help="Path to pipeline.yaml config file.")
    p.add_argument("--stage", choices=STAGES,
                   help="Run only this stage.")
    p.add_argument("--interactive", action="store_true",
                   help="Force interactive ROI/bbox selection.")
    p.add_argument("--force", action="store_true",
                   help="Re-run stages even if outputs already exist.")
    args = p.parse_args()

    cfg = load_config(args.config)

    # Override interactive flag
    if args.interactive:
        roi_cfg = cfg.setdefault("roi", {})
        roi_cfg["tap_roi"] = None
        roi_cfg["tap_divider"] = None
        sam3_cfg = cfg.setdefault("sam3", {})
        sam3_cfg["tap_bboxes"] = None

    stages_enabled = cfg.get("stages", {})

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

        # Pass interactive/force to stages that support it
        if stage_name in ("roi_selection", "sam3_tap_tracking"):
            func(cfg, interactive=args.interactive, force=args.force)
        else:
            func(cfg, force=args.force)

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Outputs in: {cfg['output_dir']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
