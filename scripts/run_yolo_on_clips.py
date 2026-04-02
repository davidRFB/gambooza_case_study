"""
Run the YOLO+SAM3 pipeline on activity clips extracted by run_simple.py.

Uses the same code path as the backend (processor.py): builds a temp YAML
config with absolute paths and pre-creates tap_roi.json, then runs
YOLODetector — identical to what the web app does.

Usage:
  # Interactive ROI selection on first clip, then batch all:
  python scripts/run_yolo_on_clips.py \
      --clips-dir results/test_ROI_FULLAREA_2h/clips \
      --output results/test_YOLO_SAM_2h \
      --interactive

  # Run on all clips with an existing ROI config:
  python scripts/run_yolo_on_clips.py \
      --clips-dir results/test_ROI_FULLAREA_2h/clips \
      --roi-config data/roi_configs/CHATTER_CAM1.json \
      --output results/test_YOLO_SAM_2h

  # Force re-run (ignore existing outputs):
  python scripts/run_yolo_on_clips.py \
      --clips-dir results/test_ROI_FULLAREA_2h/clips \
      --roi-config data/roi_configs/CHATTER_CAM1.json \
      --output results/test_YOLO_SAM_2h \
      --force
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import yaml

from backend.ml.common import crop_normalized, select_roi_interactive, select_tap_bboxes_interactive

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def select_roi_interactive_from_video(video_path: Path, save_path: Path) -> dict:
    """Run 3-step interactive ROI selection on a video's first frame.

    Steps:
      1. Select crop region on full frame
      2. Select TAP_A bbox on cropped frame
      3. Select TAP_B bbox on cropped frame

    Saves and returns ROI config in the same format as data/roi_configs/*.json.
    """
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: Could not read first frame from {video_path}")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"  First frame: {w}x{h}")

    # Step 1: crop region
    print("\n  Step 1/3: Select CROP REGION on full frame, then close window.")
    tap_roi = select_roi_interactive(frame)
    print(f"  Crop ROI: {[round(v, 4) for v in tap_roi]}")

    # Step 2+3: TAP_A and TAP_B bboxes on cropped frame
    crop = crop_normalized(frame, tap_roi)
    print("\n  Step 2/3: Select TAP_A handle on cropped frame, then close window.")
    print("  Step 3/3: Select TAP_B handle on cropped frame, then close window.")
    tap_bboxes = select_tap_bboxes_interactive(crop, ["TAP_A", "TAP_B"])

    # Build config
    roi = {
        "yolo": {
            "tap_roi": list(tap_roi),
            "sam3_tap_bboxes": tap_bboxes,
        }
    }

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(roi, indent=2))
    print(f"\n  ROI config saved to {save_path}")

    # Save annotated image
    annotated = frame.copy()
    x1, y1, x2, y2 = tap_roi
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 165, 255), 2)  # orange crop
    cv2.putText(annotated, "CROP", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # Draw tap bboxes on the crop region of the annotated image
    crop_h, crop_w = crop.shape[:2]
    for bbox, label, color in zip(
        tap_bboxes,
        ["TAP A", "TAP B"],
        [(255, 0, 0), (0, 255, 0)],
    ):
        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        # Offset to full-frame coordinates
        cv2.rectangle(annotated, (px1 + bx1, py1 + by1), (px1 + bx2, py1 + by2), color, 2)
        cv2.putText(
            annotated, label, (px1 + bx1, py1 + by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    annotated_path = save_path.parent / "roi_annotated.png"
    cv2.imwrite(str(annotated_path), annotated)
    print(f"  Annotated frame saved to {annotated_path}")

    return roi


def build_config(clip_path: Path, output_dir: Path, roi: dict, base_config: dict) -> dict:
    """Build a pipeline config for a single clip, matching processor.py logic."""
    cfg = dict(base_config)  # shallow copy

    # Override video path and output dir (absolute)
    cfg["video_path"] = str(clip_path.resolve())
    cfg["output_dir"] = str(output_dir.resolve())

    # Override ROI and SAM3 bboxes from roi config
    yolo_roi = roi["yolo"]
    cfg.setdefault("roi", {})["tap_roi"] = yolo_roi["tap_roi"]
    cfg.setdefault("sam3", {})["tap_bboxes"] = yolo_roi["sam3_tap_bboxes"]

    # Set preview_second to 0 (clips are short) and save raw YOLO video
    cfg.setdefault("yolo", {})["preview_second"] = 0
    cfg["yolo"]["save_video"] = True
    cfg.setdefault("relink", {})["save_video"] = True

    # Resolve all relative paths to absolute (same as processor.py)
    yolo_cfg = cfg.get("yolo", {})
    if "tracker" in yolo_cfg and not Path(yolo_cfg["tracker"]).is_absolute():
        yolo_cfg["tracker"] = str(PROJECT_ROOT / yolo_cfg["tracker"])
    if "model" in yolo_cfg and not Path(yolo_cfg["model"]).is_absolute():
        yolo_cfg["model"] = str(PROJECT_ROOT / yolo_cfg["model"])
    sam3_cfg = cfg.get("sam3", {})
    if "model" in sam3_cfg and not Path(sam3_cfg["model"]).is_absolute():
        sam3_cfg["model"] = str(PROJECT_ROOT / sam3_cfg["model"])

    return cfg


def run_clip(clip_path: Path, output_dir: Path, cfg: dict, force: bool) -> dict:
    """Run YOLODetector on a single clip. Returns result summary dict."""
    from backend.ml.approach_yolo.detector import YOLODetector

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create tap_roi.json so ROI stage skips interactive mode
    yolo_roi = cfg.get("roi", {}).get("tap_roi")
    sam3_bboxes = cfg.get("sam3", {}).get("tap_bboxes")
    tap_roi_json = output_dir / "tap_roi.json"
    tap_roi_json.write_text(
        json.dumps(
            {"tap_roi": yolo_roi, "sam3_tap_bboxes": sam3_bboxes},
            indent=2,
        )
    )

    # Write temp config file (convert numpy types to native Python for yaml)
    cfg_clean = json.loads(json.dumps(cfg, default=lambda x: float(x)))
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="pipeline_clip_",
        delete=False,
    )
    yaml.dump(cfg_clean, tmp)
    tmp.close()

    try:
        detector = YOLODetector(config_path=tmp.name)
        result = detector.run(interactive=False, force=force)
        return result.to_dict()
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(
        description="Run YOLO+SAM3 pipeline on activity clips (same as backend).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--clips-dir",
        required=True,
        type=Path,
        help="Directory containing clip .mp4 files",
    )
    p.add_argument(
        "--roi-config",
        type=Path,
        default=None,
        help="Path to ROI config JSON (e.g. data/roi_configs/CHATTER_CAM1.json)",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Base output directory (each clip gets a subdirectory)",
    )
    p.add_argument(
        "--base-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "pipeline.yaml",
        help="Base pipeline YAML config to use as template",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively select ROI on first clip's first frame (3-step: crop, TAP_A, TAP_B)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run stages even if outputs already exist",
    )
    args = p.parse_args()

    # Validate inputs
    if not args.clips_dir.exists():
        print(f"ERROR: Clips directory not found: {args.clips_dir}")
        sys.exit(1)
    if not args.interactive and not args.roi_config:
        print("ERROR: Must provide --roi-config or --interactive")
        sys.exit(1)
    if args.roi_config and not args.roi_config.exists():
        print(f"ERROR: ROI config not found: {args.roi_config}")
        sys.exit(1)

    # Load base pipeline config
    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)

    # Find clips
    clips = sorted(args.clips_dir.glob("*.mp4"))
    if not clips:
        print(f"No .mp4 files found in {args.clips_dir}")
        sys.exit(1)

    # Resolve ROI config
    if args.interactive:
        args.output.mkdir(parents=True, exist_ok=True)
        roi_save_path = args.output / "roi_config.json"
        print(f"\n  Interactive ROI selection using first clip: {clips[0].name}")
        roi = select_roi_interactive_from_video(clips[0], roi_save_path)
    else:
        roi = json.loads(args.roi_config.read_text())
        if "yolo" not in roi:
            print("ERROR: ROI config missing 'yolo' section")
            sys.exit(1)

    roi_label = args.roi_config or "interactive"
    print(f"\n{'=' * 60}")
    print("  YOLO+SAM3 Batch Processing")
    print(f"  Clips: {len(clips)} from {args.clips_dir}")
    print(f"  ROI config: {roi_label}")
    print(f"  Output: {args.output}")
    print(f"{'=' * 60}\n")

    args.output.mkdir(parents=True, exist_ok=True)
    all_results = []
    total_t0 = time.time()

    for i, clip in enumerate(clips):
        clip_output = args.output / clip.stem
        print(f"\n{'=' * 60}")
        print(f"  [{i + 1}/{len(clips)}] {clip.name}")
        print(f"  Output: {clip_output}")
        print(f"{'=' * 60}")

        cfg = build_config(clip, clip_output, roi, base_config)
        t0 = time.time()

        try:
            result = run_clip(clip, clip_output, cfg, args.force)
            elapsed = time.time() - t0
            result["clip"] = clip.name
            result["elapsed_s"] = round(elapsed, 1)
            all_results.append(result)

            print(
                f"\n  Result: Tap A={result['tap_a_count']}, "
                f"Tap B={result['tap_b_count']}, "
                f"Unknown={result['unknown_count']}, "
                f"Total={result['total']} ({elapsed:.1f}s)"
            )

        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  ERROR: {e} ({elapsed:.1f}s)")
            all_results.append(
                {
                    "clip": clip.name,
                    "error": str(e),
                    "elapsed_s": round(elapsed, 1),
                }
            )

    total_elapsed = time.time() - total_t0

    # Save batch summary
    summary = {
        "clips_processed": len(clips),
        "total_elapsed_s": round(total_elapsed, 1),
        "roi_config": str(args.roi_config),
        "results": all_results,
    }
    summary_path = args.output / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Print final summary
    print(f"\n{'=' * 60}")
    print(f"  BATCH COMPLETE — {len(clips)} clips in {total_elapsed:.1f}s")
    print(f"{'=' * 60}")
    total_a = sum(r.get("tap_a_count", 0) for r in all_results)
    total_b = sum(r.get("tap_b_count", 0) for r in all_results)
    total_u = sum(r.get("unknown_count", 0) for r in all_results)
    errors = sum(1 for r in all_results if "error" in r)
    print(f"  Tap A: {total_a}  |  Tap B: {total_b}  |  Unknown: {total_u}")
    if errors:
        print(f"  Errors: {errors}")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
