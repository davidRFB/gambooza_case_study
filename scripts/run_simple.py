"""
Run SimpleDetector on a video.

Usage:
  # Interactive — draw tap A and tap B handle ROIs on first frame:
  python scripts/run_simple.py --video data/videos/cerveza2.mp4 --interactive

  # Load ROIs from a previous run:
  python scripts/run_simple.py --video data/videos/cerveza2.mp4 \
      --roi-json results/simple_test_cerveza2/simple_roi.json

  # Custom output dir:
  python scripts/run_simple.py --video data/videos/cerveza2.mp4 \
      --roi-json results/simple_test_cerveza2/simple_roi.json \
      --output results/simple_test_cerveza2_v2
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from backend.ml.approach_simple.detector import SimpleDetector, plot_simple_results
from backend.ml.approach_simple.filtering import extract_clips, find_activity_windows
from backend.ml.common import select_roi_interactive

# ---------------------------------------------------------------------------
# ROI resolution
# ---------------------------------------------------------------------------


ROI_COLORS = [
    (255, 0, 0),  # BGR: blue
    (0, 255, 0),  # green
]


def _save_annotated_frame(frame: np.ndarray, rois: list[dict], save_path: Path):
    """Draw ROI rectangles on the full-res first frame and save as PNG."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for i, roi_info in enumerate(rois):
        x1, y1, x2, y2 = roi_info["roi"]
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        color = ROI_COLORS[i % len(ROI_COLORS)]
        label = roi_info["label"]
        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 3)
        cv2.putText(
            annotated,
            label,
            (px1, py1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
        )

    cv2.imwrite(str(save_path), annotated)
    print(f"  Annotated frame saved to {save_path}")


def _read_first_frame(video_path: Path) -> np.ndarray:
    """Read the first frame at full resolution."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: Could not read {video_path}")
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"  First frame: {w}x{h}")
    return frame


def _roi_definitions(num_rois: int) -> list[tuple[str, str]]:
    """Return (json_key, display_label) pairs for the requested number of ROIs."""
    if num_rois == 1:
        return [("roi_1", "Region 1")]
    return [("roi_1", "Region 1"), ("roi_2", "Region 2")]


def load_or_select_rois(
    video_path: Path,
    roi_json: Path | None,
    interactive: bool,
    num_rois: int = 2,
) -> dict:
    """Resolve ROIs from file or interactive selection.

    Returns dict with keys like "roi_1", "roi_2" plus legacy "tap_a_roi"/"tap_b_roi"
    for backward compatibility with SimpleDetector.
    """
    data = {}
    roi_defs = _roi_definitions(num_rois)
    roi_keys = [key for key, _ in roi_defs]

    if roi_json and roi_json.exists():
        data = json.loads(roi_json.read_text())
        # Check if all needed keys exist (support both new and legacy key names)
        has_new_keys = all(k in data for k in roi_keys)
        has_legacy = "tap_a_roi" in data and (num_rois == 1 or "tap_b_roi" in data)
        if not interactive and (has_new_keys or has_legacy):
            # Migrate legacy keys
            if has_legacy and not has_new_keys:
                data["roi_1"] = data["tap_a_roi"]
                if num_rois >= 2 and "tap_b_roi" in data:
                    data["roi_2"] = data["tap_b_roi"]
            print(f"Loaded ROIs from {roi_json}")
            return data

    # Read first frame at full resolution for interactive selection
    frame_0 = _read_first_frame(video_path)

    for key, label in roi_defs:
        if key in data and not interactive:
            print(f"  {label}: {data[key]}")
        else:
            print(f"  Drag rectangle over [{label}], then close window.")
            roi = select_roi_interactive(frame_0)
            data[key] = list(roi)
            print(f"  {label} selected: {[round(v, 4) for v in roi]}")

    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Run SimpleDetector")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument(
        "--roi-json", type=Path, default=None, help="Path to simple_roi.json from a previous run"
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: results/simple_<video>)",
    )
    p.add_argument("--interactive", action="store_true", help="Force interactive ROI selection")
    p.add_argument(
        "--num-rois",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of ROI areas to select (1 or 2, default: 2)",
    )
    p.add_argument("--sample-every", type=int, default=3)
    p.add_argument("--on-threshold", type=float, default=0.05)
    p.add_argument("--min-on-frames", type=int, default=10)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=3000)
    p.add_argument(
        "--extract-clips",
        action="store_true",
        help="Extract activity windows as short video clips",
    )
    p.add_argument(
        "--clip-padding",
        type=float,
        default=5.0,
        help="Seconds of padding before/after each activity window (default: 5)",
    )
    p.add_argument(
        "--clip-merge-gap",
        type=float,
        default=10.0,
        help="Merge windows closer than this many seconds (default: 10)",
    )
    p.add_argument(
        "--clip-threshold",
        type=float,
        default=None,
        help="Activity threshold for clip extraction (default: on-threshold / 2)",
    )
    args = p.parse_args()

    out = args.output or Path(f"results/simple_{args.video.stem}")
    out.mkdir(parents=True, exist_ok=True)

    # Resolve ROIs
    num_rois = args.num_rois
    rois = load_or_select_rois(args.video, args.roi_json, args.interactive, num_rois)

    # Save ROIs for re-use
    roi_save = out / "simple_roi.json"
    roi_save.write_text(json.dumps(rois, indent=2))
    print(f"ROIs saved to {roi_save}")

    # Save annotated first frame at full resolution
    frame_0 = _read_first_frame(args.video)
    roi_defs = _roi_definitions(num_rois)
    roi_annotations = [{"label": label, "roi": rois[key]} for key, label in roi_defs if key in rois]
    _save_annotated_frame(frame_0, roi_annotations, out / "roi_annotated.png")

    # Map ROIs to SimpleDetector (which expects tap_a_roi / tap_b_roi)
    roi_1 = tuple(rois["roi_1"])
    roi_2 = tuple(rois["roi_2"]) if num_rois >= 2 and "roi_2" in rois else roi_1

    # Run
    detector = SimpleDetector(
        tap_a_roi=roi_1,
        tap_b_roi=roi_2,
        sample_every=args.sample_every,
        on_threshold=args.on_threshold,
        min_on_frames=args.min_on_frames,
        n_workers=args.n_workers,
        progress_every=args.progress_every,
    )

    result = detector.run(args.video)

    # Rename signals for clarity
    signal_labels = {1: ["Region 1"], 2: ["Region 1", "Region 2"]}[num_rois]

    # Print events
    for e in result.events:
        region = signal_labels[0] if e.tap == "A" else signal_labels[-1]
        print(
            f"  {region}: {e.timestamp_start:.1f}s – {e.timestamp_end:.1f}s "
            f"({e.duration_s:.1f}s, peak={e.peak_activity:.3f})"
        )

    # Save
    (out / "summary.json").write_text(json.dumps(result.to_dict(), indent=2))

    # Save signals as CSV for further analysis
    import pandas as pd

    csv_data = {
        "frame": [result.times[i] * result.fps for i in range(len(result.times))],
        "time_s": result.times,
        "region_1": result.signals.get("Tap A", []),
    }
    if num_rois >= 2:
        csv_data["region_2"] = result.signals.get("Tap B", [])
    df = pd.DataFrame(csv_data)
    df["frame"] = df["frame"].astype(int)
    csv_path = out / "signals.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Signals CSV saved to {csv_path} ({len(df)} rows)")

    # Extract activity clips
    if args.extract_clips:
        clip_threshold = (
            args.clip_threshold if args.clip_threshold is not None else args.on_threshold / 2
        )
        print(
            f"\n  Finding activity windows (threshold={clip_threshold:.4f}, "
            f"padding={args.clip_padding}s, merge_gap={args.clip_merge_gap}s)..."
        )

        windows = find_activity_windows(
            times=result.times,
            signals=result.signals,
            threshold=clip_threshold,
            padding_s=args.clip_padding,
            merge_gap_s=args.clip_merge_gap,
            total_duration=result.duration_s,
        )

        # Save windows summary
        windows_path = out / "activity_windows.json"
        windows_path.write_text(json.dumps(windows, indent=2))
        total_clip_s = sum(w["duration_s"] for w in windows)
        print(
            f"  Found {len(windows)} activity windows ({total_clip_s:.0f}s total "
            f"out of {result.duration_s:.0f}s, {total_clip_s / result.duration_s * 100:.1f}% of video)"
        )

        if windows:
            print("\n  Extracting clips...")
            extract_clips(args.video, windows, out)
        else:
            print("  No activity windows found — try lowering --clip-threshold")

    plot_simple_results(result, out, on_threshold=args.on_threshold)
    print(f"\nAll results saved to {out}")


if __name__ == "__main__":
    main()
