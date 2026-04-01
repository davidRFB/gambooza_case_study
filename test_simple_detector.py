"""
Test SimpleDetector on a video.

Usage:
  # Interactive — draw tap A and tap B handle ROIs on first frame:
  python test_simple_detector.py --video data/videos/cerveza2.mp4 --interactive

  # Load ROIs from a previous run:
  python test_simple_detector.py --video data/videos/cerveza2.mp4 \
      --roi-json results/simple_test_cerveza2/simple_roi.json

  # Custom output dir:
  python test_simple_detector.py --video data/videos/cerveza2.mp4 \
      --roi-json results/simple_test_cerveza2/simple_roi.json \
      --output results/simple_test_cerveza2_v2
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "backend/ml/approach_simple")

from detector import SimpleDetector, plot_simple_results


# ---------------------------------------------------------------------------
# Interactive ROI selection
# ---------------------------------------------------------------------------

def select_roi_interactive(frame_bgr: np.ndarray, title: str) -> tuple:
    """Show frame, let user drag a rectangle, return normalised ROI."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    coords = {}

    def on_select(eclick, erelease):
        coords["x0"] = eclick.xdata
        coords["y0"] = eclick.ydata
        coords["x1"] = erelease.xdata
        coords["y1"] = erelease.ydata

    selector = RectangleSelector(
        ax, on_select, useblit=True, button=[1], interactive=True,
        props=dict(facecolor="cyan", edgecolor="red", alpha=0.3, linewidth=2),
    )
    plt.show()

    if not coords:
        print("No rectangle drawn. Exiting.")
        sys.exit(1)

    x0, y0, x1, y1 = coords["x0"], coords["y0"], coords["x1"], coords["y1"]
    return (min(x0, x1) / w, min(y0, y1) / h,
            max(x0, x1) / w, max(y0, y1) / h)


def load_or_select_rois(video_path: Path, roi_json: Path | None,
                        interactive: bool) -> dict:
    """Resolve tap_a_roi and tap_b_roi from file or interactive."""
    data = {}

    if roi_json and roi_json.exists():
        data = json.loads(roi_json.read_text())
        if not interactive and "tap_a_roi" in data and "tap_b_roi" in data:
            print(f"Loaded ROIs from {roi_json}")
            return data

    # Read first frame for interactive selection
    cap = cv2.VideoCapture(str(video_path))
    ret, frame_0 = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: Could not read {video_path}")
        sys.exit(1)

    for key, label in [("tap_a_roi", "TAP A handle"),
                       ("tap_b_roi", "TAP B handle")]:
        if key in data and not interactive:
            print(f"  {label}: {data[key]}")
        else:
            roi = select_roi_interactive(
                frame_0,
                f"Drag rectangle over [{label}] (small region on the handle), "
                f"then close window."
            )
            data[key] = list(roi)
            print(f"  {label} selected: {[round(v, 4) for v in roi]}")

    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Test SimpleDetector")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--roi-json", type=Path, default=None,
                   help="Path to simple_roi.json from a previous run")
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory (default: results/simple_test_<video>)")
    p.add_argument("--interactive", action="store_true",
                   help="Force interactive ROI selection")
    p.add_argument("--sample-every", type=int, default=3)
    p.add_argument("--on-threshold", type=float, default=0.05)
    p.add_argument("--min-on-frames", type=int, default=10)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=3000)
    args = p.parse_args()

    out = args.output or Path(f"results/simple_test_{args.video.stem}")
    out.mkdir(parents=True, exist_ok=True)

    # Resolve ROIs
    rois = load_or_select_rois(args.video, args.roi_json, args.interactive)

    # Save ROIs for re-use
    roi_save = out / "simple_roi.json"
    roi_save.write_text(json.dumps(rois, indent=2))
    print(f"ROIs saved to {roi_save}")

    # Run
    detector = SimpleDetector(
        tap_a_roi=tuple(rois["tap_a_roi"]),
        tap_b_roi=tuple(rois["tap_b_roi"]),
        sample_every=args.sample_every,
        on_threshold=args.on_threshold,
        min_on_frames=args.min_on_frames,
        n_workers=args.n_workers,
        progress_every=args.progress_every,
    )

    result = detector.run(args.video)

    # Print events
    for e in result.events:
        print(f"  Tap {e.tap}: {e.timestamp_start:.1f}s – {e.timestamp_end:.1f}s "
              f"({e.duration_s:.1f}s, peak={e.peak_activity:.3f})")

    # Save
    (out / "summary.json").write_text(json.dumps(result.to_dict(), indent=2))
    plot_simple_results(result, out, on_threshold=args.on_threshold)
    print(f"\nAll results saved to {out}")


if __name__ == "__main__":
    main()
