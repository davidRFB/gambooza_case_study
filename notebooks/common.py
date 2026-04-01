"""
Shared utilities for the beer tap counting pipeline.

Extracts duplicated code from 03_YOLO_track.py, 05_relink_coexistence.py,
and 06_YOLOE_seg_track.py into a single importable module.

Interactive selectors use matplotlib (TkAgg) instead of cv2.imshow, since
the opencv-python-headless package has no GUI support.
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


# ---------------------------------------------------------------------------
# Interactive selectors (matplotlib-based)
# ---------------------------------------------------------------------------

def select_roi_interactive(frame: np.ndarray) -> tuple:
    """Show frame in matplotlib, let the user drag a rectangle, return normalised ROI.

    Drag to select the tap area, then close the window.
    """
    from matplotlib.widgets import RectangleSelector

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(rgb)
    ax.set_title("Drag a rectangle over the crop region, then close the window.")
    ax.axis("off")
    plt.tight_layout()

    coords = {}

    def on_select(eclick, erelease):
        coords["x0"] = eclick.xdata
        coords["y0"] = eclick.ydata
        coords["x1"] = erelease.xdata
        coords["y1"] = erelease.ydata

    selector = RectangleSelector(
        ax, on_select, useblit=True,
        button=[1], interactive=True,
        props=dict(facecolor="cyan", edgecolor="red", alpha=0.3, linewidth=2),
    )

    plt.show()

    if not coords:
        print("No rectangle drawn. Run again with --interactive.")
        sys.exit(1)

    x0, y0, x1, y1 = coords["x0"], coords["y0"], coords["x1"], coords["y1"]
    roi = (min(x0, x1) / w, min(y0, y1) / h,
           max(x0, x1) / w, max(y0, y1) / h)
    return roi


def select_divider_interactive(crop: np.ndarray) -> tuple:
    """Show cropped frame in matplotlib, let the user click two points (top/bottom)
    to define the A|B divider line.  After both clicks the line is drawn and
    the user can verify it before closing the window.

    Returns normalised coordinates within the crop: (x0/w, y0/h, x1/w, y1/h).
    """
    h, w = crop.shape[:2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    ax.set_title("Click 2 points: TOP and BOTTOM of the A|B divider line\n"
                 "(Left = Tap A, Right = Tap B)")
    ax.axis("off")
    plt.tight_layout()

    raw = plt.ginput(n=2, timeout=0, show_clicks=True)

    if len(raw) < 2:
        plt.close(fig)
        print(f"Only {len(raw)}/2 clicks recorded for divider. Run again with --interactive.")
        sys.exit(1)

    ax.plot([raw[0][0], raw[1][0]], [raw[0][1], raw[1][1]],
            "r-", linewidth=2)
    mid_x = (raw[0][0] + raw[1][0]) / 2
    ax.text(raw[0][0] * 0.3, h * 0.05, "A", fontsize=24, color="blue",
            fontweight="bold", ha="center")
    ax.text(mid_x + (w - mid_x) * 0.5, h * 0.05, "B", fontsize=24, color="green",
            fontweight="bold", ha="center")
    ax.set_title("Divider line drawn — verify, then close the window.")
    fig.canvas.draw()
    plt.show()

    x0, y0 = raw[0]
    x1, y1 = raw[1]

    divider = (x0 / w, y0 / h, x1 / w, y1 / h)
    return divider


def select_tap_bboxes_interactive(
    frame_bgr: np.ndarray,
    labels: list[str],
) -> list[list[float]]:
    """Show one drag-rectangle window per label, return bboxes in pixel coords.

    Each window lets the user drag a rectangle around the tap area, then close
    the window to confirm.  Returns list of [x1, y1, x2, y2] bounding boxes.
    """
    from matplotlib.widgets import RectangleSelector

    bboxes = []
    for label in labels:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(rgb)
        ax.set_title(f"Drag a rectangle over [{label}], then close the window.")
        ax.axis("off")
        plt.tight_layout()

        coords = {}

        def on_select(eclick, erelease, _coords=coords):
            _coords["x0"] = eclick.xdata
            _coords["y0"] = eclick.ydata
            _coords["x1"] = erelease.xdata
            _coords["y1"] = erelease.ydata

        selector = RectangleSelector(
            ax, on_select, useblit=True,
            button=[1], interactive=True,
            props=dict(facecolor="cyan", edgecolor="red", alpha=0.3, linewidth=2),
        )

        plt.show()

        if not coords:
            raise ValueError(f"No rectangle drawn for [{label}]. Run again with --interactive.")

        x0, y0, x1, y1 = coords["x0"], coords["y0"], coords["x1"], coords["y1"]
        bbox = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        bboxes.append(bbox)
        print(f"  [{label}] bbox -> {[round(v, 1) for v in bbox]}")

    return bboxes


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def point_side_of_line(px, py, x1, y1, x2, y2) -> str:
    """Return 'A' if point is left of the line, 'B' if right.
    The line is always oriented top-to-bottom (smallest y first)."""
    if y1 > y2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return "A" if cross <= 0 else "B"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def crop_normalized(frame: np.ndarray, roi: tuple) -> np.ndarray:
    """Crop a frame using normalised (0-1) coordinates."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)
    return frame[y1:y2, x1:x2]


def savefig(fig, out_dir: Path, name: str):
    """Save a matplotlib figure to *out_dir/name* and close it."""
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def open_video(video_path: Path):
    """Open a video and return (cap, fps, total_frames, duration, frame_0)."""
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"{video_path.name}: {fps:.0f} fps  {total_frames} frames  {duration:.1f}s")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_0 = cap.read()
    if not ret:
        print(f"Could not read first frame from {video_path}")
        sys.exit(1)
    cap.release()

    return fps, total_frames, duration, frame_0


def export_cropped_video(video_path: Path, output_path: Path, roi: tuple,
                         fps: float) -> Path:
    """Export the ROI-cropped version of a video to *output_path*."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Could not read {video_path}")

    crop = crop_normalized(frame, roi)
    h_out, w_out = crop.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w_out, h_out))

    # Write first frame
    writer.write(crop)

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = crop_normalized(frame, roi)
        writer.write(crop)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  Cropping frame {frame_idx}...", end="\r")

    cap.release()
    writer.release()
    print(f"  Cropped video saved to {output_path} ({w_out}x{h_out}, {frame_idx} frames)")
    return output_path


# ---------------------------------------------------------------------------
# ROI config persistence
# ---------------------------------------------------------------------------

def load_roi_config(roi_json_path: Path) -> dict:
    """Load tap_roi.json if it exists, else return empty dict."""
    if roi_json_path.exists():
        return json.loads(roi_json_path.read_text())
    return {}


def save_roi_config(roi_json_path: Path, tap_roi: tuple,
                    tap_divider: tuple | None = None,
                    extra_keys: dict | None = None):
    """Save ROI and divider to tap_roi.json."""
    data = {"tap_roi": list(tap_roi)}
    if tap_divider:
        data["tap_divider"] = list(tap_divider)
    if extra_keys:
        data.update(extra_keys)
    roi_json_path.write_text(json.dumps(data, indent=2))
    print(f"ROI config saved to {roi_json_path}")


def resolve_roi(config_roi, roi_json_path: Path, frame_0: np.ndarray,
                interactive: bool = False) -> tuple:
    """Resolve the tap ROI from config value, JSON file, or interactive selection.

    Parameters
    ----------
    config_roi : list | None
        ROI from YAML config (or CLI arg). Used first if not None.
    roi_json_path : Path
        Path to tap_roi.json for fallback.
    frame_0 : np.ndarray
        First video frame, used for interactive selection.
    interactive : bool
        If True, force interactive selection even if config_roi is set.
    """
    if interactive or config_roi is None:
        # Try JSON fallback first (unless interactive is forced)
        if not interactive:
            data = load_roi_config(roi_json_path)
            if "tap_roi" in data:
                tap_roi = tuple(data["tap_roi"])
                print(f"TAP_ROI loaded from {roi_json_path}: "
                      f"{tuple(round(v, 4) for v in tap_roi)}")
                return tap_roi

        print("Opening ROI selector on frame 0 ...")
        tap_roi = select_roi_interactive(frame_0)
        print(f"TAP_ROI = {tuple(round(v, 4) for v in tap_roi)}")
        return tap_roi

    tap_roi = tuple(config_roi)
    print(f"TAP_ROI from config: {tuple(round(v, 4) for v in tap_roi)}")
    return tap_roi


def resolve_divider(config_divider, roi_json_path: Path,
                    tap_roi: tuple, frame_0: np.ndarray,
                    interactive: bool = False) -> tuple | None:
    """Resolve the A|B divider from config, JSON, or interactive selection."""
    if interactive or config_divider is None:
        if not interactive:
            data = load_roi_config(roi_json_path)
            if "tap_divider" in data:
                tap_divider = tuple(data["tap_divider"])
                print(f"TAP_DIVIDER loaded from {roi_json_path}: "
                      f"{tuple(round(v, 4) for v in tap_divider)}")
                return tap_divider

            if not interactive:
                print("WARNING: No tap divider defined. All pours will be unclassified.")
                return None

        print("Opening divider selector on cropped frame ...")
        crop = crop_normalized(frame_0, tap_roi)
        tap_divider = select_divider_interactive(crop)
        print(f"TAP_DIVIDER = {tuple(round(v, 4) for v in tap_divider)}")
        return tap_divider

    tap_divider = tuple(config_divider)
    print(f"TAP_DIVIDER from config: {tuple(round(v, 4) for v in tap_divider)}")
    return tap_divider
