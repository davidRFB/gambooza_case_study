"""Activity window detection and clip extraction utilities.

Used by the SimpleDetector pre-filtering pipeline to identify time windows
with tap activity and extract them as short video clips for YOLO processing.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def find_activity_windows(
    times: np.ndarray,
    signals: dict[str, np.ndarray],
    threshold: float,
    padding_s: float,
    merge_gap_s: float,
    total_duration: float,
) -> list[dict]:
    """Find time windows where either tap has activity above threshold.

    Returns list of {"start_s", "end_s", "duration_s"} dicts, padded and merged.
    """
    # Combined signal: max of both taps at each time step
    combined = np.maximum(
        signals.get("Tap A", np.zeros(len(times))),
        signals.get("Tap B", np.zeros(len(times))),
    )

    # Find active stretches (any single sample above threshold counts)
    active = combined > threshold
    raw_windows = []
    in_window = False
    start = 0.0

    for i, is_active in enumerate(active):
        if is_active and not in_window:
            in_window = True
            start = times[i]
        elif not is_active and in_window:
            in_window = False
            raw_windows.append((start, times[i]))

    if in_window:
        raw_windows.append((start, times[-1]))

    if not raw_windows:
        return []

    # Add padding
    padded = [(max(0, s - padding_s), min(total_duration, e + padding_s)) for s, e in raw_windows]

    # Merge windows closer than merge_gap_s
    merged = [padded[0]]
    for s, e in padded[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= merge_gap_s:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))

    return [
        {"start_s": round(s, 2), "end_s": round(e, 2), "duration_s": round(e - s, 2)}
        for s, e in merged
    ]


def extract_clips(
    video_path: Path,
    windows: list[dict],
    output_dir: Path,
) -> list[Path]:
    """Extract each activity window as a separate video clip."""
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    paths = []
    for i, win in enumerate(windows):
        start_frame = int(win["start_s"] * fps)
        end_frame = int(win["end_s"] * fps)
        clip_name = f"clip_{i:03d}_{win['start_s']:.0f}s_{win['end_s']:.0f}s.mp4"
        clip_path = clips_dir / clip_name

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))

        frame_idx = start_frame
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frame_idx += 1

        writer.release()
        paths.append(clip_path)
        logger.info(
            "Clip %d/%d: %.1fs - %.1fs (%.1fs) -> %s",
            i + 1,
            len(windows),
            win["start_s"],
            win["end_s"],
            win["duration_s"],
            clip_name,
        )

    cap.release()
    return paths
