"""
YOLO-World beer pour detector using model.track() for proper object tracking.

Uses YOLO-World for open-vocabulary detection (only "cup" + "person" by default)
combined with BoT-SORT/ByteTrack tracker so each object gets a persistent track
ID across frames.  A tuned BoT-SORT config (config/botsort.yaml) keeps lost
tracks alive longer to reduce ID switches during occlusions.

Usage examples
--------------
# First run: select the tap area and divider line interactively
python 03_YOLO_track.py --video ../data/videos/cerveza2.mp4 --output ../results/cerveza2_track --crop-area

# Re-run with saved coordinates (skip ROI selection)
python 03_YOLO_track.py --video ../data/videos/cerveza2.mp4 --output ../results/cerveza2_track

# Use standard YOLOv8 (all 80 COCO classes) instead of YOLO-World
python 03_YOLO_track.py --video ../data/videos/cerveza2.mp4 --output ../results/cerveza2_track \
    --model yolov8x.pt

# Detect additional classes with YOLO-World
python 03_YOLO_track.py --video ../data/videos/cerveza2.mp4 --output ../results/cerveza2_track \
    --classes cup person bottle

# Export annotated video clip for a time range (e.g. 50s to 80s)
python 03_YOLO_track.py --video ../data/videos/cerveza2.mp4 --output ../results/cerveza2_track \
    --record-range 50 80
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO, YOLOWorld

from backend.ml.common import (
    crop_normalized,
    savefig,
    select_roi_interactive,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Detect beer pour events using YOLO tracking (model.track).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, type=Path, help="Path to input video file.")
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where plots and results are saved (created if absent).",
    )

    # ROI
    roi_g = p.add_mutually_exclusive_group()
    roi_g.add_argument(
        "--crop-area",
        action="store_true",
        help="Open an interactive window to select the tap ROI. "
        "Saves coordinates to <output>/tap_roi.json, then continues.",
    )
    roi_g.add_argument(
        "--tap-roi",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Normalised tap-area ROI (0-1). Overrides any saved tap_roi.json.",
    )

    # Model / detection
    p.add_argument(
        "--model",
        default="data/models/yolov8x-worldv2.pt",
        help="YOLO weights file or Ultralytics model name. "
        "Use a *-world*.pt model for open-vocabulary detection.",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=["cup", "person"],
        help="Classes to detect (only used with YOLO-World models). "
        "Ignored for standard YOLO models that use fixed COCO classes.",
    )
    p.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Process every Nth frame (1 = every frame, recommended for tracking).",
    )
    p.add_argument(
        "--preview-second",
        type=float,
        default=60.0,
        help="Timestamp (s) used for the 'beer being served' preview frame.",
    )
    p.add_argument(
        "--tracker",
        default="config/botsort.yaml",
        help="Tracker config file (botsort.yaml or bytetrack.yaml). "
        "Default points to the project's tuned config.",
    )
    p.add_argument(
        "--conf-threshold", type=float, default=0.25, help="Minimum detection confidence."
    )

    # Annotated video output
    p.add_argument(
        "--record-range",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        help="Optional: export an annotated video for this time range (seconds). "
        "E.g. --record-range 50 80 records from 50s to 80s with "
        "bounding boxes, track IDs, and the A|B divider drawn on each frame.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers (select_roi_interactive, select_divider_interactive,
#          point_side_of_line, crop_normalized, savefig imported from common)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Core tracking function (importable by pipeline.py)
# ---------------------------------------------------------------------------


def run_yolo_tracking(
    video_path: Path,
    output_dir: Path,
    tap_roi: tuple,
    model_name: str = "data/models/yolov8x-worldv2.pt",
    classes: list[str] | None = None,
    sample_every: int = 1,
    conf_threshold: float = 0.25,
    tracker: str = "../config/botsort.yaml",
    preview_second: float = 60.0,
    record_range: tuple | None = None,
) -> Path:
    """Run YOLO tracking on a cropped video region and return path to raw_detections.csv.

    Detects and tracks objects (cups, persons) across frames. Pour classification
    and tap assignment happen downstream (relink + SAM stages).
    """
    if classes is None:
        classes = ["cup", "person"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Video metadata -----------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    logger.info("%s: %.0f fps  %d frames  %.1fs", video_path.name, fps, total_frames, duration)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_0 = cap.read()

    preview_frame_idx = min(int(preview_second * fps), total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame_idx)
    ret, frame_preview = cap.read()
    if not ret:
        frame_preview = frame_0  # fallback to first frame
    cap.release()

    logger.debug(
        "Frame 0: %s  Preview frame (%.0fs): %s", frame_0.shape, preview_second, frame_preview.shape
    )

    # -- Crop preview frames ------------------------------------------------
    crop_0 = crop_normalized(frame_0, tap_roi)
    crop_preview = crop_normalized(frame_preview, tap_roi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(cv2.cvtColor(crop_0, cv2.COLOR_BGR2RGB))
    ax1.set_title("Cropped — 0s")
    ax1.axis("off")
    ax2.imshow(cv2.cvtColor(crop_preview, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Cropped — {preview_second:.0f}s")
    ax2.axis("off")
    plt.tight_layout()
    savefig(fig, output_dir, "01_crop_preview.png")

    # -- Load YOLO ----------------------------------------------------------
    is_world = "world" in str(model_name).lower()
    if is_world:
        logger.info("Loading YOLO-World model: %s", model_name)
        model = YOLOWorld(model_name)
        model.set_classes(classes)
        logger.info("Detecting classes: %s", classes)
    else:
        logger.info("Loading YOLO model: %s", model_name)
        model = YOLO(model_name)

    # -- Full-video tracking ------------------------------------------------
    logger.info("Tracking video every %d frame(s) with %s", sample_every, tracker)
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    all_detections = []  # (frame, time, class, conf, track_id, x1, y1, x2, y2)
    track_data = defaultdict(
        lambda: {"frames": [], "times": [], "bboxes": [], "confs": [], "class": None}
    )

    # -- Optional annotated video recording ---------------------------------
    rec_start_frame = rec_stop_frame = None
    video_writer = None
    if record_range:
        rec_start_frame = int(record_range[0] * fps)
        rec_stop_frame = int(record_range[1] * fps)
        logger.info(
            "Recording annotated video: %.1fs -> %.1fs (frames %d-%d)",
            record_range[0],
            record_range[1],
            rec_start_frame,
            rec_stop_frame,
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            crop = crop_normalized(frame, tap_roi)
            results = model.track(
                crop, persist=True, tracker=tracker, conf=conf_threshold, verbose=False
            )[0]

            if results.boxes is not None and results.boxes.id is not None:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls]
                    track_id = int(box.id[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    t_s = frame_idx / fps

                    all_detections.append((frame_idx, t_s, name, conf, track_id, x1, y1, x2, y2))

                    td = track_data[track_id]
                    td["frames"].append(frame_idx)
                    td["times"].append(t_s)
                    td["bboxes"].append((x1, y1, x2, y2))
                    td["confs"].append(conf)
                    if td["class"] is None:
                        td["class"] = name

            # Write annotated frame if within recording range
            if rec_start_frame is not None and rec_start_frame <= frame_idx <= rec_stop_frame:
                annotated = results.plot()
                # Timestamp overlay
                cv2.putText(
                    annotated,
                    f"{frame_idx / fps:.1f}s",
                    (annotated.shape[1] - 120, annotated.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                if video_writer is None:
                    h_out, w_out = annotated.shape[:2]
                    rec_path = output_dir / "annotated_clip.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(rec_path), fourcc, fps / sample_every, (w_out, h_out)
                    )
                    logger.debug("Video writer opened: %s (%dx%d)", rec_path, w_out, h_out)
                video_writer.write(annotated)

            if frame_idx % (sample_every * 100) == 0:
                logger.debug(
                    "Frame %d/%d  tracks so far: %d", frame_idx, total_frames, len(track_data)
                )

        frame_idx += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
        logger.info("Annotated video saved to %s", output_dir / "annotated_clip.mp4")
    logger.info("Total detections: %d  Unique track IDs: %d", len(all_detections), len(track_data))

    # -- Save raw detections to CSV -----------------------------------------
    raw_csv = output_dir / "raw_detections.csv"
    with open(raw_csv, "w") as f:
        f.write("frame,time_s,class,confidence,track_id,x1,y1,x2,y2\n")
        for det in all_detections:
            fidx, t_s, name, conf, tid, x1, y1, x2, y2 = det
            f.write(
                f"{fidx},{t_s:.4f},{name},{conf:.4f},{tid},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}\n"
            )
    logger.info("Raw detections saved to %s", raw_csv)

    # -- Detection timeline -------------------------------------------------
    det_times: dict = {}
    for det in all_detections:
        det_times.setdefault(det[2], []).append(det[1])

    fig, ax = plt.subplots(figsize=(14, 4))
    for i, (name, times_list) in enumerate(det_times.items()):
        ax.scatter(times_list, [i] * len(times_list), s=10, label=f"{name} ({len(times_list)})")
    ax.set_yticks(range(len(det_times)))
    ax.set_yticklabels(list(det_times.keys()))
    ax.set_xlabel("Time (s)")
    ax.set_title("Detection timeline — all classes")
    ax.legend(loc="upper right")
    plt.tight_layout()
    savefig(fig, output_dir, "03_detection_timeline.png")

    # -- Filter cup tracks --------------------------------------------------
    cup_tracks = {tid: td for tid, td in track_data.items() if td["class"] == "cup"}

    for tid, td in cup_tracks.items():
        bboxes = np.array(td["bboxes"])
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
        td["movement"] = float(np.sqrt(cx.std() ** 2 + cy.std() ** 2))

    logger.info("Found %d cup tracks (by YOLO track ID)", len(cup_tracks))
    for tid in sorted(cup_tracks.keys()):
        td = cup_tracks[tid]
        dur = td["times"][-1] - td["times"][0]
        logger.debug(
            "  ID %3d: %.1fs -> %.1fs  dur=%.1fs  dets=%d  move=%.1fpx  avg_conf=%.2f",
            tid,
            td["times"][0],
            td["times"][-1],
            dur,
            len(td["frames"]),
            td["movement"],
            np.mean(td["confs"]),
        )

    # -- Cup track timeline -------------------------------------------------
    sorted_tids = sorted(cup_tracks.keys())
    fig, ax = plt.subplots(figsize=(14, max(3, len(sorted_tids) * 0.4)))
    for row, tid in enumerate(sorted_tids):
        td = cup_tracks[tid]
        dur = td["times"][-1] - td["times"][0]
        ax.barh(row, dur, left=td["times"][0], height=0.6, label=f"ID {tid} ({dur:.1f}s)")
        ax.scatter(td["times"], [row] * len(td["times"]), s=8, color="black", zorder=3)
    ax.set_yticks(range(len(sorted_tids)))
    ax.set_yticklabels([f"ID {tid}" for tid in sorted_tids])
    ax.set_xlabel("Time (s)")
    ax.set_title("Cup tracks (YOLO tracker IDs)")
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    savefig(fig, output_dir, "04_cup_tracks.png")

    # -- Save summary JSON --------------------------------------------------
    summary = {
        "video": str(video_path),
        "tap_roi": list(tap_roi),
        "tracker": tracker,
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": duration,
        "sample_every": sample_every,
        "total_detections": len(all_detections),
        "unique_track_ids": len(track_data),
        "cup_tracks": len(cup_tracks),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary saved to %s", summary_path)
    logger.info("All outputs written to: %s/", output_dir)

    return raw_csv


# ---------------------------------------------------------------------------
# Main (standalone CLI)
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    roi_json = args.output / "tap_roi.json"

    # -- Video metadata for ROI resolution ----------------------------------
    if not args.video.exists():
        print(f"Video not found: {args.video}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_0 = cap.read()
    cap.release()

    # -- Resolve TAP ROI ----------------------------------------------------
    tap_roi = None
    if args.tap_roi:
        tap_roi = tuple(args.tap_roi)
        print(f"TAP_ROI from CLI: {tuple(round(v, 4) for v in tap_roi)}")
    elif args.crop_area:
        print("Opening ROI selector on frame 0 ...")
        tap_roi = select_roi_interactive(frame_0)
        print(f"TAP_ROI = {tuple(round(v, 4) for v in tap_roi)}")
    elif roi_json.exists():
        data = json.loads(roi_json.read_text())
        tap_roi = tuple(data["tap_roi"])
        print(f"TAP_ROI loaded from {roi_json}: {tuple(round(v, 4) for v in tap_roi)}")
    else:
        print(
            "No TAP_ROI defined. Re-run with --crop-area to select one interactively,\n"
            "or pass --tap-roi X1 Y1 X2 Y2."
        )
        sys.exit(1)

    if args.crop_area:
        save_data = {"tap_roi": list(tap_roi)}
        roi_json.write_text(json.dumps(save_data, indent=2))
        print(f"ROI config saved to {roi_json}")

    # -- Run tracking -------------------------------------------------------
    run_yolo_tracking(
        video_path=args.video,
        output_dir=args.output,
        tap_roi=tap_roi,
        model_name=args.model,
        classes=args.classes,
        sample_every=args.sample_every,
        conf_threshold=args.conf_threshold,
        tracker=args.tracker,
        preview_second=args.preview_second,
        record_range=tuple(args.record_range) if args.record_range else None,
    )


if __name__ == "__main__":
    main()
