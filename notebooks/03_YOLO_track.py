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
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO, YOLOWorld

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect beer pour events using YOLO tracking (model.track).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, type=Path, help="Path to input video file.")
    p.add_argument("--output", required=True, type=Path,
                   help="Directory where plots and results are saved (created if absent).")

    # ROI
    roi_g = p.add_mutually_exclusive_group()
    roi_g.add_argument("--crop-area", action="store_true",
                       help="Open an interactive window to select the tap ROI. "
                            "Saves coordinates to <output>/tap_roi.json, then continues.")
    roi_g.add_argument("--tap-roi", nargs=4, type=float,
                       metavar=("X1", "Y1", "X2", "Y2"),
                       help="Normalised tap-area ROI (0-1). Overrides any saved tap_roi.json.")

    # Tap A/B divider
    p.add_argument("--tap-divider", nargs=4, type=float,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="Normalised divider line within the crop (0-1). "
                        "Two points (top, bottom) defining the A|B boundary. "
                        "Left of line = Tap A, right = Tap B. Overrides saved value.")

    # Model / detection
    p.add_argument("--model", default="yolov8x-worldv2.pt",
                   help="YOLO weights file or Ultralytics model name. "
                        "Use a *-world*.pt model for open-vocabulary detection.")
    p.add_argument("--classes", nargs="+", default=["cup", "person"],
                   help="Classes to detect (only used with YOLO-World models). "
                        "Ignored for standard YOLO models that use fixed COCO classes.")
    p.add_argument("--sample-every", type=int, default=1,
                   help="Process every Nth frame (1 = every frame, recommended for tracking).")
    p.add_argument("--preview-second", type=float, default=60.0,
                   help="Timestamp (s) used for the 'beer being served' preview frame.")
    p.add_argument("--tracker", default="../config/botsort.yaml",
                   help="Tracker config file (botsort.yaml or bytetrack.yaml). "
                        "Default points to the project's tuned config.")
    p.add_argument("--conf-threshold", type=float, default=0.25,
                   help="Minimum detection confidence.")

    # Pour classification
    p.add_argument("--movement-threshold", type=float, default=5.0,
                   help="Minimum position spread (px) to count a track as a pour.")

    # Merge
    p.add_argument("--merge-gap", type=float, default=5.0,
                   help="Pour tracks within this many seconds are merged into one event.")

    # Annotated video output
    p.add_argument("--record-range", nargs=2, type=float, metavar=("START", "STOP"),
                   help="Optional: export an annotated video for this time range (seconds). "
                        "E.g. --record-range 50 80 records from 50s to 80s with "
                        "bounding boxes, track IDs, and the A|B divider drawn on each frame.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# ROI selector
# ---------------------------------------------------------------------------

def select_roi_interactive(frame: np.ndarray) -> tuple:
    """Open a half-scale window, let the user click two corners, return normalised ROI."""
    scale = 0.5
    roi_frame = cv2.resize(frame.copy(), None, fx=scale, fy=scale)
    clone = roi_frame.copy()
    clicks = []

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        clicks.append((x, y))
        cv2.circle(roi_frame, (x, y), 5, (0, 255, 0), -1)
        if len(clicks) == 2:
            cv2.rectangle(roi_frame, clicks[0], clicks[1], (0, 255, 0), 2)
            cv2.putText(roi_frame, "TAP AREA", (clicks[0][0], clicks[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Select TAP area  [click x2, r=reset, q=done]", roi_frame)

    cv2.imshow("Select TAP area  [click x2, r=reset, q=done]", roi_frame)
    cv2.setMouseCallback("Select TAP area  [click x2, r=reset, q=done]", on_click)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            clicks.clear()
            roi_frame[:] = clone
            cv2.imshow("Select TAP area  [click x2, r=reset, q=done]", roi_frame)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    h_s, w_s = clone.shape[:2]
    if len(clicks) < 2:
        print(f"Only {len(clicks)}/2 clicks recorded. Run again with --crop-area.")
        sys.exit(1)

    roi = (clicks[0][0] / w_s, clicks[0][1] / h_s,
           clicks[1][0] / w_s, clicks[1][1] / h_s)
    roi = (min(roi[0], roi[2]), min(roi[1], roi[3]),
           max(roi[0], roi[2]), max(roi[1], roi[3]))
    return roi


def select_divider_interactive(crop: np.ndarray) -> tuple:
    """Open a window on the cropped frame, let the user click two points (top/bottom)
    to define the A|B divider line. Returns normalised coordinates within the crop."""
    scale = max(1.0, 600 / crop.shape[1])
    disp = cv2.resize(crop.copy(), None, fx=scale, fy=scale)
    clone = disp.copy()
    clicks = []

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        clicks.append((x, y))
        cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
        if len(clicks) == 2:
            cv2.line(disp, clicks[0], clicks[1], (0, 0, 255), 2)
            cv2.putText(disp, "A", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            mid_x = (clicks[0][0] + clicks[1][0]) // 2
            cv2.putText(disp, "B", (mid_x + 30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Select A|B divider  [click x2, r=reset, q=done]", disp)

    cv2.imshow("Select A|B divider  [click x2, r=reset, q=done]", disp)
    cv2.setMouseCallback("Select A|B divider  [click x2, r=reset, q=done]", on_click)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            clicks.clear()
            disp[:] = clone
            cv2.imshow("Select A|B divider  [click x2, r=reset, q=done]", disp)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    h_s, w_s = clone.shape[:2]
    if len(clicks) < 2:
        print(f"Only {len(clicks)}/2 clicks recorded for divider. Run again with --crop-area.")
        sys.exit(1)

    divider = (clicks[0][0] / w_s, clicks[0][1] / h_s,
               clicks[1][0] / w_s, clicks[1][1] / h_s)
    return divider


def point_side_of_line(px, py, x1, y1, x2, y2) -> str:
    """Return 'A' if point is left of the line, 'B' if right.
    The line is always oriented top-to-bottom (smallest y first)."""
    if y1 > y2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return "A" if cross <= 0 else "B"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def crop_normalized(frame: np.ndarray, roi: tuple) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)
    return frame[y1:y2, x1:x2]


def savefig(fig, out_dir: Path, name: str):
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # -- Output directory ---------------------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)
    roi_json = args.output / "tap_roi.json"

    # -- Video metadata -----------------------------------------------------
    if not args.video.exists():
        print(f"Video not found: {args.video}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"{args.video.name}: {fps:.0f} fps  {total_frames} frames  {duration:.1f}s")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_0 = cap.read()

    preview_frame_idx = int(args.preview_second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame_idx)
    ret, frame_preview = cap.read()
    cap.release()

    print(f"Frame 0: {frame_0.shape}  "
          f"Preview frame ({args.preview_second:.0f}s): {frame_preview.shape}")

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
        print("No TAP_ROI defined. Re-run with --crop-area to select one interactively,\n"
              "or pass --tap-roi X1 Y1 X2 Y2.")
        sys.exit(1)

    # -- Resolve TAP divider (A|B line) -------------------------------------
    tap_divider = None

    if args.tap_divider:
        tap_divider = tuple(args.tap_divider)
        print(f"TAP_DIVIDER from CLI: {tuple(round(v, 4) for v in tap_divider)}")

    elif args.crop_area:
        print("Opening divider selector on cropped frame ...")
        crop_for_divider = crop_normalized(frame_0, tap_roi)
        tap_divider = select_divider_interactive(crop_for_divider)
        print(f"TAP_DIVIDER = {tuple(round(v, 4) for v in tap_divider)}")

    elif roi_json.exists():
        data = json.loads(roi_json.read_text())
        if "tap_divider" in data:
            tap_divider = tuple(data["tap_divider"])
            print(f"TAP_DIVIDER loaded from {roi_json}: "
                  f"{tuple(round(v, 4) for v in tap_divider)}")

    # Save both ROI and divider together
    if args.crop_area:
        save_data = {"tap_roi": list(tap_roi)}
        if tap_divider:
            save_data["tap_divider"] = list(tap_divider)
        roi_json.write_text(json.dumps(save_data, indent=2))
        print(f"ROI config saved to {roi_json}")

    if tap_divider is None:
        print("WARNING: No tap divider defined. All pours will be unclassified.\n"
              "  Re-run with --crop-area or pass --tap-divider X1 Y1 X2 Y2.")

    # -- Crop preview frames ------------------------------------------------
    crop_0 = crop_normalized(frame_0, tap_roi)
    crop_preview = crop_normalized(frame_preview, tap_roi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(cv2.cvtColor(crop_0, cv2.COLOR_BGR2RGB))
    ax1.set_title("Cropped — 0s")
    ax1.axis("off")
    ax2.imshow(cv2.cvtColor(crop_preview, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Cropped — {args.preview_second:.0f}s")
    ax2.axis("off")
    plt.tight_layout()
    savefig(fig, args.output, "01_crop_preview.png")

    # -- Tap A|B division preview -------------------------------------------
    if tap_divider:
        crop_h, crop_w = crop_0.shape[:2]
        lx1 = tap_divider[0] * crop_w
        ly1 = tap_divider[1] * crop_h
        lx2 = tap_divider[2] * crop_w
        ly2 = tap_divider[3] * crop_h

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cv2.cvtColor(crop_0, cv2.COLOR_BGR2RGB))
        ax.plot([lx1, lx2], [ly1, ly2], color="red", linewidth=2, linestyle="--",
                label="A | B divider")
        mid_x = (lx1 + lx2) / 2
        ax.text(mid_x / 2, crop_h * 0.07, "TAP A", fontsize=18, fontweight="bold",
                color="black", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax.text(mid_x + (crop_w - mid_x) / 2, crop_h * 0.07, "TAP B", fontsize=18,
                fontweight="bold", color="black", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax.set_title("Tap A | B division")
        ax.legend(loc="upper right")
        ax.axis("off")
        plt.tight_layout()
        savefig(fig, args.output, "01b_tap_division.png")

    # -- Load YOLO ----------------------------------------------------------
    is_world = "world" in str(args.model).lower()
    if is_world:
        print(f"\nLoading YOLO-World model: {args.model}")
        model = YOLOWorld(args.model)
        model.set_classes(args.classes)
        print(f"  Detecting classes: {args.classes}")
    else:
        print(f"\nLoading YOLO model: {args.model}")
        model = YOLO(args.model)

    # -- Full-video tracking ------------------------------------------------
    print(f"\nTracking video every {args.sample_every} frame(s) "
          f"with {args.tracker} ...")
    cap = cv2.VideoCapture(str(args.video))
    frame_idx = 0
    all_detections = []          # (frame, time, class, conf, track_id, x1, y1, x2, y2)
    track_data = defaultdict(lambda: {"frames": [], "times": [], "bboxes": [],
                                       "confs": [], "class": None})

    # -- Optional annotated video recording ---------------------------------
    rec_start_frame = rec_stop_frame = None
    video_writer = None
    if args.record_range:
        rec_start_frame = int(args.record_range[0] * fps)
        rec_stop_frame = int(args.record_range[1] * fps)
        print(f"Recording annotated video: {args.record_range[0]:.1f}s -> "
              f"{args.record_range[1]:.1f}s  "
              f"(frames {rec_start_frame}-{rec_stop_frame})")

    # Pre-compute divider pixel coords for annotation
    crop_ref = crop_normalized(frame_0, tap_roi)
    div_px = None
    if tap_divider:
        ch, cw = crop_ref.shape[:2]
        div_px = (int(tap_divider[0] * cw), int(tap_divider[1] * ch),
                  int(tap_divider[2] * cw), int(tap_divider[3] * ch))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.sample_every == 0:
            crop = crop_normalized(frame, tap_roi)
            results = model.track(crop, persist=True, tracker=args.tracker,
                                  conf=args.conf_threshold, verbose=False)[0]

            if results.boxes is not None and results.boxes.id is not None:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls]
                    track_id = int(box.id[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    t_s = frame_idx / fps

                    all_detections.append((frame_idx, t_s, name, conf, track_id,
                                           x1, y1, x2, y2))

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
                # Draw A|B divider line
                if div_px:
                    cv2.line(annotated, (div_px[0], div_px[1]), (div_px[2], div_px[3]),
                             (0, 0, 255), 2)
                    cv2.putText(annotated, "A",
                                (div_px[0] // 2, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(annotated, "B",
                                ((div_px[0] + div_px[2]) // 2 + 20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
                # Timestamp overlay
                cv2.putText(annotated, f"{frame_idx / fps:.1f}s",
                            (annotated.shape[1] - 120, annotated.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if video_writer is None:
                    h_out, w_out = annotated.shape[:2]
                    rec_path = args.output / "annotated_clip.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(rec_path), fourcc,
                                                   fps / args.sample_every,
                                                   (w_out, h_out))
                    print(f"  Video writer opened: {rec_path} ({w_out}x{h_out})")
                video_writer.write(annotated)

            if frame_idx % (args.sample_every * 100) == 0:
                print(f"  Frame {frame_idx}/{total_frames}  "
                      f"tracks so far: {len(track_data)}", end="\r")

        frame_idx += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"  Annotated video saved to {args.output / 'annotated_clip.mp4'}")
    print(f"\nTotal detections: {len(all_detections)}  "
          f"Unique track IDs: {len(track_data)}")

    # -- Save raw detections to CSV -----------------------------------------
    raw_csv = args.output / "raw_detections.csv"
    with open(raw_csv, "w") as f:
        f.write("frame,time_s,class,confidence,track_id,x1,y1,x2,y2\n")
        for det in all_detections:
            fidx, t_s, name, conf, tid, x1, y1, x2, y2 = det
            f.write(f"{fidx},{t_s:.4f},{name},{conf:.4f},{tid},"
                    f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}\n")
    print(f"Raw detections saved to {raw_csv}")

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
    savefig(fig, args.output, "03_detection_timeline.png")

    # -- Filter cup tracks --------------------------------------------------
    cup_tracks = {tid: td for tid, td in track_data.items() if td["class"] == "cup"}

    # Classify each cup track as Tap A or Tap B
    crop_h, crop_w = crop_0.shape[:2]
    for tid, td in cup_tracks.items():
        bboxes = np.array(td["bboxes"])
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
        if tap_divider:
            lx1 = tap_divider[0] * crop_w
            ly1 = tap_divider[1] * crop_h
            lx2 = tap_divider[2] * crop_w
            ly2 = tap_divider[3] * crop_h
            td["tap"] = point_side_of_line(cx.mean(), cy.mean(), lx1, ly1, lx2, ly2)
        else:
            td["tap"] = "?"

        td["movement"] = float(np.sqrt(cx.std() ** 2 + cy.std() ** 2))

    print(f"\nFound {len(cup_tracks)} cup tracks (by YOLO track ID)")
    for tid in sorted(cup_tracks.keys()):
        td = cup_tracks[tid]
        dur = td["times"][-1] - td["times"][0]
        print(f"  ID {tid:3d} [Tap {td['tap']}]: "
              f"{td['times'][0]:.1f}s -> {td['times'][-1]:.1f}s  "
              f"dur={dur:.1f}s  dets={len(td['frames'])}  "
              f"move={td['movement']:.1f}px  "
              f"avg_conf={np.mean(td['confs']):.2f}")

    # -- Cup track timeline -------------------------------------------------
    sorted_tids = sorted(cup_tracks.keys())
    fig, ax = plt.subplots(figsize=(14, max(3, len(sorted_tids) * 0.4)))
    for row, tid in enumerate(sorted_tids):
        td = cup_tracks[tid]
        dur = td["times"][-1] - td["times"][0]
        ax.barh(row, dur, left=td["times"][0], height=0.6,
                label=f"ID {tid} ({dur:.1f}s)")
        ax.scatter(td["times"], [row] * len(td["times"]), s=8, color="black", zorder=3)
    ax.set_yticks(range(len(sorted_tids)))
    ax.set_yticklabels([f"ID {tid}" for tid in sorted_tids])
    ax.set_xlabel("Time (s)")
    ax.set_title("Cup tracks (YOLO tracker IDs)")
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    savefig(fig, args.output, "04_cup_tracks.png")

    # -- Cup centers on crop with A|B divider (moving tracks only) ----------
    if tap_divider:
        lx1 = tap_divider[0] * crop_w
        ly1 = tap_divider[1] * crop_h
        lx2 = tap_divider[2] * crop_w
        ly2 = tap_divider[3] * crop_h

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cv2.cvtColor(crop_0, cv2.COLOR_BGR2RGB))
        ax.plot([lx1, lx2], [ly1, ly2], color="red", linewidth=2, linestyle="--",
                label="A | B divider")

        cmap = plt.cm.get_cmap("tab10")
        plot_idx = 0
        for tid in sorted_tids:
            td = cup_tracks[tid]
            if td["movement"] <= args.movement_threshold:
                continue
            bboxes = np.array(td["bboxes"])
            cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
            cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
            c = cmap(plot_idx % 10)
            ax.scatter(cx, cy, s=20, color=c, alpha=0.7, zorder=3,
                       label=f"ID {tid} (Tap {td['tap']})")
            plot_idx += 1

        mid_x = (lx1 + lx2) / 2
        ax.text(mid_x / 2, crop_h * 0.07, "TAP A", fontsize=16, fontweight="bold",
                color="black", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax.text(mid_x + (crop_w - mid_x) / 2, crop_h * 0.07, "TAP B", fontsize=16,
                fontweight="bold", color="black", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax.set_title("Cup centres by track ID (moving tracks only)")
        ax.legend(loc="upper right", fontsize=7)
        ax.axis("off")
        plt.tight_layout()
        savefig(fig, args.output, "04b_cup_centers_by_tap.png")

    # -- Person + cup co-occurrence -----------------------------------------
    frames_with_person: set = set()
    frames_with_cup: set = set()

    for det in all_detections:
        fidx, t_s, name = det[0], det[1], det[2]
        if name == "person":
            frames_with_person.add(fidx)
        elif name == "cup":
            frames_with_cup.add(fidx)

    cooccurrence_frames = sorted(frames_with_person & frames_with_cup)
    cooccurrence_times = [f / fps for f in cooccurrence_frames]

    print(f"\nFrames with person: {len(frames_with_person)}")
    print(f"Frames with cup:    {len(frames_with_cup)}")
    print(f"Frames with BOTH:   {len(cooccurrence_frames)}")

    segments = []
    if cooccurrence_times:
        gap_threshold = args.sample_every / fps * 3
        seg_start = cooccurrence_times[0]
        prev_t = cooccurrence_times[0]
        for t_s in cooccurrence_times[1:]:
            if t_s - prev_t > gap_threshold:
                segments.append((seg_start, prev_t))
                seg_start = t_s
            prev_t = t_s
        segments.append((seg_start, prev_t))

        print("\nPour candidates (person+cup co-occurrence segments):")
        for j, (s, e) in enumerate(segments):
            print(f"  Segment {j}: {s:.1f}s -> {e:.1f}s  duration={e-s:.1f}s")

    person_times = sorted([f / fps for f in frames_with_person])
    cup_times = sorted([f / fps for f in frames_with_cup])

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.scatter(cup_times, [0] * len(cup_times), s=12, color="blue", label="cup", alpha=0.6)
    ax.scatter(person_times, [1] * len(person_times), s=12, color="orange",
               label="person", alpha=0.6)
    ax.scatter(cooccurrence_times, [2] * len(cooccurrence_times), s=12, color="red",
               label="BOTH (pour?)", alpha=0.8)
    for s, e in segments:
        ax.axvspan(s, e, alpha=0.15, color="red")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["cup", "person", "BOTH"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Person + Cup co-occurrence = pour events")
    ax.legend(loc="upper right")
    plt.tight_layout()
    savefig(fig, args.output, "05_cooccurrence.png")

    # -- Classify tracks & merge pour events --------------------------------
    if cooccurrence_times:
        print("\nCup tracks that overlap with person presence:")
        for tid in sorted_tids:
            td = cup_tracks[tid]
            dur = td["times"][-1] - td["times"][0]
            overlaps = any(td["times"][-1] >= s and td["times"][0] <= e
                           for s, e in segments)
            status = ("POUR" if overlaps and td["movement"] > args.movement_threshold
                      else "background")
            print(f"  ID {tid:3d} [Tap {td['tap']}]: "
                  f"move={td['movement']:.1f}px  dur={dur:.1f}s  "
                  f"overlaps_person={overlaps}  -> {status}")

    pour_tracks = []
    for tid in sorted_tids:
        td = cup_tracks[tid]
        overlaps = (cooccurrence_times and
                    any(td["times"][-1] >= s and td["times"][0] <= e
                        for s, e in segments))
        if overlaps and td["movement"] > args.movement_threshold:
            pour_tracks.append({"track_ids": [tid], "start": td["times"][0],
                                 "end": td["times"][-1],
                                 "movement": td["movement"], "tap": td["tap"]})

    pour_tracks.sort(key=lambda x: x["start"])
    merged_pours = []
    for pt in pour_tracks:
        if (merged_pours
                and pt["start"] - merged_pours[-1]["end"] < args.merge_gap
                and pt["tap"] == merged_pours[-1]["tap"]):
            merged_pours[-1]["end"] = max(merged_pours[-1]["end"], pt["end"])
            merged_pours[-1]["track_ids"].extend(pt["track_ids"])
        else:
            merged_pours.append(pt.copy())

    print(f"\nPOUR tracks before merge: {len(pour_tracks)}")
    print(f"Pour events after merge:  {len(merged_pours)}")
    for j, mp in enumerate(merged_pours):
        dur = mp["end"] - mp["start"]
        print(f"  Pour {j+1} [Tap {mp['tap']}]: {mp['start']:.1f}s -> {mp['end']:.1f}s  "
              f"duration={dur:.1f}s  from track IDs {mp['track_ids']}")

    # -- Final timeline plot ------------------------------------------------
    tap_colors = {"A": "green", "B": "purple", "?": "gray"}
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.scatter(cup_times, [0] * len(cup_times), s=8, color="blue", alpha=0.3, label="cup")
    ax.scatter(person_times, [1] * len(person_times), s=8, color="orange", alpha=0.3,
               label="person")
    for j, mp in enumerate(merged_pours):
        c = tap_colors.get(mp["tap"], "gray")
        ax.axvspan(mp["start"], mp["end"], alpha=0.25, color=c,
                   label=(f"Tap {mp['tap']}"
                          if mp["tap"] not in [m["tap"] for m in merged_pours[:j]]
                          else None))
        ax.text((mp["start"] + mp["end"]) / 2, 1.5, f"POUR {j+1}\nTap {mp['tap']}",
                ha="center", fontweight="bold", color=c)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["cup", "person"])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Final result: {len(merged_pours)} pour event(s) detected")
    ax.legend(loc="upper right")
    plt.tight_layout()
    savefig(fig, args.output, "06_final_pours.png")

    # -- Save summary JSON --------------------------------------------------
    summary = {
        "video": str(args.video),
        "tap_roi": list(tap_roi),
        "tap_divider": list(tap_divider) if tap_divider else None,
        "tracker": args.tracker,
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": duration,
        "sample_every": args.sample_every,
        "total_detections": len(all_detections),
        "unique_track_ids": len(track_data),
        "cup_tracks": len(cup_tracks),
        "pour_events": [
            {"id": j + 1, "tap": mp["tap"], "start_s": mp["start"], "end_s": mp["end"],
             "duration_s": mp["end"] - mp["start"], "track_ids": mp["track_ids"]}
            for j, mp in enumerate(merged_pours)
        ],
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_path}")
    print(f"All outputs written to: {args.output}/")


if __name__ == "__main__":
    main()
