"""
YOLO-based beer pour detector with Tap A / Tap B classification.

Usage examples
--------------
# First run: select the tap area and divider line interactively
python 02_exploreYOLO.py --video ../data/videos/cerveza2.mp4 --output ../results/cerveza2 --crop-area

# Re-run with saved coordinates (skip ROI selection)
python 02_exploreYOLO.py --video ../data/videos/cerveza2.mp4 --output ../results/cerveza2

# Override saved ROI and divider with explicit values
python 02_exploreYOLO.py --video ../data/videos/cerveza2.mp4 --output ../results/cerveza2 \
    --tap-roi 0.4609 0.3398 0.6724 0.8796 --tap-divider 0.5 0.0 0.5 1.0

# Tweak detection parameters
python 02_exploreYOLO.py --video ../data/videos/cerveza2.mp4 --output ../results/cerveza2 \
    --sample-every 5 --iou-threshold 0.4 --merge-gap 3.0
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from common import (
    select_roi_interactive, select_divider_interactive,
    point_side_of_line, crop_normalized, savefig,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect beer pour events in a video using YOLO.",
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
    p.add_argument("--model", default="yolov8x.pt",
                   help="YOLO weights file or Ultralytics model name.")
    p.add_argument("--sample-every", type=int, default=10,
                   help="Process every Nth frame (higher = faster but less precise).")
    p.add_argument("--preview-second", type=float, default=60.0,
                   help="Timestamp (s) used for the 'beer being served' preview frame.")

    # Tracker
    p.add_argument("--iou-threshold", type=float, default=0.3,
                   help="Minimum IoU to link a detection to an existing track.")
    p.add_argument("--max-gap-frames", type=int, default=None,
                   help="Max frame gap to extend a track (default: sample-every * 5).")
    p.add_argument("--movement-threshold", type=float, default=5.0,
                   help="Minimum position spread (px) to count a track as a pour.")

    # Merge
    p.add_argument("--merge-gap", type=float, default=5.0,
                   help="Pour tracks within this many seconds are merged into one event.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0]);  y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]);  y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


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
        # Label A (left) and B (right)
        ax.text(lx1 * 0.3, crop_h * 0.1, "TAP A", fontsize=18, fontweight="bold",
                color="green", ha="center")
        ax.text(lx1 + (crop_w - lx1) * 0.5, crop_h * 0.1, "TAP B", fontsize=18,
                fontweight="bold", color="purple", ha="center")
        ax.set_title("Tap A | B division")
        ax.legend(loc="upper right")
        ax.axis("off")
        plt.tight_layout()
        savefig(fig, args.output, "01b_tap_division.png")

    # -- Load YOLO ----------------------------------------------------------
    print(f"\nLoading YOLO model: {args.model}")
    model = YOLO(args.model)

    results_0 = model(crop_0, verbose=False)[0]
    results_preview = model(crop_preview, verbose=False)[0]

    print("=== Cropped 0s detections ===")
    for box in results_0.boxes:
        cls = int(box.cls[0]); conf = float(box.conf[0]); name = model.names[cls]
        print(f"  {name:15s}  conf={conf:.2f}  bbox={box.xyxy[0].tolist()}")

    print(f"\n=== Cropped {args.preview_second:.0f}s detections ===")
    for box in results_preview.boxes:
        cls = int(box.cls[0]); conf = float(box.conf[0]); name = model.names[cls]
        print(f"  {name:15s}  conf={conf:.2f}  bbox={box.xyxy[0].tolist()}")

    annotated_0 = results_0.plot()
    annotated_preview = results_preview.plot()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(cv2.cvtColor(annotated_0, cv2.COLOR_BGR2RGB))
    ax1.set_title("YOLO — 0s ")
    ax1.axis("off")
    ax2.imshow(cv2.cvtColor(annotated_preview, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"YOLO — {args.preview_second:.0f}s ")
    ax2.axis("off")
    plt.tight_layout()
    savefig(fig, args.output, "02_yolo_preview.png")

    # -- Full-video scan ----------------------------------------------------
    print(f"\nScanning video every {args.sample_every} frames ...")
    cap = cv2.VideoCapture(str(args.video))
    frame_idx = 0
    all_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.sample_every == 0:
            crop = crop_normalized(frame, tap_roi)
            results = model(crop, verbose=False)[0]
            for box in results.boxes:
                cls = int(box.cls[0]); conf = float(box.conf[0]); name = model.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_detections.append((frame_idx, frame_idx / fps, name, conf, x1, y1, x2, y2))
            if frame_idx % (args.sample_every * 50) == 0:
                print(f"  Frame {frame_idx}/{total_frames}", end="\r")
        frame_idx += 1

    cap.release()
    print(f"\nTotal detections: {len(all_detections)}")

    # -- Save raw detections to CSV -----------------------------------------
    raw_csv = args.output / "raw_detections.csv"
    with open(raw_csv, "w") as f:
        f.write("frame,time_s,class,confidence,x1,y1,x2,y2\n")
        for det in all_detections:
            fidx, t, name, conf, x1, y1, x2, y2 = det
            f.write(f"{fidx},{t:.4f},{name},{conf:.4f},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}\n")
    print(f"Raw detections saved to {raw_csv}")

    # -- Detection timeline -------------------------------------------------
    det_times: dict = {}
    for (fidx, t, name, conf, *bbox) in all_detections:
        det_times.setdefault(name, []).append(t)

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

    # -- IoU cup tracker ----------------------------------------------------
    dets_by_frame: dict = defaultdict(list)
    for det in all_detections:
        fidx, t, name, conf, x1, y1, x2, y2 = det
        if name == "cup":
            dets_by_frame[fidx].append({"t": t, "bbox": (x1, y1, x2, y2), "conf": conf})

    max_gap = args.max_gap_frames if args.max_gap_frames else args.sample_every * 5
    tracks = []

    for fidx in sorted(dets_by_frame.keys()):
        dets = dets_by_frame[fidx]
        matched = set()

        for det in dets:
            best_track, best_iou_score = None, args.iou_threshold

            for ti, track in enumerate(tracks):
                if ti in matched:
                    continue
                if fidx - track["frames"][-1] > max_gap:
                    continue
                score = iou(det["bbox"], track["bboxes"][-1])
                if score > best_iou_score:
                    best_iou_score = score
                    best_track = ti

            if best_track is not None:
                tracks[best_track]["frames"].append(fidx)
                tracks[best_track]["times"].append(det["t"])
                tracks[best_track]["bboxes"].append(det["bbox"])
                tracks[best_track]["confs"].append(det["conf"])
                matched.add(best_track)
            else:
                tracks.append({"frames": [fidx], "times": [det["t"]],
                               "bboxes": [det["bbox"]], "confs": [det["conf"]]})

    # Classify each track as Tap A or Tap B based on average bbox center
    for t in tracks:
        bboxes = np.array(t["bboxes"])
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
        if tap_divider:
            crop_h, crop_w = crop_normalized(frame_0, tap_roi).shape[:2]
            lx1 = tap_divider[0] * crop_w
            ly1 = tap_divider[1] * crop_h
            lx2 = tap_divider[2] * crop_w
            ly2 = tap_divider[3] * crop_h
            t["tap"] = point_side_of_line(cx.mean(), cy.mean(), lx1, ly1, lx2, ly2)
        else:
            t["tap"] = "?"

    print(f"\nFound {len(tracks)} cup tracks")
    for i, t in enumerate(tracks):
        dur = t["times"][-1] - t["times"][0]
        print(f"  Track {i} [Tap {t['tap']}]: {t['times'][0]:.1f}s -> {t['times'][-1]:.1f}s  "
              f"dur={dur:.1f}s  dets={len(t['frames'])}  avg_conf={np.mean(t['confs']):.2f}")

    # Cup track timeline
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, t in enumerate(tracks):
        dur = t["times"][-1] - t["times"][0]
        ax.barh(i, dur, left=t["times"][0], height=0.6, label=f"Track {i} ({dur:.1f}s)")
        ax.scatter(t["times"], [i] * len(t["times"]), s=8, color="black", zorder=3)
    ax.set_yticks(range(len(tracks)))
    ax.set_yticklabels([f"Track {i}" for i in range(len(tracks))])
    ax.set_xlabel("Time (s)")
    ax.set_title("Cup tracks — short-lived = potential pour events")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    savefig(fig, args.output, "04_cup_tracks.png")

    # -- Cup centers on crop with A|B divider (moving tracks only) -----------
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

        cmap = plt.cm.get_cmap("tab10")
        track_idx_plotted = 0
        for i, t in enumerate(tracks):
            bboxes = np.array(t["bboxes"])
            cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
            cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
            movement = np.sqrt(cx.std() ** 2 + cy.std() ** 2)
            if movement <= args.movement_threshold:
                continue
            c = cmap(track_idx_plotted % 10)
            ax.scatter(cx, cy, s=20, color=c, alpha=0.7, zorder=3,
                       label=f"Track {i} (Tap {t['tap']})")
            track_idx_plotted += 1

        # Position A/B labels on each side of the divider
        mid_x = (lx1 + lx2) / 2
        ax.text(mid_x / 2, crop_h * 0.07, "TAP A", fontsize=16, fontweight="bold",
                color="black", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax.text(mid_x + (crop_w - mid_x) / 2, crop_h * 0.07, "TAP B", fontsize=16,
                fontweight="bold", color="black", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        ax.set_title("Cup centres by track (moving tracks only)")
        ax.legend(loc="upper right", fontsize=7)
        ax.axis("off")
        plt.tight_layout()
        savefig(fig, args.output, "04b_cup_centers_by_tap.png")

    # -- Track position analysis --------------------------------------------
    print("\nTrack position analysis:")
    for i, t in enumerate(tracks):
        bboxes = np.array(t["bboxes"])
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
        movement = np.sqrt(cx.std() ** 2 + cy.std() ** 2)
        dur = t["times"][-1] - t["times"][0]
        print(f"  Track {i}: avg_pos=({cx.mean():.0f},{cy.mean():.0f})  "
              f"movement={movement:.1f}px  dur={dur:.1f}s  dets={len(t['frames'])}")

    # -- Person + cup co-occurrence -----------------------------------------
    frames_with_person: set = set()
    frames_with_cup: set = set()

    for det in all_detections:
        fidx, t_s, name, conf, *bbox = det
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
    ax.scatter(person_times, [1] * len(person_times), s=12, color="orange", label="person", alpha=0.6)
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
        for i, t in enumerate(tracks):
            bboxes = np.array(t["bboxes"])
            cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
            cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
            movement = np.sqrt(cx.std() ** 2 + cy.std() ** 2)
            dur = t["times"][-1] - t["times"][0]
            overlaps = any(t["times"][-1] >= s and t["times"][0] <= e for s, e in segments)
            status = "POUR" if overlaps and movement > args.movement_threshold else "background"
            print(f"  Track {i} [Tap {t['tap']}]: pos=({cx.mean():.0f},{cy.mean():.0f})  "
                  f"movement={movement:.1f}px  dur={dur:.1f}s  "
                  f"overlaps_person={overlaps}  -> {status}")

    pour_tracks = []
    for i, t in enumerate(tracks):
        bboxes = np.array(t["bboxes"])
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2
        movement = np.sqrt(cx.std() ** 2 + cy.std() ** 2)
        overlaps = (cooccurrence_times and
                    any(t["times"][-1] >= s and t["times"][0] <= e for s, e in segments))
        if overlaps and movement > args.movement_threshold:
            pour_tracks.append({"track_ids": [i], "start": t["times"][0],
                                 "end": t["times"][-1], "movement": movement,
                                 "tap": t["tap"]})

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
              f"duration={dur:.1f}s  from tracks {mp['track_ids']}")

    # -- Final timeline plot ------------------------------------------------
    tap_colors = {"A": "green", "B": "purple", "?": "gray"}
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.scatter(cup_times, [0] * len(cup_times), s=8, color="blue", alpha=0.3, label="cup")
    ax.scatter(person_times, [1] * len(person_times), s=8, color="orange", alpha=0.3,
               label="person")
    for j, mp in enumerate(merged_pours):
        c = tap_colors.get(mp["tap"], "gray")
        ax.axvspan(mp["start"], mp["end"], alpha=0.25, color=c,
                   label=f"Tap {mp['tap']}" if mp["tap"] not in [m["tap"] for m in merged_pours[:j]] else None)
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
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": duration,
        "sample_every": args.sample_every,
        "total_detections": len(all_detections),
        "cup_tracks": len(tracks),
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
