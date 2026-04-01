"""
YOLOE instance-segmentation beer pour detector with tracking.

Uses YOLOE for open-vocabulary *instance segmentation* (masks + boxes) combined
with BoT-SORT tracking.  For each cup/glass detection the script extracts
per-mask pixel statistics (mean RGB, HSV, area, fill ratio) so that the
empty-glass -> beer-glass transition can be detected by colour/brightness shift.

Usage examples
--------------
# First run: select the tap area and divider line interactively
python 06_YOLOE_seg_track.py --video ../data/videos/cerveza2.mp4 \
    --output ../results/YOLOE_seg/cerveza2_track --crop-area

# Re-run with saved coordinates (skip ROI selection)
python 06_YOLOE_seg_track.py --video ../data/videos/cerveza2.mp4 \
    --output ../results/YOLOE_seg/cerveza2_track

# Use a smaller / faster model
python 06_YOLOE_seg_track.py --video ../data/videos/cerveza2.mp4 \
    --output ../results/YOLOE_seg/cerveza2_track --model yoloe-11s-seg.pt

# Custom classes
python 06_YOLOE_seg_track.py --video ../data/videos/cerveza2.mp4 \
    --output ../results/YOLOE_seg/cerveza2_track \
    --classes cup glass "beer glass" person

# Export annotated video clip for a time range
python 06_YOLOE_seg_track.py --video ../data/videos/cerveza2.mp4 \
    --output ../results/YOLOE_seg/cerveza2_track --record-range 50 80
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLOE

from common import (
    select_roi_interactive, select_divider_interactive,
    point_side_of_line, crop_normalized, savefig,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOE instance-segmentation beer pour detector with tracking.",
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
    p.add_argument("--model", default="yoloe-11l-seg.pt",
                   help="YOLOE segmentation model weights.")
    p.add_argument("--classes", nargs="+", default=["cup", "glass", "beer glass", "person"],
                   help="Open-vocabulary classes to detect via text prompt.")
    p.add_argument("--sample-every", type=int, default=1,
                   help="Process every Nth frame (1 = every frame, recommended for tracking).")
    p.add_argument("--preview-second", type=float, default=60.0,
                   help="Timestamp (s) used for the preview frame.")
    p.add_argument("--tracker", default="../config/botsort.yaml",
                   help="Tracker config file (botsort.yaml or bytetrack.yaml).")
    p.add_argument("--conf-threshold", type=float, default=0.15,
                   help="Minimum detection confidence (lower than bbox-only since masks "
                        "give more signal).")

    # Frame saving
    p.add_argument("--save-frames", type=int, default=20,
                   help="Save this many evenly-spaced annotated frames as PNGs.")

    # Annotated video output
    p.add_argument("--record-range", nargs=2, type=float, metavar=("START", "STOP"),
                   help="Optional: export an annotated video for this time range (seconds).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask_pixel_stats(crop_bgr: np.ndarray, mask_bool: np.ndarray):
    """Compute mean RGB and HSV values for pixels under a boolean mask."""
    pixels_bgr = crop_bgr[mask_bool]
    if len(pixels_bgr) == 0:
        return 0, 0, 0, 0, 0, 0
    crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    pixels_hsv = crop_hsv[mask_bool]
    mean_b, mean_g, mean_r = pixels_bgr.mean(axis=0).tolist()
    mean_h, mean_s, mean_v = pixels_hsv.mean(axis=0).tolist()
    return mean_r, mean_g, mean_b, mean_h, mean_s, mean_v


CUP_CLASSES = {"cup", "glass", "beer glass"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # -- Output directory ---------------------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output / "frames"
    frames_dir.mkdir(exist_ok=True)
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

    # -- Load YOLOE ---------------------------------------------------------
    print(f"\nLoading YOLOE segmentation model: {args.model}")
    model = YOLOE(args.model)
    model.set_classes(args.classes)
    print(f"  Detecting classes: {args.classes}")

    # -- Full-video tracking with segmentation ------------------------------
    print(f"\nTracking video every {args.sample_every} frame(s) "
          f"with {args.tracker} ...")
    cap = cv2.VideoCapture(str(args.video))
    frame_idx = 0
    all_detections = []
    track_data = defaultdict(lambda: {
        "frames": [], "times": [], "bboxes": [], "confs": [], "class": None,
        "mask_areas": [], "bbox_areas": [], "fill_ratios": [],
        "mean_rs": [], "mean_gs": [], "mean_bs": [],
        "mean_hs": [], "mean_ss": [], "mean_vs": [],
    })

    # Determine which frames to save as annotated PNGs
    processed_frame_count = total_frames // args.sample_every
    if args.save_frames > 0 and processed_frame_count > 0:
        save_interval = max(1, processed_frame_count // args.save_frames)
    else:
        save_interval = 0
    frames_saved = 0
    processed_count = 0

    # Optional annotated video recording
    rec_start_frame = rec_stop_frame = None
    video_writer = None
    if args.record_range:
        rec_start_frame = int(args.record_range[0] * fps)
        rec_stop_frame = int(args.record_range[1] * fps)
        print(f"Recording annotated video: {args.record_range[0]:.1f}s -> "
              f"{args.record_range[1]:.1f}s  "
              f"(frames {rec_start_frame}-{rec_stop_frame})")

    # Pre-compute divider pixel coords
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
            crop_h, crop_w = crop.shape[:2]
            results = model.track(crop, persist=True, tracker=args.tracker,
                                  conf=args.conf_threshold, verbose=False)[0]

            has_boxes = results.boxes is not None and results.boxes.id is not None
            has_masks = results.masks is not None

            if has_boxes:
                n_det = len(results.boxes)
                for i in range(n_det):
                    box = results.boxes[i]
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls]
                    track_id = int(box.id[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    t_s = frame_idx / fps

                    bbox_area = (x2 - x1) * (y2 - y1)

                    # Extract mask pixel stats if available
                    mask_area = 0
                    fill_ratio = 0.0
                    mean_r = mean_g = mean_b = 0.0
                    mean_h = mean_s = mean_v = 0.0

                    if has_masks and i < len(results.masks.data):
                        mask_tensor = results.masks.data[i].cpu().numpy()
                        mask_resized = cv2.resize(
                            mask_tensor.astype(np.float32),
                            (crop_w, crop_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        mask_bool = mask_resized > 0.5
                        mask_area = int(mask_bool.sum())
                        if bbox_area > 0:
                            fill_ratio = mask_area / bbox_area
                        mean_r, mean_g, mean_b, mean_h, mean_s, mean_v = \
                            mask_pixel_stats(crop, mask_bool)

                    det_record = (
                        frame_idx, t_s, name, conf, track_id,
                        x1, y1, x2, y2,
                        mask_area, bbox_area, fill_ratio,
                        mean_r, mean_g, mean_b,
                        mean_h, mean_s, mean_v,
                    )
                    all_detections.append(det_record)

                    td = track_data[track_id]
                    td["frames"].append(frame_idx)
                    td["times"].append(t_s)
                    td["bboxes"].append((x1, y1, x2, y2))
                    td["confs"].append(conf)
                    td["mask_areas"].append(mask_area)
                    td["bbox_areas"].append(bbox_area)
                    td["fill_ratios"].append(fill_ratio)
                    td["mean_rs"].append(mean_r)
                    td["mean_gs"].append(mean_g)
                    td["mean_bs"].append(mean_b)
                    td["mean_hs"].append(mean_h)
                    td["mean_ss"].append(mean_s)
                    td["mean_vs"].append(mean_v)
                    if td["class"] is None:
                        td["class"] = name

            # Save annotated frame at regular intervals
            should_save_frame = (
                save_interval > 0
                and processed_count % save_interval == 0
                and frames_saved < args.save_frames
            )
            if should_save_frame:
                annotated = results.plot()
                if div_px:
                    cv2.line(annotated, (div_px[0], div_px[1]),
                             (div_px[2], div_px[3]), (0, 0, 255), 2)
                cv2.putText(annotated, f"{frame_idx / fps:.1f}s",
                            (annotated.shape[1] - 120, annotated.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                fname = f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frames_dir / fname), annotated)
                frames_saved += 1

            # Write annotated frame to video if in recording range
            if rec_start_frame is not None and rec_start_frame <= frame_idx <= rec_stop_frame:
                annotated = results.plot()
                if div_px:
                    cv2.line(annotated, (div_px[0], div_px[1]),
                             (div_px[2], div_px[3]), (0, 0, 255), 2)
                cv2.putText(annotated, f"{frame_idx / fps:.1f}s",
                            (annotated.shape[1] - 120, annotated.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if video_writer is None:
                    h_out, w_out = annotated.shape[:2]
                    rec_path = args.output / "annotated_clip.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(rec_path), fourcc, fps / args.sample_every, (w_out, h_out))
                    print(f"  Video writer opened: {rec_path} ({w_out}x{h_out})")
                video_writer.write(annotated)

            processed_count += 1
            if frame_idx % (args.sample_every * 100) == 0:
                print(f"  Frame {frame_idx}/{total_frames}  "
                      f"tracks so far: {len(track_data)}", end="\r")

        frame_idx += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"  Annotated video saved to {args.output / 'annotated_clip.mp4'}")
    print(f"\nTotal detections: {len(all_detections)}  "
          f"Unique track IDs: {len(track_data)}  "
          f"Frames saved: {frames_saved}")

    # -- Save detections to CSV ---------------------------------------------
    csv_path = args.output / "seg_detections.csv"
    columns = [
        "frame", "time_s", "class", "confidence", "track_id",
        "x1", "y1", "x2", "y2",
        "mask_area", "bbox_area", "fill_ratio",
        "mean_r", "mean_g", "mean_b",
        "mean_h", "mean_s", "mean_v",
    ]
    with open(csv_path, "w") as f:
        f.write(",".join(columns) + "\n")
        for det in all_detections:
            vals = [
                str(det[0]), f"{det[1]:.4f}", det[2], f"{det[3]:.4f}", str(det[4]),
                f"{det[5]:.2f}", f"{det[6]:.2f}", f"{det[7]:.2f}", f"{det[8]:.2f}",
                str(det[9]), f"{det[10]:.2f}", f"{det[11]:.4f}",
                f"{det[12]:.2f}", f"{det[13]:.2f}", f"{det[14]:.2f}",
                f"{det[15]:.2f}", f"{det[16]:.2f}", f"{det[17]:.2f}",
            ]
            f.write(",".join(vals) + "\n")
    print(f"Segmentation detections saved to {csv_path}")

    # -- Load CSV back as DataFrame for analysis ----------------------------
    df = pd.read_csv(csv_path)

    # -- Detection timeline -------------------------------------------------
    det_times: dict = {}
    for _, row in df.iterrows():
        det_times.setdefault(row["class"], []).append(row["time_s"])

    fig, ax = plt.subplots(figsize=(14, 4))
    for i, (name, times_list) in enumerate(det_times.items()):
        ax.scatter(times_list, [i] * len(times_list), s=10,
                   label=f"{name} ({len(times_list)})")
    ax.set_yticks(range(len(det_times)))
    ax.set_yticklabels(list(det_times.keys()))
    ax.set_xlabel("Time (s)")
    ax.set_title("Detection timeline — all classes (YOLOE segmentation)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    savefig(fig, args.output, "02_detection_timeline.png")

    # -- Filter cup/glass tracks --------------------------------------------
    cup_classes_in_data = set(df["class"].unique()) & CUP_CLASSES
    cup_tracks = {
        tid: td for tid, td in track_data.items()
        if td["class"] in CUP_CLASSES
    }

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

    print(f"\nCup/glass classes found in data: {cup_classes_in_data or 'none'}")
    print(f"Found {len(cup_tracks)} cup/glass tracks")
    for tid in sorted(cup_tracks.keys()):
        td = cup_tracks[tid]
        dur = td["times"][-1] - td["times"][0] if len(td["times"]) > 1 else 0
        avg_v = np.mean(td["mean_vs"]) if td["mean_vs"] else 0
        avg_s = np.mean(td["mean_ss"]) if td["mean_ss"] else 0
        print(f"  ID {tid:3d} [{td['class']:>10s}] [Tap {td['tap']}]: "
              f"{td['times'][0]:.1f}s -> {td['times'][-1]:.1f}s  "
              f"dur={dur:.1f}s  dets={len(td['frames'])}  "
              f"avg_V={avg_v:.1f}  avg_S={avg_s:.1f}")

    # -- Plot 3: Per-track mask properties over time ------------------------
    if cup_tracks:
        sorted_tids = sorted(cup_tracks.keys())
        n_tracks = len(sorted_tids)
        fig, axes = plt.subplots(n_tracks, 3, figsize=(18, max(4, n_tracks * 3)),
                                 squeeze=False)

        for row, tid in enumerate(sorted_tids):
            td = cup_tracks[tid]
            times = np.array(td["times"])

            ax_v = axes[row, 0]
            ax_v.plot(times, td["mean_vs"], color="gold", linewidth=1, alpha=0.8)
            ax_v.set_ylabel(f"ID {tid}\nmean V")
            ax_v.set_ylim(0, 260)
            if row == 0:
                ax_v.set_title("Brightness (V channel)")

            ax_s = axes[row, 1]
            ax_s.plot(times, td["mean_ss"], color="blue", linewidth=1, alpha=0.8)
            ax_s.set_ylabel("mean S")
            ax_s.set_ylim(0, 260)
            if row == 0:
                ax_s.set_title("Saturation (S channel)")

            ax_a = axes[row, 2]
            ax_a.plot(times, td["mask_areas"], color="green", linewidth=1, alpha=0.8)
            ax_a.set_ylabel("mask area (px)")
            if row == 0:
                ax_a.set_title("Mask area")

            if row == n_tracks - 1:
                ax_v.set_xlabel("Time (s)")
                ax_s.set_xlabel("Time (s)")
                ax_a.set_xlabel("Time (s)")

        plt.tight_layout()
        savefig(fig, args.output, "03_mask_properties_per_track.png")

    # -- Plot 4: Fill-state transition detection ----------------------------
    transitions = {}
    if cup_tracks:
        WINDOW = 15  # frames for rolling average
        for tid in sorted(cup_tracks.keys()):
            td = cup_tracks[tid]
            if len(td["mean_vs"]) < WINDOW * 2:
                continue
            vs = pd.Series(td["mean_vs"]).rolling(WINDOW, min_periods=1).mean()
            ss = pd.Series(td["mean_ss"]).rolling(WINDOW, min_periods=1).mean()

            # Detect a significant brightness drop + saturation rise
            v_diff = vs.diff(periods=WINDOW)
            s_diff = ss.diff(periods=WINDOW)

            # Threshold: brightness drops by >10 AND saturation rises by >5
            fill_mask = (v_diff < -10) & (s_diff > 5)
            fill_indices = np.where(fill_mask.values)[0]

            if len(fill_indices) > 0:
                first_idx = fill_indices[0]
                transitions[tid] = {
                    "frame": td["frames"][first_idx],
                    "time_s": td["times"][first_idx],
                    "v_before": float(vs.iloc[max(0, first_idx - WINDOW)]),
                    "v_after": float(vs.iloc[first_idx]),
                    "s_before": float(ss.iloc[max(0, first_idx - WINDOW)]),
                    "s_after": float(ss.iloc[first_idx]),
                }

        if transitions:
            print(f"\nDetected fill transitions in {len(transitions)} track(s):")
            for tid, tr in transitions.items():
                td = cup_tracks[tid]
                print(f"  ID {tid} [{td['class']}] [Tap {td['tap']}] at {tr['time_s']:.1f}s  "
                      f"V: {tr['v_before']:.0f}->{tr['v_after']:.0f}  "
                      f"S: {tr['s_before']:.0f}->{tr['s_after']:.0f}")
        else:
            print("\nNo clear fill transitions detected by heuristic. "
                  "Check the mask property plots for manual inspection.")

        # Plot smoothed V and S per cup track with transition markers
        n_cup = len(cup_tracks)
        sorted_cup_tids = sorted(cup_tracks.keys())
        fig, axes = plt.subplots(n_cup, 1, figsize=(14, max(4, n_cup * 3)),
                                 squeeze=False)
        for row, tid in enumerate(sorted_cup_tids):
            td = cup_tracks[tid]
            ax = axes[row, 0]
            times = np.array(td["times"])

            vs_smooth = pd.Series(td["mean_vs"]).rolling(WINDOW, min_periods=1).mean()
            ss_smooth = pd.Series(td["mean_ss"]).rolling(WINDOW, min_periods=1).mean()

            ax.plot(times, vs_smooth, color="gold", label="V (brightness)", linewidth=1.5)
            ax.plot(times, ss_smooth, color="blue", label="S (saturation)", linewidth=1.5)
            ax.set_ylabel(f"ID {tid} [{td['class']}]\nTap {td.get('tap', '?')}")
            ax.set_ylim(0, 260)

            if tid in transitions:
                tr_time = transitions[tid]["time_s"]
                ax.axvline(tr_time, color="red", linewidth=2, linestyle="--",
                           label=f"fill @ {tr_time:.1f}s")

            ax.legend(loc="upper right", fontsize=7)
            if row == n_cup - 1:
                ax.set_xlabel("Time (s)")
            if row == 0:
                ax.set_title("Fill-state analysis (smoothed V & S per cup track)")

        plt.tight_layout()
        savefig(fig, args.output, "04_fill_transitions.png")

    # -- Plot 5: Transition grid (key frames) -------------------------------
    if transitions:
        cap = cv2.VideoCapture(str(args.video))
        grid_images = []
        grid_labels = []

        for tid, tr in sorted(transitions.items()):
            td = cup_tracks[tid]
            tr_frame = tr["frame"]

            # Sample frames: well before, just before, at transition, after
            offsets = [-int(fps * 5), -int(fps * 1), 0, int(fps * 2)]
            for off in offsets:
                target = max(0, min(total_frames - 1, tr_frame + off))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, fr = cap.read()
                if ret:
                    cr = crop_normalized(fr, tap_roi)
                    grid_images.append(cv2.cvtColor(cr, cv2.COLOR_BGR2RGB))
                    label = f"ID {tid} @ {target / fps:.1f}s"
                    if off == 0:
                        label += " [FILL]"
                    grid_labels.append(label)

        cap.release()

        if grid_images:
            n_cols = 4
            n_rows = (len(grid_images) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * 4, n_rows * 4))
            axes_flat = np.array(axes).flatten() if n_rows > 1 else (
                [axes] if n_cols == 1 else list(axes))
            for idx, (img, label) in enumerate(zip(grid_images, grid_labels)):
                ax = axes_flat[idx]
                ax.imshow(img)
                ax.set_title(label, fontsize=9)
                ax.axis("off")
            for idx in range(len(grid_images), len(axes_flat)):
                axes_flat[idx].axis("off")
            plt.suptitle("Key frames around detected fill transitions", fontsize=14)
            plt.tight_layout()
            savefig(fig, args.output, "05_transition_grid.png")

    # -- Save summary JSON --------------------------------------------------
    summary = {
        "video": str(args.video),
        "model": args.model,
        "classes": args.classes,
        "tap_roi": list(tap_roi),
        "tap_divider": list(tap_divider) if tap_divider else None,
        "tracker": args.tracker,
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": duration,
        "sample_every": args.sample_every,
        "conf_threshold": args.conf_threshold,
        "total_detections": len(all_detections),
        "unique_track_ids": len(track_data),
        "cup_glass_tracks": len(cup_tracks),
        "cup_classes_detected": sorted(cup_classes_in_data),
        "track_summaries": {
            str(tid): {
                "class": td["class"],
                "tap": td.get("tap", "?"),
                "start_s": td["times"][0],
                "end_s": td["times"][-1],
                "n_detections": len(td["frames"]),
                "avg_mask_area": float(np.mean(td["mask_areas"])) if td["mask_areas"] else 0,
                "avg_mean_v": float(np.mean(td["mean_vs"])) if td["mean_vs"] else 0,
                "avg_mean_s": float(np.mean(td["mean_ss"])) if td["mean_ss"] else 0,
            }
            for tid, td in sorted(cup_tracks.items())
        },
        "fill_transitions": {
            str(tid): tr for tid, tr in transitions.items()
        } if transitions else {},
    }
    summary_path = args.output / "summary_seg.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_path}")
    print(f"All outputs written to: {args.output}/")


if __name__ == "__main__":
    main()
