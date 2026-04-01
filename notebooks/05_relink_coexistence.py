"""
Re-link fragmented cup tracks using temporal co-existence constraints
and spatial affinity.

The BoT-SORT tracker loses IDs during occlusions (bartender's arm) and
detection drops.  This script merges fragments belonging to the same
physical cup by exploiting two observations:

1. Tracks of *different* cups overlap in time (co-exist) — they cannot
   be fragments of the same cup.
2. Tracks of the *same* cup share a similar spatial region (median
   centre over the track lifetime).

Algorithm
---------
1. Build a temporal incompatibility graph: tracks overlapping by more
   than ``--overlap-threshold`` frames are declared incompatible
   (definitely different cups).
2. Greedy graph colouring (sorted by start frame) assigns each track
   to a "cup group".  When a track is compatible with multiple groups,
   the group whose spatial centroid is nearest (by median centre) is
   chosen.
3. Within each group, tracks are ordered by time and assigned a single
   canonical ID.
4. Short detection gaps (≤ ``--max-interp-gap`` frames) are filled with
   linearly-interpolated bounding boxes.

Usage
-----
source .venv-gambooza/bin/activate
python notebooks/05_relink_coexistence.py \
    --input results/YOLOworld/cerveza2_track/raw_detections.csv \
    --output results/YOLOworld/cerveza2_track

# Also render an annotated video clip (50s–80s) with relinked IDs:
python notebooks/05_relink_coexistence.py \
    --input results/YOLOworld/cerveza2_track/raw_detections.csv \
    --output results/YOLOworld/cerveza2_track \
    --video data/videos/cerveza2.mp4 \
    --record-range 50 80
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import crop_normalized, savefig

# ── Defaults ────────────────────────────────────────────────────────────────

OVERLAP_THRESHOLD = 15  # frames of co-existence → incompatible
MIN_TRACK_DETS = 2      # ignore single-detection noise
MAX_INTERP_GAP = 10     # interpolate gaps ≤ 0.5 s

# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Re-link fragmented cup tracks via temporal co-existence "
                    "constraints and spatial affinity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="raw_detections.csv from 03_YOLO_track.py")
    p.add_argument("--output", required=True, type=Path,
                   help="Output directory")
    p.add_argument("--overlap-threshold", type=int, default=OVERLAP_THRESHOLD,
                   help="Minimum frame overlap to declare two tracks incompatible.")
    p.add_argument("--min-track-dets", type=int, default=MIN_TRACK_DETS,
                   help="Ignore tracks with fewer detections than this.")
    p.add_argument("--max-interp-gap", type=int, default=MAX_INTERP_GAP,
                   help="Interpolate detection gaps up to this many frames.")
    p.add_argument("--video", type=Path,
                   help="Path to original video file (required for --record-range).")
    p.add_argument("--record-range", nargs=2, type=float,
                   metavar=("START", "STOP"),
                   help="Export annotated video for this time range in seconds. "
                        "E.g. --record-range 50 80.  Requires --video.")
    return p.parse_args()

# ── Helpers ─────────────────────────────────────────────────────────────────


def build_track_info(cups: pd.DataFrame, min_dets: int) -> dict:
    """Per-track temporal and spatial summary."""
    info = {}
    for tid, grp in cups.groupby("track_id"):
        grp = grp.sort_values("frame")
        if len(grp) < min_dets:
            continue
        tid = int(tid)
        info[tid] = {
            "first_frame": int(grp["frame"].iloc[0]),
            "last_frame":  int(grp["frame"].iloc[-1]),
            "frames":      set(grp["frame"].astype(int)),
            "median_cx":   float(grp["cx"].median()),
            "median_cy":   float(grp["cy"].median()),
            "n_dets":      len(grp),
        }
    return info


def build_incompatibility(tids: list[int], info: dict,
                          threshold: int) -> set[tuple[int, int]]:
    """Pairs (a, b) with a < b whose temporal overlap exceeds *threshold*."""
    incompat = set()
    for a, b in combinations(tids, 2):
        overlap = len(info[a]["frames"] & info[b]["frames"])
        if overlap > threshold:
            incompat.add((min(a, b), max(a, b)))
    return incompat


def greedy_coexistence_groups(
    tids: list[int],
    info: dict,
    incompat: set[tuple[int, int]],
) -> list[list[int]]:
    """Greedy interval-graph colouring guided by spatial affinity.

    Tracks are processed in start-time order.  Each track joins the
    compatible group whose median-centre centroid is nearest; if no
    compatible group exists a new group is created.
    """
    sorted_tids = sorted(tids, key=lambda t: (info[t]["first_frame"], t))
    groups: list[list[int]] = []

    for tid in sorted_tids:
        t = info[tid]

        compatible_gi = []
        for gi, group in enumerate(groups):
            if all(
                (min(tid, other), max(tid, other)) not in incompat
                for other in group
            ):
                compatible_gi.append(gi)

        if not compatible_gi:
            groups.append([tid])
        elif len(compatible_gi) == 1:
            groups[compatible_gi[0]].append(tid)
        else:
            best_gi, best_dist = compatible_gi[0], float("inf")
            for gi in compatible_gi:
                centroid_x = np.median([info[m]["median_cx"] for m in groups[gi]])
                centroid_y = np.median([info[m]["median_cy"] for m in groups[gi]])
                d = np.hypot(t["median_cx"] - centroid_x,
                             t["median_cy"] - centroid_y)
                if d < best_dist:
                    best_dist = d
                    best_gi = gi
            groups[best_gi].append(tid)

    return groups


_TRACK_COLORS = [
    (230, 159,  23),  # blue
    ( 34, 200,  78),  # green
    ( 50,  70, 230),  # red
    (210, 180,  60),  # cyan-ish
    (100,  60, 220),  # magenta
    ( 50, 210, 210),  # yellow
    (180, 105, 255),  # pink
    (255, 160,  80),  # light blue
    ( 80, 255, 160),  # light green
    (140, 140, 255),  # salmon
]


def draw_detections(frame: np.ndarray, dets: pd.DataFrame,
                    track_color_map: dict) -> np.ndarray:
    """Draw bounding boxes + labels for all detections on a frame."""
    out = frame.copy()
    for _, row in dets.iterrows():
        tid = int(row["track_id"])
        color = track_color_map.get(tid, (200, 200, 200))
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        thickness = 1 if row.get("interpolated", False) else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        cls = str(row["class"])
        conf = row["confidence"]
        label = f"{cls} #{tid}" if conf > 0 else f"{cls} #{tid} (interp)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


def render_annotated_video(video_path: Path, output_dir: Path,
                           record_range: tuple, df: pd.DataFrame):
    """Read the original video frame-by-frame and render relinked detections."""
    roi_json = output_dir / "tap_roi.json"
    if not roi_json.exists():
        print(f"  WARNING: {roi_json} not found — cannot crop. Skipping video.")
        return
    roi_cfg = json.loads(roi_json.read_text())
    tap_roi = tuple(roi_cfg["tap_roi"])
    tap_divider = tuple(roi_cfg["tap_divider"]) if "tap_divider" in roi_cfg else None

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rec_start = int(record_range[0] * fps)
    rec_stop = int(record_range[1] * fps)
    print(f"\nRendering annotated video: {record_range[0]:.1f}s → "
          f"{record_range[1]:.1f}s  (frames {rec_start}–{rec_stop})")

    frame_dets: dict[int, pd.DataFrame] = {}
    for f, grp in df.groupby("frame"):
        f = int(f)
        if rec_start <= f <= rec_stop:
            frame_dets[f] = grp

    unique_tids = sorted(df["track_id"].unique())
    track_color_map = {int(tid): _TRACK_COLORS[i % len(_TRACK_COLORS)]
                       for i, tid in enumerate(unique_tids)}

    ref_frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    ret, ref_frame = cap.read()
    crop_ref = crop_normalized(ref_frame, tap_roi)
    ch, cw = crop_ref.shape[:2]

    div_px = None
    if tap_divider:
        div_px = (int(tap_divider[0] * cw), int(tap_divider[1] * ch),
                  int(tap_divider[2] * cw), int(tap_divider[3] * ch))

    cap.set(cv2.CAP_PROP_POS_FRAMES, rec_start)
    video_writer = None
    rec_path = output_dir / "relinked_clip.mp4"

    for fidx in range(rec_start, min(rec_stop + 1, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        crop = crop_normalized(frame, tap_roi)

        if fidx in frame_dets:
            crop = draw_detections(crop, frame_dets[fidx], track_color_map)

        if div_px:
            cv2.line(crop, (div_px[0], div_px[1]), (div_px[2], div_px[3]),
                     (0, 0, 255), 2)
            cv2.putText(crop, "A", (div_px[0] // 2, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(crop, "B",
                        ((div_px[0] + div_px[2]) // 2 + 20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

        cv2.putText(crop, f"{fidx / fps:.1f}s",
                    (crop.shape[1] - 120, crop.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if video_writer is None:
            h_out, w_out = crop.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(rec_path), fourcc, fps,
                                           (w_out, h_out))
            print(f"  Video writer opened: {rec_path} ({w_out}x{h_out})")
        video_writer.write(crop)

        if (fidx - rec_start) % 100 == 0:
            print(f"  Frame {fidx}/{rec_stop}", end="\r")

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"  Annotated video saved to {rec_path}                ")


# ── Core relink function (importable by pipeline.py) ───────────────────────


def run_relink(
    input_csv: Path,
    output_dir: Path,
    overlap_threshold: int = OVERLAP_THRESHOLD,
    min_track_dets: int = MIN_TRACK_DETS,
    max_interp_gap: int = MAX_INTERP_GAP,
    video_path: Path | None = None,
    record_range: tuple | None = None,
) -> Path:
    """Run track relinking and return path to relinked_detections.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw detections ──────────────────────────────────────────
    df = pd.read_csv(input_csv)
    df["cx"] = (df["x1"] + df["x2"]) / 2
    df["cy"] = (df["y1"] + df["y2"]) / 2

    cups = df[df["class"] == "cup"].copy()
    print(f"Loaded {len(df)} detections  |  {len(cups)} cup rows")

    # ── 2. Per-track summaries ──────────────────────────────────────────
    info = build_track_info(cups, min_track_dets)
    tids = sorted(info.keys())
    print(f"\n{len(tids)} cup tracks (≥ {min_track_dets} detections):")
    for tid in tids:
        t = info[tid]
        print(f"  Track {tid:3d}: frames {t['first_frame']:4d}–{t['last_frame']:4d}  "
              f"dets={t['n_dets']:4d}  "
              f"median=({t['median_cx']:.0f}, {t['median_cy']:.0f})")

    # ── 3. Temporal incompatibility graph ───────────────────────────────
    incompat = build_incompatibility(tids, info, overlap_threshold)
    print(f"\nIncompatible pairs (overlap > {overlap_threshold} frames):")
    for a, b in sorted(incompat):
        overlap = len(info[a]["frames"] & info[b]["frames"])
        print(f"  {a:3d} ↔ {b:3d}  ({overlap} frames overlap)")

    # ── 4. Greedy coexistence grouping ──────────────────────────────────
    groups = greedy_coexistence_groups(tids, info, incompat)

    mapping: dict[int, int] = {}
    for group in groups:
        canonical = min(group)
        for tid in group:
            mapping[tid] = canonical

    print(f"\nGrouping: {len(tids)} tracks → {len(groups)} physical cups")
    for group in sorted(groups, key=lambda g: min(g)):
        canonical = min(group)
        chain = sorted(group, key=lambda t: info[t]["first_frame"])
        total_dets = sum(info[t]["n_dets"] for t in group)
        if len(group) > 1:
            print(f"  Cup {canonical:3d} ← {chain}  ({total_dets} dets)")
        else:
            print(f"  Cup {canonical:3d}    (solo, {total_dets} dets)")

    # ── 5. Apply mapping ────────────────────────────────────────────────
    df["original_track_id"] = df["track_id"]
    df["track_id"] = df["track_id"].map(
        lambda t: mapping.get(int(t), int(t))
    )
    df["interpolated"] = False

    # ── 6. Interpolate short gaps ───────────────────────────────────────
    cup_mask = df["class"] == "cup"
    interp_rows: list[dict] = []

    for tid in sorted(df.loc[cup_mask, "track_id"].unique()):
        track = df.loc[cup_mask & (df["track_id"] == tid)].sort_values("frame")
        if len(track) < 2:
            continue
        frames = track["frame"].values.astype(int)
        for idx in range(len(frames) - 1):
            gap = frames[idx + 1] - frames[idx]
            if gap <= 1 or gap > max_interp_gap:
                continue
            ra, rb = track.iloc[idx], track.iloc[idx + 1]
            for f in range(frames[idx] + 1, frames[idx + 1]):
                alpha = (f - frames[idx]) / gap
                interp_rows.append({
                    "frame": f,
                    "time_s": ra["time_s"] + alpha * (rb["time_s"] - ra["time_s"]),
                    "class": "cup",
                    "confidence": 0.0,
                    "track_id": int(tid),
                    "x1": ra["x1"] + alpha * (rb["x1"] - ra["x1"]),
                    "y1": ra["y1"] + alpha * (rb["y1"] - ra["y1"]),
                    "x2": ra["x2"] + alpha * (rb["x2"] - ra["x2"]),
                    "y2": ra["y2"] + alpha * (rb["y2"] - ra["y2"]),
                    "original_track_id": int(tid),
                    "interpolated": True,
                })

    if interp_rows:
        df = pd.concat([df, pd.DataFrame(interp_rows)], ignore_index=True)
        df = df.sort_values(["frame", "track_id"]).reset_index(drop=True)
    print(f"\nInterpolated {len(interp_rows)} rows across short gaps")

    # Recompute centres (needed for interpolated rows)
    df["cx"] = (df["x1"] + df["x2"]) / 2
    df["cy"] = (df["y1"] + df["y2"]) / 2

    # ── 7. Save relinked CSV ────────────────────────────────────────────
    out_cols = ["frame", "time_s", "class", "confidence", "track_id",
                "x1", "y1", "x2", "y2", "original_track_id", "interpolated"]
    out_csv = output_dir / "relinked_detections.csv"
    df[out_cols].to_csv(out_csv, index=False)

    relinked_cups = df[df["class"] == "cup"]
    print(f"\nSaved {out_csv}")
    print(f"  Total rows:       {len(df)}")
    print(f"  Cup rows:         {len(relinked_cups)} "
          f"({relinked_cups['interpolated'].sum()} interpolated)")
    print(f"  Unique cup IDs:   {sorted(relinked_cups['track_id'].unique())}")

    # ── 8. Before / after timeline ──────────────────────────────────────
    cups_before = pd.read_csv(input_csv)
    cups_before = cups_before[cups_before["class"] == "cup"]
    tids_before = sorted(cups_before["track_id"].unique())

    cups_after = relinked_cups
    tids_after = sorted(cups_after["track_id"].unique())

    n_before, n_after = len(tids_before), len(tids_after)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(16, max(6, (n_before + n_after) * 0.28)),
        sharex=True,
        gridspec_kw={"height_ratios": [n_before, max(n_after, 1)]},
    )

    for row, tid in enumerate(tids_before):
        sub = cups_before[cups_before["track_id"] == tid]
        t0, t1 = sub["time_s"].min(), sub["time_s"].max()
        ax1.barh(row, t1 - t0, left=t0, height=0.6, alpha=0.7)
        ax1.scatter(sub["time_s"], [row] * len(sub), s=4, color="black", zorder=3)
    ax1.set_yticks(range(n_before))
    ax1.set_yticklabels([f"ID {t}" for t in tids_before], fontsize=7)
    ax1.set_title(f"BEFORE re-linking — {n_before} cup track IDs")
    ax1.set_ylabel("Track ID")

    cmap = plt.colormaps["tab10"]
    for row, tid in enumerate(tids_after):
        sub = cups_after[cups_after["track_id"] == tid]
        real = sub[~sub["interpolated"]]
        interp = sub[sub["interpolated"]]
        t0, t1 = sub["time_s"].min(), sub["time_s"].max()
        ax2.barh(row, t1 - t0, left=t0, height=0.6, alpha=0.5,
                 color=cmap(row % 10))
        ax2.scatter(real["time_s"], [row] * len(real),
                    s=6, color="black", zorder=3)
        if len(interp):
            ax2.scatter(interp["time_s"], [row] * len(interp),
                        s=4, color="red", zorder=3, alpha=0.5, marker="x")
    ax2.set_yticks(range(n_after))
    ax2.set_yticklabels([f"Cup {t}" for t in tids_after], fontsize=9)
    ax2.set_title(f"AFTER re-linking — {n_after} physical cups  "
                  f"(red × = interpolated)")
    ax2.set_ylabel("Cup ID")
    ax2.set_xlabel("Time (s)")

    plt.tight_layout()
    savefig(fig, output_dir, "07_relink_before_after.png")

    # ── 9. x / y vs frame (after re-linking) ────────────────────────────
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(12, 8))
    for tid in tids_after:
        sub = cups_after[cups_after["track_id"] == tid].sort_values("frame")
        ax_x.plot(sub["frame"], sub["cx"], marker=".", markersize=3,
                  label=f"Cup {tid}")
        ax_y.plot(sub["frame"], sub["cy"], marker=".", markersize=3,
                  label=f"Cup {tid}")

    ax_x.set_ylabel("center_x")
    ax_x.set_title("center_x vs frame — after re-linking")
    ax_x.legend(fontsize="small")
    ax_x.grid(True)

    ax_y.set_ylabel("center_y")
    ax_y.set_xlabel("frame")
    ax_y.set_title("center_y vs frame — after re-linking")
    ax_y.legend(fontsize="small")
    ax_y.grid(True)

    plt.tight_layout()
    savefig(fig, output_dir, "08_relinked_xy_vs_frame.png")

    # ── 10. Annotated video export ──────────────────────────────────────
    if record_range:
        if not video_path:
            print("\nERROR: record_range requires video_path")
        elif not video_path.exists():
            print(f"\nERROR: video not found: {video_path}")
        else:
            render_annotated_video(video_path, output_dir, record_range, df)

    print(f"\n{'=' * 60}")
    print(f"  BEFORE: {n_before} cup track IDs")
    print(f"  AFTER:  {n_after} physical cups")
    print(f"{'=' * 60}")

    return out_csv


# ── Main (standalone CLI) ──────────────────────────────────────────────────


def main():
    args = parse_args()
    run_relink(
        input_csv=args.input,
        output_dir=args.output,
        overlap_threshold=args.overlap_threshold,
        min_track_dets=args.min_track_dets,
        max_interp_gap=args.max_interp_gap,
        video_path=args.video if hasattr(args, "video") and args.video else None,
        record_range=tuple(args.record_range) if args.record_range else None,
    )


if __name__ == "__main__":
    main()
