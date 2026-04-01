"""
Post-processing: re-link fragmented cup tracks from YOLO + BoT-SORT output.

BoT-SORT loses track IDs during occlusions (arm crosses) and detection drops
(lighting changes).  This script merges fragments that belong to the same
physical cup using two complementary strategies, verified with IoU, and
chained transitively via Union-Find.

Strategies
----------
1. **Co-location merge** — tracks whose median bbox centres are within
   COLOC_DIST_PX and whose median bboxes have IoU >= MIN_IOU are the same
   cup.  Handles fragments that appear in detection-gaps of a long-running
   track (e.g. Track 6 + 12 tiny fragments all at the same pixel location).
   A temporal-overlap guard prevents merging two genuinely different cups
   that happen to sit close together.

2. **Sequential merge** — if track A's last frame is < track B's first frame,
   the gap is <= MAX_GAP_FRAMES, the endpoint centres are within MAX_DIST_PX,
   and the endpoint bboxes have IoU >= MIN_IOU, they are the same cup.
   Handles the classic "ID switch after brief occlusion" pattern.

Both feed into the same Union-Find so transitive chains (A→B→C) collapse
into one canonical ID.

After merging, short detection gaps (<= MAX_INTERP_GAP frames) are filled
with linearly-interpolated bounding boxes.

Usage
-----
uv run python notebooks/04_relink_tracks.py \
    --input results/cerveza2_track/raw_detections.csv \
    --output results/cerveza2_track

# Also render an annotated video clip (50s–80s) with relinked IDs:
uv run python notebooks/04_relink_tracks.py \
    --input results/cerveza2_track/raw_detections.csv \
    --output results/cerveza2_track \
    --video data/videos/cerveza2.mp4 \
    --record-range 50 80
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

MAX_GAP_FRAMES = 40  # 2 s at 20 fps
MAX_DIST_PX = 50  # sequential-merge centre distance threshold
COLOC_DIST_PX = 30  # co-location merge median-centre distance threshold
MIN_IOU = 0.2  # bbox IoU gate for both strategies
MAX_INTERP_GAP = 10  # interpolate gaps <= 0.5 s only

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Re-link fragmented cup tracks from raw YOLO detections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to raw_detections.csv from 03_YOLO_track.py.",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for output files (relinked CSV, plots).",
    )
    p.add_argument("--max-gap-frames", type=int, default=MAX_GAP_FRAMES)
    p.add_argument("--max-dist-px", type=float, default=MAX_DIST_PX)
    p.add_argument("--coloc-dist-px", type=float, default=COLOC_DIST_PX)
    p.add_argument("--min-iou", type=float, default=MIN_IOU)
    p.add_argument("--max-interp-gap", type=int, default=MAX_INTERP_GAP)

    # Annotated video output
    p.add_argument(
        "--video", type=Path, help="Path to original video file (required for --record-range)."
    )
    p.add_argument(
        "--record-range",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        help="Export annotated video for this time range in seconds. "
        "E.g. --record-range 50 80.  Requires --video.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bbox_iou(a: tuple, b: tuple) -> float:
    """IoU between two (x1, y1, x2, y2) bounding boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def euclidean(a: tuple, b: tuple) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def build_track_summaries(cups: pd.DataFrame) -> dict:
    """Compute per-track spatial/temporal summary for merge candidate scoring."""
    summaries = {}
    for tid, grp in cups.groupby("track_id"):
        grp = grp.sort_values("frame")
        summaries[int(tid)] = {
            "first_frame": int(grp["frame"].iloc[0]),
            "last_frame": int(grp["frame"].iloc[-1]),
            "n_dets": len(grp),
            "first_center": (float(grp["cx"].iloc[0]), float(grp["cy"].iloc[0])),
            "last_center": (float(grp["cx"].iloc[-1]), float(grp["cy"].iloc[-1])),
            "median_center": (float(grp["cx"].median()), float(grp["cy"].median())),
            "first_bbox": tuple(
                float(v)
                for v in (
                    grp["x1"].iloc[0],
                    grp["y1"].iloc[0],
                    grp["x2"].iloc[0],
                    grp["y2"].iloc[0],
                )
            ),
            "last_bbox": tuple(
                float(v)
                for v in (
                    grp["x1"].iloc[-1],
                    grp["y1"].iloc[-1],
                    grp["x2"].iloc[-1],
                    grp["y2"].iloc[-1],
                )
            ),
            "median_bbox": tuple(
                float(v)
                for v in (
                    grp["x1"].median(),
                    grp["y1"].median(),
                    grp["x2"].median(),
                    grp["y2"].median(),
                )
            ),
            "frames_set": set(grp["frame"].astype(int)),
        }
    return summaries


def savefig(fig, out_dir: Path, name: str):
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def crop_normalized(frame: np.ndarray, roi: tuple) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    return frame[y1:y2, x1:x2]


# 10 distinct colours (BGR) for drawing boxes, cycled by track_id
_TRACK_COLORS = [
    (230, 159, 23),  # blue
    (34, 200, 78),  # green
    (50, 70, 230),  # red
    (210, 180, 60),  # cyan-ish
    (100, 60, 220),  # magenta
    (50, 210, 210),  # yellow
    (180, 105, 255),  # pink
    (255, 160, 80),  # light blue
    (80, 255, 160),  # light green
    (140, 140, 255),  # salmon
]


def draw_detections(frame: np.ndarray, dets: pd.DataFrame, track_color_map: dict) -> np.ndarray:
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
        cv2.putText(
            out,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Annotated video
# ---------------------------------------------------------------------------


def _render_annotated_video(args, df: pd.DataFrame):
    """Read the original video frame-by-frame and render relinked detections."""
    roi_json = args.output / "tap_roi.json"
    if not roi_json.exists():
        print(f"  WARNING: {roi_json} not found — cannot crop. Skipping video.")
        return
    roi_cfg = json.loads(roi_json.read_text())
    tap_roi = tuple(roi_cfg["tap_roi"])
    tap_divider = tuple(roi_cfg["tap_divider"]) if "tap_divider" in roi_cfg else None

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rec_start = int(args.record_range[0] * fps)
    rec_stop = int(args.record_range[1] * fps)
    print(
        f"\nRendering annotated video: {args.record_range[0]:.1f}s → "
        f"{args.record_range[1]:.1f}s  (frames {rec_start}–{rec_stop})"
    )

    # Build per-frame lookup from relinked detections
    frame_dets: dict[int, pd.DataFrame] = {}
    for f, grp in df.groupby("frame"):
        f = int(f)
        if rec_start <= f <= rec_stop:
            frame_dets[f] = grp

    # Assign stable colour per relinked track_id
    unique_tids = sorted(df["track_id"].unique())
    track_color_map = {
        int(tid): _TRACK_COLORS[i % len(_TRACK_COLORS)] for i, tid in enumerate(unique_tids)
    }

    # Compute divider pixel coords (on crop)
    ref_frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_idx)
    ret, ref_frame = cap.read()
    crop_ref = crop_normalized(ref_frame, tap_roi)
    ch, cw = crop_ref.shape[:2]

    div_px = None
    if tap_divider:
        div_px = (
            int(tap_divider[0] * cw),
            int(tap_divider[1] * ch),
            int(tap_divider[2] * cw),
            int(tap_divider[3] * ch),
        )

    # Seek to start and iterate
    cap.set(cv2.CAP_PROP_POS_FRAMES, rec_start)
    video_writer = None
    rec_path = args.output / "relinked_clip.mp4"

    for fidx in range(rec_start, min(rec_stop + 1, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        crop = crop_normalized(frame, tap_roi)

        # Draw detections for this frame
        if fidx in frame_dets:
            crop = draw_detections(crop, frame_dets[fidx], track_color_map)

        # Draw A|B divider
        if div_px:
            cv2.line(crop, (div_px[0], div_px[1]), (div_px[2], div_px[3]), (0, 0, 255), 2)
            cv2.putText(
                crop, "A", (div_px[0] // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
            cv2.putText(
                crop,
                "B",
                ((div_px[0] + div_px[2]) // 2 + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 255),
                2,
            )

        # Timestamp
        cv2.putText(
            crop,
            f"{fidx / fps:.1f}s",
            (crop.shape[1] - 120, crop.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if video_writer is None:
            h_out, w_out = crop.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(rec_path), fourcc, fps, (w_out, h_out))
            print(f"  Video writer opened: {rec_path} ({w_out}x{h_out})")
        video_writer.write(crop)

        if (fidx - rec_start) % 100 == 0:
            print(f"  Frame {fidx}/{rec_stop}", end="\r")

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"  Annotated video saved to {rec_path}                ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw detections ─────────────────────────────────────────────
    df = pd.read_csv(args.input)
    df["cx"] = (df["x1"] + df["x2"]) / 2
    df["cy"] = (df["y1"] + df["y2"]) / 2

    cups = df[df["class"] == "cup"].copy()
    tids = sorted(int(t) for t in cups["track_id"].unique())
    print(f"Loaded {len(df)} detections  |  {len(cups)} cup rows  |  {len(tids)} cup track IDs")

    # ── 2. Per-track summaries ─────────────────────────────────────────────
    summaries = build_track_summaries(cups)
    print("\nTrack summaries:")
    for tid in tids:
        s = summaries[tid]
        print(
            f"  Track {tid:3d}:  frames {s['first_frame']:4d}–{s['last_frame']:4d}  "
            f"dets={s['n_dets']:4d}  "
            f"median=({s['median_center'][0]:.0f},{s['median_center'][1]:.0f})"
        )

    # ── 3. Pairwise candidate matching ─────────────────────────────────────
    uf = UnionFind()
    merges: list[tuple] = []

    for i, a_id in enumerate(tids):
        a = summaries[a_id]
        for b_id in tids[i + 1 :]:
            b = summaries[b_id]

            # --- Strategy 1: co-location (fragments sharing a position) ---
            med_dist = euclidean(a["median_center"], b["median_center"])
            if med_dist <= args.coloc_dist_px:
                iou = bbox_iou(a["median_bbox"], b["median_bbox"])
                if iou >= args.min_iou:
                    overlap_frames = a["frames_set"] & b["frames_set"]
                    min_dets = min(a["n_dets"], b["n_dets"])
                    # allow tiny temporal overlap (tracker flicker), but not
                    # sustained co-detection which signals distinct objects
                    if len(overlap_frames) <= max(2, int(min_dets * 0.3)):
                        uf.union(a_id, b_id)
                        merges.append(
                            (
                                a_id,
                                b_id,
                                "coloc",
                                f"dist={med_dist:.1f}  iou={iou:.2f}  "
                                f"overlap_frames={len(overlap_frames)}",
                            )
                        )
                        continue

            # --- Strategy 2: sequential (A ends → B starts nearby) --------
            for first_id, second_id in ((a_id, b_id), (b_id, a_id)):
                fa, fb = summaries[first_id], summaries[second_id]
                if fa["last_frame"] >= fb["first_frame"]:
                    continue
                gap = fb["first_frame"] - fa["last_frame"]
                if gap > args.max_gap_frames:
                    continue
                dist = euclidean(fa["last_center"], fb["first_center"])
                if dist > args.max_dist_px:
                    continue
                iou = bbox_iou(fa["last_bbox"], fb["first_bbox"])
                if iou >= args.min_iou:
                    uf.union(a_id, b_id)
                    merges.append(
                        (
                            first_id,
                            second_id,
                            "sequential",
                            f"gap={gap}  dist={dist:.1f}  iou={iou:.2f}",
                        )
                    )
                    break  # already merged, no need to check reverse

    # ── 4. Build old→canonical mapping via Union-Find ──────────────────────
    print(f"\nMerge edges: {len(merges)}")
    for a_id, b_id, method, detail in merges:
        print(f"  {a_id:3d} ↔ {b_id:3d}  [{method:10s}]  {detail}")

    groups: dict[int, list[int]] = {}
    for tid in tids:
        root = uf.find(tid)
        groups.setdefault(root, []).append(tid)

    mapping: dict[int, int] = {}
    for members in groups.values():
        canonical = min(members)
        for tid in members:
            mapping[tid] = canonical

    print(f"\nMerge result: {len(tids)} tracks → {len(groups)} physical cups")
    for root in sorted(groups, key=lambda r: min(groups[r])):
        members = sorted(groups[root])
        canonical = min(members)
        total_dets = sum(summaries[t]["n_dets"] for t in members)
        if len(members) > 1:
            print(f"  Cup {canonical:3d} ← {members}  ({total_dets} dets)")
        else:
            print(f"  Cup {canonical:3d}    (solo, {total_dets} dets)")

    # ── 5. Apply mapping (only cup IDs remapped; other classes unchanged) ──
    df["original_track_id"] = df["track_id"]
    df["track_id"] = df["track_id"].map(lambda t: mapping.get(int(t), int(t)))
    df["interpolated"] = False

    # ── 6. Interpolate short gaps ──────────────────────────────────────────
    cup_mask = df["class"] == "cup"
    interp_rows: list[dict] = []

    for tid in sorted(df.loc[cup_mask, "track_id"].unique()):
        track = df.loc[cup_mask & (df["track_id"] == tid)].sort_values("frame")
        if len(track) < 2:
            continue
        frames = track["frame"].values.astype(int)
        for idx in range(len(frames) - 1):
            gap = frames[idx + 1] - frames[idx]
            if gap <= 1 or gap > args.max_interp_gap:
                continue
            ra = track.iloc[idx]
            rb = track.iloc[idx + 1]
            for f in range(frames[idx] + 1, frames[idx + 1]):
                alpha = (f - frames[idx]) / gap
                interp_rows.append(
                    {
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
                    }
                )

    if interp_rows:
        df = pd.concat([df, pd.DataFrame(interp_rows)], ignore_index=True)
        df = df.sort_values(["frame", "track_id"]).reset_index(drop=True)
    print(f"\nInterpolated {len(interp_rows)} rows across short cup-track gaps")

    # ── 7. Save relinked_detections.csv ────────────────────────────────────
    out_cols = [
        "frame",
        "time_s",
        "class",
        "confidence",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "original_track_id",
        "interpolated",
    ]
    out_csv = args.output / "relinked_detections.csv"
    df[out_cols].to_csv(out_csv, index=False)

    relinked_cups = df[df["class"] == "cup"]
    print(f"\nSaved {out_csv}")
    print(f"  Total rows:       {len(df)}")
    print(
        f"  Cup rows:         {len(relinked_cups)} "
        f"({relinked_cups['interpolated'].sum()} interpolated)"
    )
    print(f"  Unique cup IDs:   {relinked_cups['track_id'].nunique()}")

    # ── 8. Before / after timeline visualisation ───────────────────────────
    cups_before = pd.read_csv(args.input)
    cups_before = cups_before[cups_before["class"] == "cup"]
    tids_before = sorted(cups_before["track_id"].unique())

    cups_after = df[df["class"] == "cup"]
    tids_after = sorted(cups_after["track_id"].unique())

    n_before = len(tids_before)
    n_after = len(tids_after)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, max(6, (n_before + n_after) * 0.28)),
        sharex=True,
        gridspec_kw={"height_ratios": [n_before, max(n_after, 1)]},
    )

    # -- Before panel
    for row, tid in enumerate(tids_before):
        sub = cups_before[cups_before["track_id"] == tid]
        t0, t1 = sub["time_s"].min(), sub["time_s"].max()
        ax1.barh(row, t1 - t0, left=t0, height=0.6, alpha=0.7)
        ax1.scatter(sub["time_s"], [row] * len(sub), s=4, color="black", zorder=3)
    ax1.set_yticks(range(n_before))
    ax1.set_yticklabels([f"ID {t}" for t in tids_before], fontsize=7)
    ax1.set_title(f"BEFORE re-linking — {n_before} cup track IDs")
    ax1.set_ylabel("Track ID")

    # -- After panel
    cmap = plt.colormaps["tab10"]
    for row, tid in enumerate(tids_after):
        sub = cups_after[cups_after["track_id"] == tid]
        real = sub[~sub["interpolated"]]
        interp = sub[sub["interpolated"]]
        t0, t1 = sub["time_s"].min(), sub["time_s"].max()
        ax2.barh(row, t1 - t0, left=t0, height=0.6, alpha=0.5, color=cmap(row % 10))
        ax2.scatter(real["time_s"], [row] * len(real), s=6, color="black", zorder=3)
        if len(interp):
            ax2.scatter(
                interp["time_s"],
                [row] * len(interp),
                s=4,
                color="red",
                zorder=3,
                alpha=0.5,
                marker="x",
            )
    ax2.set_yticks(range(n_after))
    ax2.set_yticklabels([f"Cup {t}" for t in tids_after], fontsize=9)
    ax2.set_title(f"AFTER re-linking — {n_after} physical cups  (red × = interpolated)")
    ax2.set_ylabel("Cup ID")
    ax2.set_xlabel("Time (s)")

    plt.tight_layout()
    savefig(fig, args.output, "07_relink_before_after.png")

    # ── 9. Annotated video export ──────────────────────────────────────────
    if args.record_range:
        if not args.video:
            print("\nERROR: --record-range requires --video <path>")
        elif not args.video.exists():
            print(f"\nERROR: video not found: {args.video}")
        else:
            _render_annotated_video(args, df)

    print(f"\n{'=' * 60}")
    print(f"  BEFORE: {n_before} cup track IDs")
    print(f"  AFTER:  {n_after} physical cups")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
