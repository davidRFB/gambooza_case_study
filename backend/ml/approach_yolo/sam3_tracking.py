"""
SAM3 video propagation for beer tap handle tracking.

Uses SAM3VideoPredictor to propagate segmentation masks across video frames
using a memory bank, initialized with bounding boxes drawn on the first frame.
Extracts centroid trajectories to detect tap ON/OFF state transitions.

Based on results/YOLOE_seg/cerveza2_track/sam3_semantic_test.py.

Usage (standalone):
    python notebooks/sam3_tracking.py --config config/pipeline.yaml

Or imported by pipeline.py:
    from sam3_tracking import run_sam3_video_tracking
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from backend.ml.common import (
    crop_normalized, export_cropped_video, savefig,
    select_tap_bboxes_interactive,
)


# ---------------------------------------------------------------------------
# Core SAM3 tracking function
# ---------------------------------------------------------------------------

def run_sam3_video_tracking(
    video_path: Path,
    output_dir: Path,
    tap_roi: tuple,
    tap_bboxes: list[list[float]],
    object_labels: list[str] | None = None,
    colors: list[list[int]] | None = None,
    model_path: str = "data/models/sam3.pt",
    max_frames: int | None = None,
    frame_skip: int = 5,
    save_snapshot_every: int = 50,
    half: bool = True,
    frame_ranges: list[tuple[int, int]] | None = None,
) -> Path:
    """Run SAM3VideoPredictor on the cropped video region.

    1. Export a cropped video (ROI only) to a temp file.
    2. Initialize SAM3VideoPredictor with bounding boxes on the first cropped frame.
    3. Propagate masks and extract per-frame centroids.
    4. Save outputs: sam3_tracked.mp4, sam3_centroids.csv, centroid_trajectory.png

    When *frame_ranges* is provided (list of (start_frame, end_frame) tuples),
    only frames within those ranges are written to the output video. All frames
    are still processed by SAM for propagation continuity, but only the relevant
    segments appear in the output video.

    Returns path to sam3_centroids.csv.
    """
    from ultralytics.models.sam import SAM3VideoPredictor

    if object_labels is None:
        object_labels = ["TAP_A", "TAP_B"]
    if colors is None:
        colors = [[0, 255, 0], [0, 128, 255]]

    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Export cropped video -----------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()

    cropped_video_path = output_dir / "cropped_for_sam3.mp4"
    if not cropped_video_path.exists():
        print("\nExporting cropped video for SAM3...")
        export_cropped_video(video_path, cropped_video_path, tap_roi, fps)
    else:
        print(f"\nUsing existing cropped video: {cropped_video_path}")

    # -- Read first frame for dimensions ------------------------------------
    cap = cv2.VideoCapture(str(cropped_video_path))
    ret, first_frame = cap.read()
    vid_h, vid_w = first_frame.shape[:2]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert ret, f"Could not read {cropped_video_path}"

    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    # Build set of frames to include in the output video
    video_frame_set: set[int] | None = None
    if frame_ranges is not None:
        video_frame_set = set()
        for rng_start, rng_end in frame_ranges:
            video_frame_set.update(range(rng_start, rng_end + 1))
        print(f"\nCropped video: {vid_w}x{vid_h}, {total_frames} frames, {fps:.1f} fps")
        print(f"  frame_ranges: {len(frame_ranges)} segment(s), "
              f"{len(video_frame_set)} frames for output video")
    else:
        print(f"\nCropped video: {vid_w}x{vid_h}, {total_frames} frames, {fps:.1f} fps")

    for label, bbox in zip(object_labels, tap_bboxes):
        print(f"  [{label}] bbox -> {[round(v, 1) for v in bbox]}")

    # -- Initialize SAM3VideoPredictor --------------------------------------
    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=model_path,
        half=half,
    )

    predictor = SAM3VideoPredictor(overrides=overrides)

    print(f"\nTracking with SAM3 (every {frame_skip} frames)...")
    results = predictor(
        source=str(cropped_video_path),
        bboxes=tap_bboxes,
        stream=True,
    )

    # -- Process results: extract centroids, write output video -------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = output_dir / "sam3_tracked.mp4"
    out_video = cv2.VideoWriter(
        str(out_video_path), fourcc, fps, (vid_w, vid_h)
    )

    centroids = {label: [] for label in object_labels}
    frames_dir = output_dir / "sam3_frames"
    frames_dir.mkdir(exist_ok=True)

    for frame_idx, result in enumerate(results):
        if max_frames is not None and frame_idx >= max_frames:
            break

        if frame_idx % frame_skip != 0:
            continue

        frame = result.orig_img
        overlay = frame.copy()

        if result.masks is not None:
            for obj_idx, mask_tensor in enumerate(result.masks.data):
                if obj_idx >= len(object_labels):
                    break
                mask = mask_tensor.cpu().numpy().astype(bool)
                color = colors[obj_idx % len(colors)]
                label = object_labels[obj_idx]

                overlay[mask] = (
                    overlay[mask] * 0.5 + np.array(color[::-1]) * 0.5
                ).astype(np.uint8)

                ys, xs = np.where(mask)
                if len(xs):
                    cx, cy = int(xs.mean()), int(ys.mean())
                    centroids[label].append((frame_idx, cx, cy))
                    cv2.circle(overlay, (cx, cy), 5, color, -1)
                    cv2.putText(overlay, label, (cx + 10, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(overlay, f"frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Write to video only if frame is in the desired ranges (or all frames)
        write_frame = (video_frame_set is None or frame_idx in video_frame_set)
        if write_frame:
            out_video.write(overlay)

        if save_snapshot_every > 0 and frame_idx % save_snapshot_every == 0:
            print(f"  {frame_idx}/{total_frames}")
            if write_frame:
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:05d}.png"), overlay)

    out_video.release()
    print(f"\nSAM3 tracked video saved to {out_video_path}")

    # -- Save centroid trajectories -----------------------------------------
    all_rows = []
    for label, data in centroids.items():
        for frame_idx, cx, cy in data:
            all_rows.append({
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "label": label,
                "centroid_x": cx,
                "centroid_y": cy,
            })

    csv_path = output_dir / "sam3_centroids.csv"
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        print(f"Centroid trajectories saved to {csv_path}")
    else:
        print("WARNING: No centroids detected.")
        pd.DataFrame(columns=["frame", "time_s", "label", "centroid_x", "centroid_y"]).to_csv(
            csv_path, index=False)

    # Also save per-label CSVs for compatibility
    for label, data in centroids.items():
        if not data:
            continue
        arr = np.array(data)
        np.savetxt(output_dir / f"{label}_centroids.csv", arr,
                   delimiter=",", header="frame,cx,cy", comments="", fmt="%d")

    # -- Plot centroid Y over time ------------------------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    for label, data in centroids.items():
        if not data:
            continue
        arr = np.array(data)
        ax.plot(arr[:, 0], arr[:, 2], label=f"{label} (Y)", linewidth=1)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Centroid Y (pixels)")
    ax.set_title("Tap Handle Centroid Y Over Time")
    ax.legend()
    plt.tight_layout()
    savefig(fig, output_dir, "centroid_trajectory.png")

    return csv_path


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():
    import yaml

    p = argparse.ArgumentParser(
        description="SAM3 tap handle tracking (standalone or via pipeline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path,
                   help="Path to pipeline.yaml config file.")
    p.add_argument("--interactive", action="store_true",
                   help="Force interactive bbox selection.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    video_path = Path(cfg["video_path"])
    output_dir = Path(cfg["output_dir"])
    sam3_cfg = cfg.get("sam3", {})
    roi_cfg = cfg.get("roi", {})

    tap_roi = tuple(roi_cfg["tap_roi"])

    # Resolve bboxes
    tap_bboxes = sam3_cfg.get("tap_bboxes")
    if args.interactive or tap_bboxes is None:
        cap = cv2.VideoCapture(str(video_path))
        ret, frame_0 = cap.read()
        cap.release()
        crop = crop_normalized(frame_0, tap_roi)
        tap_bboxes = select_tap_bboxes_interactive(
            crop, sam3_cfg.get("object_labels", ["TAP_A", "TAP_B"])
        )

    run_sam3_video_tracking(
        video_path=video_path,
        output_dir=output_dir,
        tap_roi=tap_roi,
        tap_bboxes=tap_bboxes,
        object_labels=sam3_cfg.get("object_labels", ["TAP_A", "TAP_B"]),
        colors=sam3_cfg.get("colors", [[0, 255, 0], [0, 128, 255]]),
        model_path=sam3_cfg.get("model", "data/models/sam3.pt"),
        max_frames=sam3_cfg.get("max_frames"),
        frame_skip=sam3_cfg.get("frame_skip", 5),
        save_snapshot_every=sam3_cfg.get("save_snapshot_every", 50),
        half=sam3_cfg.get("half", True),
    )


if __name__ == "__main__":
    main()
