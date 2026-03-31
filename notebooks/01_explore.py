# %% Imports
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# %% Parameters — edit these defaults or override when running
PARAMS = dict(
    video_path      = "../data/videos/ChartterTablas_All.mp4",
    output_dir      = "../data/explore_output",
    sample_rate     = 35,        # process every Nth frame
    blur_kernel     = 21,       # Gaussian blur kernel size
    diff_threshold  = 25,       # pixel intensity change threshold
    process_scale   = None,     # optional downscale for processing (e.g. 0.5); None = full res
    sample_count    = 10,       # number of frames to sample for overview
    activity_cutoff = 75,       # percentile cutoff to highlight high-activity regions
    n_workers       = 4,        # parallel workers for video processing
)

video_path = Path(PARAMS["video_path"])
OUTPUT_DIR = Path(PARAMS["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROI_PATH = OUTPUT_DIR / "roi.json"

# %% Load video metadata
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"File:     {video_path.name}")
print(f"FPS:      {fps}")
print(f"Frames:   {total_frames}")
print(f"Duration: {duration:.1f}s")
print(f"Size:     {vid_w}x{vid_h}")

# %% Show first frame
cap = cv2.VideoCapture(str(video_path))
ret, frame = cap.read()
cap.release()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 7))
plt.imshow(frame_rgb)
plt.title("First frame — use next cell to define the taps ROI")
plt.axis("off")
plt.tight_layout()
#plt.show()

# %% Sample frames evenly across the video
cap = cv2.VideoCapture(str(video_path))
sample_frames = np.linspace(0, total_frames - 2, PARAMS["sample_count"], dtype=int)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, fidx in enumerate(sample_frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, f = cap.read()
    if ret:
        axes[i].imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        pct = int(i / (PARAMS["sample_count"] - 1) * 100)
        axes[i].set_title(f"{pct}%  ({fidx/fps:.1f}s)")
    axes[i].axis("off")

cap.release()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sample_frames.png", dpi=150, bbox_inches="tight")
#plt.show()

# ============================================================
# STEP 1 — Select a single ROI that covers both taps
# ============================================================

# %% ROI selector — click top-left then bottom-right
# Press 'r' to reset, 'q' to accept and save to roi.json.
# If roi.json already exists, this cell loads it and skips the selector.

overwrite_roi = True
if ROI_PATH.exists() and not overwrite_roi :
    roi_data = json.loads(ROI_PATH.read_text())
    TAPS_ROI = tuple(roi_data["taps_roi"])
    print(f"Loaded ROI from {ROI_PATH}: {TAPS_ROI}")
else:
    cap = cv2.VideoCapture(str(video_path))
    ret, roi_frame = cap.read()
    cap.release()

    # Full resolution for accurate selection
    clicks = []
    clone = roi_frame.copy()

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        clicks.append((x, y))
        color = (0, 255, 0)
        cv2.circle(roi_frame, (x, y), 5, color, -1)
        if len(clicks) == 2:
            cv2.rectangle(roi_frame, clicks[0], clicks[1], color, 2)
            cv2.putText(roi_frame, "Taps ROI", (clicks[0][0], clicks[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Select Taps ROI", roi_frame)

    cv2.imshow("Select Taps ROI", roi_frame)
    cv2.setMouseCallback("Select Taps ROI", on_click)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            clicks.clear()
            roi_frame[:] = clone
            cv2.imshow("Select Taps ROI", roi_frame)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(clicks) >= 2:
        TAPS_ROI = tuple(round(v, 4) for v in (
            clicks[0][0] / vid_w, clicks[0][1] / vid_h,
            clicks[1][0] / vid_w, clicks[1][1] / vid_h,
        ))
        roi_data = {"taps_roi": list(TAPS_ROI), "video": video_path.name}
        ROI_PATH.write_text(json.dumps(roi_data, indent=2))
        print(f"ROI saved to {ROI_PATH}: {TAPS_ROI}")
    else:
        raise ValueError(f"Not enough clicks ({len(clicks)}/2). Run again.")

# %% Visualize ROI on first frame
cap = cv2.VideoCapture(str(video_path))
ret, vis_frame = cap.read()
cap.release()
vis_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)

def roi_to_px(roi, w, h):
    return int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)

rx1, ry1, rx2, ry2 = roi_to_px(TAPS_ROI, vid_w, vid_h)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.imshow(vis_rgb)
ax1.add_patch(patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1,
              linewidth=2, edgecolor='lime', facecolor='none', label='Taps ROI'))
ax1.legend()
ax1.set_title("Full frame with ROI")
ax1.axis("off")

crop_rgb = vis_rgb[ry1:ry2, rx1:rx2]
ax2.imshow(crop_rgb)
ax2.set_title("Cropped ROI")
ax2.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roi_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# STEP 2 — Analyze pixel-level motion inside the ROI
#   Build a per-pixel activity heatmap + histogram to
#   reveal ON (high motion) vs OFF (low motion) regions.
# ============================================================

# %% Compute per-pixel motion heatmap inside the ROI (chunked + multiprocessing)
import multiprocessing as mp
import time

roi_h = ry2 - ry1
roi_w = rx2 - rx1
sample_rate = PARAMS["sample_rate"]
diff_thresh = PARAMS["diff_threshold"]
blur_k = PARAMS["blur_kernel"]
n_workers = PARAMS["n_workers"]
process_scale = PARAMS["process_scale"]

# Compute processing dimensions (optionally downscaled)
if process_scale is not None and process_scale < 1.0:
    proc_h = int(roi_h * process_scale)
    proc_w = int(roi_w * process_scale)
    print(f"Processing at {process_scale}x: {proc_w}x{proc_h} (full ROI: {roi_w}x{roi_h})")
else:
    proc_h, proc_w = roi_h, roi_w
    process_scale = None
    print(f"Processing at full resolution: {roi_w}x{roi_h}")


def process_chunk(args):
    """Process a frame range and return partial motion_sum, count, and diffs."""
    vpath, start_frame, end_frame, sr, dt, bk, r, pscale = args
    ry1_, ry2_, rx1_, rx2_ = r

    cap = cv2.VideoCapture(str(vpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Figure out output dims
    test_h = ry2_ - ry1_
    test_w = rx2_ - rx1_
    if pscale is not None:
        out_h, out_w = int(test_h * pscale), int(test_w * pscale)
    else:
        out_h, out_w = test_h, test_w

    local_sum = np.zeros((out_h, out_w), dtype=np.float64)
    local_count = 0
    local_diffs = []
    prev_crop = None
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, f = cap.read()
        if not ret:
            break
        if frame_idx % sr == 0:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (bk, bk), 0)
            crop = gray[ry1_:ry2_, rx1_:rx2_]
            if pscale is not None:
                crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if prev_crop is not None:
                diff = cv2.absdiff(crop, prev_crop)
                mask = (diff > dt).astype(np.float64)
                local_sum += mask
                local_count += 1
                local_diffs.append(float(np.mean(mask)))

            prev_crop = crop
        frame_idx += 1

    cap.release()
    return local_sum, local_count, local_diffs


# Split video into chunks
chunk_size = total_frames // n_workers
chunks = []
for i in range(n_workers):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < n_workers - 1 else total_frames
    # Overlap by 1 sampled frame so each chunk has a valid prev_crop at its boundary
    overlap_start = max(0, start - sample_rate) if i > 0 else start
    chunks.append((
        str(video_path), overlap_start, end, sample_rate, diff_thresh,
        blur_k, (ry1, ry2, rx1, rx2), process_scale,
    ))

print(f"Processing {total_frames} frames with {n_workers} workers "
      f"(sample_rate={sample_rate}, ~{total_frames // sample_rate} sampled frames)...")
t0 = time.time()

with mp.Pool(n_workers) as pool:
    results = pool.map(process_chunk, chunks)

# Merge results
motion_sum = np.zeros((proc_h, proc_w), dtype=np.float64)
total_count = 0
frame_diffs = []
for local_sum, local_count, local_diffs in results:
    motion_sum += local_sum
    total_count += local_count
    frame_diffs.extend(local_diffs)

elapsed = time.time() - t0
print(f"Processed {total_count} frame pairs in {elapsed:.1f}s "
      f"({total_count / max(elapsed, 1):.0f} pairs/s)")

# Normalize: fraction of frames each pixel was active
motion_heatmap = motion_sum / max(total_count, 1)
frame_diffs = np.array(frame_diffs)

# Upscale heatmap back to full ROI size for visualization if we downscaled
if process_scale is not None:
    motion_heatmap_vis = cv2.resize(motion_heatmap, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
else:
    motion_heatmap_vis = motion_heatmap

# %% Save heatmap as numpy for downstream use
np.save(OUTPUT_DIR / "motion_heatmap.npy", motion_heatmap)
print(f"Heatmap saved: {motion_heatmap.shape}")

# %% Activity overlay with cutoff to highlight high-activity areas
flat = motion_heatmap_vis.flatten()
nonzero = flat[flat > 0]
cutoff = np.percentile(nonzero, PARAMS["activity_cutoff"]) if len(nonzero) > 0 else 0

heatmap_cut = np.where(motion_heatmap_vis >= cutoff, motion_heatmap_vis, 0)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.imshow(crop_rgb)
im = ax.imshow(heatmap_cut, cmap="hot", alpha=0.6, vmin=cutoff, vmax=motion_heatmap_vis.max())
plt.colorbar(im, ax=ax, fraction=0.046, label="Activity fraction")
ax.set_title(f"High-activity regions (>= P{PARAMS['activity_cutoff']} = {cutoff:.4f})")
ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "motion_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Histogram of per-pixel activity
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(nonzero, bins=100, color="steelblue", edgecolor="none", log=True)
ax.axvline(cutoff, color="red", linestyle="--", linewidth=2,
           label=f"P{PARAMS['activity_cutoff']} cutoff = {cutoff:.4f}")
ax.axvline(np.mean(nonzero), color="orange", linestyle="--",
           label=f"mean = {np.mean(nonzero):.4f}")
ax.set_xlabel("Activity fraction (non-zero pixels only)")
ax.set_ylabel("Pixel count (log)")
ax.set_title("Distribution of per-pixel activity")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "motion_histogram.png", dpi=150, bbox_inches="tight")
plt.show()

# Save high-activity mask (at visualization resolution)
on_mask = motion_heatmap_vis >= cutoff
np.save(OUTPUT_DIR / "on_mask.npy", on_mask)
print(f"Cutoff (P{PARAMS['activity_cutoff']}): {cutoff:.4f}")
print(f"High-activity pixels: {on_mask.sum():,} ({on_mask.mean()*100:.1f}%)")
print(f"Results saved to {OUTPUT_DIR.resolve()}")

# %%
