# %% Imports
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# %% Parameters
PARAMS = dict(
    video_path      = "../data/videos/cerveza2.mp4",
    output_dir      = "../data/explore_tabs_output",
    sample_rate     = 3,        # process every Nth frame
    blur_kernel     = 21,       # Gaussian blur kernel size
    diff_threshold  = 25,       # pixel intensity change threshold
)

video_path = Path(PARAMS["video_path"])
OUTPUT_DIR = Path(PARAMS["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROI_PATH = OUTPUT_DIR / "rois.json"

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
plt.title("First frame")
plt.axis("off")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 1 — Select 4 ROIs: Tap A ON, Tap A OFF, Tap B ON, Tap B OFF
# ============================================================

# %% ROI selector
# Click 2 points per ROI (top-left, bottom-right) x 4 ROIs = 8 clicks.
# Press 'r' to reset, 'q' to accept and save to rois.json.
# If rois.json already exists, loads it and skips the selector.

ROI_LABELS = ["Tap A ON", "Tap A OFF", "Tap B ON", "Tap B OFF"]
ROI_COLORS_BGR = [(0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255)]
ROI_COLORS_MPL = ["green", "cyan", "red", "magenta"]

if ROI_PATH.exists():
    roi_data = json.loads(ROI_PATH.read_text())
    ROIS = {k: tuple(v) for k, v in roi_data["rois"].items()}
    print(f"Loaded ROIs from {ROI_PATH}:")
    for k, v in ROIS.items():
        print(f"  {k}: {v}")
else:
    cap = cv2.VideoCapture(str(video_path))
    ret, roi_frame = cap.read()
    cap.release()

    clicks = []
    clone = roi_frame.copy()

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        clicks.append((x, y))
        n = len(clicks)
        roi_idx = (n - 1) // 2
        click_in_pair = (n - 1) % 2
        if roi_idx < len(ROI_LABELS):
            color = ROI_COLORS_BGR[roi_idx]
            cv2.circle(roi_frame, (x, y), 5, color, -1)
            if click_in_pair == 1:
                p1 = clicks[roi_idx * 2]
                p2 = clicks[roi_idx * 2 + 1]
                cv2.rectangle(roi_frame, p1, p2, color, 2)
                cv2.putText(roi_frame, ROI_LABELS[roi_idx],
                            (p1[0], p1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Select 4 ROIs", roi_frame)

    cv2.imshow("Select 4 ROIs", roi_frame)
    cv2.setMouseCallback("Select 4 ROIs", on_click)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            clicks.clear()
            roi_frame[:] = clone
            cv2.imshow("Select 4 ROIs", roi_frame)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(clicks) >= 8:
        ROIS = {}
        for i, label in enumerate(ROI_LABELS):
            p1 = clicks[i * 2]
            p2 = clicks[i * 2 + 1]
            ROIS[label] = (
                round(p1[0] / vid_w, 4), round(p1[1] / vid_h, 4),
                round(p2[0] / vid_w, 4), round(p2[1] / vid_h, 4),
            )
        roi_data = {"rois": {k: list(v) for k, v in ROIS.items()}, "video": video_path.name}
        ROI_PATH.write_text(json.dumps(roi_data, indent=2))
        print(f"ROIs saved to {ROI_PATH}:")
        for k, v in ROIS.items():
            print(f"  {k}: {v}")
    else:
        raise ValueError(f"Not enough clicks ({len(clicks)}/8). Run again.")

# %% Visualize ROIs on first frame
cap = cv2.VideoCapture(str(video_path))
ret, vis_frame = cap.read()
cap.release()
vis_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)


def roi_to_px(roi, w, h):
    return int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)


fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.imshow(vis_rgb)
for label, color in zip(ROI_LABELS, ROI_COLORS_MPL):
    r = roi_to_px(ROIS[label], vid_w, vid_h)
    ax.add_patch(patches.Rectangle((r[0], r[1]), r[2]-r[0], r[3]-r[1],
                 linewidth=2, edgecolor=color, facecolor='none', label=label))
ax.legend(loc="upper right")
ax.set_title("4 ROIs on first frame")
ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rois_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# Show crops
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax_i, label, color in zip(axes, ROI_LABELS, ROI_COLORS_MPL):
    r = roi_to_px(ROIS[label], vid_w, vid_h)
    crop = vis_rgb[r[1]:r[3], r[0]:r[2]]
    ax_i.imshow(crop)
    ax_i.set_title(label, color=color)
    ax_i.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rois_crops.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# STEP 2 — Compute activity signal per ROI over time
# ============================================================

# %% Process video
cap = cv2.VideoCapture(str(video_path))

sample_rate = PARAMS["sample_rate"]
diff_thresh = PARAMS["diff_threshold"]
blur_k = PARAMS["blur_kernel"]

# Pixel coords for each ROI
rois_px = {label: roi_to_px(ROIS[label], vid_w, vid_h) for label in ROI_LABELS}

prev_crops = {label: None for label in ROI_LABELS}
scores = {label: [] for label in ROI_LABELS}
frame_indices = []
frame_idx = 0

while True:
    ret, f = cap.read()
    if not ret:
        break
    if frame_idx % sample_rate == 0:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

        has_prev = prev_crops[ROI_LABELS[0]] is not None
        for label in ROI_LABELS:
            x1, y1, x2, y2 = rois_px[label]
            crop = gray[y1:y2, x1:x2]
            if has_prev:
                diff = cv2.absdiff(crop, prev_crops[label])
                scores[label].append(float(np.mean(diff > diff_thresh)))
            prev_crops[label] = crop

        if has_prev:
            frame_indices.append(frame_idx)

        print(f"\rProcessing frame {frame_idx}/{total_frames}", end="")

    frame_idx += 1

cap.release()

for label in ROI_LABELS:
    scores[label] = np.array(scores[label])
times = np.array(frame_indices) / fps

print(f"\nProcessed {len(frame_indices)} frame pairs")

# %% Save signals
signals = {"times": times.tolist()}
for label in ROI_LABELS:
    signals[label] = scores[label].tolist()
(OUTPUT_DIR / "signals.json").write_text(json.dumps(signals))
np.savez(OUTPUT_DIR / "signals.npz", times=times, **{label: scores[label] for label in ROI_LABELS})
print(f"Signals saved to {OUTPUT_DIR}")

# ============================================================
# STEP 3 — Plot: 2 rows (Tap A / Tap B), ON vs OFF signals
# ============================================================

# %% Time series plot
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Tap A
ax_a.plot(times, scores["Tap A ON"],  color="green",  linewidth=0.8, label="Tap A ON", alpha=0.8)
ax_a.plot(times, scores["Tap A OFF"], color="cyan",    linewidth=0.8, label="Tap A OFF", alpha=0.8)
ax_a.set_ylabel("Active pixel fraction")
ax_a.set_title("Tap A — ON vs OFF signal")
ax_a.legend(loc="upper right")
ax_a.grid(True, alpha=0.3)

# Tap B
ax_b.plot(times, scores["Tap B ON"],  color="red",     linewidth=0.8, label="Tap B ON", alpha=0.8)
ax_b.plot(times, scores["Tap B OFF"], color="magenta",  linewidth=0.8, label="Tap B OFF", alpha=0.8)
ax_b.set_ylabel("Active pixel fraction")
ax_b.set_xlabel("Time (s)")
ax_b.set_title("Tap B — ON vs OFF signal")
ax_b.legend(loc="upper right")
ax_b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tap_signals.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Difference signal (ON - OFF) to see when tap is active
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

diff_a = scores["Tap A ON"] - scores["Tap A OFF"]
diff_b = scores["Tap B ON"] - scores["Tap B OFF"]

ax_a.plot(times, diff_a, color="green", linewidth=0.8)
ax_a.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax_a.fill_between(times, diff_a, 0, where=diff_a > 0, alpha=0.3, color="green", label="ON > OFF")
ax_a.fill_between(times, diff_a, 0, where=diff_a < 0, alpha=0.3, color="red", label="OFF > ON")
ax_a.set_ylabel("ON - OFF")
ax_a.set_title("Tap A — differential signal")
ax_a.legend(loc="upper right")
ax_a.grid(True, alpha=0.3)

ax_b.plot(times, diff_b, color="red", linewidth=0.8)
ax_b.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax_b.fill_between(times, diff_b, 0, where=diff_b > 0, alpha=0.3, color="green", label="ON > OFF")
ax_b.fill_between(times, diff_b, 0, where=diff_b < 0, alpha=0.3, color="red", label="OFF > ON")
ax_b.set_ylabel("ON - OFF")
ax_b.set_xlabel("Time (s)")
ax_b.set_title("Tap B — differential signal")
ax_b.legend(loc="upper right")
ax_b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tap_diff_signals.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"All results saved to {OUTPUT_DIR.resolve()}")

# %%
