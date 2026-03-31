# %% Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %% Load video
VIDEO_DIR = Path("../data/videos/")
video_path = VIDEO_DIR / "cerveza2.mp4"

cap = cv2.VideoCapture(str(video_path))

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"File:     {video_path.name}")
print(f"FPS:      {fps}")
print(f"Frames:   {total_frames}")
print(f"Duration: {duration:.1f}s")
print(f"Size:     {width}x{height}")

cap.release()

# %% Show first frame
cap = cv2.VideoCapture(str(video_path))
ret, frame = cap.read()
cap.release()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 7))
plt.imshow(frame_rgb)
plt.title("First frame — use this to define ROIs")
plt.axis("off")
plt.tight_layout()
plt.show()

# %% Sample frames evenly across the video to get a feel for content
cap = cv2.VideoCapture(str(video_path))
sample_count = 10
# total_frames from OpenCV is often off-by-one; use -2 to stay safe
sample_frames = np.linspace(0, total_frames - 2, sample_count, dtype=int)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, frame_idx in enumerate(sample_frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pct = int(i / (sample_count - 1) * 100)
        axes[i].set_title(f"{pct}%  ({frame_idx/fps:.1f}s)")
    axes[i].axis("off")

cap.release()
plt.tight_layout()
plt.show()


# %% ROI selector — click top-left then bottom-right for each tap
# Run this cell, a window pops up.
# Click 2 points for Tap A (green), then 2 points for Tap B (red).
# Press 'r' to reset, 'q' to quit and print coordinates.

cap = cv2.VideoCapture(str(video_path))
ret, roi_frame = cap.read()
cap.release()

# Scale down to fit screen
scale = 0.5
roi_frame = cv2.resize(roi_frame, None, fx=scale, fy=scale)

clicks = []
clone = roi_frame.copy()

def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    clicks.append((x, y))
    n = len(clicks)
    # First 2 clicks = Tap A (green), next 2 = Tap B (red)
    if n <= 2:
        color = (0, 255, 0)
        cv2.circle(roi_frame, (x, y), 5, color, -1)
        if n == 2:
            cv2.rectangle(roi_frame, clicks[0], clicks[1], color, 2)
            cv2.putText(roi_frame, "Tap A", (clicks[0][0], clicks[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    elif n <= 4:
        color = (0, 0, 255)
        cv2.circle(roi_frame, (x, y), 5, color, -1)
        if n == 4:
            cv2.rectangle(roi_frame, clicks[2], clicks[3], color, 2)
            cv2.putText(roi_frame, "Tap B", (clicks[2][0], clicks[2][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Select ROIs", roi_frame)

cv2.imshow("Select ROIs", roi_frame)
cv2.setMouseCallback("Select ROIs", on_click)

while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('r'):  # reset
        clicks.clear()
        roi_frame[:] = clone
        cv2.imshow("Select ROIs", roi_frame)
    elif key == ord('q'):  # done
        break

cv2.destroyAllWindows()

# Normalize to 0-1 range
h, w = clone.shape[:2]
if len(clicks) >= 4:
    # Coordinates are on the scaled image — normalize divides by scaled w/h, so result is still 0-1
    ROI_A = (clicks[0][0]/w, clicks[0][1]/h, clicks[1][0]/w, clicks[1][1]/h)
    ROI_B = (clicks[2][0]/w, clicks[2][1]/h, clicks[3][0]/w, clicks[3][1]/h)
    print(f"ROI_A = {tuple(round(v, 4) for v in ROI_A)}")
    print(f"ROI_B = {tuple(round(v, 4) for v in ROI_B)}")
else:
    print(f"Not enough clicks ({len(clicks)}/4). Run again.")


# this ROIS are on to button of the TAP
ROI_A = (0.4719, 0.3741, 0.5396, 0.7991)
ROI_B = (0.5437, 0.3731, 0.6161, 0.7843)

# %% Visualize ROIs on first frame + save cropped ROIs
TEMP_DIR = Path("../data/videos/temporal")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
ret, vis_frame = cap.read()
cap.release()

h_vis, w_vis = vis_frame.shape[:2]
roi_a_vis = (int(ROI_A[0]*w_vis), int(ROI_A[1]*h_vis), int(ROI_A[2]*w_vis), int(ROI_A[3]*h_vis))
roi_b_vis = (int(ROI_B[0]*w_vis), int(ROI_B[1]*h_vis), int(ROI_B[2]*w_vis), int(ROI_B[3]*h_vis))

# Draw boxes
vis_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.imshow(vis_rgb)
import matplotlib.patches as patches
rect_a = patches.Rectangle((roi_a_vis[0], roi_a_vis[1]),
                            roi_a_vis[2]-roi_a_vis[0], roi_a_vis[3]-roi_a_vis[1],
                            linewidth=2, edgecolor='green', facecolor='none', label='Tap A')
rect_b = patches.Rectangle((roi_b_vis[0], roi_b_vis[1]),
                            roi_b_vis[2]-roi_b_vis[0], roi_b_vis[3]-roi_b_vis[1],
                            linewidth=2, edgecolor='red', facecolor='none', label='Tap B')
ax.add_patch(rect_a)
ax.add_patch(rect_b)
ax.legend()
ax.set_title("ROIs on first frame")
ax.axis("off")
plt.tight_layout()
plt.show()

# Save cropped ROIs
crop_a = vis_rgb[roi_a_vis[1]:roi_a_vis[3], roi_a_vis[0]:roi_a_vis[2]]
crop_b = vis_rgb[roi_b_vis[1]:roi_b_vis[3], roi_b_vis[0]:roi_b_vis[2]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
ax1.imshow(crop_a)
ax1.set_title("Tap A crop")
ax1.axis("off")
ax2.imshow(crop_b)
ax2.set_title("Tap B crop")
ax2.axis("off")
plt.tight_layout()
plt.savefig(TEMP_DIR / "roi_crops.png", dpi=150)
plt.show()

# Also save the full frame with boxes
fig2, ax3 = plt.subplots(1, 1, figsize=(12, 7))
ax3.imshow(vis_rgb)
ax3.add_patch(patches.Rectangle((roi_a_vis[0], roi_a_vis[1]),
              roi_a_vis[2]-roi_a_vis[0], roi_a_vis[3]-roi_a_vis[1],
              linewidth=2, edgecolor='green', facecolor='none'))
ax3.add_patch(patches.Rectangle((roi_b_vis[0], roi_b_vis[1]),
              roi_b_vis[2]-roi_b_vis[0], roi_b_vis[3]-roi_b_vis[1],
              linewidth=2, edgecolor='red', facecolor='none'))
ax3.axis("off")
fig2.savefig(TEMP_DIR / "roi_overlay.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved to {TEMP_DIR.resolve()}")

# %% Compute activity score per frame for each ROI
SAMPLE_RATE = 3  # process every Nth frame

cap = cv2.VideoCapture(str(video_path))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Convert normalized ROIs to pixel coords
def roi_to_pixels(roi, w, h):
    return (int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h))

roi_a_px = roi_to_pixels(ROI_A, w, h)
roi_b_px = roi_to_pixels(ROI_B, w, h)

def crop_roi(gray, roi_px):
    x1, y1, x2, y2 = roi_px
    return gray[y1:y2, x1:x2]

prev_a = None
prev_b = None
scores_a = []
scores_b = []
frame_indices = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % SAMPLE_RATE == 0:
        print(f"Processing frame {frame_idx}/{total_frames}", end="\r")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        crop_a = crop_roi(gray, roi_a_px)
        crop_b = crop_roi(gray, roi_b_px)

        if prev_a is not None:
            diff_a = cv2.absdiff(crop_a, prev_a)
            diff_b = cv2.absdiff(crop_b, prev_b)
            score_a = np.mean(diff_a > 25) # fraction of pixels that changed
            score_b = np.mean(diff_b > 25)
            scores_a.append(score_a)
            scores_b.append(score_b)
            frame_indices.append(frame_idx)

        prev_a = crop_a
        prev_b = crop_b

    frame_idx += 1

cap.release()

scores_a = np.array(scores_a)
scores_b = np.array(scores_b)
times = np.array(frame_indices) / fps

print(f"Processed {len(scores_a)} frame pairs")

# %% Plot activity scores
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

ax1.plot(times, scores_a, color="green", linewidth=0.8)
ax1.set_ylabel("Activity score")
ax1.set_title("Tap A")
ax1.axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="threshold")
ax1.legend()

ax2.plot(times, scores_b, color="red", linewidth=0.8)
ax2.set_ylabel("Activity score")
ax2.set_xlabel("Time (s)")
ax2.set_title("Tap B")
ax2.axhline(0.05, color="gray", linestyle="--", alpha=0.5, label="threshold")
ax2.legend()

plt.tight_layout()
plt.show()

# %%
