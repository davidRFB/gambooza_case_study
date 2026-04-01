"""
SAM3 video tracking of beer tap handles.
- Draw bounding boxes on the first frame to initialize SAM3VideoPredictor
- SAM3 propagates masks across all frames using its memory bank
- Extracts centroid per frame to detect ON/OFF transitions
"""
from ultralytics.models.sam import SAM3VideoPredictor
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH = Path("../../../data/videos/cerveza2.mp4")
OUT_DIR    = Path("sam3_track_output")
OUT_DIR.mkdir(exist_ok=True)

OBJECT_LABELS = ["TAP_A", "TAP_B"]
COLORS = [(0, 255, 0), (0, 128, 255)]
MAX_FRAMES = None  # process all frames
FRAME_SKIP = 5     # process every Nth frame
# ─────────────────────────────────────────────────────────────────────────────

# Step 1 — interactive bbox selection on first frame
cap = cv2.VideoCapture(str(VIDEO_PATH))
ret, first_frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS) or 25
vid_h, vid_w = first_frame.shape[:2]
cap.release()
assert ret, f"Could not read {VIDEO_PATH}"

# Click top-left then bottom-right for each bbox.
# Total clicks needed: 2 per object.
n_clicks = len(OBJECT_LABELS) * 2
fig, ax = plt.subplots(figsize=(14, 8))
ax.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
ax.set_title(
    f"Click {n_clicks} times: top-left then bottom-right for each tap\n"
    f"Order: {', '.join(OBJECT_LABELS)}"
)
ax.axis("off")
plt.tight_layout()

raw = []
for i in range(n_clicks):
    pt = plt.ginput(n=1, timeout=0, show_clicks=True)
    if not pt:
        break
    raw.append(pt[0])
    ax.plot(pt[0][0], pt[0][1], "r+", markersize=15, markeredgewidth=2)
    fig.canvas.draw()
plt.close()

if len(raw) < 2:
    raise ValueError("Need at least one bounding box (2 clicks).")

bboxes = []
for i in range(0, len(raw) - 1, 2):
    x1, y1 = raw[i]
    x2, y2 = raw[i + 1]
    bboxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])

for label, bbox in zip(OBJECT_LABELS, bboxes):
    print(f"  [{label}] bbox → {[round(v, 1) for v in bbox]}")

# Step 2 — run SAM3 video tracking
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True,
)

predictor = SAM3VideoPredictor(overrides=overrides)

print(f"\nTracking with SAM3 (every {FRAME_SKIP} frames)…")
results = predictor(
    source=str(VIDEO_PATH),
    bboxes=bboxes,
    stream=True,
)

# Step 3 — process results, extract centroids, write output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter(
    str(OUT_DIR / "sam3_tracked.mp4"), fourcc, fps, (vid_w, vid_h)
)

centroids = {label: [] for label in OBJECT_LABELS}  # label → [(frame, cx, cy)]

for frame_idx, result in enumerate(results):
    if frame_idx % FRAME_SKIP != 0:
        continue

    frame = result.orig_img
    overlay = frame.copy()

    if result.masks is not None:
        for obj_idx, mask_tensor in enumerate(result.masks.data):
            if obj_idx >= len(OBJECT_LABELS):
                break
            mask = mask_tensor.cpu().numpy().astype(bool)
            color = COLORS[obj_idx % len(COLORS)]
            label = OBJECT_LABELS[obj_idx]

            # color overlay
            overlay[mask] = (
                overlay[mask] * 0.5 + np.array(color[::-1]) * 0.5
            ).astype(np.uint8)

            # centroid
            ys, xs = np.where(mask)
            if len(xs):
                cx, cy = int(xs.mean()), int(ys.mean())
                centroids[label].append((frame_idx, cx, cy))
                cv2.circle(overlay, (cx, cy), 5, color, -1)
                cv2.putText(overlay, label, (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(overlay, f"frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out_video.write(overlay)

    if frame_idx % 50 == 0:
        print(f"  {frame_idx}/{MAX_FRAMES}")
        cv2.imwrite(str(OUT_DIR / f"frame_{frame_idx:05d}.png"), overlay)

out_video.release()

# Step 4 — save centroid trajectories and plot
for label, data in centroids.items():
    if not data:
        continue
    arr = np.array(data)
    np.savetxt(OUT_DIR / f"{label}_centroids.csv", arr,
               delimiter=",", header="frame,cx,cy", comments="", fmt="%d")

# plot centroid Y over time (Y position reveals ON/OFF)
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
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(str(OUT_DIR / "centroid_trajectory.png"), dpi=150)
plt.show()

print(f"\nDone. Output saved to {OUT_DIR.resolve()}")
