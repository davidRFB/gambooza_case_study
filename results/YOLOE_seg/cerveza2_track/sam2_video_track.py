"""
SAM2 video tracking of beer tap handles.
- Click points on the first frame to initialize SAM2
- SAM2 tracks the segmented objects across all frames
- Press 'q' to confirm points, 'r' to reset
"""

from pathlib import Path

import cv2
import matplotlib
import numpy as np
from ultralytics import SAM

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH = Path("../../../data/videos/cerveza2.mp4")
OUT_DIR = Path("sam2_track_output")
OUT_DIR.mkdir(exist_ok=True)

OBJECT_LABELS = ["TAP_A", "TAP_B"]  # one label per click
COLORS = [(0, 255, 0), (0, 128, 255)]  # green / orange per object
# ─────────────────────────────────────────────────────────────────────────────

# Step 1 — interactive point selection on first frame
cap = cv2.VideoCapture(str(VIDEO_PATH))
ret, first_frame = cap.read()
cap.release()
assert ret, f"Could not read {VIDEO_PATH}"

fps = cap.get(cv2.CAP_PROP_FPS) or 25
vid_h, vid_w = first_frame.shape[:2]

fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
ax.set_title(
    f"Click {len(OBJECT_LABELS)} point(s): {', '.join(OBJECT_LABELS)}\nClose window when done."
)
ax.axis("off")
plt.tight_layout()

raw = plt.ginput(n=len(OBJECT_LABELS), timeout=0, show_clicks=True)
plt.close()

clicks = [(int(x), int(y)) for x, y in raw]

if not clicks:
    raise ValueError("No points selected.")

for label, (x, y) in zip(OBJECT_LABELS, clicks):
    print(f"  [{label}] → ({x}, {y})")

print(f"\nSelected points: {list(zip(OBJECT_LABELS[: len(clicks)], clicks))}")

# Step 2 — run SAM2 on every frame using selected points
model = SAM("sam2_b.pt")

cap = cv2.VideoCapture(str(VIDEO_PATH))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter(str(OUT_DIR / "sam2_tracked.mp4"), fourcc, fps, (vid_w, vid_h))

print(f"\nTracking across {total_frames} frames…")

total_frames = min(total_frames, 800)
for frame_idx in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    overlay = frame.copy()

    for obj_idx, (x, y) in enumerate(clicks):
        results = model(frame, points=[[x, y]], labels=[1], verbose=False)
        result = results[0]
        if result.masks is not None and len(result.masks.data) > 0:
            mask = result.masks.data[0].cpu().numpy().astype(bool)
            color = COLORS[obj_idx % len(COLORS)]
            overlay[mask] = (
                overlay[mask] * 0.5 + np.array(color[::-1]) * 0.5  # BGR
            ).astype(np.uint8)
            # draw label at centroid
            ys, xs = np.where(mask)
            if len(xs):
                cx, cy = int(xs.mean()), int(ys.mean())
                cv2.putText(
                    overlay,
                    OBJECT_LABELS[obj_idx],
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

    # frame counter
    cv2.putText(
        overlay, f"frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    out_video.write(overlay)

    if frame_idx % 50 == 0:
        print(f"  {frame_idx}/{total_frames}")
        cv2.imwrite(str(OUT_DIR / f"frame_{frame_idx:05d}.png"), overlay)

cap.release()
out_video.release()
print(f"\nDone. Output saved to {OUT_DIR.resolve()}")
