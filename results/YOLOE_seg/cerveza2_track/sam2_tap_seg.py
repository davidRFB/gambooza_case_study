from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

IMAGE = Path(__file__).parent / "01b_tap_division.png"
OUT = Path(__file__).parent / "sam2_tap_result.png"

# Approximate tap handle centers (x, y) in pixel coords
TAP_A = [130, 200]  # left dark handle
TAP_B = [310, 180]  # right chrome handle

model = SAM("sam2_b.pt")

img = cv2.imread(str(IMAGE))

for label, point in [("TAP_A", TAP_A), ("TAP_B", TAP_B)]:
    results = model(str(IMAGE), points=[point], labels=[1])
    result = results[0]
    if result.masks is not None:
        mask = result.masks.data[0].cpu().numpy().astype(np.uint8) * 255
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = img.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        cv2.circle(overlay, tuple(point), 6, (0, 0, 255), -1)
        cv2.putText(
            overlay, label, (point[0] + 8, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        out_path = Path(__file__).parent / f"sam2_{label}.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"{label}: mask saved to {out_path}")
    else:
        print(f"{label}: no mask found")
