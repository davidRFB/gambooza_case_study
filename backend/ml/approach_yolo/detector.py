"""
YOLODetector — GPU-accelerated beer tap counter.

Wraps the 4-stage YOLO + SAM3 pipeline (pipeline.py) behind a simple
interface. Requires a YAML config file for pipeline parameters.

Stages: ROI selection → YOLO tracking → Relink → SAM3 tap tracking → Tap assignment
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from backend.ml.approach_yolo.pipeline import load_config, STAGE_FUNCS, STAGES


@dataclass
class YOLODetectorResult:
    tap_a_count: int
    tap_b_count: int
    unknown_count: int
    total: int
    pour_events: list[dict]
    stage_times: dict[str, float]
    total_elapsed_s: float

    def to_dict(self) -> dict:
        return asdict(self)


class YOLODetector:
    """GPU-accelerated beer tap counter using YOLO-World + SAM3."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)

    def run(
        self,
        interactive: bool = False,
        force: bool = False,
        stage: str | None = None,
    ) -> YOLODetectorResult:
        """Run the full pipeline and return results.

        Parameters
        ----------
        interactive : force interactive ROI/bbox selection
        force : re-run stages even if outputs exist
        stage : run only this stage (None = all)
        """
        import time
        import cv2

        cfg = load_config(self.config_path)

        if interactive:
            roi_cfg = cfg.setdefault("roi", {})
            roi_cfg["tap_roi"] = None
            roi_cfg["tap_divider"] = None
            sam3_cfg = cfg.setdefault("sam3", {})
            sam3_cfg["tap_bboxes"] = None

        stages_enabled = cfg.get("stages", {})
        stage_times: dict[str, float] = {}
        t_pipeline = time.time()

        for stage_name in STAGES:
            if stage and stage != stage_name:
                continue
            if not stages_enabled.get(stage_name, True):
                continue

            func = STAGE_FUNCS[stage_name]
            t0 = time.time()

            if stage_name in ("roi_selection", "sam3_tap_tracking"):
                func(cfg, interactive=interactive, force=force)
            else:
                func(cfg, force=force)

            stage_times[stage_name] = round(time.time() - t0, 3)

        total_elapsed = round(time.time() - t_pipeline, 3)

        # Read results from summary
        output_dir = Path(cfg["output_dir"])
        assigned_path = output_dir / "pour_events_assigned.json"
        if assigned_path.exists():
            pours = json.loads(assigned_path.read_text())
        else:
            pour_path = output_dir / "pour_events.json"
            pours = json.loads(pour_path.read_text()) if pour_path.exists() else []

        tap_a = sum(1 for p in pours if p.get("tap") == "TAP_A")
        tap_b = sum(1 for p in pours if p.get("tap") == "TAP_B")
        unknown = len(pours) - tap_a - tap_b

        return YOLODetectorResult(
            tap_a_count=tap_a,
            tap_b_count=tap_b,
            unknown_count=unknown,
            total=len(pours),
            pour_events=pours,
            stage_times=stage_times,
            total_elapsed_s=total_elapsed,
        )
