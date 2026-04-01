"""Tests for the processor service."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from backend.services.processor import _load_roi_config, _run_yolo_pipeline, _map_pour_events
from backend.config import ROI_CONFIGS_DIR, YOLO_BASE_CONFIG, RESULTS_DIR


def test_load_default_roi():
    roi = _load_roi_config("default")
    assert "simple" in roi
    assert "yolo" in roi
    assert len(roi["yolo"]["tap_roi"]) == 4
    assert len(roi["yolo"]["sam3_tap_bboxes"]) == 2


def test_load_roi_not_found():
    with pytest.raises(FileNotFoundError):
        _load_roi_config("nonexistent_camera")


def test_load_roi_missing_yolo_section():
    path = ROI_CONFIGS_DIR / "_test_bad.json"
    path.write_text(json.dumps({"simple": {}}))
    try:
        with pytest.raises(ValueError, match="missing 'yolo' section"):
            _load_roi_config("_test_bad")
    finally:
        path.unlink()


def test_run_yolo_pipeline_builds_correct_config():
    """Verify the temp config has the right overrides (mock the actual detector)."""
    roi = _load_roi_config("default")

    fake_result = MagicMock()
    fake_result.pour_events = [{"cup_id": 1, "frame_start": 0, "frame_end": 100}]

    captured_config_path = {}

    def capture_detector(config_path):
        """Save the config path so we can read it before it's deleted."""
        captured_config_path["path"] = config_path
        with open(config_path) as f:
            captured_config_path["content"] = yaml.safe_load(f)
        mock_instance = MagicMock()
        mock_instance.run.return_value = fake_result
        return mock_instance

    with patch(
        "backend.services.processor._import_yolo_detector",
        return_value=capture_detector,
    ):
        events = _run_yolo_pipeline(Path("/tmp/fake.mp4"), 99, roi)

    # Check the config was built correctly
    cfg = captured_config_path["content"]
    assert cfg["video_path"] == "/tmp/fake.mp4"
    assert cfg["output_dir"] == str(RESULTS_DIR / "web_99")
    assert cfg["roi"]["tap_roi"] == roi["yolo"]["tap_roi"]
    assert cfg["sam3"]["tap_bboxes"] == roi["yolo"]["sam3_tap_bboxes"]

    # Check events passed through
    assert events == [{"cup_id": 1, "frame_start": 0, "frame_end": 100}]


def test_map_pour_events_tap_a_and_b():
    events = [
        {"cup_id": 1, "frame_start": 0, "frame_end": 100,
         "time_start": 0.0, "time_end": 5.0, "tap": "TAP_A"},
        {"cup_id": 2, "frame_start": 200, "frame_end": 350,
         "time_start": 10.0, "time_end": 17.5, "tap": "TAP_B"},
    ]
    rows = _map_pour_events(video_id=1, pour_events=events)
    assert len(rows) == 2
    assert rows[0].tap == "A"
    assert rows[0].timestamp_start == 0.0
    assert rows[1].tap == "B"
    assert rows[1].frame_end == 350
    assert rows[1].count == 1


def test_map_pour_events_skips_unknown():
    events = [
        {"cup_id": 1, "frame_start": 0, "frame_end": 100,
         "time_start": 0.0, "time_end": 5.0, "tap": "TAP_A"},
        {"cup_id": 2, "frame_start": 200, "frame_end": 350,
         "time_start": 10.0, "time_end": 17.5, "tap": "UNKNOWN"},
        {"cup_id": 3, "frame_start": 400, "frame_end": 500,
         "time_start": 20.0, "time_end": 25.0},  # no tap key at all
    ]
    rows = _map_pour_events(video_id=1, pour_events=events)
    assert len(rows) == 1
    assert rows[0].tap == "A"


def test_map_pour_events_empty():
    rows = _map_pour_events(video_id=1, pour_events=[])
    assert rows == []


def test_load_roi_missing_yolo_key():
    path = ROI_CONFIGS_DIR / "_test_bad2.json"
    path.write_text(json.dumps({
        "yolo": {"tap_roi": [0.1, 0.2, 0.3, 0.4]}
    }))
    try:
        with pytest.raises(ValueError, match="missing key: sam3_tap_bboxes"):
            _load_roi_config("_test_bad2")
    finally:
        path.unlink()
