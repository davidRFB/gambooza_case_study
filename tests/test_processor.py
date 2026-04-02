"""Tests for the processor service."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from backend.config import ROI_CONFIGS_DIR
from backend.services.processor import (
    _load_roi_config,
    _map_pour_events,
    _resolve_roi_config_name,
    _run_yolo_pipeline,
)


def test_load_roi_config():
    path = ROI_CONFIGS_DIR / "_test_valid.json"
    path.write_text(
        json.dumps(
            {
                "yolo": {
                    "tap_roi": [0.1, 0.2, 0.3, 0.4],
                    "sam3_tap_bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
                }
            }
        )
    )
    try:
        roi = _load_roi_config("_test_valid")
        assert "yolo" in roi
        assert len(roi["yolo"]["tap_roi"]) == 4
        assert len(roi["yolo"]["sam3_tap_bboxes"]) == 2
    finally:
        path.unlink()


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


def test_run_yolo_pipeline_builds_correct_config(tmp_path):
    """Verify the temp config has the right overrides (mock the actual detector)."""
    roi = {
        "yolo": {
            "tap_roi": [0.1, 0.2, 0.3, 0.4],
            "sam3_tap_bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
        }
    }

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

    test_results = tmp_path / "results"
    with (
        patch("backend.services.processor.RESULTS_DIR", test_results),
        patch(
            "backend.services.processor._import_yolo_detector",
            return_value=capture_detector,
        ),
    ):
        events = _run_yolo_pipeline(Path("/tmp/fake.mp4"), 99, roi)

    # Check the config was built correctly
    cfg = captured_config_path["content"]
    assert cfg["video_path"] == "/tmp/fake.mp4"
    assert cfg["output_dir"] == str(test_results / "web_99")
    assert cfg["roi"]["tap_roi"] == roi["yolo"]["tap_roi"]
    assert cfg["sam3"]["tap_bboxes"] == roi["yolo"]["sam3_tap_bboxes"]

    # Check events passed through
    assert events == [{"cup_id": 1, "frame_start": 0, "frame_end": 100}]


def test_map_pour_events_tap_a_and_b():
    events = [
        {
            "cup_id": 1,
            "frame_start": 0,
            "frame_end": 100,
            "time_start": 0.0,
            "time_end": 5.0,
            "tap": "TAP_A",
        },
        {
            "cup_id": 2,
            "frame_start": 200,
            "frame_end": 350,
            "time_start": 10.0,
            "time_end": 17.5,
            "tap": "TAP_B",
        },
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
        {
            "cup_id": 1,
            "frame_start": 0,
            "frame_end": 100,
            "time_start": 0.0,
            "time_end": 5.0,
            "tap": "TAP_A",
        },
        {
            "cup_id": 2,
            "frame_start": 200,
            "frame_end": 350,
            "time_start": 10.0,
            "time_end": 17.5,
            "tap": "UNKNOWN",
        },
        {
            "cup_id": 3,
            "frame_start": 400,
            "frame_end": 500,
            "time_start": 20.0,
            "time_end": 25.0,
        },  # no tap key at all
    ]
    rows = _map_pour_events(video_id=1, pour_events=events)
    assert len(rows) == 1
    assert rows[0].tap == "A"


def test_map_pour_events_empty():
    rows = _map_pour_events(video_id=1, pour_events=[])
    assert rows == []


def test_load_roi_missing_yolo_key():
    path = ROI_CONFIGS_DIR / "_test_bad2.json"
    path.write_text(json.dumps({"yolo": {"tap_roi": [0.1, 0.2, 0.3, 0.4]}}))
    try:
        with pytest.raises(ValueError, match="missing key: sam3_tap_bboxes"):
            _load_roi_config("_test_bad2")
    finally:
        path.unlink()


def test_resolve_roi_config_with_existing_file():
    """When a matching config file exists, use it."""
    path = ROI_CONFIGS_DIR / "testrest_cam1.json"
    path.write_text(json.dumps({"yolo": {"tap_roi": [], "sam3_tap_bboxes": []}}))
    try:
        assert _resolve_roi_config_name("testrest", "cam1") == "testrest_cam1"
    finally:
        path.unlink()


def test_resolve_roi_config_returns_none_when_missing():
    """When no matching config file exists, return None."""
    assert _resolve_roi_config_name("nonexistent", "cam99") is None


def test_resolve_roi_config_none_values():
    """When restaurant or camera is None, return None."""
    assert _resolve_roi_config_name(None, None) is None
    assert _resolve_roi_config_name("some_rest", None) is None
    assert _resolve_roi_config_name(None, "cam1") is None


def test_load_roi_config_with_simple_section():
    """ROI config with both 'simple' and 'yolo' sections loads correctly."""
    path = ROI_CONFIGS_DIR / "_test_full.json"
    path.write_text(
        json.dumps(
            {
                "simple": {
                    "roi_1": [0.48, 0.44, 0.52, 0.47],
                    "roi_2": [0.56, 0.41, 0.58, 0.53],
                },
                "yolo": {
                    "tap_roi": [0.1, 0.2, 0.3, 0.4],
                    "sam3_tap_bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
                },
            }
        )
    )
    try:
        roi = _load_roi_config("_test_full")
        assert "yolo" in roi
        assert "simple" in roi
        assert roi["simple"]["roi_1"] == [0.48, 0.44, 0.52, 0.47]
        assert roi["simple"]["roi_2"] == [0.56, 0.41, 0.58, 0.53]
    finally:
        path.unlink()


def test_load_roi_config_without_simple_section():
    """ROI config with only 'yolo' section (no 'simple') loads fine."""
    path = ROI_CONFIGS_DIR / "_test_yolo_only.json"
    path.write_text(
        json.dumps(
            {
                "yolo": {
                    "tap_roi": [0.1, 0.2, 0.3, 0.4],
                    "sam3_tap_bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
                }
            }
        )
    )
    try:
        roi = _load_roi_config("_test_yolo_only")
        assert "yolo" in roi
        assert "simple" not in roi
    finally:
        path.unlink()


def test_run_yolo_pipeline_with_output_subdir(tmp_path):
    """Verify output_subdir places output in a subdirectory."""
    roi = {
        "yolo": {
            "tap_roi": [0.1, 0.2, 0.3, 0.4],
            "sam3_tap_bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
        }
    }

    fake_result = MagicMock()
    fake_result.pour_events = []

    captured_config_path = {}

    def capture_detector(config_path):
        captured_config_path["path"] = config_path
        with open(config_path) as f:
            captured_config_path["content"] = yaml.safe_load(f)
        mock_instance = MagicMock()
        mock_instance.run.return_value = fake_result
        return mock_instance

    test_results = tmp_path / "results"
    with (
        patch("backend.services.processor.RESULTS_DIR", test_results),
        patch(
            "backend.services.processor._import_yolo_detector",
            return_value=capture_detector,
        ),
    ):
        _run_yolo_pipeline(Path("/tmp/fake.mp4"), 99, roi, output_subdir="yolo_clips/clip_000")

    cfg = captured_config_path["content"]
    assert cfg["output_dir"] == str(test_results / "web_99" / "yolo_clips" / "clip_000")
