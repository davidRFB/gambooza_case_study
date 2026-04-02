"""Tests for the filtering module (activity windows)."""

import numpy as np

from backend.ml.approach_simple.filtering import find_activity_windows


def test_find_activity_windows_basic():
    """Single activity window is detected, padded, and returned."""
    times = np.linspace(0, 100, 1000)
    signals = {
        "Tap A": np.zeros(1000),
        "Tap B": np.zeros(1000),
    }
    # Activity from ~20s to ~30s
    signals["Tap A"][200:300] = 0.1

    windows = find_activity_windows(
        times=times,
        signals=signals,
        threshold=0.05,
        padding_s=5.0,
        merge_gap_s=10.0,
        total_duration=100.0,
    )

    assert len(windows) == 1
    assert windows[0]["start_s"] < 20.0  # padded before
    assert windows[0]["end_s"] > 30.0  # padded after


def test_find_activity_windows_no_activity():
    """No activity returns empty list."""
    times = np.linspace(0, 100, 1000)
    signals = {"Tap A": np.zeros(1000), "Tap B": np.zeros(1000)}

    windows = find_activity_windows(
        times=times,
        signals=signals,
        threshold=0.05,
        padding_s=5.0,
        merge_gap_s=10.0,
        total_duration=100.0,
    )
    assert windows == []


def test_find_activity_windows_merge():
    """Two close windows are merged."""
    times = np.linspace(0, 100, 1000)
    signals = {"Tap A": np.zeros(1000), "Tap B": np.zeros(1000)}
    # Two bursts 5s apart (within merge_gap of 40s)
    signals["Tap A"][200:250] = 0.1  # ~20-25s
    signals["Tap A"][300:350] = 0.1  # ~30-35s

    windows = find_activity_windows(
        times=times,
        signals=signals,
        threshold=0.05,
        padding_s=5.0,
        merge_gap_s=40.0,
        total_duration=100.0,
    )

    assert len(windows) == 1  # merged into one


def test_find_activity_windows_separate():
    """Two distant windows remain separate."""
    times = np.linspace(0, 200, 2000)
    signals = {"Tap A": np.zeros(2000), "Tap B": np.zeros(2000)}
    signals["Tap A"][100:150] = 0.1  # ~10-15s
    signals["Tap A"][1500:1550] = 0.1  # ~150-155s

    windows = find_activity_windows(
        times=times,
        signals=signals,
        threshold=0.05,
        padding_s=5.0,
        merge_gap_s=40.0,
        total_duration=200.0,
    )

    assert len(windows) == 2
