"""Tests for the frontend API client."""

from unittest.mock import patch, MagicMock

from frontend.utils.api_client import (
    list_videos, get_counts_summary, upload_video,
    process_video, get_video_status, delete_video, BACKEND_URL,
)


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status_code
    resp.raise_for_status.return_value = None
    return resp


@patch("frontend.utils.api_client.requests.get")
def test_list_videos(mock_get):
    fake_data = [
        {"id": 1, "original_name": "vid.mp4", "upload_date": "2026-04-01", "status": "completed", "ml_approach": "yolo"},
    ]
    mock_get.return_value = _mock_response(fake_data)

    result = list_videos()

    mock_get.assert_called_once_with(f"{BACKEND_URL}/api/videos/")
    assert result == fake_data
    assert len(result) == 1
    assert result[0]["original_name"] == "vid.mp4"


@patch("frontend.utils.api_client.requests.get")
def test_list_videos_empty(mock_get):
    mock_get.return_value = _mock_response([])

    result = list_videos()

    assert result == []


@patch("frontend.utils.api_client.requests.get")
def test_get_counts_summary(mock_get):
    fake_data = {"tap_a_total": 5, "tap_b_total": 3, "grand_total": 8, "video_count": 2}
    mock_get.return_value = _mock_response(fake_data)

    result = get_counts_summary()

    mock_get.assert_called_once_with(f"{BACKEND_URL}/api/counts/summary")
    assert result["grand_total"] == 8
    assert result["video_count"] == 2


@patch("frontend.utils.api_client.requests.post")
def test_upload_video(mock_post):
    fake_data = {"id": 1, "filename": "abc123_test.mp4", "original_name": "test.mp4", "status": "pending"}
    mock_post.return_value = _mock_response(fake_data)

    result = upload_video("test.mp4", b"fake video bytes")

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "/api/videos/upload" in call_kwargs[0][0]
    assert result["id"] == 1
    assert result["status"] == "pending"


@patch("frontend.utils.api_client.requests.post")
def test_upload_video_returns_fields(mock_post):
    fake_data = {"id": 5, "filename": "xyz_vid.mp4", "original_name": "vid.mp4", "status": "pending"}
    mock_post.return_value = _mock_response(fake_data)

    result = upload_video("vid.mp4", b"data")

    assert result["original_name"] == "vid.mp4"
    assert result["filename"] == "xyz_vid.mp4"


@patch("frontend.utils.api_client.requests.post")
def test_process_video_accepted(mock_post):
    mock_post.return_value = _mock_response({"message": "Processing started", "video_id": 1}, status_code=202)

    status_code = process_video(1)

    mock_post.assert_called_once_with(f"{BACKEND_URL}/api/videos/1/process")
    assert status_code == 202


@patch("frontend.utils.api_client.requests.post")
def test_process_video_conflict(mock_post):
    mock_post.return_value = _mock_response({"detail": "Already processing"}, status_code=409)

    status_code = process_video(1)

    assert status_code == 409


@patch("frontend.utils.api_client.requests.get")
def test_get_video_status(mock_get):
    fake_data = {
        "id": 1, "filename": "abc.mp4", "original_name": "test.mp4",
        "upload_date": "2026-04-01", "status": "completed",
        "tap_a_count": 3, "tap_b_count": 2, "total": 5,
        "events": [{"id": 1, "tap": "A", "frame_start": 100, "frame_end": 200,
                     "timestamp_start": 3.3, "timestamp_end": 6.6, "confidence": 0.9, "count": 1}],
    }
    mock_get.return_value = _mock_response(fake_data)

    result = get_video_status(1)

    mock_get.assert_called_once_with(f"{BACKEND_URL}/api/videos/1/status")
    assert result["status"] == "completed"
    assert result["tap_a_count"] == 3
    assert result["total"] == 5
    assert len(result["events"]) == 1


@patch("frontend.utils.api_client.requests.delete")
def test_delete_video_success(mock_delete):
    mock_delete.return_value = _mock_response(None, status_code=204)

    result = delete_video(1)

    mock_delete.assert_called_once_with(f"{BACKEND_URL}/api/videos/1")
    assert result is True


@patch("frontend.utils.api_client.requests.delete")
def test_delete_video_not_found(mock_delete):
    mock_delete.return_value = _mock_response({"detail": "Not found"}, status_code=404)

    result = delete_video(999)

    assert result is False
