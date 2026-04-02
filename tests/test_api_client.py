"""Tests for the frontend API client."""

from unittest.mock import MagicMock, patch

from frontend.utils.api_client import (
    BACKEND_URL,
    check_roi_config,
    delete_video,
    get_counts_summary,
    get_restaurants,
    get_video_frame,
    get_video_status,
    list_videos,
    process_video,
    save_roi_config,
    upload_video,
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
        {
            "id": 1,
            "original_name": "vid.mp4",
            "upload_date": "2026-04-01",
            "status": "completed",
            "ml_approach": "yolo",
        },
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
    fake_data = {
        "id": 1,
        "filename": "abc123_test.mp4",
        "original_name": "test.mp4",
        "status": "pending",
    }
    mock_post.return_value = _mock_response(fake_data)

    result = upload_video("test.mp4", b"fake video bytes")

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "/api/videos/upload" in call_kwargs[0][0]
    assert result["id"] == 1
    assert result["status"] == "pending"


@patch("frontend.utils.api_client.requests.post")
def test_upload_video_returns_fields(mock_post):
    fake_data = {
        "id": 5,
        "filename": "xyz_vid.mp4",
        "original_name": "vid.mp4",
        "status": "pending",
    }
    mock_post.return_value = _mock_response(fake_data)

    result = upload_video("vid.mp4", b"data")

    assert result["original_name"] == "vid.mp4"
    assert result["filename"] == "xyz_vid.mp4"


@patch("frontend.utils.api_client.requests.post")
def test_process_video_accepted(mock_post):
    mock_post.return_value = _mock_response(
        {"message": "Processing started", "video_id": 1}, status_code=202
    )

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
        "id": 1,
        "filename": "abc.mp4",
        "original_name": "test.mp4",
        "upload_date": "2026-04-01",
        "status": "completed",
        "tap_a_count": 3,
        "tap_b_count": 2,
        "total": 5,
        "events": [
            {
                "id": 1,
                "tap": "A",
                "frame_start": 100,
                "frame_end": 200,
                "timestamp_start": 3.3,
                "timestamp_end": 6.6,
                "confidence": 0.9,
                "count": 1,
            }
        ],
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


@patch("frontend.utils.api_client.requests.post")
def test_upload_video_with_restaurant_camera(mock_post):
    fake_data = {
        "id": 1,
        "filename": "abc_test.mp4",
        "original_name": "test.mp4",
        "status": "pending",
        "restaurant_name": "mikes_pub",
        "camera_id": "cam1",
    }
    mock_post.return_value = _mock_response(fake_data)

    result = upload_video("test.mp4", b"data", restaurant_name="mikes_pub", camera_id="cam1")

    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs["params"] == {"restaurant_name": "mikes_pub", "camera_id": "cam1"}
    assert result["restaurant_name"] == "mikes_pub"


@patch("frontend.utils.api_client.requests.get")
def test_get_restaurants(mock_get):
    fake_data = {"restaurants": ["mikes_pub"], "cameras": {"mikes_pub": ["cam1"]}}
    mock_get.return_value = _mock_response(fake_data)

    result = get_restaurants()

    mock_get.assert_called_once_with(f"{BACKEND_URL}/api/videos/restaurants")
    assert "mikes_pub" in result["restaurants"]


@patch("frontend.utils.api_client.requests.get")
def test_check_roi_config(mock_get):
    fake_data = {"exists": True, "config_name": "mikes_pub_cam1"}
    mock_get.return_value = _mock_response(fake_data)

    result = check_roi_config("mikes_pub", "cam1")

    assert result["exists"] is True


@patch("frontend.utils.api_client.requests.get")
def test_get_video_frame(mock_get):
    resp = MagicMock()
    resp.content = b"\xff\xd8\xff\xe0"  # JPEG header bytes
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp

    result = get_video_frame(1)

    mock_get.assert_called_once_with(f"{BACKEND_URL}/api/videos/1/frame")
    assert result == b"\xff\xd8\xff\xe0"


@patch("frontend.utils.api_client.requests.post")
def test_save_roi_config(mock_post):
    fake_data = {"config_name": "mikes_pub_cam1", "path": "/some/path"}
    mock_post.return_value = _mock_response(fake_data)

    roi_data = {"yolo": {"tap_roi": [0.1, 0.2, 0.3, 0.4], "sam3_tap_bboxes": [[1, 2, 3, 4]]}}
    result = save_roi_config("mikes_pub", "cam1", roi_data)

    assert result["config_name"] == "mikes_pub_cam1"
