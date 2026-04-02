import logging
import os
from functools import wraps

import requests

logger = logging.getLogger(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


class BackendUnavailable(Exception):
    """Raised when the backend API cannot be reached."""

    pass


def _handle_connection(func):
    """Wrap API calls to raise BackendUnavailable on connection failure."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.ConnectionError:
            raise BackendUnavailable(f"Cannot connect to backend at {BACKEND_URL}")

    return wrapper


@_handle_connection
def list_videos():
    """Get all uploaded videos, newest first."""
    resp = requests.get(f"{BACKEND_URL}/api/videos/")
    resp.raise_for_status()
    return resp.json()


@_handle_connection
def get_counts_summary():
    """Get global totals across all completed videos."""
    resp = requests.get(f"{BACKEND_URL}/api/counts/summary")
    resp.raise_for_status()
    return resp.json()


@_handle_connection
def upload_video(name, file_bytes, restaurant_name=None, camera_id=None):
    """Upload a video file. Returns {id, filename, original_name, status}."""
    logger.info("Uploading video: %s", name)
    params = {}
    if restaurant_name:
        params["restaurant_name"] = restaurant_name
    if camera_id:
        params["camera_id"] = camera_id
    resp = requests.post(
        f"{BACKEND_URL}/api/videos/upload",
        files={"file": (name, file_bytes, "video/mp4")},
        params=params,
    )
    resp.raise_for_status()
    return resp.json()


@_handle_connection
def process_video(video_id):
    """Start background processing. Returns HTTP status code (202 or 409)."""
    resp = requests.post(f"{BACKEND_URL}/api/videos/{video_id}/process")
    return resp.status_code


@_handle_connection
def get_video_status(video_id):
    """Get full video status including counts and events."""
    resp = requests.get(f"{BACKEND_URL}/api/videos/{video_id}/status")
    resp.raise_for_status()
    return resp.json()


@_handle_connection
def delete_video(video_id):
    """Delete a video and its events. Returns True on success."""
    logger.info("Deleting video: %d", video_id)
    resp = requests.delete(f"{BACKEND_URL}/api/videos/{video_id}")
    return resp.status_code == 204


@_handle_connection
def get_restaurants():
    """Get known restaurants and their cameras."""
    resp = requests.get(f"{BACKEND_URL}/api/videos/restaurants")
    resp.raise_for_status()
    return resp.json()


@_handle_connection
def check_roi_config(restaurant_name, camera_id):
    """Check if ROI config exists for a restaurant+camera combo."""
    resp = requests.get(
        f"{BACKEND_URL}/api/videos/roi-config-exists",
        params={"restaurant_name": restaurant_name, "camera_id": camera_id},
    )
    resp.raise_for_status()
    return resp.json()


@_handle_connection
def get_video_frame(video_id):
    """Get the first frame of a video as JPEG bytes."""
    resp = requests.get(f"{BACKEND_URL}/api/videos/{video_id}/frame")
    resp.raise_for_status()
    return resp.content


@_handle_connection
def save_roi_config(restaurant_name, camera_id, roi_data):
    """Save ROI config for a restaurant+camera combo."""
    resp = requests.post(
        f"{BACKEND_URL}/api/videos/roi-config",
        json={
            "restaurant_name": restaurant_name,
            "camera_id": camera_id,
            "roi_data": roi_data,
        },
    )
    resp.raise_for_status()
    return resp.json()
