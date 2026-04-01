import logging
import os

import requests

logger = logging.getLogger(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def list_videos():
    """Get all uploaded videos, newest first."""
    resp = requests.get(f"{BACKEND_URL}/api/videos/")
    resp.raise_for_status()
    return resp.json()


def get_counts_summary():
    """Get global totals across all completed videos."""
    resp = requests.get(f"{BACKEND_URL}/api/counts/summary")
    resp.raise_for_status()
    return resp.json()


def upload_video(name, file_bytes):
    """Upload a video file. Returns {id, filename, original_name, status}."""
    logger.info("Uploading video: %s", name)
    resp = requests.post(
        f"{BACKEND_URL}/api/videos/upload",
        files={"file": (name, file_bytes, "video/mp4")},
    )
    resp.raise_for_status()
    return resp.json()


def process_video(video_id):
    """Start background processing. Returns HTTP status code (202 or 409)."""
    resp = requests.post(f"{BACKEND_URL}/api/videos/{video_id}/process")
    return resp.status_code


def get_video_status(video_id):
    """Get full video status including counts and events."""
    resp = requests.get(f"{BACKEND_URL}/api/videos/{video_id}/status")
    resp.raise_for_status()
    return resp.json()


def delete_video(video_id):
    """Delete a video and its events. Returns True on success."""
    logger.info("Deleting video: %d", video_id)
    resp = requests.delete(f"{BACKEND_URL}/api/videos/{video_id}")
    return resp.status_code == 204
