import os
import requests

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
