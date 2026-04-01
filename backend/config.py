"""Application settings, paths, and constants."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"    # original test videos
UPLOADS_DIR = DATA_DIR / "uploads"   # user-uploaded videos
DB_DIR = DATA_DIR / "db_files"
MODELS_DIR = DATA_DIR / "models"
ROI_CONFIGS_DIR = DATA_DIR / "roi_configs"

# ── Database ───────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_DIR / 'app.db'}")

# ── ML approach toggle ─────────────────────────────────────────────
ML_APPROACH = os.getenv("ML_APPROACH", "simple")  # "simple" | "yolo"

# ── Default ROI (from results/simple_test_cerveza2/simple_roi.json) ─
DEFAULT_ROI = {
    "tap_roi": (0.4678, 0.3737, 0.6311, 0.5540),
    "tap_a_roi": (0.4827, 0.4470, 0.5245, 0.4705),
    "tap_b_roi": (0.5638, 0.4103, 0.5764, 0.5340),
}

# ── YOLO pipeline base config (used as template for web runs) ──────
YOLO_BASE_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"
