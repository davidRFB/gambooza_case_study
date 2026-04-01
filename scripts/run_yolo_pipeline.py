"""
Run the YOLO + SAM3 pipeline on a video.

Thin CLI wrapper around backend.ml.approach_yolo.pipeline.main().

Usage:
  python scripts/run_yolo_pipeline.py --config config/pipeline.yaml --interactive
  python scripts/run_yolo_pipeline.py --config config/pipeline.yaml
  python scripts/run_yolo_pipeline.py --config config/pipeline.yaml --stage relink
  python scripts/run_yolo_pipeline.py --config config/pipeline.yaml --force
"""

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ml.approach_yolo.pipeline import main

if __name__ == "__main__":
    main()
