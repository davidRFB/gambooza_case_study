import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging():
    """Configure logging for the application.

    Env vars:
        LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO)
        LOG_TO_FILE: set to 'true' to enable file logging to data/logs/app.log
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler (always on)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    # Optional file handler
    if os.environ.get("LOG_TO_FILE", "").lower() == "true":
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "app.log", maxBytes=10_000_000, backupCount=3
        )
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
