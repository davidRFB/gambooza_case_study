# CLAUDE.md — Beer Tap Counter: Implementation Guide

## Project Overview

A self-contained application that counts beers served from a dual-tap beer dispenser by analyzing uploaded video footage. The system identifies events on Tap A and Tap B independently, persists results in SQLite, and displays counts through a minimal web UI.

**Architecture:** Docker Compose → Streamlit (frontend) + FastAPI (backend + ML pipeline) + SQLite (persistence)

---

## 1. Project Structure

```
gambooza_case_study/
├── CLAUDE.md
├── pyproject.toml                  # Project config: deps, ruff, pytest
├── uv.lock                        # Reproducible dependency lockfile
├── .pre-commit-config.yaml         # Ruff lint + format on commit
├── docker-compose.yml              # Docker Compose for full stack
│
├── backend/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app entry point
│   ├── config.py                   # Settings, paths, constants
│   ├── logging_config.py           # Logging setup (console + optional file)
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py           # Engine, SessionLocal, init_db, get_db
│   │   ├── models.py              # SQLAlchemy ORM: Video, TapEvent
│   │   └── schemas.py             # Pydantic response models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── videos.py              # Upload, list, status, delete, process, ROI, frame
│   │   └── counts.py              # Query counts, summary
│   ├── services/
│   │   ├── __init__.py
│   │   └── processor.py           # Background processor (YOLO pipeline bridge)
│   └── ml/
│       ├── __init__.py
│       ├── common.py               # Shared utilities (ROI, cropping, interactive selectors)
│       ├── approach_simple/
│       │   ├── __init__.py
│       │   ├── detector.py         # SimpleDetector — CPU pixel differencing
│       │   └── filtering.py        # Activity window detection + clip extraction
│       └── approach_yolo/
│           ├── __init__.py
│           ├── detector.py         # YOLODetector — GPU wrapper + tap assignment
│           ├── pipeline.py         # 4-stage orchestrator (ROI → YOLO → Relink → SAM3)
│           ├── yolo_track.py       # YOLO-World + BoT-SORT tracking
│           ├── relink.py           # Track relinking + pour classification
│           └── sam3_tracking.py    # SAM3 tap handle segmentation
│
├── tests/
│   ├── __init__.py
│   ├── test_app.py                 # FastAPI app health check
│   ├── test_database.py            # DB models, columns, cascade delete
│   ├── test_schemas.py             # Pydantic schema validation
│   ├── test_videos_router.py       # Upload, list, status, delete, process, ROI endpoints
│   ├── test_counts_router.py       # Counts query and summary endpoints
│   ├── test_processor.py           # ROI resolution, config loader, YOLO config builder, event mapper
│   └── test_api_client.py          # Frontend API client (mocked HTTP calls)
│
├── frontend/
│   ├── app.py                      # Main Streamlit app (two tabs + ROI wizard)
│   ├── requirements.txt            # streamlit, requests, streamlit-cropper, Pillow
│   └── utils/
│       ├── __init__.py
│       └── api_client.py           # Backend HTTP client (11 functions)
│
├── scripts/
│   ├── run_simple.py               # CLI: run SimpleDetector on a video (+ clip extraction)
│   ├── run_yolo_pipeline.py        # CLI: run YOLO+SAM3 pipeline on a video
│   └── run_yolo_on_clips.py        # CLI: batch YOLO+SAM3 on activity clips
│
├── config/
│   ├── pipeline.yaml               # Default YOLO pipeline config
│   ├── pipeline[1-7].yaml          # Per-video pipeline configs
│   └── botsort.yaml                # BoT-SORT tracker config
│
├── data/
│   ├── videos/                     # Source videos (cerveza1–7.mp4, etc.)
│   ├── models/                     # ML weights (yolov8x-worldv2.pt, sam3.pt, etc.)
│   ├── db_files/                   # SQLite database (app.db)
│   ├── roi_configs/                # ROI configs: {restaurant}_{camera}.json
│   └── results/                    # Pipeline outputs + uploaded videos
│       └── web_{id}/               # Per-video: uploaded video + pipeline outputs
│
├── notes.txt                       # Development notes, TODOs, known issues
│
└── notebooks/                      # Exploration & development (not deployed)
    ├── 01_explore.py               # Full-ROI pixel motion heatmap
    ├── 01_exploreTABS.py           # Per-tap pixel activity signals
    ├── 02_exploreYOLO.py           # YOLO detection experiments
    ├── 03_YOLO_track.py            # (original) YOLO tracking
    ├── 04_relink_tracks.py         # (original) track relinking v1
    ├── 05_relink_coexistence.py    # (original) track relinking v2
    ├── 06_YOLOE_seg_track.py       # YOLO-E segmentation experiments
    ├── sam3_tracking.py            # (original) SAM3 tracking
    ├── pipeline.py                 # (original) pipeline orchestrator
    └── common.py                   # (original) shared utilities
```

---

## 2. Docker Setup

### docker-compose.yml

Two services: backend with GPU passthrough, frontend depending on backend health.
Host `./data/` directory is bind-mounted to `/app/mount_data` in the backend container.

```yaml
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - DATA_DIR=/app/mount_data
      - DATABASE_URL=sqlite:////app/mount_data/db_files/app.db
      - ML_APPROACH=${ML_APPROACH:-yolo}
    volumes:
      - ./data:/app/mount_data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python3.11", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped
```

### Key Docker Decisions

- **GPU passthrough:** Use `nvidia-container-toolkit` so the backend container can access the host GPU. Required for both YOLO inference and SAM3 segmentation.
- **Bind mount:** `./data` on host → `/app/mount_data` in container. The `DATA_DIR` env var tells `backend/config.py` where to find models, DB, ROI configs, and results. This decouples the host data layout from the container's `/app` directory.
- **Always YOLO:** Processing always runs the full YOLO+SAM3 pipeline. SimpleDetector is used as a pre-filter for long videos (> 2500 frames) when simple ROIs are available.

### Backend Dockerfile (`Dockerfile.backend`)

Multi-stage build: builder (uv + deps) → runtime (CUDA + Python 3.11).

```dockerfile
# ── Builder stage: install dependencies with uv ──────────────────
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 UV_COMPILE_BYTECODE=0 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ── Runtime stage: CUDA + Python 3.11 ────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg YOLO_AUTOINSTALL=False
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    libgl1-mesa-glx libglib2.0-0 ffmpeg gcc g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
RUN ln -sf /usr/bin/python3.11 /app/.venv/bin/python && \
    ln -sf /usr/bin/python3.11 /app/.venv/bin/python3
ENV PATH="/app/.venv/bin:$PATH"
COPY backend/ ./backend/
COPY config/ ./config/
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Key details:
- **`build-essential` in builder:** Required for compiling `lap` (C extension used by BoT-SORT tracker).
- **`gcc g++ python3.11-dev` in runtime:** Required for PyTorch Triton JIT compilation (SAM3 uses torch.compile/inductor which compiles CUDA kernels at runtime).
- **`YOLO_AUTOINSTALL=False`:** Prevents ultralytics from trying to `pip install` optional packages at runtime. All required packages (`timm`, `clip`, `lap`) are declared in `pyproject.toml` and installed at build time.
- **Shebang fix:** Builder venv has `python` symlinks pointing to the builder's Python; runtime only has `python3.11` from deadsnakes, so symlinks are patched.

### Frontend Dockerfile (`Dockerfile.frontend`)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY frontend/ ./
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Path Resolution

In Docker, data lives at `/app/mount_data/` (bind-mounted from `./data`), but `config/pipeline.yaml` has relative paths like `data/models/sam3.pt`. The processor (`backend/services/processor.py`) resolves these using `_resolve_path()`:
- Paths starting with `data/` → resolved relative to `DATA_DIR` (`/app/mount_data/`)
- Other relative paths (e.g. `config/botsort.yaml`) → resolved relative to `PROJECT_ROOT` (`/app/`)

This ensures the same `pipeline.yaml` works both locally and in Docker without modification.

---

## 3. Database Layer (SQLite + SQLAlchemy)

### ORM Models (`backend/database/models.py`)

```
Table: videos
├── id                    INTEGER PK AUTOINCREMENT
├── filename              TEXT NOT NULL          # UUID name on disk (in data/results/web_{id}/)
├── original_name         TEXT NOT NULL          # user's original filename
├── upload_date           DATETIME DEFAULT now
├── status                TEXT DEFAULT 'pending' # pending | processing | completed | error
├── duration_sec          FLOAT NULLABLE
├── error_message         TEXT NULLABLE
├── ml_approach           TEXT NULLABLE          # "yolo"
├── processing_started_at DATETIME NULLABLE
├── processing_finished_at DATETIME NULLABLE
├── output_dir            TEXT NULLABLE          # path to pipeline intermediate files
├── restaurant_name       TEXT NULLABLE          # e.g. "cerveceria_centro"
└── camera_id             TEXT NULLABLE          # e.g. "cam1"

Table: tap_events
├── id              INTEGER PK AUTOINCREMENT
├── video_id        INTEGER FK → videos.id (cascade delete)
├── tap             TEXT NOT NULL             # 'A' | 'B'
├── frame_start     INTEGER NOT NULL
├── frame_end       INTEGER NOT NULL
├── timestamp_start FLOAT NOT NULL           # Seconds into video
├── timestamp_end   FLOAT NOT NULL
├── confidence      FLOAT NULLABLE
└── count           INTEGER DEFAULT 1        # Beers served in this event

Counts are computed as SUM(count) per tap, not COUNT(*).
```

### Pydantic Schemas (`backend/database/schemas.py`)

Response models (not DB tables — shapes for API JSON):
- `VideoUploadResponse(id, filename, original_name, status, restaurant_name, camera_id)`
- `VideoStatusResponse(id, filename, original_name, upload_date, status, restaurant_name, camera_id, duration_sec, error_message, ml_approach, processing_started_at, processing_finished_at, output_dir, tap_a_count, tap_b_count, total, events)`
- `VideoListItem(id, original_name, upload_date, status, ml_approach, restaurant_name, camera_id)`
- `TapEventResponse(id, tap, frame_start, frame_end, timestamp_start, timestamp_end, confidence, count)`
- `CountResult(video_id, original_name, upload_date, restaurant_name, camera_id, tap_a, tap_b, total)`
- `CountSummary(tap_a_total, tap_b_total, grand_total, video_count)`

### Migration Strategy

`init_db()` in `connection.py` calls `Base.metadata.create_all(engine)` on startup. No Alembic. If schema changes, delete `data/db_files/app.db` and restart.

---

## 4. Backend — FastAPI

### Entry Point (`backend/main.py`)

```python
app = FastAPI(title="Beer Tap Counter API", lifespan=lifespan)
# lifespan calls init_db() on startup
app.include_router(videos_router, prefix="/api/videos")
app.include_router(counts_router, prefix="/api/counts")
```

### Router: Videos (`backend/routers/videos.py`)

| Method | Endpoint                        | Purpose                                                    |
|--------|---------------------------------|------------------------------------------------------------|
| POST   | `/api/videos/upload`            | Accept mp4/mov + restaurant/camera, save to `data/results/web_{id}/` |
| POST   | `/api/videos/{id}/process`      | Launch background YOLO pipeline, returns 202               |
| GET    | `/api/videos/{id}/status`       | Return status + counts + events if completed               |
| GET    | `/api/videos/`                  | List all uploaded videos                                   |
| DELETE | `/api/videos/{id}`              | Remove video dir + DB records (cascade)                    |
| GET    | `/api/videos/restaurants`       | List known restaurant+camera combos (from DB + config files) |
| GET    | `/api/videos/roi-config-exists` | Check if ROI config exists for restaurant+camera           |
| POST   | `/api/videos/roi-config`        | Save ROI config JSON for a restaurant+camera combo         |
| GET    | `/api/videos/{id}/frame`        | Return first frame of video as JPEG                        |

#### Upload Flow

1. Validate file extension (.mp4, .mov only).
2. Generate unique filename: `{uuid8}_{original_name}`.
3. Insert row in `videos` table with `status='pending'`, `restaurant_name`, `camera_id`.
4. Create directory `data/results/web_{id}/` and save video file there.
5. Return `VideoUploadResponse`.

#### Process Flow

1. Validate video exists (404) and not already processing (409).
2. Launch `BackgroundTasks.add_task(_run_processing, video_id, roi_config)`.
3. Return `202 Accepted` immediately.
4. Background task creates its own `SessionLocal()` (request session closes on response).
5. Processor runs YOLO pipeline, writes `tap_events`, updates status.

#### Process Endpoint

No parameters needed — ROI config is resolved automatically from the video's `restaurant_name` + `camera_id` fields. The config file `data/roi_configs/{restaurant}_{camera}.json` must exist before processing (no default fallback).

### Router: Counts (`backend/routers/counts.py`)

| Method | Endpoint              | Purpose                                              |
|--------|-----------------------|------------------------------------------------------|
| GET    | `/api/counts/`        | Query counts with filters: video_id, date range, tap |
| GET    | `/api/counts/summary` | Aggregate totals across all completed videos         |

### Background Processing (`backend/services/processor.py`)

The processor automatically chooses between two paths based on video length and ROI config:

- **Short videos** (≤ 2500 frames) or **no simple ROIs**: direct YOLO pipeline
- **Long videos** (> 2500 frames) with **simple ROIs** in config: filtered pipeline (SimpleDetector pre-filter → clip extraction → YOLO on clips)

```
process_video(video_id, db):
    1. Set status='processing', record processing_started_at
    2. Detect video length (total_frames, fps, duration)
    3. Resolve ROI config from video's restaurant_name + camera_id:
       - Looks for data/roi_configs/{restaurant}_{camera}.json
       - Raises error if not found (no default fallback)
    4. Branch: if total_frames > 2500 AND "simple" section in ROI config:
       → _run_filtered_pipeline()  (see below)
       Otherwise:
       → _run_yolo_pipeline()  (direct YOLO on full video)
    5. Clear any existing tap_events for this video (re-processing safe)
    6. Map pour_events to DB TapEvent rows:
       - "TAP_A" → "A", "TAP_B" → "B"
       - "time_start"/"time_end" → timestamp_start/timestamp_end
       - UNKNOWN or unassigned events are skipped
    7. Set status='completed', record processing_finished_at, output_dir
    8. On exception: status='error', save error_message
```

#### Filtered Pipeline (`_run_filtered_pipeline`)

For long videos, running YOLO on every frame is too slow. The filtered pipeline uses SimpleDetector (fast CPU pixel differencing) to find activity windows, then runs the full YOLO+SAM3 pipeline only on those clips.

```
_run_filtered_pipeline(video_path, video_id, roi, video, db, fps):
    Stage 1: SimpleDetector (status='processing_filter')
       - Uses simple ROIs from config (roi_1, roi_2 — normalized 0-1)
       - Parameters: sample_every=3, on_threshold=0.05, min_on_frames=10, n_workers=1
       - Produces per-tap activity signals (fraction of changed pixels per frame)
       - Saves simple_summary.json

    Stage 2: Activity Windows + Clip Extraction (status='processing_clips')
       - find_activity_windows(): combines tap signals, finds stretches > threshold
         - threshold=0.05, padding_s=5.0, merge_gap_s=40.0
         - Merges nearby windows to avoid fragmenting single pour events
       - extract_clips(): writes MP4 clips for each window
         - Named: clip_{i:03d}_{start}s_{end}s.mp4
       - Saves activity_windows.json, records num_clips + filtered_duration_s

    Stage 3: YOLO on Each Clip (status='processing_yolo')
       - Runs _run_yolo_pipeline() on each clip independently
       - Maps timestamps back to original video:
         event.time_start += window.start_s
         event.frame_start += int(window.start_s * fps)
       - Aggregates pour events from all clips
```

**Example:** A 2-hour video (216,000 frames at 30fps) might have 8 activity windows totaling 12 minutes. SimpleDetector scans the full video in ~15 seconds on CPU, then YOLO processes only those 12 minutes instead of 2 hours.

#### Direct YOLO Pipeline (`_run_yolo_pipeline`)

```
_run_yolo_pipeline(video_path, video_id, roi, output_subdir=None):
    1. Load base config from config/pipeline.yaml
    2. Override: video_path, output_dir, roi.tap_roi, sam3.tap_bboxes
    3. Resolve relative paths: `data/` paths via DATA_DIR, others via PROJECT_ROOT
    4. Pre-create tap_roi.json in output_dir (skip interactive mode)
    5. Write temp YAML config, run YOLODetector(config).run()
    6. Return pour_events list
```

### ROI Config System (`data/roi_configs/`)

ROI configs are JSON files named by restaurant+camera: `{restaurant_name}_{camera_id}.json`. There is **no default fallback** — each restaurant+camera combo must have its own config. Configs can be created via the frontend's visual ROI wizard or the CLI interactive tools.

```json
{
  "simple": {
    "roi_1": [0.5172, 0.4583, 0.5469, 0.7889],
    "roi_2": [0.5359, 0.4972, 0.5781, 0.8111]
  },
  "yolo": {
    "tap_roi": [0.4483, 0.3702, 0.6655, 0.8278],
    "sam3_tap_bboxes": [
      [115.59, 155.02, 294.97, 215.52],
      [442.50, 78.60, 481.78, 270.71]
    ]
  }
}
```

- **`yolo`** section (mandatory):
  - `tap_roi`: normalized (0–1) crop region on the full frame
  - `sam3_tap_bboxes`: pixel-space bounding boxes on the cropped frame for TAP_A and TAP_B
- **`simple`** section (optional — enables filtered pipeline for long videos):
  - `roi_1`: normalized (0–1) bounding box around Tap A handle area (for pixel differencing)
  - `roi_2`: normalized (0–1) bounding box around Tap B handle area (defaults to roi_1 if omitted)
  - When present and video > 2500 frames, SimpleDetector pre-filter runs before YOLO
- Resolution: `_resolve_roi_config_name(restaurant, camera)` → returns `"{restaurant}_{camera}"` if file exists, else `None` (error)
- Naming validated: only `[a-zA-Z0-9_-]` allowed in restaurant_name and camera_id

### Error Handling Patterns

- Wrap processing in try/except; always update DB status on failure.
- Validate file type on upload (reject non mp4/mov).
- Return proper HTTP codes: 404 for unknown video_id, 409 if already processing, 422 for bad input.

---

## 5. Frontend — Streamlit (DONE)

### Layout (`frontend/app.py`)

Two-tab layout with ROI wizard:

```
Page: Beer Tap Counter
│
├── Tab 1: Upload & Process
│   ├── Recent videos list (last 5, with status icons)
│   ├── Restaurant selector (dropdown + "Add new...")
│   ├── Camera ID selector (dropdown + "Add new...", filtered by restaurant)
│   ├── Preview ROI checkbox (only when config exists for selected combo)
│   ├── File uploader (mp4/mov) — auto-uploads on file select
│   │   ├── If ROI config exists + preview unchecked: auto-triggers processing
│   │   ├── If ROI config exists + preview checked: shows ROI confirmation page
│   │   │   ├── First frame with ROI boxes (crop=orange, TAP A=blue, TAP B=green)
│   │   │   ├── "Confirm & Process" / "Re-draw ROI" / "Cancel"
│   │   └── If no ROI config: enters ROI wizard (3-step)
│   │       ├── Step 1: Select crop region on full frame (streamlit-cropper)
│   │       ├── Step 2: Select TAP A handle on cropped image
│   │       ├── Step 3: Select TAP B handle on cropped image → save & process
│   └── Active video status:
│       ├── If pending/queued: warning + auto-poll
│       ├── If processing:     warning + auto-poll
│       ├── If completed:      success + metrics (Tap A, Tap B, Total) + events table
│       └── If error:          error message
│
└── Tab 2: Dashboard
    ├── Refresh button
    ├── Global summary: 4x st.metric (Tap A Total, Tap B Total, Grand Total, Videos Processed)
    └── Video list: expandable per video (shows restaurant/camera in label)
        ├── Completed: metrics + events table
        ├── Error: error message
        ├── Other: status info
        └── Delete button per video
```

### API Client (`frontend/utils/api_client.py`)

Thin wrapper using `requests`, configured via `BACKEND_URL` env var (default `http://localhost:8000`):

| Function | Endpoint | Returns |
|----------|----------|---------|
| `list_videos()` | GET /api/videos/ | list[dict] |
| `get_video_status(video_id)` | GET /api/videos/{id}/status | dict |
| `upload_video(name, file_bytes, restaurant_name, camera_id)` | POST /api/videos/upload | dict |
| `process_video(video_id)` | POST /api/videos/{id}/process | HTTP status code |
| `delete_video(video_id)` | DELETE /api/videos/{id} | bool |
| `get_counts_summary()` | GET /api/counts/summary | dict |
| `get_counts()` | GET /api/counts/ | list[dict] |
| `get_restaurants()` | GET /api/videos/restaurants | dict |
| `check_roi_config(restaurant, camera)` | GET /api/videos/roi-config-exists | dict |
| `get_video_frame(video_id)` | GET /api/videos/{id}/frame | bytes (JPEG) |
| `save_roi_config(restaurant, camera, roi_data)` | POST /api/videos/roi-config | dict |

### Queue Management (Frontend-side)

Only one video processes at a time (GPU constraint). No backend changes needed:

1. On upload, check `list_videos()` — if any video has `status == "processing"`, don't trigger process
2. Video stays as "pending" with "queued" message
3. On "Refresh Status" click, if nothing else is processing, auto-triggers the pending video

### UX Notes

- File uploader hides while a video is processing (prevents confusion)
- No auto-polling — user clicks "Refresh Status" manually (avoids jarring reloads)
- `st.toast()` for upload/process/delete notifications
- Delete available inside video expanders on Dashboard tab

---

## 6. ML Pipeline — Implemented Architecture

Processing always runs the full YOLO+SAM3 pipeline. SimpleDetector is available as a CLI tool and planned as a future pre-filter for long videos (see Optimization section below).

### Approach A: SimpleDetector (CPU, fast) — Pre-filter + CLI

`backend/ml/approach_simple/detector.py` — pixel differencing on two small tap-handle ROIs.

- Two normalised ROIs per tap handle → frame differencing → activity signal per tap
- ON/OFF thresholding → event detection (each OFF→ON→OFF cycle = 1 pour event)
- Multiprocessing on chunks for long videos (> 3000 frames)
- Runs in seconds on CPU, no GPU needed
- Outputs: per-tap activity signals, tap heatmaps, event list
- **Used by the web app** as a pre-filter for long videos (> 2500 frames) when the ROI config has a `simple` section — finds activity windows, extracts clips, then YOLO runs only on those clips
- Also available as a standalone CLI tool

```bash
# Interactive ROI selection:
python scripts/run_simple.py --video data/videos/cerveza2.mp4 --interactive

# From saved ROIs:
python scripts/run_simple.py --video data/videos/cerveza2.mp4 \
    --roi-json results/simple_cerveza2/simple_roi.json
```

### Approach B: YOLO + SAM3 Pipeline (GPU, accurate) — Used by web app

`backend/ml/approach_yolo/` — 4-stage system + tap assignment, driven by a YAML config file. Each stage produces output files that the next stage consumes. Stages are skippable and idempotent — existing outputs are reused unless `--force` is set.

### Pipeline Stages

```
Stage 1: ROI Selection      → tap_roi.json (crop region + SAM3 tap bboxes)
Stage 2: YOLO Tracking      → raw_detections.csv (per-frame cup/person tracks)
Stage 3: Relink              → relinked_detections.csv + pour_events.json
Stage 4: SAM3 Tap Tracking   → sam3_centroids.csv (tap handle centroid trajectories)
Final:   Tap Assignment      → pour_events_assigned.json (done in detector.py)
```

### Stage 1: ROI Selection

Interactive or config-driven. User selects:
1. **Crop region** — normalized (0–1) bounding box on the full frame
2. **TAP_A / TAP_B bboxes** — pixel-space bounding boxes on the cropped frame (used to initialize SAM3)

No A|B divider line — tap side is determined later by SAM3 centroid Y movement.
Saved to `tap_roi.json` in the output directory.

For web uploads, `tap_roi.json` is pre-created from the named ROI config to skip interactive mode.

### Stage 2: YOLO Tracking

Uses **YOLO-World** (`yolov8x-worldv2.pt`) for zero-shot open-vocabulary detection of `cup` and `person` classes on the cropped video. Tracking via **BoT-SORT** (`config/botsort.yaml`) assigns persistent IDs across frames.

Key parameters (from config):
- `sample_every: 1` — process every frame (configurable)
- `conf_threshold: 0.25`
- `record_range` — optional time window `[start_s, stop_s]`
- `save_video: false` — when true, saves `yolo_raw_tracking.mp4` (pre-relink annotated video)

Output: `raw_detections.csv` with per-frame bounding boxes and track IDs. `summary.json` includes `tracking_elapsed_s`. Execution time is tracked at all levels: per-stage in `yolo_track.py`, per-pipeline in `detector.py`/`pipeline.py`, and per-batch in `run_yolo_on_clips.py`.

### Stage 3: Relink

Post-processes YOLO tracks to fix fragmentation and classify pour events:
1. **Merge fragmented tracks** — tracks with overlapping time/space are relinked into single continuous tracks
2. **Classify pours** — a track is a "pour" if it meets all three criteria:
   - Enough frames (`min_pour_frames: 30`)
   - Enough spatial spread (`movement_threshold: 5.0 px`)
   - **Not stationary** — cups that sit in one spot (within `stationary_px: 10` of their median position) for more than `stationary_ratio: 80%` of their lifespan are filtered out (e.g. cups left on the counter, waiting glasses)

Key parameters:
- `overlap_threshold: 15` frames of co-existence → incompatible tracks
- `min_track_dets: 2` — ignore tiny tracks
- `max_interp_gap: 10` — interpolate gaps up to this many frames
- `stationary_ratio: 0.8` — fraction of frames near median position to be considered stationary
- `stationary_px: 10.0` — pixel radius for "same spot" check

#### Resolution-Aware Scaling

Pixel-based thresholds (`movement_threshold`, `stationary_px`) were calibrated on ~800px-wide crops from 4K video. For lower-resolution videos (e.g. 640x360 with a ~111px crop), these thresholds are too strict and miss real pour events.

The relink stage auto-detects the crop width from `tap_roi.json` + video resolution and scales thresholds proportionally when the crop is smaller than 90% of the 800px reference:

```
scale = crop_width / 800.0
movement_threshold_effective = movement_threshold * scale
stationary_px_effective = stationary_px * scale
```

Example: 111px crop → scale=0.14 → `movement_threshold: 5.0 → 0.7px`, `stationary_px: 10.0 → 1.4px`.

Font sizes, line thicknesses, and timestamp overlays in annotated videos (`draw_detections`, `render_annotated_video`, `yolo_raw_tracking.mp4`) also scale proportionally to crop width via `_font_scale_for_width()`.

Output: `relinked_detections.csv`, `pour_events.json`, `pour_frame_ranges.json`.

### Stage 4: SAM3 Tap Handle Tracking

Uses **SAM3VideoPredictor** (Segment Anything Model 3) to track tap handles across the video via mask propagation with a memory bank. Initialized with bounding boxes from Stage 1.

- Tracks TAP_A and TAP_B handle positions (centroid X/Y per frame)
- Only processes pour-event frame ranges (from Stage 3) for efficiency
- `frame_skip: 5`, `half: true` for speed

Output: `sam3_centroids.csv` with per-frame centroid coordinates for each tap label.

### Tap Assignment (Final — in `detector.py`)

Correlates pour events with SAM3 centroid data:
- For each pour's frame range, compute **centroid-Y standard deviation** for each tap
- The tap with **more Y movement** during the pour is the one that poured
- If no meaningful movement difference → `UNKNOWN`

Output: `pour_events_assigned.json` (enriched pour events with `tap` field).

**Note:** Tap assignment is called in `YOLODetector.run()` after all 4 stages complete. It was originally only in `pipeline.py:main()` (the CLI entry point) and was missing from the detector class.

### Pipeline Config (`config/pipeline.yaml`)

```yaml
video_path: data/videos/cerveza2.mp4
output_dir: results/pipeline_cerveza2

stages:
  roi_selection: true
  yolo_tracking: true
  relink: true
  sam3_tap_tracking: true

roi:
  roi_json: null       # reuse from another run, or null for interactive
  tap_roi: null        # [x1, y1, x2, y2] normalized, or null
  tap_divider: null    # optional, not used by YOLO/relink

yolo:
  model: data/models/yolov8x-worldv2.pt
  classes: [cup, person]
  sample_every: 1
  conf_threshold: 0.25
  tracker: config/botsort.yaml
  save_video: false        # save yolo_raw_tracking.mp4 (pre-relink annotated video)

relink:
  overlap_threshold: 15
  min_pour_frames: 30
  movement_threshold: 5.0  # auto-scaled by crop resolution (ref: 800px)
  stationary_ratio: 0.8    # skip cups stationary for >80% of their lifespan
  stationary_px: 10.0      # auto-scaled by crop resolution (ref: 800px)
  save_video: false         # save relinked_full.mp4 (post-relink annotated video)

sam3:
  model: data/models/sam3.pt
  object_labels: [TAP_A, TAP_B]
  tap_bboxes: null     # pixel-space on cropped frame, or null for interactive
  frame_skip: 5
  half: true
```

### Running the YOLO Pipeline

```bash
# Full pipeline (interactive ROI + bbox selection on first run):
python scripts/run_yolo_pipeline.py --config config/pipeline.yaml --interactive

# Re-run with saved coordinates:
python scripts/run_yolo_pipeline.py --config config/pipeline.yaml

# Run a single stage:
python scripts/run_yolo_pipeline.py --config config/pipeline.yaml --stage relink

# Force re-run even if outputs exist:
python scripts/run_yolo_pipeline.py --config config/pipeline.yaml --force
```

### Key Implementation Files

| File | Purpose |
|------|---------|
| `backend/ml/common.py` | Shared utilities — ROI selection, cropping, interactive selectors |
| `backend/ml/approach_simple/detector.py` | SimpleDetector — CPU pixel differencing |
| `backend/ml/approach_simple/filtering.py` | Activity window detection + clip extraction |
| `backend/ml/approach_yolo/detector.py` | YOLODetector — GPU wrapper + tap assignment |
| `backend/ml/approach_yolo/pipeline.py` | 4-stage orchestrator (ROI → YOLO → Relink → SAM3) |
| `backend/ml/approach_yolo/yolo_track.py` | YOLO-World + BoT-SORT tracking |
| `backend/ml/approach_yolo/relink.py` | Track relinking + pour classification |
| `backend/ml/approach_yolo/sam3_tracking.py` | SAM3 tap handle segmentation |
| `scripts/run_simple.py` | CLI entry for SimpleDetector (+ activity clip extraction) |
| `scripts/run_yolo_pipeline.py` | CLI entry for YOLO pipeline |
| `scripts/run_yolo_on_clips.py` | Batch YOLO+SAM3 on activity clips (same code path as backend) |
| `config/pipeline.yaml` | Default YOLO pipeline config |
| `config/botsort.yaml` | BoT-SORT tracker config |

---

## 7. Testing

### Test Suite (59 tests)

Run with: `uv run pytest tests/ -v`

| File | Tests | What it covers |
|------|-------|----------------|
| `test_app.py` | 2 | Health check, /docs available |
| `test_database.py` | 6 | Table existence, column names (incl. restaurant_name, camera_id), insert, cascade delete |
| `test_schemas.py` | 5 | Pydantic model instantiation and serialization |
| `test_videos_router.py` | 16 | Upload (with/without restaurant+camera), reject bad extension, list, status, 404s, process 202, delete, ROI config endpoints (exists, save, validation), restaurants list |
| `test_counts_router.py` | 4 | Counts query, filter by tap, empty results, summary |
| `test_processor.py` | 11 | ROI config loading/validation, ROI resolution (existing file, missing, None values), YOLO config builder (mocked), event mapping |
| `test_api_client.py` | 15 | Frontend API client: list, upload (with restaurant+camera), process, status, delete, restaurants, ROI config check, frame, save ROI (mocked HTTP) |

### Testing patterns

- In-memory SQLite with `StaticPool` for router tests (avoids disk, fast, isolated per test)
- `dependency_overrides` on `get_db` to inject test DB session
- Mock `_import_yolo_detector` to test config building without GPU/ML deps
- Each test fixture creates a fresh DB

---

## 8. Development Environment

- **Python:** 3.11 via `.venv` (managed with `uv` and `pyproject.toml`)
- **Install deps:** `uv sync` (installs all deps + dev group from `pyproject.toml`)
- **Add a dependency:** `uv add <package>` or `uv add --dev <package>`
- **Run backend:** `uv run uvicorn backend.main:app --port 8000 --reload`
- **Run frontend:** `cd frontend && uv run streamlit run app.py`
- **Run tests:** `uv run pytest tests/ -v`
- **Lint:** `uv run ruff check .` (auto-fix: `uv run ruff check --fix .`)
- **Format:** `uv run ruff format .`
- **DB location:** `data/db_files/app.db` (delete to reset schema)

### Code Quality

- **Ruff** — linter + formatter (replaces Black, flake8, isort). Config in `pyproject.toml`.
- **Pre-commit** — runs `ruff check --fix` and `ruff format` on every `git commit`.
  - Install hooks: `uv run pre-commit install`
  - Run manually: `uv run pre-commit run --all-files`
- **Ruff rules:** `E` (pycodestyle), `W` (warnings), `F` (pyflakes), `I` (isort), `UP` (pyupgrade)
- **Excluded:** `notebooks/` (exploration code, not production)

### Logging

- **Config:** `backend/logging_config.py` — called on app startup via `setup_logging()`
- **Console:** always on, format: `%(asctime)s | %(name)s | %(levelname)s | %(message)s`
- **File:** optional, enabled via `LOG_TO_FILE=true` env var → `data/logs/app.log` (rotating, 10MB max, 3 backups)
- **Level:** configurable via `LOG_LEVEL` env var (default: `INFO`)
- **Usage:** every module uses `logger = logging.getLogger(__name__)`
- **Noisy loggers silenced:** `ultralytics`, `urllib3` set to WARNING

---

## 9. Development Workflow

### Phase 1: ML Pipeline (DONE)

Notebook-based exploration → two production detectors:
1. Explored videos, calibrated ROIs interactively (`notebooks/01_explore*.py`).
2. Built YOLO-World + BoT-SORT + Relink + SAM3 pipeline (now in `backend/ml/approach_yolo/`).
3. Built SimpleDetector — CPU pixel differencing (now in `backend/ml/approach_simple/`).
4. Validated YOLO pipeline across multiple videos (pipeline1–7 configs).
5. Organized: production code in `backend/ml/`, exploration in `notebooks/`, CLIs in `scripts/`.

### Phase 2: Application Skeleton (DONE — backend, frontend TODO)

1. ✅ Database models, connection, schemas (`backend/database/`).
2. ✅ FastAPI endpoints — videos router (upload, list, status, delete, process, ROI, frame, restaurants).
3. ✅ Counts router (query with filters, summary).
4. ✅ Background processor service — bridges YOLO pipeline to DB.
5. ✅ ROI config system — `{restaurant}_{camera}.json` configs, no default fallback.
6. ✅ Restaurant + Camera ID on Video model, resolved to ROI config automatically.
7. ✅ 59 tests covering all layers (including frontend api_client).
8. ✅ End-to-end tested: upload → process → completed with tap events.
9. ✅ Streamlit UI — two tabs, restaurant/camera dropdowns, ROI wizard (streamlit-cropper), ROI preview, queue management, delete.
10. ✅ Dockerfiles (multi-stage backend + frontend) and docker-compose.yml.

### Phase 3: Polish

1. Test with all provided videos, fix edge cases.
2. Add error handling and input validation.
3. Write README with setup instructions.
4. Clean up code.
5. Test the Docker build from scratch to ensure it works cleanly.
6. **Goal:** Demo-ready. `docker-compose up` → open browser → upload new video → correct results.

---

## 10. Critical Trade-offs to Document (for the presentation)

These should go in the README or be prepared as talking points:

1. **Simple vs YOLO (hybrid approach implemented):** Simple is faster and requires no GPU but is fragile to environmental changes. YOLO is robust but heavier. The hybrid approach is implemented: SimpleDetector scans the full video on CPU (~15s for 2h), finds activity windows, extracts clips, then YOLO+SAM3 runs only on those clips. Activated automatically for videos > 2500 frames when the ROI config has a `simple` section.

2. **Frame sampling rate:** Processing every frame is accurate but slow (a 2h video at 30fps = 216,000 frames). Sampling every 3rd frame (10 effective fps) cuts processing by 3x with negligible accuracy loss for events lasting several seconds.

3. **SQLite vs Postgres:** SQLite is single-writer, which is fine for single-user demo. Postgres would be needed for concurrent users. SQLite keeps Docker setup trivial.

4. **Streamlit vs React:** Streamlit is less customizable but delivers the required UI in ~100 lines. React would require separate build tooling, more code, and a Node.js container.

5. **Background processing:** FastAPI BackgroundTasks is simple but limited (no retry, no queue). For production, you'd use Celery + Redis. For this demo, BackgroundTasks is sufficient.

6. **ROI configs:** ROIs are stored as `{restaurant}_{camera}.json` configs. The frontend provides a visual 3-step ROI wizard using `streamlit-cropper`. No default fallback — each restaurant+camera combo must be calibrated. A production system could add auto-detection.

7. **Tap assignment in detector vs pipeline:** The tap assignment logic (`_assign_pours_to_taps`) was originally only in `pipeline.py:main()` (CLI path). It's now also called in `YOLODetector.run()` so the web flow gets tap-assigned results.

---

## 11. Validation Checklist (Pre-Demo)

Before the presentation, verify:

- [x] Backend starts without errors: `uvicorn backend.main:app --port 8000`
- [x] All 59 tests pass: `uv run pytest tests/ -v`
- [x] Upload a video with restaurant+camera: `POST /api/videos/upload?restaurant_name=X&camera_id=Y`
- [x] Processing starts: `POST /api/videos/{id}/process` returns 202
- [x] Processing completes: status updates to 'completed'
- [x] Pipeline outputs + video saved to `data/results/web_{id}/`
- [ ] Tap A count, Tap B count, and Total are displayed correctly
- [ ] Counts match manual count from watching the video
- [ ] Uploading a second video works and shows separate results
- [ ] Query by date/tap works via `/api/counts/`
- [x] Error case: upload a non-video file → 422 error
- [ ] `docker-compose up` builds and starts both services
- [ ] Frontend is accessible at `localhost:8501`
- [ ] Processing a 2h video completes in reasonable time (< 5 min with GPU)

---

## 12. Dependencies Summary

All dependencies are managed in `pyproject.toml` with `uv`.

### Production dependencies (`[project].dependencies`)

```
opencv-python, numpy, matplotlib, pandas, ultralytics,
clip (git+https://github.com/ultralytics/CLIP.git),
lap, timm,
fastapi, uvicorn[standard], sqlalchemy, aiofiles,
python-multipart, pyyaml, streamlit, requests,
streamlit-cropper
```

### Dev dependencies (`[dependency-groups].dev`)

```
pytest, ruff, pre-commit, httpx, ipython, ipykernel
```

### Environment

Managed with `uv` in `.venv`:
```bash
uv sync              # install all deps + dev group
uv run pytest        # run in managed env
uv add <package>     # add production dep
uv add --dev <pkg>   # add dev dep
```
