# Beer Tap Counter

Counts beers served from a dual-tap beer dispenser by analyzing uploaded video footage. Identifies pour events on **Tap A** and **Tap B** independently, persists results in SQLite, and displays counts through a web UI.

Built as a case study for **Intern Full Stack & AI Developer** at Gambooza.

## Architecture

```
Streamlit (frontend)  ──HTTP──>  FastAPI (backend + ML pipeline)  ──>  SQLite
     :8501                              :8000                        app.db
```

- **Backend:** FastAPI REST API. Accepts video uploads, runs ML processing in background, exposes counts.
- **Frontend:** Streamlit app. Upload videos, configure ROI (tap regions), view per-tap counts and pour event timeline.
- **ML Pipeline:** YOLO-World (zero-shot object detection) + BoT-SORT (tracking) + SAM3 (tap handle segmentation) for accurate, robust counting.
- **Database:** SQLite with SQLAlchemy ORM. Videos + TapEvents tables.

## ML Pipeline

### Smart Filtering (long videos)

For videos longer than ~80 seconds (2500 frames), a **SimpleDetector pre-filter** runs first on CPU using pixel differencing. It scans the full video in seconds, identifies activity windows (moments when tap handles move), and extracts short clips. The YOLO+SAM3 pipeline then processes only those clips instead of the entire video. This is activated automatically when the ROI config includes a `simple` section with per-tap handle ROIs.

Example: a 2-hour video might yield 8 activity clips totaling 12 minutes — reducing YOLO processing time from hours to minutes.

### YOLO+SAM3 Pipeline (4 stages)

1. **ROI Selection** — Crop region + tap handle bounding boxes (interactive or from saved config)
2. **YOLO Tracking** — YOLO-World detects cups/persons on cropped video, BoT-SORT assigns persistent IDs
3. **Relink** — Merges fragmented tracks, classifies pour events (filters stationary cups, too-short tracks)
4. **SAM3 Tap Tracking** — Segments tap handles, tracks centroid movement to assign pours to Tap A or Tap B

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPU + CUDA (for YOLO/SAM3 inference)
- ML model weights in `data/models/` (yolov8x-worldv2.pt, sam3.pt)

### Local (without Docker)

```bash
# Install dependencies
uv sync

# Start backend (terminal 1)
uv run uvicorn backend.main:app --port 8000 --reload

# Start frontend (terminal 2)
cd frontend && uv run streamlit run app.py
```

Open http://localhost:8501, upload a video, draw the ROI regions, and process.

### Docker Compose

```bash
# Ensure data directories and model weights exist
mkdir -p data/{db_files,models,roi_configs,results}
# Place yolov8x-worldv2.pt and sam3.pt in data/models/

# Build and start
docker compose up --build
```

- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- Data persists in `./data/` on the host (bind-mounted to `/app/mount_data` in the container)

> **Requirements:** `nvidia-container-toolkit` for GPU passthrough, NVIDIA GPU with CUDA support.

The backend uses a multi-stage Docker build: dependencies are installed with `uv` in a builder stage, then copied to a CUDA runtime image with Python 3.11. Key env vars:
- `DATA_DIR=/app/mount_data` — tells the backend where bind-mounted data lives
- `YOLO_AUTOINSTALL=False` — prevents ultralytics from auto-installing optional packages at runtime
- `gcc/g++` are included in the runtime image for PyTorch Triton JIT compilation (used by SAM3)

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/videos/upload` | Upload mp4/mov video |
| POST | `/api/videos/{id}/process` | Start ML processing |
| GET | `/api/videos/{id}/status` | Status + counts + events |
| GET | `/api/videos/` | List all videos |
| DELETE | `/api/videos/{id}` | Delete video + events |
| GET | `/api/counts/` | Query counts (filters: video_id, date, tap) |
| GET | `/api/counts/summary` | Aggregate totals |

## Testing

```bash
uv run pytest tests/ -v    # 59 tests
uv run ruff check .        # lint
uv run ruff format .       # format
```

## Project Structure

```
backend/
  main.py              # FastAPI entry point
  database/            # SQLAlchemy models, schemas, connection
  routers/             # videos + counts endpoints
  services/            # background processor
  ml/
    approach_yolo/     # YOLO + SAM3 pipeline (used by web app)
    approach_simple/   # CPU pixel-differencing (pre-filter for long videos + CLI)
frontend/
  app.py               # Streamlit UI (upload, ROI wizard, dashboard)
  utils/api_client.py  # Backend HTTP client
config/                # Pipeline YAML configs + BoT-SORT config
data/                  # Videos, models, DB, ROI configs, results
tests/                 # pytest suite (59 tests)
```

## Key Trade-offs

| Decision | Why |
|----------|-----|
| **Hybrid: pixel pre-filter + YOLO+SAM3** | Pixel differencing (CPU, seconds) finds activity windows; YOLO+SAM3 (GPU, accurate) processes only those clips. Best of both worlds for long videos. |
| **SQLite over Postgres** | Single-user demo, trivial setup. Postgres needed for concurrency. |
| **Streamlit over React** | Delivers required UI in ~600 lines. React would need separate build tooling. |
| **BackgroundTasks over Celery** | Simple, no extra infra. Celery+Redis for production scale. |
| **Per-restaurant ROI configs** | Each camera angle needs calibration. No universal default — accuracy over convenience. |
| **Process every frame** | Accurate for short videos. For 2h+ videos, `sample_every: 3` cuts time 3x with minimal accuracy loss. |
