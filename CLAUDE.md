# CLAUDE.md — Beer Tap Counter: Implementation Guide

## Project Overview

A self-contained application that counts beers served from a dual-tap beer dispenser by analyzing uploaded video footage. The system identifies events on Tap A and Tap B independently, persists results in SQLite, and displays counts through a minimal web UI.

**Architecture:** Docker Compose → Streamlit (frontend) + FastAPI (backend + ML pipeline) + SQLite (persistence)

---

## 1. Project Structure

```
gambooza_case_study/
├── CLAUDE.md
├── docker-compose.yml              # (TODO) Docker Compose for full stack
│
├── backend/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app entry point
│   ├── config.py                   # Settings, paths, constants
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py           # Engine, SessionLocal, init_db, get_db
│   │   ├── models.py              # SQLAlchemy ORM: Video, TapEvent
│   │   └── schemas.py             # Pydantic response models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── videos.py              # Upload, list, status, delete, process
│   │   └── counts.py              # Query counts, summary
│   ├── services/
│   │   ├── __init__.py
│   │   └── processor.py           # Background processor (YOLO pipeline bridge)
│   └── ml/
│       ├── __init__.py
│       ├── common.py               # Shared utilities (ROI, cropping, interactive selectors)
│       ├── approach_simple/
│       │   ├── __init__.py
│       │   └── detector.py         # SimpleDetector — CPU pixel differencing
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
│   ├── test_videos_router.py       # Upload, list, status, delete, process endpoints
│   ├── test_counts_router.py       # Counts query and summary endpoints
│   └── test_processor.py           # ROI config loader, YOLO config builder, event mapper
│
├── frontend/                       # (TODO) Streamlit UI
│
├── scripts/
│   ├── run_simple.py               # CLI: run SimpleDetector on a video
│   └── run_yolo_pipeline.py        # CLI: run YOLO+SAM3 pipeline on a video
│
├── config/
│   ├── pipeline.yaml               # Default YOLO pipeline config
│   ├── pipeline[1-7].yaml          # Per-video pipeline configs
│   └── botsort.yaml                # BoT-SORT tracker config
│
├── data/
│   ├── videos/                     # Source videos (cerveza1–7.mp4, etc.)
│   ├── uploads/                    # User-uploaded videos (via API)
│   ├── models/                     # ML weights (yolov8x-worldv2.pt, sam3.pt, etc.)
│   ├── db_files/                   # SQLite database (app.db)
│   └── roi_configs/                # Named ROI configs (default.json, etc.)
│
├── results/                        # Pipeline outputs (per-video directories)
│   ├── web_{id}/                   # Web upload pipeline outputs
│   ├── pipeline_cerveza*/          # CLI pipeline outputs
│   └── simple_test_cerveza*/       # SimpleDetector outputs
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

Two services sharing a named volume for videos and the database.

```yaml
version: "3.9"
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - upload_data:/app/data/uploads
      - db_data:/app/data/db_files
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]  # Enable GPU passthrough

  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    depends_on:
      - backend
    environment:
      - BACKEND_URL=http://backend:8000

volumes:
  upload_data:
  db_data:
```

### Key Docker Decisions

- **GPU passthrough:** Use `nvidia-container-toolkit` so the backend container can access the host GPU. Required for both YOLO inference and faster OpenCV processing.
- **Shared volumes:** Uploads saved by the backend; both services read the SQLite DB (though only the backend writes).
- **Always YOLO:** Processing always runs the full YOLO+SAM3 pipeline. SimpleDetector is planned as a future pre-filter for long videos (see notes.txt).

### Backend Dockerfile Skeleton

```dockerfile
FROM python:3.11-slim
# Install system deps for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 ffmpeg
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile Skeleton

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 3. Database Layer (SQLite + SQLAlchemy)

### ORM Models (`backend/database/models.py`)

```
Table: videos
├── id                    INTEGER PK AUTOINCREMENT
├── filename              TEXT NOT NULL          # UUID name on disk (in data/uploads/)
├── original_name         TEXT NOT NULL          # user's original filename
├── upload_date           DATETIME DEFAULT now
├── status                TEXT DEFAULT 'pending' # pending | processing | completed | error
├── duration_sec          FLOAT NULLABLE
├── error_message         TEXT NULLABLE
├── ml_approach           TEXT NULLABLE          # "yolo"
├── processing_started_at DATETIME NULLABLE
├── processing_finished_at DATETIME NULLABLE
└── output_dir            TEXT NULLABLE          # path to pipeline intermediate files

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
- `VideoUploadResponse(id, filename, original_name, status)`
- `VideoStatusResponse(id, filename, original_name, upload_date, status, duration_sec, error_message, ml_approach, processing_started_at, processing_finished_at, output_dir, tap_a_count, tap_b_count, total, events)`
- `VideoListItem(id, original_name, upload_date, status, ml_approach)`
- `TapEventResponse(id, tap, frame_start, frame_end, timestamp_start, timestamp_end, confidence, count)`
- `CountResult(video_id, original_name, upload_date, tap_a, tap_b, total)`
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

| Method | Endpoint                   | Purpose                                              |
|--------|----------------------------|------------------------------------------------------|
| POST   | `/api/videos/upload`       | Accept mp4/mov, save to `data/uploads/`, create DB row |
| POST   | `/api/videos/{id}/process` | Launch background YOLO pipeline, returns 202         |
| GET    | `/api/videos/{id}/status`  | Return status + counts + events if completed         |
| GET    | `/api/videos/`             | List all uploaded videos                             |
| DELETE | `/api/videos/{id}`         | Remove video file + DB records (cascade)             |

#### Upload Flow

1. Validate file extension (.mp4, .mov only).
2. Generate unique filename: `{uuid8}_{original_name}`.
3. Save to `data/uploads/`.
4. Insert row in `videos` table with `status='pending'`.
5. Return `VideoUploadResponse`.

#### Process Flow

1. Validate video exists (404) and not already processing (409).
2. Launch `BackgroundTasks.add_task(_run_processing, video_id, roi_config)`.
3. Return `202 Accepted` immediately.
4. Background task creates its own `SessionLocal()` (request session closes on response).
5. Processor runs YOLO pipeline, writes `tap_events`, updates status.

#### Process Endpoint Parameters

- `roi_config` (query, default `"default"`) — name of ROI config in `data/roi_configs/`

### Router: Counts (`backend/routers/counts.py`)

| Method | Endpoint              | Purpose                                              |
|--------|-----------------------|------------------------------------------------------|
| GET    | `/api/counts/`        | Query counts with filters: video_id, date range, tap |
| GET    | `/api/counts/summary` | Aggregate totals across all completed videos         |

### Background Processing (`backend/services/processor.py`)

```
process_video(video_id, db, roi_config="default"):
    1. Set status='processing', record processing_started_at
    2. Load ROI config from data/roi_configs/{roi_config}.json
    3. Create temp YAML config from config/pipeline.yaml:
       - Override video_path → data/uploads/{filename}
       - Override output_dir → results/web_{video_id}/
       - Override roi.tap_roi + sam3.tap_bboxes from ROI config
       - Resolve all relative paths to absolute (tracker, models)
       - Pre-create tap_roi.json in output_dir (skip interactive mode)
    4. Run YOLODetector(temp_config).run()
    5. Map pour_events to DB TapEvent rows:
       - "TAP_A" → "A", "TAP_B" → "B"
       - "time_start"/"time_end" → timestamp_start/timestamp_end
       - UNKNOWN or unassigned events are skipped
    6. Set status='completed', record processing_finished_at, output_dir
    7. On exception: status='error', save error_message
```

### ROI Config System (`data/roi_configs/`)

Named JSON files with ROI coordinates for different camera setups:

```json
{
  "simple": {
    "tap_roi": [0.4678, 0.3737, 0.6311, 0.5540],
    "tap_a_roi": [0.4827, 0.4470, 0.5245, 0.4705],
    "tap_b_roi": [0.5638, 0.4103, 0.5764, 0.5340]
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

- `simple` section: normalized ROIs for SimpleDetector (future use)
- `yolo` section: crop region + pixel-space SAM3 bounding boxes
- `default.json` ships with cerveza camera values
- New cameras → new JSON file (via CLI interactive tools, frontend picker in Phase 3)

### Error Handling Patterns

- Wrap processing in try/except; always update DB status on failure.
- Validate file type on upload (reject non mp4/mov).
- Return proper HTTP codes: 404 for unknown video_id, 409 if already processing, 422 for bad input.

---

## 5. Frontend — Streamlit

### Layout (`frontend/app.py`)

```
Page: Beer Tap Counter
│
├── Sidebar
│   ├── Upload Section
│   │   ├── st.file_uploader (accept mp4, mov)
│   │   └── Upload button → POST /api/videos/upload
│   └── History Section
│       └── st.selectbox listing past videos → GET /api/videos/
│
├── Main Area
│   ├── Status Banner
│   │   ├── If pending:    "Ready to process" + [Process] button
│   │   ├── If processing: spinner + auto-refresh (st.rerun with sleep)
│   │   ├── If completed:  success banner
│   │   └── If error:      error message display
│   │
│   ├── Results Section (only when completed)
│   │   ├── Three columns:
│   │   │   ├── st.metric("Tap A", count_a)
│   │   │   ├── st.metric("Tap B", count_b)
│   │   │   └── st.metric("Total", total)
│   │   └── (Optional) Timeline table of individual events
│   │
│   └── (Optional) Video preview with st.video()
│
└── Footer
    └── Query section: filter by date range, tap
```

### API Client (`frontend/utils/api_client.py`)

A thin wrapper using `requests` or `httpx`:
- `upload_video(file) → VideoUploadResponse`
- `process_video(video_id) → status_code`
- `get_status(video_id) → VideoStatus`
- `list_videos() → List[VideoSummary]`
- `get_counts(filters) → List[CountResult]`

### Polling Strategy for Processing Status

```
When user clicks "Process":
    1. POST /api/videos/{id}/process
    2. Enter polling loop:
        - GET /api/videos/{id}/status every 2 seconds
        - Show spinner + progress message
        - Break when status is 'completed' or 'error'
    3. Display results or error
```

### UX Notes

- Disable the Process button while status is 'processing'.
- Show video metadata (filename, upload date, duration) in the history list.
- Use `st.toast()` for success/error notifications.
- Keep the UI minimal: no auth, no multi-user, no pagination needed.

---

## 6. ML Pipeline — Implemented Architecture

Processing always runs the full YOLO+SAM3 pipeline. SimpleDetector is available as a CLI tool and planned as a future pre-filter for long videos (see Optimization section below).

### Approach A: SimpleDetector (CPU, fast) — CLI only

`backend/ml/approach_simple/detector.py` — pixel differencing on two small tap-handle ROIs.

- Two normalised ROIs per tap handle → frame differencing → activity signal per tap
- ON/OFF thresholding → event detection (each OFF→ON→OFF cycle = 1 pour event)
- Multiprocessing on chunks for long videos (> 3000 frames)
- Runs in seconds on CPU, no GPU needed
- Outputs: per-tap time series, tap heatmaps, event list
- **Not used by the web app** — planned as pre-filter for YOLO (see notes.txt)

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

Output: `raw_detections.csv` with per-frame bounding boxes and track IDs.

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

relink:
  overlap_threshold: 15
  min_pour_frames: 30
  movement_threshold: 5.0
  stationary_ratio: 0.8   # skip cups stationary for >80% of their lifespan
  stationary_px: 10.0     # px radius for "same spot"

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
| `backend/ml/approach_yolo/detector.py` | YOLODetector — GPU wrapper + tap assignment |
| `backend/ml/approach_yolo/pipeline.py` | 4-stage orchestrator (ROI → YOLO → Relink → SAM3) |
| `backend/ml/approach_yolo/yolo_track.py` | YOLO-World + BoT-SORT tracking |
| `backend/ml/approach_yolo/relink.py` | Track relinking + pour classification |
| `backend/ml/approach_yolo/sam3_tracking.py` | SAM3 tap handle segmentation |
| `scripts/run_simple.py` | CLI entry for SimpleDetector |
| `scripts/run_yolo_pipeline.py` | CLI entry for YOLO pipeline |
| `config/pipeline.yaml` | Default YOLO pipeline config |
| `config/botsort.yaml` | BoT-SORT tracker config |

---

## 7. Testing

### Test Suite (34 tests)

Run with: `source .venv-gambooza/bin/activate && python -m pytest tests/ -v`

| File | Tests | What it covers |
|------|-------|----------------|
| `test_app.py` | 2 | Health check, /docs available |
| `test_database.py` | 6 | Table existence, column names, insert, cascade delete |
| `test_schemas.py` | 5 | Pydantic model instantiation and serialization |
| `test_videos_router.py` | 9 | Upload, reject bad extension, list, status, 404s, process 202, delete |
| `test_counts_router.py` | 4 | Counts query, filter by tap, empty results, summary |
| `test_processor.py` | 8 | ROI config loading/validation, YOLO config builder (mocked), event mapping |

### Testing patterns

- In-memory SQLite with `StaticPool` for router tests (avoids disk, fast, isolated per test)
- `dependency_overrides` on `get_db` to inject test DB session
- Mock `_import_yolo_detector` to test config building without GPU/ML deps
- Each test fixture creates a fresh DB

---

## 8. Development Environment

- **Python:** 3.11 via `.venv-gambooza` (managed with `uv`)
- **Install deps:** `source .venv-gambooza/bin/activate && uv pip install -r requirements.txt`
- **Run server:** `uvicorn backend.main:app --port 8000 --reload`
- **Run tests:** `python -m pytest tests/ -v`
- **DB location:** `data/db_files/app.db` (delete to reset schema)

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
2. ✅ FastAPI endpoints — videos router (upload, list, status, delete, process).
3. ✅ Counts router (query with filters, summary).
4. ✅ Background processor service — bridges YOLO pipeline to DB.
5. ✅ ROI config system — named JSON configs for different camera setups.
6. ✅ 34 tests covering all layers.
7. ✅ End-to-end tested: upload → process → completed with tap events.
8. 🔲 Build Streamlit UI (`frontend/`).
9. 🔲 Create Dockerfiles and docker-compose.yml.

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

1. **Simple vs YOLO:** Simple is faster and requires no GPU but is fragile to environmental changes. YOLO is robust but heavier. The planned hybrid (Simple as pre-filter → YOLO on active windows only) is the best of both worlds but requires solving tracker continuity across time gaps.

2. **Frame sampling rate:** Processing every frame is accurate but slow (a 2h video at 30fps = 216,000 frames). Sampling every 3rd frame (10 effective fps) cuts processing by 3x with negligible accuracy loss for events lasting several seconds.

3. **SQLite vs Postgres:** SQLite is single-writer, which is fine for single-user demo. Postgres would be needed for concurrent users. SQLite keeps Docker setup trivial.

4. **Streamlit vs React:** Streamlit is less customizable but delivers the required UI in ~100 lines. React would require separate build tooling, more code, and a Node.js container.

5. **Background processing:** FastAPI BackgroundTasks is simple but limited (no retry, no queue). For production, you'd use Celery + Redis. For this demo, BackgroundTasks is sufficient.

6. **ROI configs:** ROIs are stored as named JSON configs per camera setup. A production system would need a calibration UI or auto-detection. For the demo, `default.json` with cerveza camera values works for all provided videos.

7. **Tap assignment in detector vs pipeline:** The tap assignment logic (`_assign_pours_to_taps`) was originally only in `pipeline.py:main()` (CLI path). It's now also called in `YOLODetector.run()` so the web flow gets tap-assigned results.

---

## 11. Validation Checklist (Pre-Demo)

Before the presentation, verify:

- [x] Backend starts without errors: `uvicorn backend.main:app --port 8000`
- [x] All 34 tests pass: `python -m pytest tests/ -v`
- [x] Upload a video: `POST /api/videos/upload`
- [x] Processing starts: `POST /api/videos/{id}/process` returns 202
- [x] Processing completes: status updates to 'completed'
- [x] Pipeline outputs saved to `results/web_{id}/`
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

### Backend (`requirements.txt`)

```
opencv-python
numpy
matplotlib
ipython
ipykernel
pandas
ultralytics
fastapi
uvicorn[standard]
sqlalchemy
aiofiles
python-multipart
pyyaml
```

### Test dependencies

```
pytest
```

### Environment

Managed with `uv` in `.venv-gambooza`:
```bash
source .venv-gambooza/bin/activate
uv pip install -r requirements.txt
```
