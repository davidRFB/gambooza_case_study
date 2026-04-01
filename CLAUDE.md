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
│   ├── main.py                     # (TODO) FastAPI app entry point
│   ├── config.py                   # (TODO) Settings, paths, constants
│   ├── database/                   # (TODO) SQLAlchemy models, schemas
│   ├── routers/                    # (TODO) API endpoints
│   ├── services/                   # (TODO) Background processor
│   └── ml/
│       ├── __init__.py
│       ├── common.py               # Shared utilities (ROI, cropping, interactive selectors)
│       ├── approach_simple/
│       │   ├── __init__.py
│       │   └── detector.py         # SimpleDetector — CPU pixel differencing
│       └── approach_yolo/
│           ├── __init__.py
│           ├── detector.py         # YOLODetector — GPU wrapper
│           ├── pipeline.py         # 4-stage orchestrator (ROI → YOLO → Relink → SAM3)
│           ├── yolo_track.py       # YOLO-World + BoT-SORT tracking
│           ├── relink.py           # Track relinking + pour classification
│           └── sam3_tracking.py    # SAM3 tap handle segmentation
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
│   ├── models/                     # ML weights (yolov8x-worldv2.pt, sam3.pt, etc.)
│   └── db/                         # (TODO) SQLite database
│
├── results/                        # Pipeline outputs (per-video directories)
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
      - video_data:/app/data/videos
      - db_data:/app/data/db
    environment:
      - ML_APPROACH=simple  # Toggle: "simple" | "yolo"
      - DATABASE_URL=sqlite:///app/data/db/app.db
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
  video_data:
  db_data:
```

### Key Docker Decisions

- **GPU passthrough:** Use `nvidia-container-toolkit` so the backend container can access the host GPU. Required for both YOLO inference and faster OpenCV processing.
- **Shared volumes:** Videos are saved by the backend; both services read the SQLite DB (though only the backend writes).
- **Toggle ML approach:** The `ML_APPROACH` env var selects which detector class gets instantiated. Both share the same interface (see Section 6).

### Backend Dockerfile Skeleton

```dockerfile
FROM python:3.11-slim
# Install system deps for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 ffmpeg
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
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
├── id              INTEGER PK AUTOINCREMENT
├── filename        TEXT NOT NULL
├── original_name   TEXT NOT NULL
├── upload_date     DATETIME DEFAULT now
├── status          TEXT DEFAULT 'pending'   # pending | processing | completed | error
├── duration_sec    FLOAT NULLABLE
├── error_message   TEXT NULLABLE
└── ml_approach     TEXT NULLABLE            # Which detector was used

Table: tap_events
├── id              INTEGER PK AUTOINCREMENT
├── video_id        INTEGER FK → videos.id
├── tap             TEXT NOT NULL             # 'A' | 'B'
├── frame_start     INTEGER NOT NULL
├── frame_end       INTEGER NOT NULL
├── timestamp_start FLOAT NOT NULL           # Seconds into video
├── timestamp_end   FLOAT NOT NULL
└── confidence      FLOAT NULLABLE           # Only for YOLO approach

View/Query: counts_summary (derived, not a physical table)
├── video_id
├── tap_a_count     COUNT WHERE tap = 'A'
├── tap_b_count     COUNT WHERE tap = 'B'
└── total           tap_a_count + tap_b_count
```

### Pydantic Schemas (`backend/database/schemas.py`)

Define request/response models:
- `VideoUploadResponse(id, filename, status)`
- `VideoStatus(id, status, tap_a_count, tap_b_count, total, error_message)`
- `TapEvent(tap, frame_start, frame_end, timestamp_start, timestamp_end)`
- `CountQuery(video_id?, date_from?, date_to?, tap?)`
- `CountResult(video_id, filename, upload_date, tap_a, tap_b, total)`

### Migration Strategy

Keep it simple: a single `init_db.py` script that calls `Base.metadata.create_all(engine)` on startup. No Alembic needed at intern level.

---

## 4. Backend — FastAPI

### Entry Point (`backend/main.py`)

```python
# Pseudocode structure
app = FastAPI(title="Beer Tap Counter API")

@app.on_event("startup")
def startup():
    init_database()
    load_ml_detector(approach=os.getenv("ML_APPROACH", "simple"))

app.include_router(videos_router, prefix="/api/videos")
app.include_router(counts_router, prefix="/api/counts")
```

### Router: Videos (`backend/routers/videos.py`)

| Method | Endpoint              | Purpose                        |
|--------|-----------------------|--------------------------------|
| POST   | `/api/videos/upload`  | Accept mp4/mov, save to disk, create DB row with status='pending' |
| POST   | `/api/videos/{id}/process` | Launch background processing task |
| GET    | `/api/videos/{id}/status`  | Return current status + counts if completed |
| GET    | `/api/videos/`        | List all uploaded videos with their status |
| DELETE | `/api/videos/{id}`    | Remove video file + DB records  |

#### Upload Flow

1. Receive `UploadFile` from request.
2. Generate unique filename: `{uuid}_{original_name}`.
3. Save to `/app/data/videos/`.
4. Insert row in `videos` table with `status='pending'`.
5. Return `VideoUploadResponse`.

#### Process Flow

1. Set `status='processing'` in DB.
2. Launch `BackgroundTasks.add_task(process_video, video_id)`.
3. Return `202 Accepted` immediately.
4. Background task runs ML detector, writes `tap_events`, updates `status='completed'` or `status='error'`.

### Router: Counts (`backend/routers/counts.py`)

| Method | Endpoint              | Purpose                        |
|--------|-----------------------|--------------------------------|
| GET    | `/api/counts/`        | Query counts with filters: video_id, date range, tap |
| GET    | `/api/counts/summary` | Aggregate totals across all videos |

### Background Processing (`backend/services/processor.py`)

```
function process_video(video_id):
    1. Load video path from DB
    2. Instantiate detector (simple or yolo based on config)
    3. Call detector.count(video_path) → List[TapEvent]
    4. Bulk insert tap_events into DB
    5. Update video status to 'completed'
    6. On exception: update status to 'error', save error_message
```

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

Two ML approaches share the same output interface (event list + counts). The backend selects which to run based on config (`ML_APPROACH=simple|yolo`).

### Approach A: SimpleDetector (CPU, fast)

`backend/ml/approach_simple/detector.py` — pixel differencing on two small tap-handle ROIs.

- Two normalised ROIs per tap handle → frame differencing → activity signal per tap
- ON/OFF thresholding → event detection (each OFF→ON→OFF cycle = 1 pour event)
- Multiprocessing on chunks for long videos (> 3000 frames)
- Runs in seconds on CPU, no GPU needed
- Outputs: per-tap time series, tap heatmaps, event list

```bash
# Interactive ROI selection:
python scripts/run_simple.py --video data/videos/cerveza2.mp4 --interactive

# From saved ROIs:
python scripts/run_simple.py --video data/videos/cerveza2.mp4 \
    --roi-json results/simple_cerveza2/simple_roi.json
```

### Approach B: YOLO + SAM3 Pipeline (GPU, accurate)

`backend/ml/approach_yolo/` — 4-stage system driven by a YAML config file. Each stage produces output files that the next stage consumes. Stages are skippable and idempotent — existing outputs are reused unless `--force` is set.

### Pipeline Stages

```
Stage 1: ROI Selection      → tap_roi.json (crop region + SAM3 tap bboxes)
Stage 2: YOLO Tracking      → raw_detections.csv (per-frame cup/person tracks)
Stage 3: Relink              → relinked_detections.csv + pour_events.json
Stage 4: SAM3 Tap Tracking   → sam3_centroids.csv (tap handle centroid trajectories)
Final:   Tap Assignment      → pour_events_assigned.json + summary.json
```

### Stage 1: ROI Selection

Interactive or config-driven. User selects:
1. **Crop region** — normalized (0–1) bounding box on the full frame
2. **TAP_A / TAP_B bboxes** — pixel-space bounding boxes on the cropped frame (used to initialize SAM3)

No A|B divider line — tap side is determined later by SAM3 centroid Y movement.
Saved to `tap_roi.json` in the output directory.

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

### Tap Assignment (Final)

Correlates pour events with SAM3 centroid data:
- For each pour's frame range, compute **centroid-Y standard deviation** for each tap
- The tap with **more Y movement** during the pour is the one that poured
- If no meaningful movement difference → `UNKNOWN`

Output: `pour_events_assigned.json` (enriched pour events with `tap` field) and `summary.json` with final counts.

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
| `backend/ml/approach_yolo/detector.py` | YOLODetector — GPU wrapper |
| `backend/ml/approach_yolo/pipeline.py` | 4-stage orchestrator (ROI → YOLO → Relink → SAM3) |
| `backend/ml/approach_yolo/yolo_track.py` | YOLO-World + BoT-SORT tracking |
| `backend/ml/approach_yolo/relink.py` | Track relinking + pour classification |
| `backend/ml/approach_yolo/sam3_tracking.py` | SAM3 tap handle segmentation |
| `scripts/run_simple.py` | CLI entry for SimpleDetector |
| `scripts/run_yolo_pipeline.py` | CLI entry for YOLO pipeline |
| `config/pipeline.yaml` | Default YOLO pipeline config |
| `config/botsort.yaml` | BoT-SORT tracker config |

---

## 9. Development Workflow

### Phase 1: ML Pipeline (DONE)

Notebook-based exploration → two production detectors:
1. Explored videos, calibrated ROIs interactively (`notebooks/01_explore*.py`).
2. Built YOLO-World + BoT-SORT + Relink + SAM3 pipeline (now in `backend/ml/approach_yolo/`).
3. Built SimpleDetector — CPU pixel differencing (now in `backend/ml/approach_simple/`).
4. Validated YOLO pipeline across multiple videos (pipeline1–7 configs).
5. Organized: production code in `backend/ml/`, exploration in `notebooks/`, CLIs in `scripts/`.

### Phase 2: Application Skeleton (NEXT)

1. Implement database models and connection (`backend/database/`).
2. Build FastAPI endpoints — integrate both detectors as backend processor.
3. Build Streamlit UI (`frontend/`).
4. Create Dockerfiles and docker-compose.yml.
5. **Goal:** End-to-end flow works: upload → processing → display counts.

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

1. **Simple vs YOLO:** Simple is faster and requires no GPU but is fragile to environmental changes. YOLO is robust but heavier. The hybrid is best but most complex.

2. **Frame sampling rate:** Processing every frame is accurate but slow (a 2h video at 30fps = 216,000 frames). Sampling every 3rd frame (10 effective fps) cuts processing by 3x with negligible accuracy loss for events lasting several seconds.

3. **SQLite vs Postgres:** SQLite is single-writer, which is fine for single-user demo. Postgres would be needed for concurrent users. SQLite keeps Docker setup trivial.

4. **Streamlit vs React:** Streamlit is less customizable but delivers the required UI in ~100 lines. React would require separate build tooling, more code, and a Node.js container.

5. **Background processing:** FastAPI BackgroundTasks is simple but limited (no retry, no queue). For production, you'd use Celery + Redis. For this demo, BackgroundTasks is sufficient.

6. **ROI hardcoding:** The ROIs are manually defined per camera setup. A production system would need a calibration UI or auto-detection. For the demo, hardcoded values from the provided videos are acceptable.

---

## 11. Validation Checklist (Pre-Demo)

Before the presentation, verify:

- [ ] `docker-compose up` builds and starts both services without errors.
- [ ] Frontend is accessible at `localhost:8501`.
- [ ] Upload a new video (not used during development if possible).
- [ ] Processing starts, status updates to 'processing', spinner shows.
- [ ] Processing completes, status updates to 'completed'.
- [ ] Tap A count, Tap B count, and Total are displayed correctly.
- [ ] Counts match manual count from watching the video.
- [ ] Uploading a second video works and shows separate results.
- [ ] Query by date/tap works.
- [ ] Error case: upload a non-video file → proper error message.
- [ ] Processing a 2h video completes in reasonable time (< 5 min with GPU).

---

## 12. Dependencies Summary

### Backend (`requirements.txt`)

