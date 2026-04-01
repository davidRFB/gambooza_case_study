# CLAUDE.md — Beer Tap Counter: Implementation Guide

## Project Overview

A self-contained application that counts beers served from a dual-tap beer dispenser by analyzing uploaded video footage. The system identifies events on Tap A and Tap B independently, persists results in SQLite, and displays counts through a minimal web UI.

**Architecture:** Docker Compose → Streamlit (frontend) + FastAPI (backend + ML pipeline) + SQLite (persistence)

---

## 1. Project Structure

```
beer-tap-counter/
├── docker-compose.yml
├── .env
├── README.md
├── CLAUDE.md
│
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                  # Streamlit UI
│   └── utils/
│       └── api_client.py       # Helper to call FastAPI endpoints
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Settings, paths, constants
│   ├── database/
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   ├── schemas.py          # Pydantic schemas
│   │   ├── connection.py       # Engine + session factory
│   │   └── migrations/         # Alembic or manual SQL scripts
│   ├── routers/
│   │   ├── videos.py           # Upload, list, delete endpoints
│   │   └── counts.py           # Query counts by video/date/tap
│   ├── services/
│   │   ├── processor.py        # Orchestrates the ML pipeline
│   │   └── background.py       # Background task runner
│   └── ml/
│       ├── __init__.py
│       ├── approach_simple/
│       │   ├── detector.py     # ROI + frame differencing logic
│       │   └── config.py       # Thresholds, ROI coordinates
│       └── approach_yolo/
│           ├── detector.py     # YOLO + tracking logic
│           ├── config.py       # Model paths, confidence thresholds
│           └── models/         # YOLO weights (.pt files)
│
├── data/
│   ├── videos/                 # Uploaded videos land here (Docker volume)
│   └── db/
│       └── app.db              # SQLite database file (Docker volume)
│
└── notebooks/                  # Dev/exploration (not deployed)
    ├── 01_explore_videos.ipynb
    ├── 02_roi_calibration.ipynb
    └── 03_yolo_experiments.ipynb
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

## 6. ML Pipeline — Shared Interface

Both approaches implement the same abstract interface so the backend can swap them via config.

### Base Detector Interface

```python
class BaseDetector(ABC):
    @abstractmethod
    def count(self, video_path: str) -> List[TapEvent]:
        """
        Analyze video and return list of detected tap events.
        Each TapEvent has: tap ('A'|'B'), frame_start, frame_end,
        timestamp_start, timestamp_end, confidence (optional).
        """
        pass
```

### Shared Preprocessing Steps (both approaches use these)

1. **Open video** with OpenCV `cv2.VideoCapture`.
2. **Extract metadata:** FPS, total frames, duration.
3. **Frame sampling:** Process every Nth frame (e.g., every 3rd for 30fps video → 10 effective fps). This saves compute without losing events that last multiple seconds.
4. **ROI definition:** Both approaches need to know where Tap A and Tap B are in the frame. Store as normalized coordinates (0-1 range) so they work across resolutions.

### ROI Calibration (one-time manual step)

Use `notebooks/02_roi_calibration.ipynb`:
1. Load first frame of a reference video.
2. Draw rectangles around Tap A zone and Tap B zone.
3. Save normalized coordinates to a config file.
4. These ROIs define "where to look" for both approaches.

The ROI should cover the area under each tap where the pour action occurs (tap handle + glass zone beneath it).

---


## 8. ML Approach B — Advanced: YOLO + Tracking + SAM

**Best for:** Higher accuracy, handles more visual variation, provides bounding boxes for UI.

### Core Idea

Use YOLOv8 (Ultralytics) to detect objects relevant to a pour event (glasses, hands, tap handles) in each frame. Then use tracking and spatial rules to determine when a pour starts and ends at each tap.

### Object Detection Setup

#### Option B1: Zero-Shot with Pretrained YOLO

Use `yolov8n.pt` (nano, fast) or `yolov8s.pt` (small, more accurate). COCO-pretrained YOLO detects `cup`, `bottle`, `person`. It does NOT detect tap handles or beer flow directly. here we used indead YOLO WORLD models to detec the cups and the person. only. In this way we avoid noise

Strategy:
- Detect `cup` class within each tap's ROI.
- A pour event = a `cup` object appears in the ROI, stays for several seconds, then leaves.
- Combine with motion detection from Approach A for the "flow" signal.


### Tracking Logic

```
For each sampled frame:
    1. Run YOLO inference → list of detections (class, bbox, confidence)
    2. Filter detections inside ROI_A and ROI_B
    3. For each ROI, track the "cup" object across frames:
        - Use simple IoU-based tracker (no need for DeepSORT)
        - Or use Ultralytics built-in tracker: model.track()

State machine per ROI:
    States: IDLE → CUP_DETECTED → POURING → COMPLETE

    IDLE:
        - If cup detected in ROI for > N consecutive frames → CUP_DETECTED

    CUP_DETECTED:
        - If motion activity in ROI increases (reuse Approach A's signal) → POURING
        - If cup leaves ROI → back to IDLE (false alarm)

    POURING:
        - If motion activity drops AND cup still present → COMPLETE
        - If cup leaves → COMPLETE

    COMPLETE:
        - Record event (tap, frame_start, frame_end, timestamps)
        - Emit confidence = average YOLO confidence across tracked frames
        - Transition to IDLE after cooldown period
```


### Hybrid Strategy (Recommended)

The strongest approach for this case combines both:
- **YOLO** confirms a glass is present at the tap (reduces false positives from hand movement).
- **Frame differencing** from Approach A confirms liquid is flowing (reduces false positives from stationary glasses).
- **Both signals must agree** for an event to be counted.

This hybrid catches cases that either approach alone would miss or miscount.

### Pros and Cons

| Pros | Cons |
|------|------|
| Object-level understanding | Requires GPU for reasonable speed |
| Provides bounding boxes (useful for debugging UI) | Pretrained YOLO may not detect beer-specific objects well |
| Confidence scores per detection | More complex pipeline and more parameters to tune |
| Fine-tuning option for high accuracy | Fine-tuning requires manual labeling effort |

---

## 9. Development Workflow


### Phase 1: Simple ML (Day 1-2, ~4-6 hours)

1. Open provided videos in a notebook, inspect frames, mark ROIs.
2. Implement Approach A (frame differencing) in the notebook first.
3. Tune thresholds against the provided videos.
4. Wrap the logic into `SimpleDetector` class implementing `BaseDetector`.
5. Integrate into the backend processor service.
6. **Goal:** Upload video → real ML processing → correct counts displayed.

### Phase 3: Skeleton (Day 1, ~4-6 hours)

1. Set up the project structure as defined in Section 1.
2. Create Dockerfiles and docker-compose.yml.
3. Implement database models and connection.
4. Build FastAPI upload + status endpoints (without real ML, just set status='completed' with dummy counts).
5. Build Streamlit UI that talks to the backend.
6. **Goal:** End-to-end flow works: upload → dummy process → display counts.
### Phase 4: Polish (Day 3, ~3-4 hours)

1. Test with all provided videos, fix edge cases.
2. Add error handling and input validation.
3. Write README with setup instructions.
4. Clean up code, add type hints and docstrings.
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

