# Beer Tap Counter

An application that counts beers served from a dual-tap beer dispenser by analyzing uploaded video footage. The system watches video recordings of a bar's tap area, identifies individual pour events, and assigns each one to **Tap A** or **Tap B**. Results are persisted in a database and displayed through a web interface.

The goal is to provide bar owners with accurate, automated beer counts from their existing security or monitoring cameras, without requiring any hardware modifications to the taps themselves.

Built as a case study for **Intern Full Stack & AI Developer** at Gambooza.

---

## Architecture

```
Streamlit (frontend)  ──HTTP──>  FastAPI (backend + ML pipeline)  ──>  SQLite
     :8501                              :8000                        app.db
```

- **Backend:** FastAPI REST API. Accepts video uploads, runs ML processing in the background, and exposes beer counts and pour events through REST endpoints.
- **Frontend:** Streamlit app. Provides a UI to upload videos, configure ROI regions (where the taps are in the frame), view per-tap counts, and browse a timeline of pour events.
- **ML Pipeline:** A multi-stage computer vision pipeline combining YOLO-World (zero-shot object detection), BoT-SORT (multi-object tracking), track relinking, and SAM3 (tap handle segmentation). An optional fast pre-filter based on pixel differencing handles long videos efficiently.
- **Database:** SQLite with SQLAlchemy ORM. Stores video metadata and individual pour events.

---

## Environment Setup

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** package manager (handles virtual environments and dependency resolution)
- **NVIDIA GPU + CUDA** (required for YOLO and SAM3 inference)
- **ML model weights** placed in `data/models/`:
  - `yolov8x-worldv2.pt` (YOLO-World model)
  - `sam3.pt` (Segment Anything Model 3)

### Install Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd gambooza_case_study

# Install all dependencies (creates .venv automatically)
uv sync
```

This installs all production and development dependencies defined in `pyproject.toml`, including PyTorch, Ultralytics, FastAPI, Streamlit, and testing tools.

---

## Running Locally

You need two terminals, one for the backend and one for the frontend.

**Terminal 1 -- Start the backend:**

```bash
uv run uvicorn backend.main:app --port 8000 --reload
```

The backend starts at `http://localhost:8000`. On first run it creates the SQLite database at `data/db_files/app.db`.

**Terminal 2 -- Start the frontend:**

```bash
cd frontend
uv run streamlit run app.py
```

The frontend starts at `http://localhost:8501`.

**Verify everything works:**

1. Open `http://localhost:8501` in your browser.
2. Select a restaurant name and camera ID (or create new ones).
3. Upload a video (.mp4 or .mov).
4. If no ROI config exists for that restaurant+camera combo, the ROI wizard will guide you through selecting the tap regions.
5. Processing starts automatically. Refresh to check progress.

### Running Tests

```bash
uv run pytest tests/ -v    # 59 tests
```

### Linting and Formatting

```bash
uv run ruff check .        # lint (add --fix to auto-fix)
uv run ruff format .       # format
```

Pre-commit hooks are available to run these automatically on every commit:

```bash
uv run pre-commit install
```

---

## Running with Docker

Docker Compose runs both the backend and frontend in containers, with GPU passthrough for ML inference.

### Prerequisites

- **Docker** and **Docker Compose**
- **nvidia-container-toolkit** installed and configured (for GPU access inside containers)
- An NVIDIA GPU with CUDA support

### Step-by-step

```bash
# 1. Create the required data directories on the host
mkdir -p data/{db_files,models,roi_configs,results}

# 2. Place model weights in data/models/
#    - yolov8x-worldv2.pt
#    - sam3.pt

# 3. Build and start both services
docker compose up --build
```

- Frontend: `http://localhost:8501`
- Backend API: `http://localhost:8000`

All data persists in the `./data/` directory on the host, which is bind-mounted into the backend container at `/app/mount_data`. This means your database, uploaded videos, ROI configs, and pipeline outputs survive container restarts.

The backend uses a multi-stage Docker build: dependencies are compiled in a builder stage with `uv`, then copied to a CUDA runtime image. The runtime image includes `gcc/g++` because SAM3 uses PyTorch Triton, which JIT-compiles CUDA kernels at runtime.

---

## ML Pipeline

The pipeline has two main approaches that work together. For short videos, the YOLO+SAM3 pipeline processes the entire video directly. For longer videos (over ~80 seconds), a fast pixel-based pre-filter first identifies the moments where something is actually happening, and then the YOLO+SAM3 pipeline processes only those segments.

### Phase 1: SimpleDetector Pre-filter (CPU, fast)

For long videos (e.g., a 2-hour security camera recording), running the full GPU pipeline on every single frame would take an impractical amount of time. The SimpleDetector solves this by quickly scanning the entire video on CPU to find the moments when the taps are actually being used.

**How it works:**

The detector monitors two small regions of interest (ROIs), one around each tap handle. For every frame, it computes pixel differences against a reference background. When pixels in a tap handle region change significantly, it means someone is pulling the handle, a cup is moving under it, or liquid is flowing. This produces a simple activity signal per tap: a number between 0 and 1 indicating how much motion is happening in that region.

**Multiprocessing for speed:** For long videos, the detector splits the video into chunks and processes them in parallel using Python's multiprocessing. This allows it to scan a 2-hour video in a matter of seconds on CPU.

**Accuracy trade-off:** The SimpleDetector is not accurate enough to count pours on its own. Background changes, lighting shifts, or people walking by can trigger false activity signals. Its purpose is purely as a pre-filter: it identifies *when* something interesting is happening so the accurate (but slower) YOLO pipeline only needs to process those segments.

**Example:** A 2-hour video might yield 8 activity windows totaling 12 minutes of actual tap usage. Instead of running YOLO on 216,000 frames, the pipeline processes only ~21,600 frames -- a 10x reduction in GPU processing time.

The activity windows are extracted as short video clips, and each clip is then processed independently by the YOLO+SAM3 pipeline. Timestamps are mapped back to the original video so all events reference the correct time.

### Understanding the ROI Configuration

Before any processing can happen, the system needs to know *where* in the video frame the taps are located. Each camera angle is different, so this must be configured per restaurant and camera.

The ROI config defines two types of regions:

1. **YOLO crop region** (`tap_roi`): A rectangular area of the frame that contains both taps, the cups, and the people being served. The YOLO model processes only this cropped region, which reduces noise from the rest of the scene (other customers, TVs, decorations) and speeds up inference.

2. **SAM3 tap handle bounding boxes** (`sam3_tap_bboxes`): Two small bounding boxes, one for each tap handle, within the cropped region. These initialize the SAM3 segmentation model so it knows which objects to track.

For the SimpleDetector pre-filter, there are two additional ROIs (`roi_1`, `roi_2`) that define small regions around each tap handle for pixel differencing. These are simpler and only need to cover the handle and the area where liquid flows.

ROI configs are stored as JSON files in `data/roi_configs/` and can be created through the frontend's visual wizard or the CLI interactive tools.

### Phase 2: YOLO Tracking (GPU-intensive)

This is the core detection stage. It uses **YOLO-World** (`yolov8x-worldv2.pt`), a zero-shot open-vocabulary object detection model. The key advantage of YOLO-World is that we can specify exactly which object classes to detect -- in our case, **"person"** and **"cup"** -- without needing to train a custom model. A standard YOLO model trained on COCO would detect dozens of irrelevant classes (bottles, chairs, TVs, etc.) that add noise and slow down processing. By focusing on just two classes, we get cleaner detections.

**What it detects:** The model looks for cups and people in the cropped tap region. The primary signal for a pour event is the simultaneous presence of a person and a cup near the tap -- someone is holding a cup under the tap to fill it.

**Tracking with BoT-SORT:** Raw detections are frame-by-frame, but we need to follow individual cups across multiple frames to understand pour events. BoT-SORT assigns persistent IDs to detected objects, creating *tracks* that follow each cup from when it appears to when it leaves the frame.

**Output:** A CSV file (`raw_detections.csv`) with per-frame bounding boxes, class labels, confidence scores, and track IDs for every detected object.

**Optimization options:** The `sample_every` parameter controls how many frames are analyzed. Setting `sample_every: 1` processes every frame (most accurate). Setting `sample_every: 3` processes every third frame, cutting GPU time by roughly 3x with minimal accuracy loss for most videos. A `record_range` parameter can also limit processing to a specific time window within the video.

YOLO tracking is the most GPU-intensive stage of the pipeline. Processing time depends heavily on video resolution, length, and the number of objects in the scene.

### Phase 3: Track Relinking and Pour Classification

After YOLO tracking, the raw tracks often have problems. The tracker may lose an object for a few frames and then pick it up again with a new ID, creating two separate tracks for what was actually one continuous cup movement. Or a cup sitting on the counter waiting to be picked up might be tracked for a long time without actually being part of a pour.

The relink stage addresses both issues:

**Merging fragmented tracks:** When two tracks have similar spatial positions and don't overlap much in time, they are likely the same object. The algorithm compares tracks based on their bounding box positions and temporal gaps. If a track ends and another begins nearby (within configurable pixel and frame thresholds), they are merged into a single track with one ID. This prevents double-counting a single pour that got split into multiple track fragments.

**Pour classification:** Not every tracked cup is part of a pour event. The relink stage filters tracks using three criteria:
- **Duration:** The track must span enough frames (`min_pour_frames: 30`) to represent a real pour, not just a momentary detection.
- **Movement:** The cup must show enough spatial movement (`movement_threshold`) -- a cup being carried to and from the tap moves across the frame.
- **Not stationary:** Cups that sit in one position for most of their tracked life (e.g., a glass left on the counter) are filtered out. If a cup stays within a small pixel radius of its median position for more than 80% of its lifespan, it is classified as stationary and excluded.

**Resolution-aware scaling:** These pixel-based thresholds were calibrated on ~800px-wide crops from 4K video. When processing lower-resolution video (e.g., 360p), the crop region is much smaller in pixels. The relink stage automatically detects the crop width and scales all thresholds proportionally, so the same config works across different video resolutions.

**Room for improvement:** The relinking parameters (overlap threshold, interpolation gap, movement thresholds) offer significant room for fine-tuning per deployment. Different camera angles, distances, and video qualities may benefit from adjusted parameters. This is an area where future work could improve accuracy.

### Phase 4: SAM3 Tap Handle Tracking (GPU, slow)

At this point, we know *when* pour events happened and we have tracked the cups involved, but we don't yet know *which tap* each pour came from. YOLO-World does not detect tap handles well -- they are small metallic objects that don't appear in standard detection vocabularies.

To solve this, we use **SAM3 (Segment Anything Model 3)**, an instance segmentation model that can segment and track arbitrary objects across video frames. SAM3 is initialized with bounding boxes around each tap handle (from the ROI config), and then propagates those segmentations across the video using a memory bank mechanism.

**What it produces:** For each frame during a pour event, SAM3 outputs the centroid (center point) coordinates of each tap handle's segmentation mask. When a tap handle is being pulled, its centroid moves -- particularly in the vertical (Y) direction.

**Tap assignment logic:** For each pour event's frame range, the system computes the standard deviation of the Y-coordinate for each tap handle's centroid. The tap with more vertical movement during that pour is the one being used. If neither tap shows meaningful movement, the pour is marked as `UNKNOWN`.

**Efficiency:** SAM3 only processes the frame ranges where pour events were detected (from Phase 3), not the entire video. It also uses frame skipping (`frame_skip: 5`) and half-precision inference (`half: true`) to reduce GPU time.

**Trade-off:** SAM3 is the slowest stage of the pipeline. Instance segmentation with memory propagation is computationally expensive. However, it provides reliable tap assignment that would be very difficult to achieve with bounding-box detection alone.

---

## Backend

The FastAPI backend manages video uploads, triggers ML processing, and serves results.

### Database Tables

**Videos table:** Stores metadata for each uploaded video -- filename, upload date, processing status (`pending`, `processing`, `completed`, `error`), duration, and timing information. Each video is also associated with a `restaurant_name` and `camera_id`.

**TapEvents table:** Stores individual pour events detected by the pipeline. Each event records which tap (`A` or `B`), the start/end frame numbers, start/end timestamps (in seconds), a confidence score, and a count (number of beers in that event, typically 1). Events are linked to their source video via a foreign key with cascade delete.

Beer counts are computed as `SUM(count)` per tap, not `COUNT(*)` of events.

### Restaurant and Camera ID

Each video is tagged with a restaurant name and camera ID. This serves two purposes:

1. **ROI config lookup:** Different cameras have different angles, resolutions, and positions. The ROI configuration (where the taps are in the frame) is stored per restaurant+camera combination. When processing starts, the system loads the matching ROI config automatically.

2. **Multi-location support:** A bar chain could have multiple locations, each with one or more cameras. The restaurant+camera tagging keeps counts organized and ensures each video is processed with the correct spatial calibration.

### API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/videos/upload` | Upload mp4/mov video |
| POST | `/api/videos/{id}/process` | Start ML processing |
| GET | `/api/videos/{id}/status` | Status + counts + events |
| GET | `/api/videos/` | List all videos |
| DELETE | `/api/videos/{id}` | Delete video + events |
| GET | `/api/counts/` | Query counts (filters: video_id, date, tap) |
| GET | `/api/counts/summary` | Aggregate totals |

---

## Frontend

The Streamlit frontend provides two main views:

### Upload & Process Tab

- Select a restaurant and camera (or create new ones).
- Upload a video file. If an ROI config already exists for that restaurant+camera, processing starts automatically.
- If no ROI config exists, a 3-step visual wizard guides you through selecting the crop region, Tap A handle, and Tap B handle on the first frame of the video.
- While processing, the UI shows the current status. Only one video processes at a time (GPU constraint), so additional uploads are queued as "pending" and auto-trigger when the GPU becomes available.
- When complete, the tab displays Tap A count, Tap B count, total, and a table of individual pour events.

### Dashboard Tab

- Shows global summary metrics: total pours per tap across all videos, grand total, and number of videos processed.
- Lists all videos with expandable details, per-video metrics, and delete buttons.

### Intermediate Outputs for Debugging

All intermediate results from the ML pipeline are saved in `data/results/web_{video_id}/` for each processed video. This includes:

- **Annotated YOLO tracking video** (`yolo_raw_tracking.mp4`): Shows bounding boxes and track IDs overlaid on the cropped video, useful for verifying that detections and tracking are working correctly.
- **Relinked detections and plots**: CSVs and visualizations showing how tracks were merged and which were classified as pours vs. filtered out.
- **SAM3 centroid trajectories** (`sam3_centroids.csv`): Per-frame handle positions, useful for verifying tap assignment logic.
- **Pour events JSON files**: Both pre-assignment (`pour_events.json`) and post-assignment (`pour_events_assigned.json`), showing the full pipeline progression.
- **SimpleDetector outputs** (for long videos): Activity signals, heatmaps, and extracted clips.

These outputs are valuable for debugging false positives/negatives, tuning pipeline parameters, and understanding why the system made specific counting decisions.

---

## Testing

```bash
uv run pytest tests/ -v    # 59 tests
uv run ruff check .        # lint
uv run ruff format .       # format
```

Tests use an in-memory SQLite database and mock the ML dependencies, so they run without a GPU.

## Project Structure

```
backend/
  main.py              # FastAPI entry point
  config.py            # Settings, paths, constants
  database/            # SQLAlchemy models, schemas, connection
  routers/             # videos + counts endpoints
  services/            # background processor (orchestrates ML pipeline)
  ml/
    common.py          # Shared utilities (ROI, cropping, interactive selectors)
    approach_yolo/     # YOLO + SAM3 pipeline (4 stages)
    approach_simple/   # CPU pixel-differencing (pre-filter + CLI)
frontend/
  app.py               # Streamlit UI (upload, ROI wizard, dashboard)
  utils/api_client.py  # Backend HTTP client
config/                # Pipeline YAML configs + BoT-SORT tracker config
data/                  # Videos, models, DB, ROI configs, results
tests/                 # pytest suite (59 tests)
scripts/               # CLI tools for running pipeline stages independently
```
