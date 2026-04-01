"""
SimpleDetector — CPU-friendly tap activity analyser using pixel differencing.

Only two ROIs needed: one per tap handle (normalised 0–1, full frame).
Frame differencing on these small regions produces a per-tap activity signal.
A heatmap is built over both tap regions combined for visual context.

Uses multiprocessing to split long videos into chunks for speed.

Outputs:
  - Per-tap activity time series (numpy arrays)
  - Heatmap of pixel activity over the tap handle regions
  - List of detected ON/OFF transition events with frame ranges + timestamps
"""

import multiprocessing as mp
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TapEvent:
    tap: str  # "A" or "B"
    frame_start: int
    frame_end: int
    timestamp_start: float  # seconds
    timestamp_end: float
    duration_s: float
    peak_activity: float  # max activity fraction during event


@dataclass
class SimpleDetectorResult:
    fps: float
    total_frames: int
    duration_s: float
    events: list[TapEvent]
    # Arrays — not serialised to JSON, consumed by plotting
    times: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    signals: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    heatmaps: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    crop_frames: dict[str, np.ndarray] = field(repr=False, default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_s": round(self.duration_s, 3),
            "events": [asdict(e) for e in self.events],
        }


# ---------------------------------------------------------------------------
# Chunk worker for multiprocessing
# ---------------------------------------------------------------------------


def _process_chunk(args: tuple) -> tuple:
    """Process a frame range, return partial signals + heatmaps.

    Returns (frame_indices, signals_a, signals_b, heatmap_a, heatmap_b,
             hmap_count, crop_a_rgb, crop_b_rgb)
    """
    (
        video_path,
        start_frame,
        end_frame,
        sample_every,
        blur_kernel,
        diff_threshold,
        tap_a_px,
        tap_b_px,
    ) = args

    ax1, ay1, ax2, ay2 = tap_a_px
    bx1, by1, bx2, by2 = tap_b_px
    hmap_a = np.zeros((ay2 - ay1, ax2 - ax1), dtype=np.float64)
    hmap_b = np.zeros((by2 - by1, bx2 - bx1), dtype=np.float64)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    signals_a, signals_b = [], []
    frame_indices = []
    hmap_count = 0
    prev_gray = None
    crop_a_rgb = None
    crop_b_rgb = None
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)

                # Tap A
                tap_diff_a = diff[ay1:ay2, ax1:ax2]
                mask_a = (tap_diff_a > diff_threshold).astype(np.float64)
                hmap_a += mask_a
                signals_a.append(float(np.mean(mask_a)))

                # Tap B
                tap_diff_b = diff[by1:by2, bx1:bx2]
                mask_b = (tap_diff_b > diff_threshold).astype(np.float64)
                hmap_b += mask_b
                signals_b.append(float(np.mean(mask_b)))

                frame_indices.append(frame_idx)
                hmap_count += 1

                if crop_a_rgb is None:
                    crop_a_rgb = cv2.cvtColor(frame[ay1:ay2, ax1:ax2], cv2.COLOR_BGR2RGB)
                    crop_b_rgb = cv2.cvtColor(frame[by1:by2, bx1:bx2], cv2.COLOR_BGR2RGB)

            prev_gray = gray
        frame_idx += 1

    cap.release()
    return (frame_indices, signals_a, signals_b, hmap_a, hmap_b, hmap_count, crop_a_rgb, crop_b_rgb)


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------


class SimpleDetector:
    """CPU-only tap activity analyser via pixel differencing."""

    def __init__(
        self,
        tap_a_roi: tuple[float, float, float, float],
        tap_b_roi: tuple[float, float, float, float],
        sample_every: int = 3,
        blur_kernel: int = 21,
        diff_threshold: int = 25,
        on_threshold: float = 0.05,
        min_on_frames: int = 10,
        cooldown_frames: int = 5,
        n_workers: int = 4,
        progress_every: int = 3000,
    ):
        """
        Parameters
        ----------
        tap_a_roi : normalised coords of tap A handle region
        tap_b_roi : normalised coords of tap B handle region
        sample_every : process every Nth frame
        blur_kernel : Gaussian blur kernel size (must be odd)
        diff_threshold : pixel intensity change threshold (0–255)
        on_threshold : fraction of active pixels to consider tap "ON"
        min_on_frames : minimum consecutive ON frames to count as event
        cooldown_frames : frames after OFF before a new event can start
        n_workers : number of parallel workers for video processing
        progress_every : show progress every N frames (0 to disable)
        """
        self.tap_a_roi = tap_a_roi
        self.tap_b_roi = tap_b_roi
        self.sample_every = sample_every
        self.blur_kernel = blur_kernel
        self.diff_threshold = diff_threshold
        self.on_threshold = on_threshold
        self.min_on_frames = min_on_frames
        self.cooldown_frames = cooldown_frames
        self.n_workers = n_workers
        self.progress_every = progress_every

    # -- public API --------------------------------------------------------

    def run(self, video_path: str | Path) -> SimpleDetectorResult:
        """Analyse video and return activity signals + events."""
        video_path = Path(video_path)

        # Video metadata
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"  Video: {video_path.name}")
        print(
            f"  {vid_w}x{vid_h} | {fps:.1f} fps | {total_frames} frames | "
            f"{duration:.1f}s ({duration / 60:.1f} min)"
        )

        # Pixel coords for each tap
        tap_a_px = self._roi_to_px(self.tap_a_roi, vid_w, vid_h)
        tap_b_px = self._roi_to_px(self.tap_b_roi, vid_w, vid_h)

        # Decide: multiprocessing for long videos, single-thread for short
        use_mp = total_frames > self.progress_every and self.n_workers > 1
        if use_mp:
            result_parts = self._run_multiprocess(str(video_path), total_frames, tap_a_px, tap_b_px)
        else:
            result_parts = self._run_single(str(video_path), total_frames, tap_a_px, tap_b_px)

        (frame_indices, sig_a, sig_b, hmap_a, hmap_b, hmap_count, crop_a_rgb, crop_b_rgb) = (
            result_parts
        )

        # Normalise heatmaps
        hmap_a = hmap_a / max(hmap_count, 1)
        hmap_b = hmap_b / max(hmap_count, 1)

        # Times array
        times = np.array(frame_indices) / fps
        sig_a = np.array(sig_a)
        sig_b = np.array(sig_b)

        # Detect ON/OFF transitions
        events_a = self._detect_events(sig_a, frame_indices, fps, "A")
        events_b = self._detect_events(sig_b, frame_indices, fps, "B")
        all_events = sorted(events_a + events_b, key=lambda e: e.frame_start)

        print(
            f"  Processed {len(frame_indices)} frame pairs | "
            f"Tap A: {len(events_a)} events | Tap B: {len(events_b)} events"
        )

        return SimpleDetectorResult(
            fps=fps,
            total_frames=total_frames,
            duration_s=duration,
            events=all_events,
            times=times,
            signals={"Tap A": sig_a, "Tap B": sig_b},
            heatmaps={"Tap A": hmap_a, "Tap B": hmap_b},
            crop_frames={"Tap A": crop_a_rgb, "Tap B": crop_b_rgb},
        )

    # -- processing modes --------------------------------------------------

    def _run_single(self, video_path, total_frames, tap_a_px, tap_b_px):
        """Single-threaded processing with progress reporting."""
        ax1, ay1, ax2, ay2 = tap_a_px
        bx1, by1, bx2, by2 = tap_b_px
        hmap_a = np.zeros((ay2 - ay1, ax2 - ax1), dtype=np.float64)
        hmap_b = np.zeros((by2 - by1, bx2 - bx1), dtype=np.float64)

        cap = cv2.VideoCapture(video_path)
        signals_a, signals_b = [], []
        frame_indices = []
        hmap_count = 0
        prev_gray = None
        crop_a_rgb = crop_b_rgb = None
        frame_idx = 0
        show_progress = total_frames > self.progress_every and self.progress_every > 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.sample_every == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)

                    tap_diff_a = diff[ay1:ay2, ax1:ax2]
                    mask_a = (tap_diff_a > self.diff_threshold).astype(np.float64)
                    hmap_a += mask_a
                    signals_a.append(float(np.mean(mask_a)))

                    tap_diff_b = diff[by1:by2, bx1:bx2]
                    mask_b = (tap_diff_b > self.diff_threshold).astype(np.float64)
                    hmap_b += mask_b
                    signals_b.append(float(np.mean(mask_b)))

                    frame_indices.append(frame_idx)
                    hmap_count += 1

                    if crop_a_rgb is None:
                        crop_a_rgb = cv2.cvtColor(frame[ay1:ay2, ax1:ax2], cv2.COLOR_BGR2RGB)
                        crop_b_rgb = cv2.cvtColor(frame[by1:by2, bx1:bx2], cv2.COLOR_BGR2RGB)

                prev_gray = gray

            frame_idx += 1
            if show_progress and frame_idx % self.progress_every == 0:
                pct = frame_idx / total_frames * 100
                print(f"\r  Progress: {frame_idx}/{total_frames} ({pct:.0f}%)", end="", flush=True)

        cap.release()
        if show_progress:
            print()  # newline after progress

        return (
            frame_indices,
            signals_a,
            signals_b,
            hmap_a,
            hmap_b,
            hmap_count,
            crop_a_rgb,
            crop_b_rgb,
        )

    def _run_multiprocess(self, video_path, total_frames, tap_a_px, tap_b_px):
        """Split video into chunks and process in parallel."""
        n = self.n_workers
        chunk_size = total_frames // n

        chunks = []
        for i in range(n):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n - 1 else total_frames
            # Overlap by sample_every so each chunk has a valid prev_gray
            overlap_start = max(0, start - self.sample_every) if i > 0 else start
            chunks.append(
                (
                    video_path,
                    overlap_start,
                    end,
                    self.sample_every,
                    self.blur_kernel,
                    self.diff_threshold,
                    tap_a_px,
                    tap_b_px,
                )
            )

        print(f"  Multiprocessing: {n} workers, ~{chunk_size} frames each")

        with mp.Pool(n) as pool:
            results = pool.map(_process_chunk, chunks)

        # Merge
        ax1, ay1, ax2, ay2 = tap_a_px
        bx1, by1, bx2, by2 = tap_b_px
        hmap_a = np.zeros((ay2 - ay1, ax2 - ax1), dtype=np.float64)
        hmap_b = np.zeros((by2 - by1, bx2 - bx1), dtype=np.float64)
        all_indices, all_sig_a, all_sig_b = [], [], []
        total_hmap_count = 0
        first_crop_a = first_crop_b = None

        for indices, sig_a, sig_b, ha, hb, hc, ca, cb in results:
            all_indices.extend(indices)
            all_sig_a.extend(sig_a)
            all_sig_b.extend(sig_b)
            hmap_a += ha
            hmap_b += hb
            total_hmap_count += hc
            if first_crop_a is None and ca is not None:
                first_crop_a = ca
                first_crop_b = cb

        return (
            all_indices,
            all_sig_a,
            all_sig_b,
            hmap_a,
            hmap_b,
            total_hmap_count,
            first_crop_a,
            first_crop_b,
        )

    # -- signal processing -------------------------------------------------

    def _detect_events(
        self,
        signal: np.ndarray,
        frame_indices: list[int],
        fps: float,
        tap: str,
    ) -> list[TapEvent]:
        """Detect ON/OFF transitions in a per-tap activity signal."""
        if len(signal) == 0:
            return []

        on = signal > self.on_threshold
        events = []
        in_event = False
        start_idx = 0
        cooldown = 0

        for i, is_on in enumerate(on):
            if cooldown > 0:
                cooldown -= 1
                continue

            if is_on and not in_event:
                in_event = True
                start_idx = i
            elif not is_on and in_event:
                in_event = False
                length = i - start_idx
                if length >= self.min_on_frames:
                    events.append(
                        self._make_event(tap, start_idx, i - 1, signal, frame_indices, fps)
                    )
                    cooldown = self.cooldown_frames

        # Event extending to end of video
        if in_event:
            length = len(signal) - start_idx
            if length >= self.min_on_frames:
                events.append(
                    self._make_event(tap, start_idx, len(signal) - 1, signal, frame_indices, fps)
                )

        return events

    def _make_event(self, tap, start_idx, end_idx, signal, frame_indices, fps) -> TapEvent:
        f_start = frame_indices[start_idx]
        f_end = frame_indices[end_idx]
        peak = float(np.max(signal[start_idx : end_idx + 1]))
        return TapEvent(
            tap=tap,
            frame_start=f_start,
            frame_end=f_end,
            timestamp_start=round(f_start / fps, 3),
            timestamp_end=round(f_end / fps, 3),
            duration_s=round((f_end - f_start) / fps, 3),
            peak_activity=round(peak, 4),
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _roi_to_px(
        roi: tuple[float, float, float, float], w: int, h: int
    ) -> tuple[int, int, int, int]:
        return int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)


# ---------------------------------------------------------------------------
# Plotting — minimal, called externally
# ---------------------------------------------------------------------------


def plot_simple_results(
    result: SimpleDetectorResult,
    output_dir: Path,
    on_threshold: float = 0.05,
):
    """Save two plots: tap heatmaps overlay + per-tap time series."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Plot 1: Heatmaps over tap handle regions --------------------------
    has_heatmaps = (
        result.heatmaps.get("Tap A") is not None and result.crop_frames.get("Tap A") is not None
    )
    if has_heatmaps:
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

        for ax, label in [(ax_a, "Tap A"), (ax_b, "Tap B")]:
            hmap = result.heatmaps[label]
            crop = result.crop_frames[label]
            ax.imshow(crop)
            vmax = max(hmap.max(), 0.01)
            im = ax.imshow(hmap, cmap="hot", alpha=0.55, vmin=0, vmax=vmax)
            ax.set_title(f"{label} — pixel activity")
            ax.axis("off")

        plt.colorbar(im, ax=[ax_a, ax_b], fraction=0.046, label="Activity fraction")
        fig.suptitle("Tap handle activity heatmaps", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / "tap_heatmaps.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # -- Plot 2: Per-tap time series ---------------------------------------
    if len(result.times) > 0:
        fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

        for ax, label, color, tap_key in [
            (ax_a, "Tap A", "green", "A"),
            (ax_b, "Tap B", "red", "B"),
        ]:
            sig = result.signals[label]
            ax.plot(result.times, sig, color=color, linewidth=0.8, alpha=0.8)
            ax.axhline(
                y=on_threshold,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
                label="ON threshold",
            )
            for e in result.events:
                if e.tap == tap_key:
                    ax.axvspan(e.timestamp_start, e.timestamp_end, alpha=0.2, color=color)
            n_events = sum(1 for e in result.events if e.tap == tap_key)
            ax.set_ylabel("Active pixel fraction")
            ax.set_title(f"{label} — {n_events} event(s)")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        ax_b.set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(output_dir / "tap_signals.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Plots saved to {output_dir}")
