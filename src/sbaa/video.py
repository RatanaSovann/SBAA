"""Video ingestion: probing and frame sampling.

Two ways to pull one frame every N seconds out of a video:

- **Sequential**: decode every frame in order, keep the ones that land on
  the interval. Cost is fixed — proportional to the *whole video's* frame
  count — no matter how sparse the sample is.
- **Seek**: jump straight to each target frame with
  cv2.set(CAP_PROP_POS_FRAMES). Cost is proportional to the *number of
  samples*, but each jump isn't free: the decoder can only randomly access
  keyframes, so it decodes forward from the nearest prior keyframe to reach
  the target.

Neither wins outright. A coarse interval means few seeks, so seeking wins;
a fine interval approaches one sample per frame, where sequential's fixed
cost wins and seeking's per-jump overhead is pure waste. `calibrate()`
measures both costs on the actual file (codec and keyframe spacing vary
enough between files that a hardcoded threshold would be wrong as often as
right), and `plan_sampling()` picks whichever is cheaper for the requested
interval.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


@dataclass
class FrameMeta:
    frame_index: int
    timestamp_sec: float
    file_path: str
    sharpness: float = 0.0


@dataclass
class Calibration:
    sequential_fps: float  # frames/sec this machine can decode this file at, in order
    seek_cost_sec: float   # avg wall-clock cost of one cv2.set() + read()


@dataclass
class SamplingPlan:
    strategy: str  # "sequential" or "seek"
    sample_count: int
    estimated_seconds: float


# Laplacian-variance blur score below which a frame is treated as unusable. Calibrated
# against one match's actual footage (motion-blur/pan frames scored under ~110, genuine
# board appearances scored 300+), not a universal constant -- broadcasts with different
# compression or camera work may need a different cutoff.
MIN_SHARPNESS = 150.0


def compute_sharpness(image) -> float:
    """Laplacian-variance blur score for an already-decoded BGR/grayscale frame.

    Low-detail, motion-blurred, or out-of-focus frames produce near-featureless CLIP
    embeddings that can land deceptively close to many unrelated reference images --
    see retrieval.py's docstring. This is a cheap (no ML) way to drop them before they
    ever reach CLIP ranking, for every asset at once rather than per-asset.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def filter_sharp(frames: list[FrameMeta], min_sharpness: float = MIN_SHARPNESS) -> list[FrameMeta]:
    """Drop frames below the blur threshold. Order-preserving."""
    return [f for f in frames if f.sharpness >= min_sharpness]


def file_cache_key(path: str | Path, chunk_size: int = 1_048_576) -> str:
    """Cheap content-based cache key: file size + hash of the first chunk.

    Hashing a multi-gigabyte match video in full is a real cost paid on
    every upload; the first megabyte plus the file size is enough to avoid
    collisions between distinct uploads without reading the whole file.
    """
    path = Path(path)
    h = hashlib.sha1()
    h.update(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk_size))
    return h.hexdigest()[:16]


def probe_video(path: str | Path) -> VideoInfo:
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    duration = frame_count / fps if fps else 0.0
    return VideoInfo(
        path=path, fps=fps, frame_count=frame_count,
        duration_sec=duration, width=width, height=height,
    )


def estimate_sample_count(info: VideoInfo, interval_sec: float) -> int:
    step = max(1, round(interval_sec * info.fps))
    return (info.frame_count + step - 1) // step if step else 0


def calibrate(path: str | Path, info: VideoInfo | None = None, probes: int = 3, sequential_frames: int = 60) -> Calibration:
    """Benchmark this file's decode cost, ~1-2s of work.

    Reused across an analyst's slider adjustments for the same video —
    it's cheap relative to a full sampling run, but not free enough to
    redo on every interaction.
    """
    path = str(path)
    info = info or probe_video(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        t0 = time.time()
        n = 0
        for _ in range(sequential_frames):
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        seq_elapsed = time.time() - t0
        sequential_fps = n / seq_elapsed if seq_elapsed > 0 and n > 0 else 1.0

        targets = [
            min(info.frame_count - 1, int(info.frame_count * f))
            for f in (0.25, 0.5, 0.75)[:probes]
        ]
        t0 = time.time()
        for target in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            cap.read()
        seek_elapsed = time.time() - t0
        seek_cost_sec = seek_elapsed / len(targets) if targets else 0.2
    finally:
        cap.release()

    return Calibration(sequential_fps=sequential_fps, seek_cost_sec=seek_cost_sec)


def plan_sampling(info: VideoInfo, interval_sec: float, calibration: Calibration) -> SamplingPlan:
    sample_count = estimate_sample_count(info, interval_sec)
    sequential_est = info.frame_count / calibration.sequential_fps
    seek_est = sample_count * calibration.seek_cost_sec
    if seek_est < sequential_est:
        return SamplingPlan("seek", sample_count, seek_est)
    return SamplingPlan("sequential", sample_count, sequential_est)


def sample_frames(
    path: str | Path,
    out_dir: str | Path,
    interval_sec: float,
    max_frames: int | None = None,
    jpeg_quality: int = 90,
    calibration: Calibration | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[FrameMeta]:
    """Sample frames at a fixed wall-clock interval, writing each to disk.

    Frames are written to `out_dir` as JPEGs rather than kept in memory:
    a full match sampled every couple of seconds is thousands of frames,
    and holding those as raw arrays at once is not something this
    machine's RAM budget for CPU-only inference can absorb.

    Picks sequential-decode or seek-based extraction per `plan_sampling`;
    without a `calibration`, defaults to sequential (the safe choice for a
    one-off or a fine-grained interval).

    `on_progress(done, total)` is throttled internally (not called every
    frame) so a caller driving a UI progress bar isn't swamped with updates
    on a run that decodes hundreds of thousands of frames.
    """
    path = str(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe_video(path)
    step = max(1, round(interval_sec * info.fps))

    strategy = "sequential"
    if calibration is not None:
        strategy = plan_sampling(info, interval_sec, calibration).strategy

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    try:
        if strategy == "seek":
            return _sample_seek(cap, info, step, out_dir, max_frames, jpeg_quality, on_progress)
        return _sample_sequential(cap, info, step, out_dir, max_frames, jpeg_quality, on_progress)
    finally:
        cap.release()


def _sample_sequential(cap, info: VideoInfo, step: int, out_dir: Path, max_frames, jpeg_quality, on_progress=None) -> list[FrameMeta]:
    results: list[FrameMeta] = []
    frame_index = 0
    # Sequential cost is proportional to frames decoded, not frames kept, so
    # progress tracks decode position against the whole video, not the sample count.
    report_every = max(1, info.frame_count // 200)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step == 0:
            results.append(_write_frame(frame, frame_index, info.fps, out_dir, jpeg_quality))
            if max_frames is not None and len(results) >= max_frames:
                break
        if on_progress and frame_index % report_every == 0:
            on_progress(frame_index, info.frame_count)
        frame_index += 1
    if on_progress:
        # Report where decoding actually stopped, not the video's full frame
        # count -- an early exit via max_frames should not read as 100% done.
        on_progress(frame_index, info.frame_count)
    return results


def _sample_seek(cap, info: VideoInfo, step: int, out_dir: Path, max_frames, jpeg_quality, on_progress=None) -> list[FrameMeta]:
    results: list[FrameMeta] = []
    frame_index = 0
    total = max_frames or ((info.frame_count + step - 1) // step)
    report_every = max(1, total // 100)
    while frame_index < info.frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        # Trust the decoder's own account of where it landed over the
        # target: some containers won't seek to the exact requested frame.
        reported = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        actual_index = int(reported) if reported >= 0 else frame_index
        results.append(_write_frame(frame, actual_index, info.fps, out_dir, jpeg_quality))
        if on_progress and len(results) % report_every == 0:
            on_progress(len(results), total)
        if max_frames is not None and len(results) >= max_frames:
            break
        frame_index += step
    if on_progress:
        on_progress(len(results), total)
    return results


def load_cached_frames(
    out_dir: str | Path, fps: float, on_progress: Callable[[int, int], None] | None = None,
) -> list[FrameMeta]:
    """Reconstruct frame metadata from an already-sampled cache directory.

    Lets a slow sampling pass (potentially tens of minutes for a fine interval over a full
    match, on this CPU-only machine) be run once — e.g. via scripts/sample_video.py ahead of
    an interactive session — and then picked up instantly on later runs instead of re-decoding.

    Sharpness isn't stored alongside the JPEGs the way timestamp/index are (parsed straight
    from the filename), so it means re-reading each one -- ~100s for a full match's worth of
    frames measured on this machine, cheap per-frame (no ML) but not negligible in aggregate.
    Cached to a single `sharpness.json` in `out_dir` so that cost is paid once per cache
    directory rather than on every "Load cached frames" click.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return []
    paths = sorted(out_dir.glob("frame_*.jpg"))

    sharpness_cache_path = out_dir / "sharpness.json"
    sharpness_cache: dict[str, float] = {}
    if sharpness_cache_path.exists():
        sharpness_cache = json.loads(sharpness_cache_path.read_text())

    results = []
    cache_dirty = False
    for i, p in enumerate(paths):
        frame_index = int(p.stem.split("_")[1])
        if p.name in sharpness_cache:
            sharpness = sharpness_cache[p.name]
        else:
            image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            sharpness = compute_sharpness(image) if image is not None else 0.0
            sharpness_cache[p.name] = sharpness
            cache_dirty = True
        results.append(FrameMeta(
            frame_index=frame_index, timestamp_sec=frame_index / fps, file_path=str(p), sharpness=sharpness,
        ))
        if on_progress and (i % max(1, len(paths) // 100) == 0 or i == len(paths) - 1):
            on_progress(i + 1, len(paths))

    if cache_dirty:
        sharpness_cache_path.write_text(json.dumps(sharpness_cache))
    return results


def _write_frame(frame, frame_index: int, fps: float, out_dir: Path, jpeg_quality: int) -> FrameMeta:
    timestamp = frame_index / fps
    file_path = out_dir / f"frame_{frame_index:08d}.jpg"
    cv2.imwrite(str(file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    # Computed from the frame already decoded in memory -- effectively free here, unlike
    # load_cached_frames() where it means re-reading each JPEG from disk.
    sharpness = compute_sharpness(frame)
    return FrameMeta(frame_index=frame_index, timestamp_sec=timestamp, file_path=str(file_path), sharpness=sharpness)
