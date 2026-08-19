"""Video ingestion: probing and frame sampling.

Sampling walks the stream sequentially rather than seeking to each target
frame with cv2.set(CAP_PROP_POS_FRAMES). Compressed video only has cheap
random access to keyframes (the GOP structure) — seeking to an arbitrary
frame decodes forward from the nearest prior keyframe anyway, so doing that
once per sample is both slower and, on some containers, off by a few frames.
Reading the stream once and picking frames off it as they pass is cheaper
and exact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

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


def sample_frames(
    path: str | Path,
    out_dir: str | Path,
    interval_sec: float,
    max_frames: int | None = None,
    jpeg_quality: int = 90,
) -> list[FrameMeta]:
    """Sample frames at a fixed wall-clock interval, writing each to disk.

    Frames are written to `out_dir` as JPEGs rather than kept in memory:
    a full match sampled every couple of seconds is thousands of frames,
    and holding those as raw arrays at once is not something this
    machine's RAM budget for CPU-only inference can absorb.
    """
    path = str(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe_video(path)
    step = max(1, round(interval_sec * info.fps))

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    results: list[FrameMeta] = []
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                timestamp = frame_index / info.fps
                file_path = out_dir / f"frame_{frame_index:08d}.jpg"
                cv2.imwrite(
                    str(file_path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                results.append(FrameMeta(
                    frame_index=frame_index,
                    timestamp_sec=timestamp,
                    file_path=str(file_path),
                ))
                if max_frames is not None and len(results) >= max_frames:
                    break
            frame_index += 1
    finally:
        cap.release()

    return results
