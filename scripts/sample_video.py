"""Offline frame sampling.

Run this ahead of an interactive session so the slow decode pass (potentially tens of
minutes for a fine interval over a full match, on this CPU-only machine per CLAUDE.md)
doesn't block the Streamlit UI. Writes into the same data/cache/<file-hash>/interval_<N>/
layout app.py already reads, so a sample produced here is picked up instantly there.

Usage:
    ./.venv/Scripts/python.exe scripts/sample_video.py <video-file-or-name> --interval 5.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sbaa.video import calibrate, file_cache_key, plan_sampling, probe_video, sample_frames  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to a video file, or a filename under sample_data/video/")
    parser.add_argument("--interval", type=float, default=2.0, help="Sampling interval in seconds")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        video_path = ROOT / "sample_data" / "video" / args.video
    if not video_path.exists():
        parser.error(f"video not found: {args.video}")

    cache_dir = ROOT / "data" / "cache" / file_cache_key(video_path) / f"interval_{args.interval}"

    print(f"Probing {video_path.name}...")
    info = probe_video(video_path)
    print(f"{info.duration_sec:.0f}s, {info.fps:.1f} fps, {info.frame_count:,} frames")

    print("Calibrating decode strategy...")
    calibration = calibrate(video_path, info)
    plan = plan_sampling(info, args.interval, calibration)
    print(f"~{plan.sample_count:,} frames via {plan.strategy} decoding, est. {plan.estimated_seconds / 60:.1f} min")

    frames = sample_frames(video_path, cache_dir, args.interval, calibration=calibration)
    print(f"Wrote {len(frames)} frames to {cache_dir}")


if __name__ == "__main__":
    main()
