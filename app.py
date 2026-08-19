"""SBAA — Sport Brand Annotation Assistant.

Milestone 1: upload a match video and reference brand-asset images, sample
frames from the video at a configurable interval, and preview them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sbaa.video import calibrate, estimate_sample_count, file_cache_key, plan_sampling, probe_video, sample_frames  # noqa: E402

ROOT = Path(__file__).parent
UPLOADS_DIR = ROOT / "data" / "uploads"
CACHE_DIR = ROOT / "data" / "cache"
VIDEO_DIR = ROOT / "sample_data" / "video"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="SBAA", layout="wide")
st.title("Sport Brand Annotation Assistant")

if "assets" not in st.session_state:
    st.session_state.assets = []  # list of {brand, asset, image_path}
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "video_info" not in st.session_state:
    st.session_state.video_info = None
if "sampled_frames" not in st.session_state:
    st.session_state.sampled_frames = None
if "calibration" not in st.session_state:
    st.session_state.calibration = None


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# --- Step 1: pick match video -------------------------------------------------
st.header("1. Match video")
st.caption(
    f"Match videos run several GB, too large for a browser upload buffered in "
    f"memory. Drop the file into `{VIDEO_DIR.relative_to(ROOT)}` and pick it below."
)

video_candidates = sorted(
    p for p in VIDEO_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
) if VIDEO_DIR.exists() else []

if not video_candidates:
    st.info(f"No video files found in `{VIDEO_DIR.relative_to(ROOT)}`. Drop one there and refresh.")
else:
    selected = st.selectbox(
        "Video file", video_candidates, format_func=lambda p: p.name,
    )
    if st.session_state.video_path != str(selected):
        st.session_state.video_path = str(selected)
        st.session_state.video_info = probe_video(selected)
        st.session_state.sampled_frames = None  # new video invalidates any prior sample
        with st.spinner("Benchmarking decode speed on this file..."):
            st.session_state.calibration = calibrate(selected, st.session_state.video_info)

if st.session_state.video_info:
    info = st.session_state.video_info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", fmt_time(info.duration_sec))
    c2.metric("FPS", f"{info.fps:.1f}")
    c3.metric("Resolution", f"{info.width}x{info.height}")
    c4.metric("Total frames", f"{info.frame_count:,}")

# --- Step 2: register brand assets -----------------------------------------------
st.header("2. Brand assets")
st.caption("Add a reference image for each brand asset you want SBAA to look for.")

with st.form("add_asset", clear_on_submit=True):
    fc1, fc2, fc3 = st.columns([2, 2, 3])
    brand = fc1.text_input("Brand")
    asset = fc2.text_input("Asset (e.g. 'main logo', 'shirt sponsor')")
    ref_image = fc3.file_uploader("Reference image", type=["png", "jpg", "jpeg"], key="ref_upload")
    submitted = st.form_submit_button("Add asset")
    if submitted:
        if not brand or not asset or ref_image is None:
            st.warning("Brand, asset name and a reference image are all required.")
        else:
            asset_dir = UPLOADS_DIR / "assets" / brand / asset
            asset_dir.mkdir(parents=True, exist_ok=True)
            image_path = asset_dir / ref_image.name
            with open(image_path, "wb") as f:
                f.write(ref_image.getbuffer())
            st.session_state.assets.append({
                "brand": brand, "asset": asset, "image_path": str(image_path),
            })

if st.session_state.assets:
    for i, a in enumerate(st.session_state.assets):
        ac1, ac2, ac3 = st.columns([1, 3, 1])
        ac1.image(a["image_path"], width=80)
        ac2.write(f"**{a['brand']}** — {a['asset']}")
        if ac3.button("Remove", key=f"remove_{i}"):
            st.session_state.assets.pop(i)
            st.rerun()
else:
    st.info("No brand assets registered yet.")

# --- Step 3: sample frames ---------------------------------------------------
st.header("3. Sample frames")

if st.session_state.video_info is None:
    st.info("Upload a video first.")
else:
    info = st.session_state.video_info
    interval = st.slider(
        "Sampling interval (seconds)", min_value=0.5, max_value=5.0, value=2.0, step=0.5,
        help="Smaller = more frames, better recall of brief appearances, more CPU time. "
             "This machine has no GPU, so keep this coarse for a full match.",
    )
    calibration = st.session_state.calibration
    plan = plan_sampling(info, interval, calibration) if calibration else None
    if plan:
        st.caption(
            f"~{plan.sample_count:,} frames, using **{plan.strategy}** decoding "
            f"— est. {fmt_time(plan.estimated_seconds)}."
        )
    else:
        st.caption(f"~{estimate_sample_count(info, interval):,} frames will be sampled.")

    if st.button("Sample frames", type="primary"):
        cache_key = file_cache_key(st.session_state.video_path)
        out_dir = CACHE_DIR / cache_key / f"interval_{interval}"
        eta = fmt_time(plan.estimated_seconds) if plan else "unknown"
        with st.spinner(f"Sampling (est. {eta})..."):
            frames = sample_frames(
                st.session_state.video_path, out_dir, interval, calibration=calibration,
            )
        st.session_state.sampled_frames = frames
        st.success(f"Sampled {len(frames)} frames.")

    frames = st.session_state.sampled_frames
    if frames:
        st.caption(f"Preview (first 24 of {len(frames)}):")
        cols = st.columns(6)
        for i, fr in enumerate(frames[:24]):
            with cols[i % 6]:
                st.image(fr.file_path, caption=fmt_time(fr.timestamp_sec), use_container_width=True)
