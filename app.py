"""SBAA — Sport Brand Annotation Assistant.

Milestone 1: upload a match video and reference brand-asset images, sample
frames from the video at a configurable interval, and preview them.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sbaa.export import build_brief_workbook, burn_box  # noqa: E402
from sbaa.retrieval import deduplicate_by_time, rank_frames, verify_with_ocr  # noqa: E402
from sbaa.video import (  # noqa: E402
    calibrate, estimate_sample_count, file_cache_key, filter_sharp, load_cached_frames,
    plan_sampling, probe_video, sample_frames,
)

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
if "rankings" not in st.session_state:
    st.session_state.rankings = {}  # (brand, asset) -> list[RankedFrame]
if "ranking_stats" not in st.session_state:
    st.session_state.ranking_stats = {}  # (brand, asset) -> {"clip_candidates", "ocr_confirmed"}
if "approvals" not in st.session_state:
    st.session_state.approvals = {}  # (brand, asset) -> list of up to 2 {file_path, frame_index, timestamp_sec, bbox}


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
st.caption(
    "For wordmark/text logos, a flat crop on a white background ranks noticeably better than "
    "a transparent vector/SVG export — see CLAUDE.md's domain gap notes."
)

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

    cache_key = file_cache_key(st.session_state.video_path)
    out_dir = CACHE_DIR / cache_key / f"interval_{interval}"
    # Cheap existence/count check only -- load_cached_frames() now also computes a blur
    # score per frame (decodes every JPEG), too expensive to run on every script rerun
    # just to decide which button to show. That full load happens on the actual click below.
    cached_count = len(list(out_dir.glob("frame_*.jpg"))) if out_dir.exists() else 0
    expected_count = estimate_sample_count(info, interval)

    def run_sampling() -> None:
        progress = st.progress(0.0)
        status = st.empty()

        def on_progress(done: int, total: int, _status=status, _progress=progress) -> None:
            _progress.progress(done / total if total else 1.0)
            _status.caption(f"Sampling: {done:,}/{total:,} frames")

        frames = sample_frames(
            st.session_state.video_path, out_dir, interval,
            calibration=calibration, on_progress=on_progress,
        )
        progress.empty()
        status.empty()
        st.session_state.sampled_frames = frames
        st.session_state.rankings = {}  # new sample invalidates any prior ranking
        st.session_state.ranking_stats = {}
        st.session_state.approvals = {}  # and any manual approvals made against the old frames
        st.success(f"Sampled {len(frames)} frames.")

    if cached_count:
        if cached_count < 0.95 * expected_count:
            st.warning(
                f"Found {cached_count} cached frames, but a full sample at this interval "
                f"should be ~{expected_count:,}. This cache looks incomplete — likely an "
                "earlier sampling run (this session's or `scripts/sample_video.py`) was "
                "interrupted partway through. Loading it will silently skip whatever portion "
                "of the video was never sampled; re-sample to fill the gap, or delete "
                f"`{out_dir.relative_to(ROOT)}` and start over."
            )
        else:
            st.success(f"Found {cached_count} already-sampled frames cached for this video/interval.")
        bc1, bc2 = st.columns(2)
        if bc1.button("Load cached frames", type="primary"):
            progress = st.progress(0.0)
            status = st.empty()

            def on_load_progress(done: int, total: int, _status=status, _progress=progress) -> None:
                _progress.progress(done / total if total else 1.0)
                _status.caption(f"Reading cached frames: {done:,}/{total:,}")

            cached = load_cached_frames(out_dir, info.fps, on_progress=on_load_progress)
            progress.empty()
            status.empty()
            st.session_state.sampled_frames = cached
            st.session_state.rankings = {}
            st.session_state.ranking_stats = {}
            st.session_state.approvals = {}
        if bc2.button("Re-sample (slow)"):
            run_sampling()
    else:
        st.caption(
            "Tip: sampling a full match can take tens of minutes on this hardware. Run "
            f"`scripts/sample_video.py --interval {interval}` offline ahead of time to "
            "avoid blocking this session — it writes to the same cache this page reads."
        )
        if st.button("Sample frames", type="primary"):
            run_sampling()

    frames = st.session_state.sampled_frames
    if frames:
        st.caption(f"Preview (first 24 of {len(frames)}):")
        cols = st.columns(6)
        for i, fr in enumerate(frames[:24]):
            with cols[i % 6]:
                st.image(fr.file_path, caption=fmt_time(fr.timestamp_sec), use_container_width=True)

# --- Step 4: rank frames per brand asset --------------------------------------
st.header("4. Rank frames per brand asset")
st.caption(
    "Coarse pass: rank sampled frames by CLIP similarity to each asset's reference image. "
    "CLIP matches a sign's color and layout more than the text on it, so for assets with a "
    "legible reference (a wordmark), an OCR pass re-checks the top candidates for the actual "
    "brand name before anything reaches the expensive detector stage."
)

OCR_TOP_K = 50
DEDUP_MIN_GAP_SEC = 10.0


def gather_reference_paths(asset: dict) -> list[str]:
    """The original uploaded logo, plus any analyst-confirmed real crops saved for this asset.

    Confirmed crops (`confirmed_<frame_index>.jpg`, written at approval time in Step 5) live
    right next to the original reference image, so a fresh asset registration and a re-rank
    both see the same growing reference set with no separate bookkeeping.
    """
    asset_dir = Path(asset["image_path"]).parent
    confirmed = sorted(str(p) for p in asset_dir.glob("confirmed_*.jpg"))
    return [asset["image_path"]] + confirmed


def rank_one_asset(asset, sharp_frames, embedding_cache_dir, on_embed_progress, on_ocr_progress):
    ref_paths = gather_reference_paths(asset)
    clip_ranked = rank_frames(
        ref_paths, sharp_frames, embedding_cache_dir, on_progress=on_embed_progress,
    )
    verified = verify_with_ocr(
        clip_ranked, asset["image_path"], top_k=OCR_TOP_K, on_progress=on_ocr_progress,
    )
    ocr_applied = verified is not clip_ranked  # unchanged list means no legible reference text
    deduped = deduplicate_by_time(verified, min_gap_sec=DEDUP_MIN_GAP_SEC)
    stats = {
        "sharp_frames": len(sharp_frames),
        "total_frames": len(st.session_state.sampled_frames),
        "clip_candidates": len(clip_ranked),
        "ocr_confirmed": len(verified) if ocr_applied else None,
        "deduped": len(deduped),
        "reference_count": len(ref_paths),
    }
    return deduped, stats


frames = st.session_state.sampled_frames
if frames is None:
    st.info("Sample frames first.")
elif not st.session_state.assets:
    st.info("Register at least one brand asset first.")
else:
    if st.button("Rank frames", type="primary"):
        cache_key = file_cache_key(st.session_state.video_path)
        embedding_cache_dir = CACHE_DIR / cache_key / f"interval_{interval}" / "embeddings"
        progress = st.progress(0.0)
        status = st.empty()

        def on_embed_progress(done: int, total: int, _status=status, _progress=progress) -> None:
            _progress.progress(done / total if total else 1.0)
            _status.caption(f"Embedding frames: {done:,}/{total:,}")

        def on_ocr_progress(done: int, total: int, _status=status, _progress=progress) -> None:
            _progress.progress(done / total if total else 1.0)
            _status.caption(f"Verifying candidates with OCR: {done}/{total}")

        # Blur filtering is asset-independent (a frame's sharpness doesn't depend on which
        # brand it's being ranked against), so it's done once here and reused for every
        # asset below, rather than repeated inside rank_frames() per asset.
        sharp_frames = filter_sharp(frames)
        status.caption(f"{len(sharp_frames):,} of {len(frames):,} sampled frames passed the blur filter.")

        rankings = {}
        ranking_stats = {}
        for a in st.session_state.assets:
            key = (a["brand"], a["asset"])
            status.caption(f"Ranking for {a['brand']} — {a['asset']}...")
            deduped, stats = rank_one_asset(a, sharp_frames, embedding_cache_dir, on_embed_progress, on_ocr_progress)
            rankings[key] = deduped
            ranking_stats[key] = stats
        st.session_state.rankings = rankings
        st.session_state.ranking_stats = ranking_stats
        progress.empty()
        status.empty()
        st.success("Ranking complete.")

    for a in st.session_state.assets:
        key = (a["brand"], a["asset"])
        if key not in st.session_state.rankings:
            continue  # not ranked yet at all -- distinct from "ranked, zero survived"
        ranked = st.session_state.rankings[key]
        st.subheader(f"{a['brand']} — {a['asset']}")
        stats = st.session_state.ranking_stats.get(key)
        if stats:
            funnel = (
                f"{stats['total_frames']:,} sampled → {stats['sharp_frames']:,} sharp enough → "
                f"{stats['clip_candidates']:,} CLIP-ranked"
            )
            if stats["ocr_confirmed"] is not None:
                funnel += f" → top {OCR_TOP_K} OCR-checked → {stats['ocr_confirmed']} confirmed"
            else:
                funnel += " (no legible text on reference — OCR check skipped)"
            st.caption(f"{funnel} → {stats['deduped']} distinct appearances (deduped).")
            if stats["reference_count"] > 1:
                st.caption(
                    f"Ranked using 1 logo + {stats['reference_count'] - 1} analyst-confirmed "
                    "example(s) from earlier approvals."
                )
        if not ranked:
            st.warning(
                "No candidates survived. Either this asset doesn't appear in the sampled "
                "frames, or OCR couldn't read its text on any of the top CLIP candidates "
                "(small/blurry text, or a low sampling rate missing the appearance)."
            )
            continue
        top = ranked[:12]
        cols = st.columns(6)
        for i, rf in enumerate(top):
            with cols[i % 6]:
                st.image(
                    rf.frame.file_path,
                    caption=f"{fmt_time(rf.frame.timestamp_sec)} · {rf.score:.3f}",
                    use_container_width=True,
                )

# --- Step 5: review & approve --------------------------------------------------
st.header("5. Review & approve")
st.caption(
    "Pick up to two examples per asset. There's no automated localization on this hardware "
    "yet (see CLAUDE.md), so draw a box around the asset by hand before approving."
)

MAX_APPROVALS = 2

for a in st.session_state.assets:
    key = (a["brand"], a["asset"])
    ranked = st.session_state.rankings.get(key)
    if not ranked:
        continue
    key_str = f"{a['brand']}__{a['asset']}"
    st.subheader(f"{a['brand']} — {a['asset']}")
    approved = st.session_state.approvals.setdefault(key, [])

    if approved:
        st.caption(f"Approved ({len(approved)}/{MAX_APPROVALS}):")
        acols = st.columns(MAX_APPROVALS)
        for i, entry in enumerate(approved):
            with acols[i]:
                st.image(
                    burn_box(entry["file_path"], entry["bbox"]),
                    caption=f"{fmt_time(entry['timestamp_sec'])} · frame {entry['frame_index']}",
                    use_container_width=True,
                )
                if st.button("Remove", key=f"remove_approval_{key_str}_{i}"):
                    approved.pop(i)
                    st.rerun()

    if len(approved) >= MAX_APPROVALS:
        st.info(f"{MAX_APPROVALS} examples already approved for this asset — remove one to pick a different candidate.")
        continue

    if 1 <= len(approved) < MAX_APPROVALS:
        rc1, rc2 = st.columns([1, 3])
        if rc1.button("Re-rank using confirmed example(s)", key=f"rerank_{key_str}"):
            cache_key = file_cache_key(st.session_state.video_path)
            embedding_cache_dir = CACHE_DIR / cache_key / f"interval_{interval}" / "embeddings"
            sharp_frames = filter_sharp(st.session_state.sampled_frames)
            progress = st.progress(0.0)
            status = st.empty()

            def on_embed_progress(done: int, total: int, _status=status, _progress=progress) -> None:
                _progress.progress(done / total if total else 1.0)
                _status.caption(f"Embedding frames: {done:,}/{total:,}")

            def on_ocr_progress(done: int, total: int, _status=status, _progress=progress) -> None:
                _progress.progress(done / total if total else 1.0)
                _status.caption(f"Verifying candidates with OCR: {done}/{total}")

            deduped, stats = rank_one_asset(a, sharp_frames, embedding_cache_dir, on_embed_progress, on_ocr_progress)
            st.session_state.rankings[key] = deduped
            st.session_state.ranking_stats[key] = stats
            progress.empty()
            status.empty()
            st.rerun()
        rc2.caption(
            "Re-embeds using the original logo plus every confirmed example approved so far "
            "for this asset — surfaces appearances a clean-logo-only ranking missed."
        )

    approved_indices = {e["frame_index"] for e in approved}
    remaining = [rf for rf in ranked if rf.frame.frame_index not in approved_indices]
    if not remaining:
        st.info("Every surfaced candidate for this asset is already approved.")
        continue

    choice = st.selectbox(
        "Candidate to review",
        range(len(remaining)),
        format_func=lambda i, _r=remaining: f"{fmt_time(_r[i].frame.timestamp_sec)} · score {_r[i].score:.3f}",
        key=f"candidate_select_{key_str}",
    )
    candidate = remaining[choice]
    image = Image.open(candidate.frame.file_path).convert("RGB")
    box = st_cropper(
        image, box_color="red", return_type="box", realtime_update=True,
        key=f"cropper_{key_str}_{choice}",
    )
    if st.button("Approve this frame", key=f"approve_{key_str}_{choice}"):
        approved.append({
            "file_path": candidate.frame.file_path,
            "frame_index": candidate.frame.frame_index,
            "timestamp_sec": candidate.frame.timestamp_sec,
            "bbox": box,
        })
        # Save the approved crop as a future reference -- both for an immediate re-rank and
        # for the next time this brand/asset is registered against a different match video.
        crop = image.crop((box["left"], box["top"], box["left"] + box["width"], box["top"] + box["height"]))
        confirmed_path = Path(a["image_path"]).parent / f"confirmed_{candidate.frame.frame_index}.jpg"
        crop.save(confirmed_path)
        st.rerun()

# --- Step 6: export brief -------------------------------------------------------
st.header("6. Export brief")
st.caption(
    "The approved examples, one Excel workbook, one sheet per brand — matching the manual "
    "spreadsheet process this replaces. No rejected candidates or decision history included."
)

has_any_approval = any(
    st.session_state.approvals.get((a["brand"], a["asset"])) for a in st.session_state.assets
)
if not has_any_approval:
    st.info("Approve at least one example above before exporting.")
else:
    if st.button("Build export", type="primary"):
        wb = build_brief_workbook(st.session_state.approvals)
        buf = io.BytesIO()
        wb.save(buf)
        st.session_state["export_buffer"] = buf.getvalue()

    if "export_buffer" in st.session_state:
        st.download_button(
            "Download brief.xlsx",
            data=st.session_state["export_buffer"],
            file_name="brief.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
