"""SBAA — Sport Brand Annotation Assistant.

Milestone 1: upload a match video and reference brand-asset images, sample
frames from the video at a configurable interval, and preview them.
"""

from __future__ import annotations

import io
import shutil
import sys
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sbaa.category import (  # noqa: E402
    BUILTIN_CATEGORIES, add_category, load_categories, remove_category,
)
from sbaa.export import build_brief_workbook, burn_box  # noqa: E402
# Imported as bare functions, not as a `progress` module: several handlers below bind a local
# `progress = st.progress(0.0)` at module scope, which would shadow the module and break the
# save at the bottom of this script.
from sbaa.progress import load_progress, progress_path, save_progress  # noqa: E402
from sbaa.retrieval import (  # noqa: E402
    deduplicate_by_time, merge_candidates, rank_frames, search_by_text, verify_with_ocr,
)
from sbaa.vlm import classify_frame, is_configured as vlm_is_configured  # noqa: E402
from sbaa.video import (  # noqa: E402
    calibrate, estimate_sample_count, file_cache_key, filter_sharp, load_cached_frames,
    plan_sampling, probe_video, sample_frames,
)

ROOT = Path(__file__).parent
UPLOADS_DIR = ROOT / "data" / "uploads"
CACHE_DIR = ROOT / "data" / "cache"
PROGRESS_DIR = ROOT / "data" / "progress"
# The taxonomy is global across videos (unlike progress, which is per video+interval), so it
# lives beside them rather than inside one video's folder.
CATEGORIES_FILE = ROOT / "data" / "categories.json"
VIDEO_DIR = ROOT / "sample_data" / "video"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="SBAA", layout="wide")
st.title("Sport Brand Annotation Assistant")

if "assets" not in st.session_state:
    st.session_state.assets = []  # list of {brand, asset, image_path}
if "auto_assets" not in st.session_state:
    st.session_state.auto_assets = []  # list of {brand, image_path} pending category discovery
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
if "progress_path" not in st.session_state:
    st.session_state.progress_path = None  # set once a video+interval is chosen; see Step 3


# Re-read every rerun rather than cached in session_state: it's a small file, and this way an
# edit made in the form below is reflected everywhere on the very next run with no invalidation.
CATEGORIES = load_categories(CATEGORIES_FILE)


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

CUSTOM_ASSET_OPTION = "Custom…"

with st.form("add_asset", clear_on_submit=True):
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 3])
    brand = fc1.text_input("Brand")
    asset_choice = fc2.selectbox(
        "Asset / category",
        list(CATEGORIES) + [CUSTOM_ASSET_OPTION],
        help="Pick a category to keep naming consistent with auto-discovered ones below (see "
             "CLAUDE.md). Use Custom for a one-off asset outside the taxonomy — note that a "
             "Custom name skips the vision-model category check, since it has no grounding "
             "description to check evidence against.",
    )
    # Always rendered, not conditionally shown -- a selection change inside st.form doesn't
    # rerun the script until submit, so a "reveal on Custom" field wouldn't update in time.
    custom_asset = fc3.text_input("Custom name (if Custom above)")
    ref_image = fc4.file_uploader("Reference image", type=["png", "jpg", "jpeg"], key="ref_upload")
    submitted = st.form_submit_button("Add asset")
    if submitted:
        brand = brand.strip()
        asset = custom_asset.strip() if asset_choice == CUSTOM_ASSET_OPTION else asset_choice
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

with st.expander(f"Placement categories ({len(CATEGORIES)})"):
    st.caption(
        "The placement types auto-discovery sorts a brand's appearances into. A missing category "
        "doesn't produce a blank — the model is asked to pick the best of the list, so a "
        "placement with no bucket comes back confidently labelled as the nearest one instead "
        "(a Hisense backdrop wall was classified as an LED Board before `Wall` was added). Add "
        "one here whenever a new placement type shows up in real footage."
    )
    for name, description in CATEGORIES.items():
        cc1, cc2 = st.columns([5, 1])
        builtin = name in BUILTIN_CATEGORIES
        cc1.markdown(f"**{name}**{' · built-in' if builtin else ''}  \n<small>{description}</small>",
                     unsafe_allow_html=True)
        if not builtin and cc2.button("Remove", key=f"remove_cat_{name}"):
            remove_category(CATEGORIES_FILE, name)
            st.rerun()

    with st.form("add_category", clear_on_submit=True):
        new_name = st.text_input("New category name", placeholder="e.g. Pitch Decal")
        new_description = st.text_area(
            "How the model should recognise it",
            placeholder="e.g. sponsor branding painted or projected onto the pitch surface "
                        "itself, not on perimeter signage or a screen",
            help="This is sent to the vision model verbatim and does real work — bare category "
                 "names caused false positives in testing. Describe what counts, and say what "
                 "it must NOT be confused with (naming the nearest existing category is what "
                 "fixed the LED Board / Wall overlap).",
        )
        if st.form_submit_button("Add category"):
            try:
                add_category(CATEGORIES_FILE, new_name, new_description)
                st.rerun()
            except ValueError as e:
                st.warning(str(e))

st.markdown("**Or auto-discover categories from a logo:**")
st.caption(
    "Ranks the brand against every sampled frame, then a vision model classifies each "
    f"OCR-confirmed candidate into one of {', '.join(CATEGORIES)} — no seed example needed. "
    "Requires `ANTHROPIC_API_KEY` to be set (see CLAUDE.md); each candidate frame is sent to "
    "the Anthropic API."
)
if not vlm_is_configured():
    st.warning(
        "`ANTHROPIC_API_KEY` isn't set — auto-discovery will be skipped at ranking time until "
        "it is."
    )

with st.form("add_auto_asset", clear_on_submit=True):
    gc1, gc2 = st.columns([2, 3])
    auto_brand = gc1.text_input("Brand", key="auto_brand")
    auto_ref_image = gc2.file_uploader("Logo image", type=["png", "jpg", "jpeg"], key="auto_ref_upload")
    auto_submitted = st.form_submit_button("Add brand for auto-discovery")
    if auto_submitted:
        auto_brand = auto_brand.strip()
        if not auto_brand or auto_ref_image is None:
            st.warning("Brand and a logo image are both required.")
        else:
            auto_dir = UPLOADS_DIR / "auto" / auto_brand
            auto_dir.mkdir(parents=True, exist_ok=True)
            image_path = auto_dir / auto_ref_image.name
            with open(image_path, "wb") as f:
                f.write(auto_ref_image.getbuffer())
            st.session_state.auto_assets.append({"brand": auto_brand, "image_path": str(image_path)})

if st.session_state.auto_assets:
    for i, pa in enumerate(st.session_state.auto_assets):
        gac1, gac2, gac3 = st.columns([1, 3, 1])
        gac1.image(pa["image_path"], width=80)
        gac2.write(f"**{pa['brand']}** — pending category discovery")
        if gac3.button("Remove", key=f"remove_auto_{i}"):
            st.session_state.auto_assets.pop(i)
            st.rerun()

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

    # Saved progress is keyed by this same video+interval pair, so picking a different interval
    # loads that interval's own work instead of needing any "your rankings are now stale" check
    # -- a restored ranking always points at the frames it was actually computed against.
    saved_path = progress_path(PROGRESS_DIR, cache_key, interval)
    if st.session_state.progress_path != str(saved_path):
        st.session_state.progress_path = str(saved_path)
        restored = load_progress(saved_path)
        st.session_state.assets = restored["assets"] if restored else []
        st.session_state.auto_assets = restored["auto_assets"] if restored else []
        st.session_state.rankings = restored["rankings"] if restored else {}
        st.session_state.ranking_stats = restored["ranking_stats"] if restored else {}
        st.session_state.approvals = restored["approvals"] if restored else {}
        # Steps 1-3 have already rendered against the pre-load state on this run, so rerun
        # rather than leaving restored assets invisible until the next interaction.
        st.rerun()

    if st.session_state.approvals or st.session_state.assets:
        st.caption(
            "Registered assets, rankings and approvals for this video and interval are saved "
            f"automatically to `{PROGRESS_DIR.relative_to(ROOT)}` and restored when you come back."
        )
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
        # Rankings and approvals are deliberately *not* cleared here. They used to be, on the
        # theory that a new sample invalidates work done against the old one -- but sampling is
        # deterministic, so re-sampling the same video at the same interval reproduces the same
        # frame indices and the same file paths. Now that progress is keyed by video+interval
        # (see the progress load above), a genuinely different set of frames is a different key
        # and reads its own file, which is the invalidation this clearing was standing in for.
        # Clearing here would instead be a silent data-loss bug: the analyst's approvals would
        # be wiped and then auto-saved over.
        st.session_state.sampled_frames = frames
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
            # Same reasoning as run_sampling(): these are the exact frames any restored
            # ranking was computed against, so nothing here is stale.
            st.session_state.sampled_frames = cached
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


def rank_one_asset(
    asset, sharp_frames, embedding_cache_dir, ocr_cache_dir,
    on_embed_progress, on_ocr_progress, on_text_progress, on_vlm_progress,
):
    ref_paths = gather_reference_paths(asset)
    clip_ranked = rank_frames(
        ref_paths, sharp_frames, embedding_cache_dir, on_progress=on_embed_progress,
    )
    verified = verify_with_ocr(
        clip_ranked, asset["image_path"], top_k=OCR_TOP_K, on_progress=on_ocr_progress,
    )
    ocr_applied = verified is not clip_ranked  # unchanged list means no legible reference text
    # Independent of CLIP/OCR above: a direct search of every sharp frame's own text for the
    # asset name, which finds real appearances a bad or graphic-only reference image would
    # otherwise bury far outside the CLIP top-K (see retrieval.search_by_text's docstring).
    text_hits = search_by_text(sharp_frames, asset["brand"], ocr_cache_dir, on_progress=on_text_progress)
    combined = merge_candidates(verified, text_hits)

    # Manual registration had no VLM confirmation at all -- only the auto-discover path did.
    # A brand-agnostic reference (a pure graphic mark like Adidas's three stripes, no legible
    # text) gets no OCR check either, so CLIP's color/layout confusion (e.g. a Visa perimeter
    # board outranking a genuine Adidas appearance) went straight to the analyst unfiltered.
    # Reuses the exact classify_frame() call already proven for auto-discover -- same taxonomy,
    # same brand+category evidence requirement -- checking whether the model's pick matches
    # this asset's own category. Only applies to assets using the fixed taxonomy (CATEGORIES);
    # a free-text "Custom…" asset name has no grounding description to check evidence against,
    # so it's left exactly as before.
    vlm_classified = None
    survivors = combined
    if vlm_is_configured() and asset["asset"] in CATEGORIES:
        capped = combined[:OCR_TOP_K]
        vlm_classified = len(capped)
        survivors = []
        for i, rf in enumerate(capped):
            category = classify_frame(rf.frame.file_path, CATEGORIES, asset["brand"], asset["image_path"])
            if category == asset["asset"]:
                survivors.append(rf)
            on_vlm_progress(i + 1, len(capped))

    deduped = deduplicate_by_time(survivors, min_gap_sec=DEDUP_MIN_GAP_SEC)
    stats = {
        "sharp_frames": len(sharp_frames),
        "total_frames": len(st.session_state.sampled_frames),
        "clip_candidates": len(clip_ranked),
        "ocr_confirmed": len(verified) if ocr_applied else None,
        "text_matches": len(text_hits),
        "vlm_classified": vlm_classified,
        "deduped": len(deduped),
        "reference_count": len(ref_paths),
    }
    return deduped, stats


frames = st.session_state.sampled_frames
if frames is None:
    st.info(
        "Load the cached frames (or sample them) above before ranking. Any ranking restored "
        "from an earlier session is still shown below."
    )
elif not st.session_state.assets and not st.session_state.auto_assets:
    st.info("Register at least one brand asset first.")
else:
    if st.button("Rank frames", type="primary"):
        cache_key = file_cache_key(st.session_state.video_path)
        embedding_cache_dir = CACHE_DIR / cache_key / f"interval_{interval}" / "embeddings"
        ocr_cache_dir = CACHE_DIR / cache_key / f"interval_{interval}" / "ocr_text"
        progress = st.progress(0.0)
        status = st.empty()

        def on_embed_progress(done: int, total: int, _status=status, _progress=progress) -> None:
            _progress.progress(done / total if total else 1.0)
            _status.caption(f"Embedding frames: {done:,}/{total:,}")

        def on_ocr_progress(done: int, total: int, _status=status, _progress=progress) -> None:
            _progress.progress(done / total if total else 1.0)
            _status.caption(f"Verifying candidates with OCR: {done}/{total}")

        def on_text_progress(done: int, total: int, _status=status, _progress=progress) -> None:
            _progress.progress(done / total if total else 1.0)
            _status.caption(f"Searching frame text for brand name: {done:,}/{total:,}")

        def on_manual_vlm_progress(done: int, total: int, _status=status, _progress=progress) -> None:
            _progress.progress(done / total if total else 1.0)
            _status.caption(f"Confirming candidates with VLM: {done}/{total}")

        # Blur filtering is asset-independent (a frame's sharpness doesn't depend on which
        # brand it's being ranked against), so it's done once here and reused for every
        # asset below, rather than repeated inside rank_frames() per asset.
        sharp_frames = filter_sharp(frames)
        status.caption(f"{len(sharp_frames):,} of {len(frames):,} sampled frames passed the blur filter.")

        # Seeded from what's already in session (including anything restored from a previous
        # session) rather than starting empty: this dict is assigned wholesale back over
        # st.session_state.rankings below, so starting empty would silently drop the ranking of
        # any asset this pass doesn't recompute.
        rankings = dict(st.session_state.rankings)
        ranking_stats = dict(st.session_state.ranking_stats)
        discovered_keys = set()

        if st.session_state.auto_assets:
            if not vlm_is_configured():
                st.warning(
                    "`ANTHROPIC_API_KEY` isn't set, so auto-discovery can't classify candidates "
                    "right now. Set it and rank again — the pending brand(s) below are kept for "
                    "that retry."
                )
            else:
                for pending in st.session_state.auto_assets:
                    status.caption(f"Discovering categories for {pending['brand']}...")
                    clip_ranked = rank_frames(
                        pending["image_path"], sharp_frames, embedding_cache_dir,
                        on_progress=on_embed_progress,
                    )
                    verified = verify_with_ocr(
                        clip_ranked, pending["image_path"], top_k=OCR_TOP_K,
                        on_progress=on_ocr_progress,
                    )
                    ocr_applied = verified is not clip_ranked
                    # Independent of CLIP/OCR above: search every sharp frame's own text for the
                    # brand name directly. This is what actually rescues a brand whose reference
                    # image is a bad match for its real on-screen appearance (see
                    # retrieval.search_by_text's docstring) -- CLIP can bury a genuine appearance
                    # far outside OCR_TOP_K when the reference doesn't resemble it closely enough,
                    # but a direct text match doesn't depend on the reference image at all.
                    text_hits = search_by_text(
                        sharp_frames, pending["brand"], ocr_cache_dir, on_progress=on_text_progress,
                    )
                    combined = merge_candidates(verified, text_hits)
                    if not combined:
                        continue
                    # Cap at OCR_TOP_K regardless of source -- a graphic-only logo (no legible
                    # reference text) makes verify_with_ocr a no-op, which would otherwise send
                    # every sharp frame to a paid API call. Text-search hits sort first (score
                    # 1.0), so they survive this cap ahead of weaker CLIP-only candidates. This
                    # is the same "cheap pass narrows before the expensive stage runs" discipline
                    # CLAUDE.md's hardware-constraint section already applies everywhere else,
                    # just newly load-bearing now that classification costs money per call.
                    candidates = combined[:OCR_TOP_K]
                    buckets: dict[str, list] = {}
                    for i, rf in enumerate(candidates):
                        category = classify_frame(
                            rf.frame.file_path, CATEGORIES, pending["brand"], pending["image_path"],
                        )
                        if category is not None:
                            buckets.setdefault(category, []).append(rf)
                        status.caption(
                            f"Classifying {pending['brand']} candidates with VLM: "
                            f"{i + 1}/{len(candidates)}"
                        )
                    if not buckets:
                        st.warning(
                            f"{pending['brand']}: the vision model didn't classify any surfaced "
                            "candidate into a known category."
                        )
                    for category, bucket_ranked in buckets.items():
                        key = (pending["brand"], category)
                        if not any(
                            a["brand"] == pending["brand"] and a["asset"] == category
                            for a in st.session_state.assets
                        ):
                            asset_dir = UPLOADS_DIR / "assets" / pending["brand"] / category
                            asset_dir.mkdir(parents=True, exist_ok=True)
                            dest = asset_dir / Path(pending["image_path"]).name
                            if not dest.exists():
                                shutil.copy(pending["image_path"], dest)
                            st.session_state.assets.append({
                                "brand": pending["brand"], "asset": category, "image_path": str(dest),
                            })
                        deduped = deduplicate_by_time(bucket_ranked, min_gap_sec=DEDUP_MIN_GAP_SEC)
                        rankings[key] = deduped
                        ranking_stats[key] = {
                            "sharp_frames": len(sharp_frames),
                            "total_frames": len(st.session_state.sampled_frames),
                            "clip_candidates": len(clip_ranked),
                            "ocr_confirmed": len(verified) if ocr_applied else None,
                            "text_matches": len(text_hits),
                            "vlm_classified": len(candidates),
                            "deduped": len(deduped),
                            "reference_count": 1,
                        }
                        discovered_keys.add(key)
                st.session_state.auto_assets = []

        # Assets that already have a ranking are skipped, not recomputed. Before rankings
        # persisted this was moot (a fresh session had none), but now clicking this button with
        # three restored assets and one new one would re-pay up to OCR_TOP_K VLM calls for each
        # of the three -- real money, for a result already on disk. Deliberate recomputation is
        # what Step 5's per-asset "Re-rank" button is for.
        manual_assets = [
            a for a in st.session_state.assets
            if (a["brand"], a["asset"]) not in discovered_keys
            and (a["brand"], a["asset"]) not in st.session_state.rankings
        ]
        already_ranked = len(st.session_state.assets) - len(manual_assets) - len(discovered_keys)
        if already_ranked > 0:
            st.caption(
                f"Skipping {already_ranked} asset(s) that already have a ranking — use "
                "**Re-rank** under an asset in Step 5 to recompute one."
            )
        if manual_assets and any(a["asset"] in CATEGORIES for a in manual_assets) and not vlm_is_configured():
            st.info(
                "`ANTHROPIC_API_KEY` isn't set, so manually registered assets skip the vision-model "
                "brand/category confirmation — candidates are CLIP+OCR+text-search only, which is "
                "more likely to include a wrong-brand or wrong-category false positive to reject by eye."
            )

        for a in manual_assets:
            key = (a["brand"], a["asset"])
            status.caption(f"Ranking for {a['brand']} — {a['asset']}...")
            deduped, stats = rank_one_asset(
                a, sharp_frames, embedding_cache_dir, ocr_cache_dir,
                on_embed_progress, on_ocr_progress, on_text_progress, on_manual_vlm_progress,
            )
            rankings[key] = deduped
            ranking_stats[key] = stats
        st.session_state.rankings = rankings
        st.session_state.ranking_stats = ranking_stats
        progress.empty()
        status.empty()
        st.success("Ranking complete.")
        if discovered_keys:
            # Step 2 (rendered earlier in this same script run) already drew the pending
            # auto_assets list before this handler cleared it -- rerun once so it reflects
            # the newly materialized (brand, category) entries instead of showing stale
            # "pending category discovery" rows for another interaction cycle.
            st.rerun()

# Deliberately outside the guards above: a ranking restored from a previous session is
# self-contained (each RankedFrame carries its own frame path and timestamp), so having the
# sampled frames loaded into memory is a prerequisite for *computing* a ranking, not for
# displaying one.
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
        if stats.get("text_matches"):
            funnel += f" → +{stats['text_matches']} found by direct brand-name text search"
        if stats.get("vlm_classified") is not None:
            funnel += f" → {stats['vlm_classified']} classified by VLM"
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
        # Unlike the restored ranking displayed in Step 4, re-ranking recomputes from scratch,
        # so it genuinely needs the sampled frames in memory -- which a session resumed from
        # saved progress doesn't have until the analyst loads them.
        can_rerank = st.session_state.sampled_frames is not None
        if rc1.button(
            "Re-rank using confirmed example(s)", key=f"rerank_{key_str}", disabled=not can_rerank,
        ):
            cache_key = file_cache_key(st.session_state.video_path)
            embedding_cache_dir = CACHE_DIR / cache_key / f"interval_{interval}" / "embeddings"
            ocr_cache_dir = CACHE_DIR / cache_key / f"interval_{interval}" / "ocr_text"
            sharp_frames = filter_sharp(st.session_state.sampled_frames)
            bar = st.progress(0.0)
            status = st.empty()

            def on_embed_progress(done: int, total: int, _status=status, _bar=bar) -> None:
                _bar.progress(done / total if total else 1.0)
                _status.caption(f"Embedding frames: {done:,}/{total:,}")

            def on_ocr_progress(done: int, total: int, _status=status, _bar=bar) -> None:
                _bar.progress(done / total if total else 1.0)
                _status.caption(f"Verifying candidates with OCR: {done}/{total}")

            def on_text_progress(done: int, total: int, _status=status, _bar=bar) -> None:
                _bar.progress(done / total if total else 1.0)
                _status.caption(f"Searching frame text for brand name: {done:,}/{total:,}")

            def on_vlm_progress(done: int, total: int, _status=status, _bar=bar) -> None:
                _bar.progress(done / total if total else 1.0)
                _status.caption(f"Confirming candidates with VLM: {done}/{total}")

            deduped, stats = rank_one_asset(
                a, sharp_frames, embedding_cache_dir, ocr_cache_dir,
                on_embed_progress, on_ocr_progress, on_text_progress, on_vlm_progress,
            )
            st.session_state.rankings[key] = deduped
            st.session_state.ranking_stats[key] = stats
            bar.empty()
            status.empty()
            st.rerun()
        rc2.caption(
            "Re-embeds using the original logo plus every confirmed example approved so far "
            "for this asset — surfaces appearances a clean-logo-only ranking missed."
            + ("" if can_rerank else " Load the cached frames in Step 3 to enable this.")
        )

    approved_indices = {e["frame_index"] for e in approved}
    remaining = [rf for rf in ranked if rf.frame.frame_index not in approved_indices]
    if not remaining:
        st.info("Every surfaced candidate for this asset is already approved.")
        continue

    # Step through candidates with Prev/Next rather than a dropdown: reviewing means looking at
    # each one in ranked order, and the position is clamped rather than reset because `remaining`
    # shrinks by one on every approval -- an index pointing at the last candidate would otherwise
    # go out of range on the next rerun.
    idx_key = f"cand_idx_{key_str}"
    choice = min(st.session_state.get(idx_key, 0), len(remaining) - 1)
    st.session_state[idx_key] = choice

    nav_prev, nav_next, nav_label = st.columns([1, 1, 6])
    if nav_prev.button("← Previous", key=f"prev_{key_str}", disabled=choice == 0):
        st.session_state[idx_key] = choice - 1
        st.rerun()
    if nav_next.button("Next →", key=f"next_{key_str}", disabled=choice >= len(remaining) - 1):
        st.session_state[idx_key] = choice + 1
        st.rerun()

    candidate = remaining[choice]
    nav_label.caption(
        f"Candidate **{choice + 1} of {len(remaining)}** — {fmt_time(candidate.frame.timestamp_sec)} "
        f"· score {candidate.score:.3f}"
    )
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

# --- Save progress --------------------------------------------------------------
# Written here at the end of the script rather than inside each mutating handler: Streamlit
# reruns top to bottom on every interaction, so this one call site always observes the final
# state of whatever the analyst just did, and no future handler can forget to call it.
if st.session_state.progress_path:
    save_progress(
        st.session_state.progress_path,
        assets=st.session_state.assets,
        auto_assets=st.session_state.auto_assets,
        rankings=st.session_state.rankings,
        ranking_stats=st.session_state.ranking_stats,
        approvals=st.session_state.approvals,
    )
