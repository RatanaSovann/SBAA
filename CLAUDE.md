# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SBAA (Sport Brand Annotation Assistant): given a sports match video and reference images for
brand assets (one image per brand/asset), retrieves and ranks likely on-screen appearances, and
lets an analyst draw a bounding box and approve the best two examples per asset. Replaces a
manual frame-by-frame spreadsheet workflow.

The output is a lightweight brief, not a full audit record: the two approved example frames
per asset, their frame numbers, and the asset details. That brief is handed off to the ML team,
who run the actual brand-exposure-value scan downstream using it. SBAA's job stops at producing
those two examples fast and correctly (non-hallucinated) — it does not need to retain rejected
candidates, scores, or a decision trail; that reproducibility burden belongs to a different
system, if anything ever needs it.

## Hardware constraint (drives most design decisions)

This machine has **no usable GPU** — AMD integrated graphics with 2GB VRAM, no CUDA, no ROCm
support on Windows for this chip. Treat everything as CPU-only. This is why the retrieval
design is two-stage (cheap coarse pass before any expensive model runs) rather than running a
detector over every frame.

## Commands

```bash
# one-time setup
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# run the app
./.venv/Scripts/python.exe -m streamlit run app.py

# compile-check after edits (no test suite exists yet)
./.venv/Scripts/python.exe -m py_compile app.py src/sbaa/*.py
```

No test framework is set up yet — validate changes to `src/sbaa/` by running functions
directly against a real file in `sample_data/video/` (see the module docstrings for why: the
decode-cost tradeoffs in `video.py` only show up against real footage, not synthetic input).

## Architecture

**Video input is a local-folder picker, not a browser upload.** Match videos run several GB;
Streamlit's `st.file_uploader` buffers the whole file in memory and caps at 200MB by default,
which doesn't work on a 16GB machine. `app.py` instead lists files already dropped into
`sample_data/video/` and reads them directly by path. Reference images (KB-sized) still go
through the browser uploader into `data/uploads/assets/<brand>/<asset>/`.

**Frame sampling picks between two decode strategies at runtime** (`src/sbaa/video.py`):
- *Sequential*: decode every frame in order, keep the ones on the interval. Fixed cost —
  proportional to the whole video's frame count, independent of the sampling interval.
- *Seek*: jump to each target frame with `cv2.set(CAP_PROP_POS_FRAMES)`. Cost is proportional
  to the number of samples, but each jump decodes forward from the nearest keyframe, so it's
  not free.

  Neither strategy wins for every interval, and the crossover point depends on the specific
  file's codec/keyframe spacing — `calibrate()` benchmarks both costs against the actual video
  (~1-2s of work, cached in `st.session_state` per video) and `plan_sampling()` picks whichever
  is cheaper for the requested interval. The chosen strategy and an ETA are surfaced in the UI
  rather than hidden, since on this hardware the difference is tens of minutes.

**Sampled frames are written to disk, never held in memory as a batch.** A full match sampled
every couple of seconds is thousands of frames; keeping those as raw arrays at once doesn't
fit this machine's RAM budget. Cache layout: `data/cache/<file-hash>/interval_<N>/frame_<idx>.jpg`,
where the hash comes from `file_cache_key()` (file size + hash of the first 1MB, not the whole
multi-GB file).

**Streamlit's rerun model**: every widget interaction reruns the whole script top to bottom.
State that must survive a rerun (selected video, calibration results, registered brand assets,
sampled frames) lives in `st.session_state`, initialized once near the top of `app.py`.

## Planned pipeline (milestones 1, 2, 4, 5, 6 done; milestone 3 deferred — see below)

Retrieval is CLIP-embedding similarity ranking candidate frames per brand asset (`src/sbaa/retrieval.py`),
narrowed further by a blur filter (`src/sbaa/video.py`) and, for assets whose reference image has
legible text, an OCR verification pass (`src/sbaa/ocr.py`) — CLIP matches a sign's color/layout more
than the text on it, so OCR catches false positives CLIP can't. All three stages are cheap enough to
run on every sampled frame's shortlist; this is what "coarse pass" means on this hardware.

**Reference image domain gap.** CLIP ranks noticeably worse when the reference is a clean/vector
logo than when it's a photo-style crop that looks like what a broadcast camera would actually see
— lighting, angle, and JPEG artifacting all matter to the embedding. Empirically, for wordmark/text
assets a flat logo crop on a plain white background ranks much better than a transparent vector/SVG
export of the same mark (surfaced as a UI hint in `app.py` Step 2). This is the same gap a
cold-start asset (no real match footage to crop from yet) always starts from.

**Confirm-and-re-rank closes the gap using the analyst's own approvals, not a bigger model.**
`rank_frames()` accepts either a single reference path or a list; with a list, a frame's score is
the *max* cosine similarity across all references, not an average (averaging a clean logo with a
lit, angled broadcast crop blurs into a vector resembling neither well). `app.py`'s
`gather_reference_paths()` builds that list as `[original uploaded logo] + [any confirmed_*.jpg
files in the asset's folder]`, so every ranking call — the first "Rank frames" click and any later
"Re-rank" click in Step 5 — asks the same question with no separate multi-ref code path. Approving
a candidate in Step 5 crops it to the analyst's box and saves it as
`data/uploads/assets/<brand>/<asset>/confirmed_<frame_index>.jpg`, right alongside the original
reference image (covered by the same "gitignored, safe to delete and regenerate" rule as the rest
of `data/uploads/`) — so a later match video registered against the same brand/asset starts with a
real in-domain reference already available, not just the clean logo. Live-tested on the Hyundai
LED-screen asset: re-ranking after one approval surfaced a genuine second appearance the
clean-logo-only pass had missed entirely. Not universal, though — tested against Coca-Cola's
reference (already closer to broadcast style to begin with, per the white-background finding
above), a confirmed crop changed nothing, because the original logo's similarity already dominated
the max for every candidate. The feature helps in proportion to how large the domain gap actually
is for a given asset's reference image, and is a no-op (not a regression) when it isn't.

**An open-vocabulary detector stage was investigated and deferred, not built.** The original plan
(region proposals from an open-vocab detector re-scoring the top-ranked frames — first scoped for
YOLO-World, then YOLOE) was tested against real match footage across three model scales, text
prompts, and image-based visual prompts. Findings: it never once detected sports perimeter/sponsor
boards across any prompt or scale; it detected a genuine trophy and clearly-visible jersey brand
logos (Adidas three-stripe) in zero of the frames tested despite both being unmistakably present;
its one real success (a gold medal at 0.71 confidence) came only from prompt-free mode using its own
built-in vocabulary, not custom text prompts, and medals don't carry brand marks anyway, so that win
doesn't serve brand-exposure detection. Conclusion: not a reliable verification signal for any of
this project's actual checklist categories (signage, trophy, jersey sponsor) as currently testable —
CLIP+OCR alone is what's shipped and trusted. Revisit only with a more targeted approach (e.g. a
fine-tuned detector) if a real need justifies the investigation cost again.

Revised milestones, in order: (1) ingest & sample — done; (2) coarse CLIP ranking + blur filter +
OCR verification — done; (3) ~~open-vocab detector region proposals~~ — deferred, see above; (4)
de-duplication of near-identical/adjacent candidates — done, by timestamp proximity
(`deduplicate_by_time()` in `src/sbaa/retrieval.py`) directly on the CLIP+OCR-confirmed list
(frames within `DEDUP_MIN_GAP_SEC` of an already-kept one are the same appearance), no
detector/bounding-box dependency needed; (5) analyst review UI — done (`app.py` Step 5): pick up
to two per asset, with a hand-drawn bounding box (`streamlit-cropper`) at approval time, not an
automated one — milestone 3 being deferred means there's no model to propose a box, so the
analyst draws it themselves in the same motion as confirming the candidate; (6) export the brief —
done (`src/sbaa/export.py`, `app.py` Step 6): an Excel workbook, one sheet per brand, one row per
approved example with frame number, timestamp, and the frame image with the analyst's box burned
in — matching the shape of the manual spreadsheet process this replaces, so there's no relearning
on the receiving end. No audit trail of rejected candidates or decision history; that's
explicitly out of scope.

## Data handling

- `sample_data/video/` is gitignored — match footage is large and typically not
  redistributable broadcast content. Never commit anything here.
- `sample_data/reference_images/` is tracked — small, one folder per brand/asset.
- `data/uploads/` and `data/cache/` are gitignored working state, safe to delete and
  regenerate.
