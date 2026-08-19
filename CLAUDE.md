# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SBAA (Sport Brand Annotation Assistant): given a sports match video and reference images for
brand assets (one image per brand/asset), retrieves and ranks likely on-screen appearances,
proposes bounding boxes, and lets an analyst approve the best two examples per asset. Replaces
a manual frame-by-frame spreadsheet workflow. Exports CSV, an Excel review workbook with
embedded annotated images, and JSON audit files.

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

## Planned pipeline (only stage 1 exists so far)

The full design is two-stage retrieval: cheap CLIP-embedding similarity ranks candidate frames
per brand asset first; an open-vocabulary detector (YOLO-World) then runs region proposals
only on the top-ranked frames, re-scored against the reference image. Running the detector
over every frame of a 90-minute match on CPU is not viable — the coarse pass exists to bound
how much of the expensive stage 2 work is needed.

Milestones, in order (each depends on the last): (1) ingest & sample — done; (2) coarse CLIP
ranking of sampled frames per brand asset; (3) region proposals on top-ranked frames only; (4)
de-duplication of near-identical/adjacent detections; (5) analyst review UI (approve top 2 per
asset); (6) export to CSV, an Excel workbook with embedded annotated crops, and a JSON audit
trail of all candidates plus decisions.

## Data handling

- `sample_data/video/` is gitignored — match footage is large and typically not
  redistributable broadcast content. Never commit anything here.
- `sample_data/reference_images/` is tracked — small, one folder per brand/asset.
- `data/uploads/` and `data/cache/` are gitignored working state, safe to delete and
  regenerate.
