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

**Auto-discovering categories from a logo, shipped after milestones 1-6.** The user asked whether
uploading just a brand logo could surface *every* placement the brand appears in (LED board,
trophy, jersey, ...) instead of the analyst manually registering one `(brand, asset)` pair per
placement they already know about. A CLIP zero-shot text-prompt classifier was tried first
(`"a sports perimeter LED advertising board"`, `"a trophy"`, `"a player's jersey with a sponsor
logo"`) and rejected by a spike against real match frames: 1/4 correct, with every frame's three
category scores clustered within ~0.02-0.06 of each other — not a usable signal, the same shape
of failure as the YOLOE deferral above, for the likely same reason (these categories are poorly
represented in what these models were trained on).


**First shipped design had no zero-shot cold start (superseded — see below).** The initial
`src/sbaa/category.py` defined a fixed taxonomy (`CATEGORIES = ["LED Board", "Trophy", "Jersey"]`)
and classified by pooling analyst-confirmed crops across every brand
(`data/uploads/assets/*/<category>/confirmed_*.jpg`), scoring a candidate by max-similarity
against a category's pooled crops only — the same combination logic `rank_frames` uses for
multiple brand references. A category with zero pooled crops wasn't offered as a target at all,
which meant bootstrapping one ("Step 0") required an analyst to manually register, rank, review,
and approve one brand under that exact category name *before* auto-discovery could use it at
all — i.e. crop an LED board out of the match by hand first, the exact manual step this feature
was meant to remove. Kept only long enough to prove the rest of the pipeline (ranking → OCR →
per-category bucketing → materializing into `st.session_state.assets`) end to end; live-verified
against the real match video (seeding "LED Board" with one approved Hyundai crop let Coca-Cola and
Rexona auto-discover their own "LED Board" appearances from their logos alone).

**Shipped design: VLM classification, no seed crop required at all.** The CLIP-embedding pooling
above is gone. `src/sbaa/vlm.py` sends each OCR-confirmed candidate frame to a vision-language
model (Claude Haiku, via the Anthropic API) with the fixed category list and asks which one (if
any) the frame clearly shows — open-ended visual reasoning over the image, not an embedding
compared to a fixed reference point, which is the specific thing that let CLIP's zero-shot attempt
fail above but doesn't require this one to fail the same way. `vlm.classify_frame()` returns
`None` rather than force-fitting when the model finds no clear match, same "don't guess" contract
the old pooled-crop `classify()` had. `src/sbaa/category.py` now holds only the `CATEGORIES`
constant, kept as the shared taxonomy both Step 2's manual
selector and the auto-discover pipeline key off (still a fixed Python list, not a managed UI —
extend by editing the list).

This is the project's first external dependency: every classification call is a real network
request to the Anthropic API, a deliberate departure from the fully-local design used everywhere
else (driven by the no-GPU hardware constraint in this file's first section). Requires
`ANTHROPIC_API_KEY` in the environment — the auto-discover form warns and no-ops without it,
leaving pending brands queued for retry rather than failing the whole ranking pass. Candidates
sent to the VLM are still capped at `OCR_TOP_K` (50) even when OCR verification is a no-op (a
graphic-only logo with no legible text), which would otherwise send every sharp sampled frame —
potentially thousands — to a paid API call; this is the same "cheap pass narrows before the
expensive stage runs" discipline the hardware-constraint section already applies everywhere else
in this pipeline, just newly load-bearing now that the expensive stage costs real money per call
instead of being free local compute. The manual registration flow (Step 2) is unaffected — an
analyst who already knows an asset's category still types/selects it directly, no classification
involved — so Step 2's category selector now exists purely to keep naming consistent for the
export, not to gate a pooling mechanism that no longer exists.

**Spike-tested against real cached match frames, not assumed to work.** First pass, with bare
category names in the prompt (`"LED Board"`, `"Trophy"`, `"Jersey"`), got real trophy and jersey
appearances right but produced false positives on `"LED Board"`: a stylized promotional bumper
(dancers on a neon-lit car) and an opening-ceremony graphic were both misclassified as an LED
board, apparently on bright/screen-like color composition alone. Fixed by grounding each category
in the prompt with a short description (`_CATEGORY_DESCRIPTIONS` in `vlm.py`) that explicitly
excludes broadcast graphics and promotional clips for the board category. Re-tested against the
same two false positives (now correctly `None`) plus the earlier true positives (still correctly
classified) and a genuine perimeter advertising board found further down a real ranking (correctly
`LED Board`) — all five held. Notably, this succeeds on exactly the categories (trophy, perimeter
board) where the deferred YOLOE open-vocab detector failed completely across every prompt and
scale tested (see above) — real evidence that open-ended VLM visual reasoning is a materially
different operation from both CLIP's embedding similarity and a region-proposal detector, not
just a bigger version of either.

**End-to-end verified on a full match, from a logo alone.** Coca-Cola, registered with nothing
but its logo and no seed crop, auto-discovered into `LED Board` and surfaced 23 distinct
appearances: 6,334 sampled → 4,602 sharp → 4,602 CLIP-ranked → top 50 OCR-checked → 2 confirmed
→ +90 found by direct brand-name text search → 50 VLM-classified → 23 deduped. Five candidates
spot-checked by eye across the full range of the list (highest-scoring through weakest) were all
genuine Coca-Cola perimeter boards, no false positives. Note the funnel's shape: OCR-vs-reference
confirmed only 2, while the brand-name text search contributed 90 — on a wordmark brand the text
search, not CLIP, is doing most of the recall work.

**Category-only classification wasn't enough — real usage surfaced brand-agnostic false
positives.** The five-frame spike above validated "does this frame show a board/trophy/jersey",
but a live browser walkthrough on Hyundai found the actual failure mode: a category question with
no brand context answers yes constantly on real match footage regardless of whether the searched
brand appears at all — a referee's jersey, fans in team kit, and the small FIFA trophy icon baked
into the broadcast's persistent scoreboard graphic all satisfy "is there a jersey/trophy here."
Fixed by sending the brand's reference logo alongside the candidate frame and requiring the model
to confirm that brand's branding is actually visible before answering with anything but `None`
(`classify_frame()`'s new `brand`/`reference_image_path` params). Re-verified: the exact watermark
frame from the live walkthrough, plus a crowd-jersey frame, a real trophy with no Hyundai branding
on it, and a real jersey with no Hyundai branding on it, all now correctly return `None`.

This only closes the gap for brands where OCR is a no-op (graphic-only logos, like Hyundai's --
see below). For brands with legible reference text, `verify_with_ocr()` already rejects
wrong-brand candidates before they ever reach the VLM (confirmed: a real perimeter board showing
VISA's branding was correctly OCR-rejected when ranked against Coca-Cola's logo, whose reference
has legible "COCACOLA" text) — the VLM brand check is a backstop for the case OCR structurally
can't cover, not a replacement for it.

**The brand check is still only as good as the reference image, and Hyundai's is a bad one.**
Broader testing surfaced two more false "LED Board" hits for Hyundai — both broadcast bumpers
with no board and no Hyundai anywhere in frame. The cause traces to the registered reference
image itself: a glossy 3D chrome car-badge emblem on white, not a flat crop resembling how
sponsor branding actually appears on painted pitch-side signage. This is the same reference-image
domain gap already documented above for CLIP ("a flat crop on a plain white background ranks
much better... than a transparent vector/SVG export") — it turns out to break VLM brand-matching
for the same underlying reason: the reference and the real on-screen appearance don't visually
resemble each other closely enough for either a CLIP embedding or a VLM's visual comparison to
reliably connect them.

**A different embedding model wouldn't fix this — it's a property of embedding similarity
itself, not a CLIP weakness.** Considered swapping in a bigger/newer checkpoint (a larger CLIP
variant, SigLIP, EVA-CLIP). Rejected without needing a spike: contrastive embeddings cluster
images by low/mid-level visual style (glossy 3D render vs. flat painted signage, vector vs.
photo) about as much as by abstract "same brand" identity, so a stylistically extreme gap like
Hyundai's would survive a bigger model, just somewhat narrowed. A heavier model also directly
fights the hardware constraint that specifically justified picking the smallest common CLIP
checkpoint in the first place (CPU-only, coarse pass over thousands of frames needs to stay
cheap) — the wrong direction to spend the tradeoff on for a gap this large.

**The actual fix: search for the brand's name in frame text directly, bypassing both CLIP and
the reference image.** Real sponsor boards are almost always legible text. `ocr.py` gained
`matches_brand_name()` (fuzzy-matches OCR text against the brand name the analyst *typed*, not
the reference image's own OCR content — works whether the reference has legible text or not) and
`detect_text_cached()` (per-frame disk cache, since raw text extraction is brand-independent: the
first brand searched against a video's frames pays the OCR cost, every brand after that is a fast
in-memory comparison against already-cached text). `retrieval.search_by_text()` scans every sharp
frame with these and returns hits at a fixed score of 1.0; `merge_candidates()` unions them with
the existing CLIP+OCR-verified list, deduped by frame, so text hits always survive the
`OCR_TOP_K` cut ahead of weaker CLIP-only candidates. Wired into both the manual per-asset path
(`rank_one_asset()`) and the auto-discover path in `app.py` Step 4.

Verified end-to-end against the exact failure this was built for: the real Hyundai perimeter
board (`frame_00563500.jpg`, also confirmed visually — it shows "COCA-COLA" and "HYUNDAI" text
side by side) ranks 298th of 6,334 in Hyundai's own CLIP ranking (score 0.471, far outside any
practical top-K), so the existing pipeline could never have found it for Hyundai no matter how
the VLM prompt was tuned. Direct OCR on that frame reads `['CECACOLE', 'QHYUNOAI', ...]` --
garbled by real broadcast text quality, but `matches_brand_name()`'s fuzzy threshold still
correctly matches both "Hyundai" and "Coca Cola" against it. `search_by_text()` found this frame
(plus its immediate neighbors) directly, and `merge_candidates()` correctly sorted it to the top
of the combined candidate list.

This only helps textual signage — it's a real gap-closer for boards and most jerseys (brand names
are usually printed somewhere), not for a placement with no text anywhere (a pure graphic mark).
That residual case still has no fix but a reference image that actually resembles the real
appearance, or a first analyst-approved crop.

**Real cost, surfaced not hidden.** Unlike CLIP ranking or OCR-vs-reference (both scoped to a
top-K shortlist), this scans *every* sharp frame, since restricting it to CLIP's top-K would
defeat the entire point (that's exactly what buries the genuine candidate today). Measured on the
full match video (6,334 sampled frames): ~26s per 50 frames uncached, ~55 minutes for a full
pass. That cost is paid once per video, not once per brand — `detect_text_cached()`'s disk cache
means every subsequent brand searched against the same sampled frames is near-instant. Progress
is surfaced via the same caption/progress-bar pattern already used for sampling and embedding,
consistent with this project's standing "surfaced not hidden" rule for CPU-only wait times.

**Manual registration had no VLM confirmation at all -- a real gap, not a hypothetical one.**
Live use on Adidas (a graphic-only mark: three stripes, no legible text on either its Jersey or
LED Board reference crop) surfaced it directly: `rank_one_asset()` (the manual Step 2 path) only
ever ran CLIP + OCR-vs-reference + brand-name text search, with no VLM step -- that only existed
on the auto-discover path. For a brand whose reference has no legible text, OCR-vs-reference is
a no-op, so CLIP's color/layout confusion (a Visa perimeter board outranking every genuine Adidas
candidate for the "LED Board" asset, confirmed by direct inspection of the top-ranked frames) and
`search_by_text()`'s brand-only matching (which found real "ADIDAS" text on a stadium jumbotron
award graphic and surfaced it identically under *both* the Jersey and LED Board assets, since
text search has no category awareness) both went straight to the analyst unfiltered. Fixed by
extending `rank_one_asset()` to run the same `classify_frame()` check auto-discover already used,
capped at `OCR_TOP_K` for cost, checking the model's category pick against the asset's own name
-- skipped for a free-text "Custom…" asset name outside `CATEGORIES`, since there's no grounding
description to check evidence against. Verified against the exact live cache: before the fix,
Adidas Jersey and Adidas LED Board shared several identical top candidates (the same jumbotron
graphic in both); after, the two asset's top-12 diverged as expected, and a genuine Adidas-jersey
crowd shot (a CLIP-only match, not a text hit, so purely on-graphic-mark evidence) survived the
filter and confirmed the graphic-mark case can work, not just the text-legible case.

**That same live debugging session found a real, silent bug affecting every VLM classification
call made so far this session, not just Adidas's.** `classify_frame()`'s evidence-forcing prompt
(see above) asks the model to state its visual evidence before answering, but was capped at
`max_tokens=60` -- too small: the evidence phrase alone routinely runs 60-90 tokens (e.g. `"tap
in with VISA" text visible on the LED board perimeter, though this is VISA branding, not
Adidas...`), so the response was silently truncated *before it ever reached the "Answer:" line*.
The old parsing fell back to scanning the entire (truncated) raw text for a category name when no
"Answer:" line was found -- which means it was matching category words mentioned while the model
was explaining they *didn't* apply (`"...I cannot clearly read Adidas text on any physical LED
boards..."` contains the literal substring `"LED board"`), silently inverting a correct rejection
into a false positive. Confirmed via a raw un-parsed API call on the exact frame that had
misfired: the model's real answer was `"None of these"`, entirely absent from the truncated
60-token response the app actually received. Fixed by raising `max_tokens` to 200 and making the
parser strict -- no "Answer:" line found now returns `None` outright instead of scanning the raw
text, the same "don't guess" contract the rest of this module already follows. This bug predates
today's Adidas work; every VLM classification made earlier this session (Hyundai included) was
running under the same truncation risk, silently.

**Residual limitation, reported honestly rather than chased further: the VLM still hallucinates
evidence on some frames, and there's no temperature control available to reduce it.** After both
fixes above, two Visa perimeter-board frames were still confirmed as Adidas "LED Board" -- not a
parsing bug this time; the raw response genuinely states invented evidence (`"'adidas' text
visible on the blue LED board..."` on a frame that, on direct visual inspection, shows no such
text). The installed `anthropic` SDK for this project (1.1.0) doesn't expose a `temperature`
parameter on `messages.create()` at all, so the usual mitigation isn't available here. This also
explains an earlier-observed flip-flop (`LED Board`/`Jersey` alternating across identical repeated
calls on one frame that genuinely shows both a real jersey and a stadium screen) that survived
the `max_tokens` fix -- real sampling variance on a genuinely dual-category frame, not a bug.
Given no automated stage in this pipeline has ever claimed to be ground truth (see this file's
opening section on what SBAA's output actually is), the Step 5 analyst review -- eyeball each
candidate, approve at most two -- is the intended and sufficient backstop for this residual rate,
not a gap to close with more prompt iteration. Separately, `_CATEGORY_DESCRIPTIONS["LED Board"]`
was hardened to explicitly exclude the stadium's own jumbotron/scoreboard screen (it was
misclassifying a screen showing an award-ceremony graphic as a perimeter board on the strength of
legible sponsor text alone) -- verified against that exact frame plus a regression check against
a genuine perimeter board, which stayed correctly classified.

**A missing category doesn't degrade gracefully — it silently mislabels.** Live use on Hisense
surfaced this: the VAR-review room's printed Hisense backdrop wall was classified `LED Board`.
Not a model failure — `CATEGORIES` had no bucket for a backdrop, so the model's only options
were the nearest of three or `None`, and a flat panel carrying a sponsor wordmark genuinely is
closest to "LED Board" among them. The forced-choice shape of the prompt means an absent
category reappears as a confident wrong answer in a neighbouring one, rather than as a visible
`None`. Fixed by adding `"Wall"` to `CATEGORIES` with a grounding description covering
press-conference/interview backdrops, mixed-zone boards and studio/VAR-room walls, plus an
explicit "not an indoor backdrop wall (use Wall for those)" exclusion on the `LED Board`
description. Verified: both VAR-room frames now classify `Wall`, while four genuine perimeter
boards (Hisense, Visa, two Coca-Cola) and two real jerseys all held their previous labels.
Because Step 2's selector and the auto-discover caption both render from `CATEGORIES` directly,
extending the list is genuinely a one-line change — worth checking for the same failure whenever
a new placement type shows up in real footage.

**The taxonomy is analyst-editable from the UI, and a category is a name *plus* a description.**
Since a missing bucket silently mislabels (above), waiting on a code change to add one is the
wrong shape for something discovered mid-review. `category.py` now owns both halves —
`BUILTIN_CATEGORIES` maps each built-in name to its grounding description, and anything the
analyst adds in Step 2's "Placement categories" expander is persisted to `data/categories.json`
and merged on top (built-ins first, so the common cases stay at the top of the dropdown). The
descriptions moved here *from* `vlm.py`, which used to hold them in a private
`_CATEGORY_DESCRIPTIONS` dict and fall back to the bare category name for anything missing.
That fallback is now gone and `classify_frame()` takes a `dict[str, str]` instead of a
`list[str]`: a name-only category is precisely the configuration that was spike-tested and
failed, so the type makes it unrepresentable rather than merely discouraged. One module owns the
taxonomy; `vlm.py` owns the API call.

Guards worth keeping: built-ins can't be removed (assets are keyed on those names), duplicates
and blank fields are rejected, and removing a custom category is non-destructive — an asset
already registered under it keeps working, since Steps 5 and 6 never consult the taxonomy; the
name just stops being offered and stops being a classification target. Verified end to end
through the real UI: adding "Pitch Decal" took the header 4→5, appeared in the auto-discover
caption, reached the model, and left four genuine LED Board / Wall / Jersey classifications
unchanged; removing it through the UI restored the built-in-only taxonomy.

## Session persistence

**The analyst's work survives closing the app** (`src/sbaa/progress.py`). Streamlit's
`st.session_state` is per-browser-session, so before this everything an analyst did — registered
assets, rankings, approvals — died with the tab. Approvals are the one thing in this project that
genuinely can't be regenerated: which two frames were picked per asset, and the box drawn on
each, exist nowhere else, and losing them means redoing by hand the exact review this app exists
to speed up. Rankings ride along too — regenerable in principle, but a full pass costs ~55
minutes of OCR plus paid VLM calls per video.

Progress is keyed by **(video cache key, sampling interval)**, mirroring the frame-cache layout,
so one progress file always describes exactly one set of sampled frames. That key is doing real
work: it's why switching interval needs no explicit staleness logic anywhere (a different
interval simply reads a different file), and a restored ranking is by construction pointing at
the frames it was computed against. Load happens in Step 3 once the interval is known, followed
by `st.rerun()` since Steps 1-3 have already rendered by then; save happens once at the very
bottom of the script, where Streamlit's top-to-bottom rerun model guarantees it observes the
final state of whatever the analyst just did — no handler can forget to call it.

Introducing this turned two pre-existing lines into data-loss bugs and required fixing them:
`run_sampling()` and the "Load cached frames" branch both used to clear
`rankings`/`ranking_stats`/`approvals`, which would now wipe the analyst's approvals *and then
auto-save the wiped state over the file*. Both clears are gone — sampling is deterministic, so
re-sampling the same video at the same interval reproduces the same frame indices and paths, and
the video+interval progress key is the real invalidation mechanism those clears were standing in
for. Step 4's candidate grid also moved outside the `sampled_frames is None` guard: a restored
ranking is self-contained (each `RankedFrame` carries its own frame path and timestamp), so
having the frames in memory is a prerequisite for *computing* a ranking, not for displaying one.
Step 5's "Re-rank" button does still need them, and is disabled with a hint until they're loaded.

It also changed what "Rank frames" should do. That handler used to build a fresh `rankings` dict
and assign it wholesale over `st.session_state.rankings`; with nothing persisted that was
harmless, but against restored state it would drop the ranking of any asset the pass didn't
recompute. It now seeds from the existing dict, and **skips assets that already have a ranking**
rather than recomputing them — otherwise adding one new asset to three restored ones would
silently re-spend up to `OCR_TOP_K` paid VLM calls on each of the three for a result already on
disk. Step 5's per-asset "Re-rank" button is the deliberate way to recompute one.

On-disk format notes: `(brand, asset)` tuple keys aren't valid JSON object keys, and joining
them with a separator would break on any brand name containing it, so both halves are stored as
named fields; `st_cropper` returns numpy integers in its box dict, which `json` can't serialize,
so bboxes are coerced to plain ints on save.

## Data handling

- `sample_data/video/` is gitignored — match footage is large and typically not
  redistributable broadcast content. Never commit anything here.
- `sample_data/reference_images/` is tracked — small, one folder per brand/asset.
- `data/uploads/` and `data/cache/` are gitignored working state, safe to delete and
  regenerate.
- `data/progress/` is gitignored too, but is the one directory here that is **not** safe to
  delete: it holds the analyst's approvals (see the session-persistence section below).
