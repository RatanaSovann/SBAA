"""OCR-based text verification for wordmark-style brand assets.

CLIP's coarse pass (see retrieval.py) matches broadcast frames by overall visual gestalt --
a sign's color and layout -- rather than by reading the specific brand name on it. Two
differently-branded perimeter boards with a similar color scheme can outscore a genuine
appearance of the asset being searched for.

For assets whose reference image contains legible text, OCR gives a much cheaper and more
precise second check than the stage-3 detector: does the candidate frame actually contain
that brand's name, not just a similarly-colored sign. It's applied only to the CLIP-ranked
shortlist, not every sampled frame, so it stays cheap -- the same "coarse pass narrows,
second pass confirms" structure as the rest of the pipeline.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Callable

_ocr = None


def _load_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    return _ocr


def _normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def detect_text(image_path: str | Path) -> list[str]:
    """Return the text lines found in an image, normalized (uppercase, alnum-only)."""
    ocr = _load_ocr()
    result, _ = ocr(str(image_path))
    if not result:
        return []
    return [t for line in result if (t := _normalize(line[1]))]


def _fuzzy_match(a: str, b: str, min_ratio: float) -> bool:
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= min_ratio


def matches_any(candidate_lines: list[str], target_lines: list[str], min_ratio: float = 0.7) -> bool:
    """True if any candidate text line fuzzy-matches any target line.

    Fuzzy rather than exact: OCR on small, motion-blurred broadcast text routinely drops
    or misreads a character or two (e.g. a logo icon read as a stray leading letter).
    """
    return any(_fuzzy_match(c, t, min_ratio) for c in candidate_lines for t in target_lines)


def matches_brand_name(candidate_lines: list[str], brand: str, min_ratio: float = 0.7) -> bool:
    """True if any detected text line names the brand directly.

    Unlike `matches_any` (which compares a candidate against the *reference image's* own OCR
    text -- a no-op when the reference is a graphic-only logo with nothing legible on it), this
    compares against the brand name the analyst actually typed. It works the same whether the
    registered reference is a photo, a stylized vector mark, or has no legible text at all --
    see `retrieval.search_by_text()` for why that independence matters.
    """
    return matches_any(candidate_lines, [_normalize(brand)], min_ratio=min_ratio)


def _text_cache_path(cache_dir: Path, image_path: Path) -> Path:
    key = hashlib.sha1(str(image_path.resolve()).encode()).hexdigest()[:16]
    return cache_dir / f"{key}.json"


def detect_text_cached(
    paths: list[str | Path],
    cache_dir: str | Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[str]]:
    """`detect_text()` for many frames, reusing a disk cache keyed by frame path.

    Text extraction is brand-independent -- the same detected lines on a frame get checked
    against every brand name searched for. Caching here means the first brand searched against
    a given sample of frames pays the full OCR cost, and every brand searched afterward against
    the same frames is a fast in-memory text comparison, not a re-run of OCR. Mirrors
    `retrieval.embed_images_cached()`'s per-file disk-cache pattern.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[str]] = {}
    pending: list[tuple[Path, Path]] = []
    for raw_path in paths:
        p = Path(raw_path)
        cpath = _text_cache_path(cache_dir, p)
        if cpath.exists():
            results[str(p)] = json.loads(cpath.read_text())
        else:
            pending.append((p, cpath))

    done = len(paths) - len(pending)
    if on_progress:
        on_progress(done, len(paths))

    for p, cpath in pending:
        lines = detect_text(p)
        cpath.write_text(json.dumps(lines))
        results[str(p)] = lines
        done += 1
        if on_progress:
            on_progress(done, len(paths))

    return results
