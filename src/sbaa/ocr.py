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
