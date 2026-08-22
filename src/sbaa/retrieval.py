"""Coarse retrieval: rank sampled frames by CLIP similarity to a reference image.

Milestone 2 of the two-stage pipeline described in CLAUDE.md. This is the cheap pass that
narrows thousands of sampled frames down to the handful worth running the expensive stage-2
detector on later. CPU-only, so this uses the smallest commonly-used CLIP checkpoint
(ViT-B-32) rather than a larger one — accuracy is traded for speed here on purpose, since
stage 2 is what actually confirms a match.

Frame embeddings are cached to disk keyed by frame path: they don't depend on which brand
asset is being ranked, so a frame embedded once is reused across every asset's ranking pass
instead of being recomputed per asset.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import open_clip
import torch
from PIL import Image

from . import ocr
from .video import FrameMeta

# quickgelu variant: the "openai" checkpoint was trained with QuickGELU activations, and
# open_clip warns (silently degrading embedding quality) if that's not matched explicitly.
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"

_model = None
_preprocess = None


@dataclass
class RankedFrame:
    frame: FrameMeta
    score: float


def _load_model():
    global _model, _preprocess
    if _model is None:
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        model.eval()
        _model = model
        _preprocess = preprocess
    return _model, _preprocess


def embed_image(path: str | Path) -> np.ndarray:
    """Embed a single image (e.g. a brand-asset reference) and L2-normalize it."""
    model, preprocess = _load_model()
    image = Image.open(path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features /= features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).numpy()


def _cache_path(cache_dir: Path, image_path: Path) -> Path:
    key = hashlib.sha1(str(image_path.resolve()).encode()).hexdigest()[:16]
    return cache_dir / f"{key}.npy"


def embed_images_cached(
    paths: list[str | Path],
    cache_dir: str | Path,
    batch_size: int = 32,
    on_progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Embed a batch of images, reusing cached per-image embeddings on disk.

    Processes uncached images in chunks rather than one big batch — thousands of sampled
    frames at once would spike memory on a CPU-only machine with no VRAM headroom to fall
    back on.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model, preprocess = _load_model()

    results: list[np.ndarray | None] = [None] * len(paths)
    pending: list[tuple[int, Path, Path]] = []
    for i, p in enumerate(paths):
        p = Path(p)
        cpath = _cache_path(cache_dir, p)
        if cpath.exists():
            results[i] = np.load(cpath)
        else:
            pending.append((i, p, cpath))

    done = len(paths) - len(pending)
    if on_progress:
        on_progress(done, len(paths))

    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        tensors = torch.stack([preprocess(Image.open(p).convert("RGB")) for _, p, _ in chunk])
        with torch.no_grad():
            features = model.encode_image(tensors)
            features /= features.norm(dim=-1, keepdim=True)
        features = features.numpy()
        for (i, _, cpath), vec in zip(chunk, features):
            np.save(cpath, vec)
            results[i] = vec
        done += len(chunk)
        if on_progress:
            on_progress(done, len(paths))

    return np.stack(results)


def rank_frames(
    reference_image_path: str | Path,
    frames: list[FrameMeta],
    embedding_cache_dir: str | Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[RankedFrame]:
    """Rank sampled frames by cosine similarity to a brand-asset reference image."""
    ref_embedding = embed_image(reference_image_path)
    frame_embeddings = embed_images_cached(
        [f.file_path for f in frames], embedding_cache_dir, on_progress=on_progress,
    )
    scores = frame_embeddings @ ref_embedding
    order = np.argsort(-scores)
    return [RankedFrame(frame=frames[i], score=float(scores[i])) for i in order]


def verify_with_ocr(
    ranked: list[RankedFrame],
    reference_image_path: str | Path,
    top_k: int = 50,
    min_ratio: float = 0.7,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[RankedFrame]:
    """Filter a CLIP ranking down to candidates whose text OCR-matches the reference.

    CLIP matches a sign's color and layout, not the brand name printed on it (see this
    module's docstring), so a similarly-styled but wrong sign can outrank a genuine
    appearance. If the reference image has legible text, this re-checks the top-`top_k`
    CLIP candidates and drops any where OCR finds no match -- catching exactly the false
    positives CLIP can't distinguish, at a fraction of stage-3 detector cost.

    Assets with no legible reference text (a pure graphic logo) can't be checked this way;
    the CLIP ranking is returned unchanged rather than discarding every candidate.
    """
    target_lines = ocr.detect_text(reference_image_path)
    if not target_lines:
        return ranked

    shortlist = ranked[:top_k]
    verified = []
    for i, rf in enumerate(shortlist):
        candidate_lines = ocr.detect_text(rf.frame.file_path)
        if ocr.matches_any(candidate_lines, target_lines, min_ratio=min_ratio):
            verified.append(rf)
        if on_progress:
            on_progress(i + 1, len(shortlist))
    return verified
