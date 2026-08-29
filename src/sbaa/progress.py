"""Persist an analyst's in-progress work so a session can be closed and picked up later.

Everything else this app writes under `data/` is derived state that can be thrown away and
rebuilt from the video plus the reference images -- CLAUDE.md's data-handling rule says exactly
that. Approvals are the one exception: which two frames an analyst picked per asset, and the box
they drew on each, exist nowhere else. Losing them means redoing by hand the review this whole
app was built to speed up. Rankings are regenerable in principle, but a full pass costs ~55
minutes of OCR plus paid VLM calls per video, so they ride along rather than being recomputed
every time the analyst reopens the app.

Progress is keyed by (video cache key, sampling interval) -- the same pair that identifies a
frame-cache directory -- so one progress file always describes exactly one set of sampled frames.
That's what makes switching the interval safe with no explicit "this is now stale" logic
anywhere: a different interval simply reads a different file, and the frames a restored ranking
points at are by construction the frames that ranking was computed against.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .retrieval import RankedFrame
from .video import FrameMeta


def progress_path(progress_dir: str | Path, cache_key: str, interval: float) -> Path:
    """Where the progress for one video+interval lives, mirroring the frame-cache layout."""
    return Path(progress_dir) / cache_key / f"interval_{interval}.json"


def _encode_keyed(mapping: dict) -> list[dict]:
    """Flatten a `(brand, asset)`-keyed dict into a JSON-representable list.

    Tuples aren't valid JSON object keys, and joining them into one string would break on any
    brand or asset name containing the separator -- so both halves are stored as named fields.
    """
    return [{"brand": brand, "asset": asset, "value": value} for (brand, asset), value in mapping.items()]


def _decode_keyed(entries: list[dict]) -> dict:
    return {(e["brand"], e["asset"]): e["value"] for e in entries}


def _encode_ranked(ranked: list[RankedFrame]) -> list[dict]:
    return [{"score": float(rf.score), "frame": asdict(rf.frame)} for rf in ranked]


def _decode_ranked(entries: list[dict]) -> list[RankedFrame]:
    return [RankedFrame(frame=FrameMeta(**e["frame"]), score=e["score"]) for e in entries]


def _encode_approval(entry: dict) -> dict:
    # st_cropper returns numpy integers in its box dict, which json can't serialize directly.
    return {**entry, "bbox": {k: int(v) for k, v in entry["bbox"].items()}}


def save_progress(
    path: str | Path,
    *,
    assets: list[dict],
    auto_assets: list[dict],
    rankings: dict,
    ranking_stats: dict,
    approvals: dict,
) -> None:
    """Write the analyst's current state. Overwrites; there's no history to keep."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "assets": assets,
        "auto_assets": auto_assets,
        "rankings": _encode_keyed({k: _encode_ranked(v) for k, v in rankings.items()}),
        "ranking_stats": _encode_keyed(ranking_stats),
        "approvals": _encode_keyed(
            {k: [_encode_approval(e) for e in v] for k, v in approvals.items()}
        ),
    }
    path.write_text(json.dumps(payload, indent=2))


def load_progress(path: str | Path) -> dict | None:
    """Read back a saved state, or None if this video+interval has no saved progress yet.

    Returns plain values ready to drop straight into `st.session_state` -- the caller shouldn't
    need to know anything about the on-disk shape.
    """
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return {
        "assets": payload["assets"],
        "auto_assets": payload["auto_assets"],
        "rankings": {k: _decode_ranked(v) for k, v in _decode_keyed(payload["rankings"]).items()},
        "ranking_stats": _decode_keyed(payload["ranking_stats"]),
        "approvals": _decode_keyed(payload["approvals"]),
    }
