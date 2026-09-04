"""Build the self-contained documentation page.

Reads `sbaa-doc.src.html`, replaces each `{{img:filename}}` placeholder with a base64 data URI
of `assets/filename`, and writes `sbaa-doc.html`.

Everything is inlined rather than linked because the page is published as a Claude Artifact,
which serves a single HTML file -- a relative `assets/` path would 404 there. The same property
makes the built file portable: one file, opens anywhere, no folder next to it.

Non-ASCII characters are escaped to numeric entities so the page renders correctly even when a
server sends no charset header (this fixed real mojibake seen when previewing over http.server).

    python docs/build.py
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

DOCS = Path(__file__).parent
SRC = DOCS / "sbaa-doc.src.html"
OUT = DOCS / "sbaa-doc.html"
ASSETS = DOCS / "assets"

MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif"}


def _data_uri(match: re.Match) -> str:
    path = ASSETS / match.group(1)
    if not path.exists():
        raise FileNotFoundError(f"{path} is referenced by {SRC.name} but missing")
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:{MIME[path.suffix.lower()]};base64,{encoded}"


def main() -> None:
    source = SRC.read_text(encoding="utf-8")
    source = "".join(c if ord(c) < 128 else f"&#{ord(c)};" for c in source)
    built = re.sub(r"\{\{img:([^}]+)\}\}", _data_uri, source)
    OUT.write_text(built, encoding="utf-8")
    print(f"wrote {OUT.name} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
