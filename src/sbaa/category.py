"""The taxonomy of placement types a brand's appearances get sorted into.

A category is a **name plus a grounding description**, never a name alone. The description is
load-bearing, not documentation: bare category names were spike-tested and produced real false
positives (a neon promotional bumper classified as "LED Board" on bright-screen colour
composition alone), and the exclusion clauses added since -- "not the stadium's own jumbotron",
"not an indoor backdrop wall" -- are each traceable to a specific misclassification on real
footage. So a category added from the UI has to carry a description too; there is deliberately
no name-only path into this taxonomy.

The `Wall` category exists because of a second, subtler finding: with no bucket for a
press-conference/VAR-room backdrop, the model didn't abstain, it answered `LED Board` --
confidently, since a flat panel carrying a sponsor wordmark genuinely is the nearest of the
categories on offer. A forced-choice prompt turns a *missing* category into a wrong neighbouring
label rather than a visible `None`, which is exactly the failure this module lets an analyst fix
without a code change.

Built-ins are fixed in code; anything the analyst adds in the UI is persisted as JSON and merged
on top. Classification itself lives in `vlm.py`, which takes the merged mapping as an argument
rather than reaching for it -- this module owns the taxonomy, that one owns the API call.
"""

from __future__ import annotations

import json
from pathlib import Path

BUILTIN_CATEGORIES: dict[str, str] = {
    "LED Board": (
        "a physical LED advertising board mounted around the perimeter of the pitch or court, "
        "showing sponsor branding, actually present in the shot -- not the stadium's own "
        "jumbotron/scoreboard screen (even when that screen is itself showing a sponsor name or "
        "award graphic), not an indoor backdrop wall (use Wall for those), and not a broadcast "
        "graphic, animation, or promotional bumper"
    ),
    "Trophy": (
        "a physical trophy or medal actually present in the shot -- not a broadcast graphic, "
        "watermark, or scoreboard icon"
    ),
    "Jersey": "a player's jersey, clearly visible",
    "Wall": (
        "sponsor branding printed or mounted on a flat vertical backdrop behind people, rather "
        "than on pitch-side perimeter signage -- a press-conference or post-match interview "
        "backdrop, a mixed-zone board, or a studio/VAR-room wall"
    ),
}


def load_custom(path: str | Path) -> dict[str, str]:
    """Analyst-added categories, or an empty mapping if none have been added yet."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_categories(path: str | Path) -> dict[str, str]:
    """The full taxonomy: built-ins first, then anything added from the UI.

    Built-ins come first so the categories an analyst sees most often stay at the top of the
    dropdown, and because dict order is what the classification prompt lists them in.
    """
    return {**BUILTIN_CATEGORIES, **load_custom(path)}


def add_category(path: str | Path, name: str, description: str) -> None:
    """Persist a new category. Raises ValueError on a blank field or a name already in use."""
    name = name.strip()
    description = description.strip()
    if not name or not description:
        raise ValueError("A category needs both a name and a description.")
    if name in load_categories(path):
        raise ValueError(f"{name!r} is already a category.")
    custom = load_custom(path)
    custom[name] = description
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(custom, indent=2), encoding="utf-8")


def remove_category(path: str | Path, name: str) -> None:
    """Drop an analyst-added category. Built-ins can't be removed.

    Assets already registered under this name keep working -- their `asset` field is just a
    string, and Step 5/6 never consult the taxonomy. The name simply stops being offered in the
    dropdown and stops being a classification target, so this is non-destructive.
    """
    if name in BUILTIN_CATEGORIES:
        raise ValueError(f"{name!r} is a built-in category and can't be removed.")
    custom = load_custom(path)
    custom.pop(name, None)
    Path(path).write_text(json.dumps(custom, indent=2), encoding="utf-8")
