"""Zero-shot category classification via a vision-language model -- no seed image required.

CLIP's zero-shot text-prompt classification (compare a frame's *embedding* to a text prompt
like "a trophy", no image reference) was spike-tested and rejected for this project: 1/4
correct on real match frames, with every frame's category scores clustered within ~0.02-0.06
of each other -- not a usable signal (see CLAUDE.md). That failure was specific to embedding
similarity, a shallow "does this image vector sit near this text vector" comparison. A VLM
answering "what does this image show" is open-ended visual reasoning, a different and more
capable operation, so it gets its own real judgment per frame instead of a fixed reference
point -- which is what removes the earlier design's requirement for at least one analyst-
approved crop before a category could be classified into at all.

Classification is brand-aware, not just category-aware: `classify_frame()` sends the brand's
reference logo alongside the candidate frame and requires the model to confirm *that brand's*
branding is actually visible, not just "does this look like a jersey/trophy/board in general."
A category-only question fails constantly on real match footage -- a referee's jersey, fans in
team kit, a trophy icon in the broadcast's persistent scoreboard graphic all satisfy "is there
a jersey/trophy here" with no relation to the brand being searched for. See `classify_frame()`'s
docstring for the live example that surfaced this.

This is SBAA's first external dependency: every call here leaves the machine as an Anthropic
API request, a deliberate departure from the fully-local design used everywhere else in this
project (driven by the no-GPU hardware constraint). Requires `ANTHROPIC_API_KEY` in the
environment; callers should check `is_configured()` and warn rather than let requests fail.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

MODEL = "claude-haiku-4-5-20251001"
NONE_OPTION = "None of these"

_client: anthropic.Anthropic | None = None
_client_checked = False


def _get_client() -> anthropic.Anthropic | None:
    global _client, _client_checked
    if not _client_checked:
        # Load a project-root .env if present -- Streamlit doesn't do this itself, and this
        # is the one place in the app that actually needs ANTHROPIC_API_KEY. Doesn't override
        # a key already set in the real environment (e.g. by the shell or a deployment).
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            _client = anthropic.Anthropic(api_key=api_key)
        _client_checked = True
    return _client


def is_configured() -> bool:
    return _get_client() is not None


def _image_block(image_path: str | Path) -> dict:
    image_path = Path(image_path)
    media_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    image_b64 = base64.standard_b64encode(image_path.read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}}


def classify_frame(
    image_path: str | Path,
    categories: dict[str, str],
    brand: str,
    reference_image_path: str | Path,
) -> str | None:
    """Which category (if any) this frame shows clear evidence of *this brand's* placement.

    An earlier version asked only "which category does this frame show", with no brand
    context at all -- category-agnostic to brand identity, it was really just asking "is there
    a jersey/trophy-shaped thing anywhere in this image", which real match footage answers yes
    to constantly (a referee's jersey, a crowd of fans in team kit, the small FIFA trophy icon
    baked into the broadcast's persistent scoreboard graphic) regardless of whether this brand
    appears at all. That's the CLIP+OCR stage's job upstream, but for a brand whose logo has no
    legible text (OCR is a no-op) and only middling CLIP similarity, the candidates reaching
    this function were often noise the category question alone couldn't reject.

    So the reference logo goes in the same call: the model is asked to confirm the specific
    brand's branding is visible before it's allowed to answer with anything but `None` -- the
    same visual-comparison job humans already do when they eyeball a candidate, not a separate
    verification pass. Returns one of `categories` verbatim, or None if the model isn't
    confident it's this brand specifically, or if the call fails -- callers treat all of these
    the same way the old embedding classify() treated "no pooled crop yet": don't force a
    bucket, just skip the frame.

    `categories` maps each category name to the grounding description shown to the model. That
    description is load-bearing rather than cosmetic -- bare names misclassified promotional
    bumpers as "LED Board" in a spike test -- so the mapping comes from `category.py`, which
    guarantees every entry has one, instead of being defaulted here.
    """
    client = _get_client()
    if client is None:
        return None

    option_lines = "\n".join(f"- {name}: {description}" for name, description in categories.items())
    category_names = ", ".join(categories)

    try:
        response = client.messages.create(
            model=MODEL,
            # The evidence phrase alone routinely runs 60-90 tokens (e.g. "'tap in with VISA'
            # text visible on the LED board perimeter, though this is VISA branding, not
            # Adidas..."), so a tight cap truncated the response before it ever reached the
            # "Answer:" line -- see the strict line-based parsing below for what that silently
            # broke.
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    _image_block(reference_image_path),
                    _image_block(image_path),
                    {
                        "type": "text",
                        "text": (
                            f"The first image is a reference logo for {brand}. The second image "
                            "is a frame from a live sports broadcast.\n\n"
                            "First, state in one short phrase the SPECIFIC visual evidence in "
                            f"the second image (exact text you can read, or a distinctive shape/"
                            f"color combination) that would let someone who has never seen the "
                            f"reference confirm it is {brand} and not any other brand. A generic "
                            "bright color, a generic screen, or \"it looks similar\" does not "
                            "count as evidence -- if you can't point to specific legible text or "
                            "an unmistakable distinctive logo shape, say NO_EVIDENCE.\n"
                            f"Then, on a new line, give your final answer: which placement type "
                            f"{brand}'s branding is on, if any:\n{option_lines}\n\n"
                            f"Answer one of {category_names}, or \"{NONE_OPTION}\" (always "
                            f"\"{NONE_OPTION}\" if you said NO_EVIDENCE above).\n\n"
                            f"Format:\nEvidence: <phrase or NO_EVIDENCE>\nAnswer: <category or "
                            f"{NONE_OPTION}>"
                        ),
                    },
                ],
            }],
        )
    except anthropic.APIError:
        return None

    text = response.content[0].text.strip()
    answer_line = next((line for line in text.splitlines() if line.lower().startswith("answer")), None)
    if answer_line is None:
        # No "Answer:" line at all means the response was cut off mid-evidence (or otherwise
        # malformed) before it stated one -- falling back to scanning the whole raw text used
        # to misfire on category names mentioned *while explaining they don't apply* (e.g. "I
        # cannot see Adidas on any physical LED boards" contains the substring "LED board").
        # No answer stated is exactly the same "don't guess" case as an explicit None.
        return None
    answer = answer_line.lower()
    for category in categories:
        if category.lower() in answer:
            return category
    return None
