# SBAA documentation

The presentation-length walkthrough of SBAA: the three-brand auto-discovery demo, the exported
brief, the supercars cost projection, and the limitations.

There are two versions of the same document. **They are separate sources — editing one does not
update the other.**

| | Word version | Web version |
| --- | --- | --- |
| Edit | `sbaa-doc.docx` directly, in Word | `sbaa-doc.src.html`, then rebuild |
| Output | `sbaa-doc.docx` (6 pages) | `sbaa-doc.html` (one self-contained file) |
| Published | — | <https://claude.ai/code/artifact/aa2feacc-3313-424f-8117-61e369866ea7> |

## Word

`sbaa-doc.docx` **is** the source — edit it directly in Word. It uses real Word Title, Heading and
Table styles, so the navigation pane and styles panel work normally.

There is deliberately no Markdown intermediate any more. The original one was generated *into* the
.docx, which meant regenerating it would silently destroy hand edits; deleting it removes that
trap. Changes now flow the other way: edit the .docx, then port the changes into
`sbaa-doc.src.html` by hand and rebuild.

## Web

```bash
./.venv/Scripts/python.exe docs/build.py
```

`build.py` reads `sbaa-doc.src.html`, inlines every `{{img:name}}` placeholder from `assets/` as a
base64 data URI, and writes `sbaa-doc.html`. Everything is inlined because the page is published as
a Claude Artifact, which serves a single file.

## Assets

`assets/` holds the screenshots shared by both versions, plus a few format-specific ones:

- `demo.gif` — the animated demo (web version only; Word shows a still instead)
- `demo-still.png` — a frame from that GIF, for the Word version
- `inputs.png` — the blank-state and three-brands screenshots side by side, for the Word version,
  which has no two-column layout
- `placement-types.jpg` — the editable placement-category panel
- `excel-export.jpg` — the brief open in Excel, showing the per-brand sheet tabs
- `brief.pdf` — the exported `brief.xlsx` printed to PDF; useful if you want to show the real
  workbook live rather than the screenshot

## Notes

- Keep it to roughly 10–15 minutes of presenting. The deeper technical material (rejected
  approaches, persistence design) deliberately lives in `CLAUDE.md`, not here.
- The web version is theme-aware and renders light or dark to match the viewer's browser.
- `sbaa-doc.html`, `sbaa-doc.docx` and `assets/demo.gif` are large binaries — decide whether you
  want them in git before committing this folder.
