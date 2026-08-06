# Notebook documentation standard

How companion docs for notebooks are written and built in this repo.
**Claude Code must follow this whenever it produces a written explainer for a
notebook.**

## The rule

One notebook, one companion doc, two files:

```
case_studies/etfs/02_labels.ipynb            the notebook
case_studies/etfs/02_labels_walkthrough.md   source of truth  (hand-written)
case_studies/etfs/02_labels_walkthrough.html generated        (never hand-edit)
```

Naming is `<notebook_stem>_walkthrough.md`. The `.html` sits beside it with the
same stem. Do not invent parallel tiers (`_guide`, `_notes`, `_summary`) — if a
short version is wanted, it is a section at the top of the walkthrough, not a
second file.

## Building the HTML

Never write the HTML by hand and never edit a generated `.html`. Run:

```bash
.venv/bin/python tools/build_docs.py case_studies/etfs/02_labels_walkthrough.md
.venv/bin/python tools/build_docs.py --all      # rebuild every walkthrough
.venv/bin/python tools/build_docs.py --all --check   # CI: fail if stale
```

**After editing any `*_walkthrough.md`, rebuild its HTML in the same turn.**
A stale HTML twin is worse than none — it looks current and is not.

## Why the build script exists (the mojibake bug)

A standalone HTML file with no `<meta charset="utf-8">` is decoded by browsers
as windows-1252, not UTF-8. Every non-ASCII character then renders as garbage:

| written | shown without charset |
|---------|-----------------------|
| `—` em dash | `â€"` |
| `⚠️` warning | `âš ï¸` |
| `→` arrow | `â†'` |
| `·` middot | `Â·` |

This is a **rendering** fault, not a file-corruption fault — the bytes are
correct UTF-8 the whole time, so grepping the file for `â€` finds nothing.
The symptom appears only when the file is opened in a browser.

It bites specifically when HTML is authored as an *artifact body* — a fragment
starting at `<title>` or `<div>` with no `<!doctype>`, `<html>`, or `<head>`.
That form is valid for publishing tools that wrap it, and broken as a local
file. `tools/build_docs.py` always emits the full skeleton with the charset
declared, which is the point of routing every doc through it.

If you ever do hand-write standalone HTML, it must begin:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>…</title>
</head>
<body>
```

## Writing the markdown

The script derives the page from document structure, so structure carries
meaning:

- **`# H1`** — becomes the page title and masthead heading. Exactly one.
- **First paragraph after the H1** — becomes the standfirst (lead blurb).
  Keep it to 2–4 sentences describing what the doc covers.
- **`## H2`** — becomes a numbered `<section>` and one table-of-contents entry.
  One per notebook cell or cell group.
- **`## H2` containing ⚠️** — rendered as an alert section and flagged in the
  contents. Use for open issues, broken cells, environment mismatches.
- **`### H3` / `#### H4`** — subheadings inside a section; not in the contents.
- **Fenced code blocks** — syntax-highlighted via Pygments. Always tag the
  language (` ```python `, ` ```text `) so highlighting and the light/dark
  themes work.
- **Tables** — auto-wrapped in a horizontal scroll container.
- **Headings inside blockquotes or lists** — treated as content, not section
  breaks. Safe to use.

### Every section opens with a brief

**Required.** Each numbered section starts with a three-item list — nothing
else may come between the heading and it:

```markdown
## 4. Section C — build the labels

- **Aim** — what the section is trying to achieve, in one sentence.
- **Shows** — what it demonstrates or puts beyond doubt; the failure it
  would expose.
- **Outcome** — what actually comes out: the variable, artifact, figure
  shape, or recorded number.

<the mechanical detail follows>
```

The build script detects a list whose first item begins with `**Aim**` and
renders it as a styled standfirst block (`ul.brief`). Matching is on the
`Aim` label, not on position, so an ordinary list that happens to open a
section is left alone.

Keep each item to one or two lines. The brief exists so a reader can decide
whether to read the section; if it needs a paragraph, it is doing the
section's job instead of its own.

Sections that are not notebook cells — the ⚠️ open-issue block, `Downstream`,
`Known limitations` — do not take a brief.

Content conventions:

- Explain *what each cell does mechanically*, which functions it calls, and
  where values come from. The notebook itself explains *why*.
- Quote real output values rather than describing them vaguely.
- Say plainly when a check is weaker than it looks — partial, vacuous, or
  true only by construction. A reader who trusts a vacuous assertion is
  worse off than one who was told nothing.
- Record open issues at the top under a ⚠️ heading, and state plainly whether
  anything was changed. Never silently fix a notebook to make a doc true.

## Styling

All presentation lives in `tools/doc_style.css`, inlined into each generated
page so the HTML is self-contained and portable. It supports light and dark
via `prefers-color-scheme` plus a `data-theme` override. Change the CSS there
and rebuild with `--all`; do not add per-document styles.

## Checklist for Claude Code

When asked to document a notebook:

1. Write/edit `<stem>_walkthrough.md` only.
2. Run `.venv/bin/python tools/build_docs.py <that file>`.
3. Confirm the output reports `wrote …` and mention both files to the user.
4. Never create `_guide.*`, never hand-edit the `.html`, never leave the twin
   stale.
