#!/usr/bin/env python
"""Render project markdown docs to standalone, self-contained HTML.

Markdown is the source of truth. The HTML twin is generated and must never
be hand-edited -- rerun this script instead.

    .venv/bin/python tools/build_docs.py case_studies/etfs/02_labels_walkthrough.md
    .venv/bin/python tools/build_docs.py --all
    .venv/bin/python tools/build_docs.py --all --check

Every emitted document declares <meta charset="utf-8">. Without it a browser
opening a local file falls back to windows-1252 and renders em dashes as
"a euro trade-mark" mojibake, which is the bug this script exists to prevent.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

ROOT = Path(__file__).resolve().parent.parent
STYLE = Path(__file__).resolve().parent / "doc_style.css"

# Docs follow the <notebook>_walkthrough.md convention; see docs/notebook-docs.md.
DOC_GLOB = "**/*_walkthrough.md"
SKIP_DIRS = {".venv", ".git", "node_modules", "data", "data_old", "__pycache__"}


# --------------------------------------------------------------------------
# markdown -> html
# --------------------------------------------------------------------------
def _highlight(code: str, lang: str, _attrs: str) -> str:
    """Fence renderer: Pygments-highlight when the language is known."""
    try:
        lexer = get_lexer_by_name(lang or "text")
    except ClassNotFound:
        return f"<pre><code>{html.escape(code)}</code></pre>"
    formatter = HtmlFormatter(nowrap=True)
    return f'<pre><code class="language-{html.escape(lang)}">{highlight(code, lexer, formatter)}</code></pre>'


def _renderer() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"highlight": _highlight, "typographer": False})
    md.enable(["table", "strikethrough"])
    return md


def _slug(text: str) -> str:
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    return re.sub(r"[\s_]+", "-", s) or "section"


# --------------------------------------------------------------------------
# document assembly
# --------------------------------------------------------------------------
def _split_front_matter(text: str) -> tuple[str, str, str]:
    """Return (h1, standfirst, remaining body) from the markdown source."""
    lines = text.splitlines()
    title, standfirst, start = "", "", 0

    for i, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip()
            start = i + 1
            break

    para: list[str] = []
    for line in lines[start:]:
        if line.strip() in {"", "---"} and para:
            break
        if line.startswith("#"):
            break
        if line.strip() and line.strip() != "---":
            para.append(line.strip())
        start += 1
    standfirst = " ".join(para)

    body = "\n".join(lines[start:]).strip("\n")
    return title, standfirst, body


def _sections(md: MarkdownIt, body: str) -> tuple[str, list[tuple[str, str, bool]]]:
    """Split on top-level <h2> into <section id>; return (html, toc entries).

    Splitting is done on the token stream, not on rendered HTML: an h2 nested
    inside a blockquote or list is content, not a section break, and cutting
    the rendered string there would orphan the enclosing open/close tags.
    """
    tokens = md.parse(body)

    # Group indices: each group is (heading_token_index | None, [content tokens]).
    groups: list[tuple[int | None, list]] = []
    current: list = []
    heading_at: int | None = None
    depth = 0

    for tok in tokens:
        top_level_h2 = depth == 0 and tok.type == "heading_open" and tok.tag == "h2"
        if top_level_h2:
            groups.append((heading_at, current))
            current, heading_at = [], len(groups)
        current.append(tok)
        depth += tok.nesting
    groups.append((heading_at, current))

    out: list[str] = []
    toc: list[tuple[str, str, bool]] = []
    n = 0

    for idx, (marker, toks) in enumerate(groups):
        if marker is None:  # lead content before the first h2
            if toks:
                out.append(md.renderer.render(toks, md.options, {}))
            continue

        # toks == [heading_open, inline, heading_close, *body]
        inner = md.renderer.render([toks[1]], md.options, {})
        rest = md.renderer.render(toks[3:], md.options, {}) if len(toks) > 3 else ""

        # A section-opening list whose first item is "**Aim** — …" is the
        # Aim/Shows/Outcome brief (see docs/notebook-docs.md); style it as a
        # standfirst. Matching on the Aim label rather than on position keeps
        # an ordinary list that merely opens a section from being restyled.
        rest = re.sub(r"\A(\s*)<ul>(\s*<li><strong>Aim</strong>.*?)</ul>",
                      r'\1<ul class="brief">\2</ul>', rest, count=1, flags=re.S)

        anchor = f"c{n}"
        flagged = "⚠" in inner
        label = re.sub(r"<[^>]+>", "", inner).replace("⚠️", "").replace("⚠", "").strip()
        toc.append((anchor, label, flagged))

        cls = ' class="alert"' if flagged else ""
        out.append(
            f'<section id="{anchor}"{cls}>\n'
            f'  <div class="cell-head"><span class="cell-num">{n}</span>'
            f"<h2>{inner}</h2></div>\n{rest}\n</section>"
        )
        n += 1

    return "\n".join(out), toc


def _toc_html(toc: list[tuple[str, str, bool]]) -> str:
    if not toc:
        return ""
    items = "\n".join(
        f'      <li><a{" class=\"flag\"" if flag else ""} href="#{a}">'
        f'<span class="n">{i}</span><span>{html.escape(label)}'
        f'{" ⚠" if flag else ""}</span></a></li>'
        for i, (a, label, flag) in enumerate(toc)
    )
    return f'  <nav class="toc">\n    <h2>Contents</h2>\n    <ol>\n{items}\n    </ol>\n  </nav>\n'


def _pygments_css() -> str:
    light = HtmlFormatter(style="friendly").get_style_defs("pre code")
    dark = HtmlFormatter(style="monokai").get_style_defs("pre code")
    return (
        f"\n/* syntax highlighting */\n{light}\n"
        f"@media (prefers-color-scheme: dark) {{\n{dark}\n}}\n"
        f':root[data-theme="dark"] {{\n{dark}\n}}\n'
        f':root[data-theme="light"] {{\n{light}\n}}\n'
    )


def render(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    title, standfirst, body = _split_front_matter(text)

    md = _renderer()
    sections_html, toc = _sections(md, body)
    sections_html = re.sub(r"(<table>.*?</table>)", r'<div class="table-wrap">\1</div>',
                           sections_html, flags=re.S)

    rel = md_path.resolve().relative_to(ROOT).as_posix()
    eyebrow = html.escape(rel.rsplit("/", 1)[0].replace("/", " · "))
    plain_title = re.sub(r"[`*]", "", title)

    css = STYLE.read_text(encoding="utf-8") + _pygments_css()

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(plain_title)}</title>
<style>
{css}
</style>
</head>
<body>

<div class="shell">

  <header class="masthead">
    <div class="eyebrow">{eyebrow}</div>
    <h1>{_renderer().renderInline(title)}</h1>
    <p class="standfirst">{_renderer().renderInline(standfirst)}</p>
    <div class="masthead-meta">
      <span>{html.escape(rel)}</span>
      <span>generated by tools/build_docs.py &mdash; do not edit</span>
    </div>
  </header>

{_toc_html(toc)}
  <article>
{sections_html}
  </article>
</div>

</body>
</html>
"""


# --------------------------------------------------------------------------
def discover() -> list[Path]:
    return sorted(
        p for p in ROOT.glob(DOC_GLOB)
        if not any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, help="markdown files to render")
    ap.add_argument("--all", action="store_true", help=f"render every {DOC_GLOB}")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any HTML twin is missing or stale (no writes)")
    args = ap.parse_args()

    targets = discover() if args.all else [p.resolve() for p in args.paths]
    if not targets:
        ap.error("no input: pass markdown paths or --all")

    stale = 0
    for md_path in targets:
        if not md_path.exists():
            print(f"missing: {md_path}", file=sys.stderr)
            return 1
        out = md_path.with_suffix(".html")
        rendered = render(md_path)
        rel = out.resolve().relative_to(ROOT)
        if args.check:
            current = out.read_text(encoding="utf-8") if out.exists() else None
            if current != rendered:
                print(f"STALE  {rel}")
                stale += 1
            else:
                print(f"ok     {rel}")
        else:
            out.write_text(rendered, encoding="utf-8")
            print(f"wrote  {rel}  ({len(rendered):,} bytes)")

    if args.check and stale:
        print(f"\n{stale} file(s) out of date -- run: "
              f".venv/bin/python tools/build_docs.py --all", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
