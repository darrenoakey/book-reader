"""Plain-text / Markdown story ingest.

Turns a .md/.txt short story into the same ``chapters/`` layout the EPUB
extractor produces, so the rest of the pipeline (characters → voices →
scripts → audio → m4b → movie) is identical for both input kinds.

Chaptering is word-driven: target ~900 words per chapter, always breaking at
a paragraph boundary (a short story has no inherent chapters, and equal-size
chunks keep per-step latencies uniform). Markdown headings become chapter
titles when present; otherwise chapters are "Part 1", "Part 2", ...
"""

from __future__ import annotations

import re
from pathlib import Path

from src.epub_extract import normalize_name

TARGET_WORDS = 900

_HEADING_RE = re.compile(r"^#{1,3}\s+(.*)$")


# ##################################################################
# strip markdown
# reduce markdown to plain spoken text: drop heading hashes, emphasis
# markers, images/links keep their text, collapse whitespace runs
def strip_markdown(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links → text
    text = re.sub(r"[*_~`]+", "", text)  # emphasis/code markers
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)  # blockquotes
    return text


# ##################################################################
# split sections
# split markdown into (title, body) sections at headings; leading content
# before the first heading becomes one untitled section
def split_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    title: str | None = None
    body: list[str] = []
    for line in text.split("\n"):
        m = _HEADING_RE.match(line.strip())
        if m:
            if body and "".join(body).strip():
                sections.append((title, "\n".join(body).strip()))
            title = m.group(1).strip()
            body = []
        else:
            body.append(line)
    if body and "".join(body).strip():
        sections.append((title, "\n".join(body).strip()))
    return sections


# ##################################################################
# chunk paragraphs
# word-driven chunking of one body of text at paragraph boundaries
def chunk_paragraphs(text: str, target_words: int = TARGET_WORDS) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    words = 0
    for para in paragraphs:
        pw = len(para.split())
        if current and words + pw > target_words:
            chunks.append("\n\n".join(current))
            current = []
            words = 0
        current.append(para)
        words += pw
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ##################################################################
# ingest text
# main entry: write chapters/ for a .md/.txt story; returns title, author,
# written paths — the same contract as extract_epub
def ingest_text(story_path: Path, output_dir: Path, author: str = "Unknown Author") -> tuple[str, str, list[Path]]:
    raw = story_path.read_text(encoding="utf-8")
    sections = split_sections(raw)

    # Title: first heading if it heads a tiny/no body (a title page), else the
    # file stem. The dragonboy stories are "Title\n\nby Author\n\n<body>".
    title = story_path.stem.replace("-", " ").title()
    if sections and sections[0][0]:
        first_title, first_body = sections[0]
        if len(first_body.split()) < 20:
            title = first_title
            byline = re.search(r"\bby\s+([A-Z][A-Za-z .']+)", first_body)
            if byline:
                author = byline.group(1).strip()
            sections = sections[1:]
        elif first_title and first_body is not None:
            title = first_title

    # No-heading plain text: detect a leading "Title" / "by Author" byline
    # pair (the dragonboy stories are exactly that shape).
    if len(sections) == 1 and sections[0][0] is None:
        body = sections[0][1]
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        if len(paras) >= 3 and len(paras[0].split()) <= 10 and re.match(r"(?i)^by\s+\w", paras[1]):
            title = paras[0]
            author = re.sub(r"(?i)^by\s+", "", paras[1]).strip()
            sections = [(None, "\n\n".join(paras[2:]))]

    chapters_dir = output_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    intro_path = chapters_dir / "00-intro.txt"
    if not intro_path.exists():
        intro_path.write_text(f"{title} by {author}, narrated by Darren's Book Reader.", encoding="utf-8")
    written.append(intro_path)

    number = 1
    for section_title, body in sections:
        plain = strip_markdown(body)
        for chunk in chunk_paragraphs(plain):
            chapter_title = section_title or f"Part {number}"
            path = chapters_dir / f"{number:02d}-{normalize_name(chapter_title)[:40]}.txt"
            if not path.exists():
                path.write_text(chunk, encoding="utf-8")
            written.append(path)
            number += 1
    return title, author, written


# ##################################################################
# extract any
# dispatch on input kind: epub goes through the EPUB extractor, .md/.txt
# stories through the text ingester — both produce the same chapters/ layout
def extract_any(source_path: Path, output_dir: Path) -> tuple[str, str, list[Path]]:
    if source_path.suffix.lower() in (".md", ".txt"):
        return ingest_text(source_path, output_dir)
    from src.epub_extract import extract_epub

    return extract_epub(source_path, output_dir)
