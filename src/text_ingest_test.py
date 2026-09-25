"""Real tests for text_ingest: markdown stripping, section splitting,
word-driven chunking, and full ingest of a real temp .md story into the
chapters/ layout the rest of the pipeline consumes. No mocks, no services.
"""

import tempfile
from pathlib import Path

from src.text_ingest import chunk_paragraphs, extract_any, ingest_text, split_sections, strip_markdown


# ##################################################################
# test strip markdown
# headings markers, emphasis and links reduce to plain spoken text
def test_strip_markdown() -> None:
    assert strip_markdown("**bold** and *italic*") == "bold and italic"
    assert strip_markdown("[a link](http://x.example)") == "a link"
    assert strip_markdown("![img](x.png) gone") == " gone"
    assert strip_markdown("> quoted") == "quoted"


# ##################################################################
# test split sections
# markdown headings split a story into (title, body) sections
def test_split_sections() -> None:
    text = "Intro paragraph.\n\n# Part One\n\nBody one.\n\n# Part Two\n\nBody two.\n"
    sections = split_sections(text)
    assert sections[0] == (None, "Intro paragraph.")
    assert sections[1] == ("Part One", "Body one.")
    assert sections[2] == ("Part Two", "Body two.")


# ##################################################################
# test chunk paragraphs
# chunking targets the word budget and never splits a paragraph
def test_chunk_paragraphs() -> None:
    paras = [" ".join(["word"] * 100) for _ in range(10)]
    chunks = chunk_paragraphs("\n\n".join(paras), target_words=300)
    assert len(chunks) == 4  # 300+300+300+100
    for chunk in chunks[:-1]:
        assert len(chunk.split()) == 300


# ##################################################################
# test ingest dragonboy shape
# a "Title / by Author / body" markdown story ingests with correct title,
# author, intro file, and numbered chapter files
def test_ingest_dragonboy_shape() -> None:
    story = (
        "Smallest Dragonboy\n\nby Anne McCaffrey\n\n"
        + "\n\n".join(" ".join(["Keevan walked on"] * 80) for _ in range(12))
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        story_path = tmp / "the-smallest-dragonboy.md"
        story_path.write_text(story, encoding="utf-8")
        out = tmp / "out"
        title, author, written = ingest_text(story_path, out)
        assert title == "Smallest Dragonboy"
        assert author == "Anne McCaffrey"
        intro = out / "chapters" / "00-intro.txt"
        assert intro.exists()
        assert "Smallest Dragonboy" in intro.read_text()
        chapters = sorted(p for p in (out / "chapters").glob("*.txt") if p.name != "00-intro.txt")
        assert len(chapters) >= 2  # 960 words at ~900/chapter
        assert len(written) == len(chapters) + 1


# ##################################################################
# test extract any dispatch
# .md routes to the text ingester and produces the same contract
def test_extract_any_markdown() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        story_path = tmp / "story.txt"
        story_path.write_text("My Tale\n\nby A Writer\n\n" + " ".join(["they lived"] * 50), encoding="utf-8")
        title, author, written = extract_any(story_path, tmp / "out")
        assert title == "My Tale"
        assert author == "A Writer"
        assert (tmp / "out" / "chapters" / "00-intro.txt").exists()
        assert len(written) >= 2
