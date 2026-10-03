import random
import re
import subprocess
import tempfile
from pathlib import Path

from ebooklib import epub

from jeff_book import (
    SENTENCES_PER_PARA,
    TARGET_CHAPTERS,
    WORDS_PER_PAGE,
    assemble_m4b,
    generate_book,
    make_sentence,
    plan_chapters,
    split_sentences,
    write_epub,
)
from src import m4b_assemble as m4b

JEFF_WORD = re.compile(r"^jeff$", re.IGNORECASE)


# ##################################################################
# strip decoration
# reduce a sentence to its bare words so only "jeff" tokens remain
def strip_decoration(sentence: str) -> list[str]:
    cleaned = re.sub(r"[.,!?…—]", " ", sentence)
    return cleaned.split()


# ##################################################################
# create tone
# a real short wav produced by ffmpeg
def create_tone(path: Path, duration: float) -> None:
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=330:duration={duration}", "-ar", "24000", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


# ##################################################################
# test make sentence is only jeff
# every generated sentence is made purely of jeff words and ends in punctuation
def test_make_sentence_is_only_jeff() -> None:
    rng = random.Random(7)
    for _ in range(300):
        sentence = make_sentence(rng)
        words = strip_decoration(sentence)
        assert 1 <= len(words) <= 12
        assert all(JEFF_WORD.match(w) for w in words), sentence
        assert words[0] == "Jeff" or words[0] == "JEFF"
        assert sentence[-1] in ".!?…—"


# ##################################################################
# test make sentence varies
# casing and punctuation are genuinely varied across a sample
def test_make_sentence_varies() -> None:
    rng = random.Random(11)
    sentences = [make_sentence(rng) for _ in range(500)]
    assert len({s[-1] for s in sentences}) >= 5
    joined = " ".join(sentences)
    assert "JEFF" in joined and "jeff" in joined and "Jeff" in joined
    assert len({len(strip_decoration(s)) for s in sentences}) >= 6


# ##################################################################
# test generate book deterministic
# the same seed reproduces the same book and a different seed does not
def test_generate_book_deterministic() -> None:
    assert generate_book(5, 123) == generate_book(5, 123)
    assert generate_book(5, 123) != generate_book(5, 124)


# ##################################################################
# test generate book size
# the book reaches its word target, split into numbered chapters of paragraphs
def test_generate_book_size_and_structure() -> None:
    pages = 40
    chapters = generate_book(pages, 3)
    total_words = sum(len(" ".join(paras).split()) for _, paras in chapters)
    assert total_words >= pages * WORDS_PER_PAGE
    assert total_words < pages * WORDS_PER_PAGE + 2 * 8 * 12 * SENTENCES_PER_PARA[1]
    assert [title for title, _ in chapters] == [f"Chapter {i}" for i in range(1, len(chapters) + 1)]
    assert TARGET_CHAPTERS - 1 <= len(chapters) <= TARGET_CHAPTERS + 1
    for _, paras in chapters:
        assert paras
        for para in paras:
            assert SENTENCES_PER_PARA[0] <= len(split_sentences(para)) <= SENTENCES_PER_PARA[1] + 12


# ##################################################################
# test generate book tiny
# a one-page book still produces at least one chapter
def test_generate_book_tiny() -> None:
    chapters = generate_book(1, 5)
    assert len(chapters) >= 1
    assert sum(len(" ".join(p).split()) for _, p in chapters) >= WORDS_PER_PAGE


# ##################################################################
# test split sentences
# terminal punctuation stays attached and em-dash/ellipsis endings split
def test_split_sentences() -> None:
    text = "Jeff Jeff. JEFF?! jeff... Jeff — Jeff… Jeff!  \n Jeff?"
    assert split_sentences(text) == ["Jeff Jeff.", "JEFF?!", "jeff...", "Jeff —", "Jeff…", "Jeff!", "Jeff?"]
    assert split_sentences("   ") == []
    # a mid-sentence em-dash aside is followed by whitespace, so it splits there too
    assert split_sentences("Jeff, Jeff — Jeff.") == ["Jeff, Jeff —", "Jeff."]
    assert split_sentences("Jeff...Jeff") == ["Jeff...Jeff"]


# ##################################################################
# test split sentences round trip
# splitting generated paragraphs loses no words
def test_split_sentences_round_trip() -> None:
    for _, paras in generate_book(3, 9):
        for para in paras:
            assert " ".join(split_sentences(para)).split() == para.split()


# ##################################################################
# test write epub round trip
# the written epub reads back with its metadata, chapters and escaped text intact
def test_write_epub_round_trip() -> None:
    chapters = [("Chapter 1", ["Jeff Jeff. Jeff!", "jeff & JEFF <Jeff>."]), ("Chapter 2", ["Jeff?"])]
    with tempfile.TemporaryDirectory() as tmp:
        out = write_epub("Jeff Test", "Some Jeff", chapters, Path(tmp) / "nested" / "dir" / "book.epub")
        assert out.exists() and out.stat().st_size > 0
        book = epub.read_epub(str(out))
        assert book.get_metadata("DC", "title")[0][0] == "Jeff Test"
        assert book.get_metadata("DC", "creator")[0][0] == "Some Jeff"
        docs = {item.file_name: item.get_content().decode("utf-8") for item in book.get_items_of_type(9)}
        ch1 = docs["chapter_1.xhtml"]
        assert "<h1>Chapter 1</h1>" in ch1
        assert "jeff &amp; JEFF &lt;Jeff&gt;." in ch1
        assert "Chapter 2" in docs["chapter_2.xhtml"]


# ##################################################################
# test plan chapters
# one job per sentence across the whole book, with per-chapter wav targets
def test_plan_chapters() -> None:
    chapters = [("Chapter 1", ["Jeff Jeff. Jeff!"]), ("Chapter 2", ["Jeff?", "JEFF. jeff..."])]
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / "audio"
        plans, jobs = plan_chapters(chapters, audio, "am_michael", 1.25)
        assert [p[0] for p in plans] == ["Chapter 1", "Chapter 2"]
        assert [p[2] for p in plans] == [audio / "001.wav", audio / "002.wav"]
        assert [len(p[1]) for p in plans] == [2, 3]
        assert len(jobs) == 5
        assert [j["text"] for j in jobs] == ["Jeff Jeff.", "Jeff!", "Jeff?", "JEFF.", "jeff..."]
        assert all(j["voice"] == "am_michael" and j["speed"] == 1.25 for j in jobs)
        assert [j["output_path"] for j in jobs] == plans[0][1] + plans[1][1]
        assert plans[1][1][0] == audio / ".lines_002" / "00000.wav"
        assert (audio / ".lines_001").is_dir() and (audio / ".lines_002").is_dir()


# ##################################################################
# test assemble m4b
# real ffmpeg builds an m4b with chapter markers and metadata at the right times
def test_assemble_m4b_chapters_and_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wav1, wav2 = tmp_path / "001.wav", tmp_path / "002.wav"
        create_tone(wav1, 1.0)
        create_tone(wav2, 2.0)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        path = assemble_m4b([("Chapter 1", wav1), ("Chapter 2", wav2)], "Jeff Title", "Jeff Author", out_dir)
        assert path == out_dir / "Jeff Title.m4b"
        # 1s + gap + 2s + gap
        assert 4.8 < m4b.get_audio_duration(path) < 5.3
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_chapters", "-show_format", "-of", "json", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        import json

        info = json.loads(probe.stdout)
        titles = [c["tags"]["title"] for c in info["chapters"]]
        assert titles == ["Chapter 1", "Chapter 2"]
        assert abs(float(info["chapters"][0]["start_time"])) < 0.05
        assert abs(float(info["chapters"][1]["start_time"]) - 2.0) < 0.1
        tags = info["format"]["tags"]
        assert tags["title"] == "Jeff Title" and tags["artist"] == "Jeff Author" and tags["album"] == "Jeff Title"


# ##################################################################
# test assemble m4b bad input
# an unreadable chapter audio file surfaces as an error rather than a silent output
def test_assemble_m4b_missing_audio_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        try:
            assemble_m4b([("Chapter 1", tmp_path / "missing.wav")], "T", "A", tmp_path)
        except RuntimeError:
            return
        raise AssertionError("expected RuntimeError for missing audio")
