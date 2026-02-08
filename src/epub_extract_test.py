import subprocess
import tempfile
from pathlib import Path

from ebooklib import epub

from src.epub_extract import classify_is_content, extract_chapters, extract_cover_image, extract_epub, html_to_text, is_substantial_item, normalize_name


# ##################################################################
# create test jpeg
# use ffmpeg to generate a 1x1 JPEG image for testing
def create_test_jpeg(output_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=1x1:d=1",
        "-frames:v", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg jpeg creation failed: {result.stderr}")


# ##################################################################
# create test epub
# builds a minimal epub file for testing
def create_test_epub(path: Path, title: str = "Test Book", author: str = "Test Author") -> None:
    book = epub.EpubBook()
    book.set_identifier("test-book-001")
    book.set_title(title)
    book.set_language("en")
    book.add_author(author)
    chapter1 = epub.EpubHtml(title="Chapter One", file_name="chapter1.xhtml", lang="en")
    chapter1.content = """
    <html><body>
    <h1>Chapter One</h1>
    <p>John walked into the room slowly. He looked around nervously, taking in every detail
    of the old house. The floorboards creaked beneath his feet as he made his way toward
    the kitchen. Dust motes floated in the afternoon light streaming through the windows.</p>
    <p>"Hello," said Mary, emerging from the shadows. "I've been waiting for you. It has
    been far too long since we last spoke, and there is much to discuss about the estate."</p>
    <p>John nodded solemnly. He thought about what to say next. It was difficult to find
    the right words after all these years apart. The tension in the room was palpable.</p>
    <p>"I know," he replied at last. "I'm sorry I'm late. The traffic from the city was
    terrible, and I had to stop twice for directions. This old place is harder to find
    than I remembered from our childhood summers here."</p>
    <p>Mary smiled sadly and gestured toward the sitting room. There was much history
    between them, and the weight of unspoken words hung heavy in the dusty air.</p>
    </body></html>
    """
    chapter2 = epub.EpubHtml(title="Chapter Two", file_name="chapter2.xhtml", lang="en")
    chapter2.content = """
    <html><body>
    <h1>Chapter Two</h1>
    <p>The next morning was cold and grey. Mary made coffee while John slept upstairs
    in their mother's old bedroom. The kitchen still smelled of lavender, just as it
    had when they were children playing in the garden outside.</p>
    <p>She wondered if he would stay this time. "Please don't leave," she whispered to
    herself, watching the steam rise from her cup. The old house needed repairs, and
    she couldn't manage them alone anymore. The roof leaked and the foundation was crumbling.</p>
    <p>John woke up and came downstairs, his footsteps heavy on the worn wooden stairs.
    "Good morning," he said with a tentative smile. "Did you sleep well? I barely slept
    at all. Too many memories in this old place, and the wind kept rattling the shutters."</p>
    <p>Mary handed him a cup of coffee without speaking. They both knew that today would
    bring difficult conversations about the future of the family home and their shared past.</p>
    </body></html>
    """
    book.add_item(chapter1)
    book.add_item(chapter2)
    book.toc = [epub.Link("chapter1.xhtml", "Chapter One", "ch1"), epub.Link("chapter2.xhtml", "Chapter Two", "ch2")]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter1, chapter2]
    epub.write_epub(str(path), book)


# ##################################################################
# create test epub with cover
# builds an epub file with a cover image for testing
def create_test_epub_with_cover(path: Path, cover_jpeg: Path) -> None:
    book = epub.EpubBook()
    book.set_identifier("test-book-cover-001")
    book.set_title("Cover Test Book")
    book.set_language("en")
    book.add_author("Cover Author")
    cover_data = cover_jpeg.read_bytes()
    book.set_cover("cover.jpeg", cover_data)
    chapter1 = epub.EpubHtml(title="Chapter One", file_name="chapter1.xhtml", lang="en")
    chapter1.content = """
    <html><body>
    <h1>Chapter One</h1>
    <p>John walked into the room slowly. He looked around nervously, taking in every detail
    of the old house. The floorboards creaked beneath his feet as he made his way toward
    the kitchen. Dust motes floated in the afternoon light streaming through the windows.</p>
    <p>"Hello," said Mary, emerging from the shadows. "I've been waiting for you. It has
    been far too long since we last spoke, and there is much to discuss about the estate."</p>
    <p>John nodded solemnly. He thought about what to say next. It was difficult to find
    the right words after all these years apart. The tension in the room was palpable.</p>
    <p>"I know," he replied at last. "I'm sorry I'm late. The traffic from the city was
    terrible, and I had to stop twice for directions. This old place is harder to find
    than I remembered from our childhood summers here."</p>
    <p>Mary smiled sadly and gestured toward the sitting room. There was much history
    between them, and the weight of unspoken words hung heavy in the dusty air.</p>
    </body></html>
    """
    book.add_item(chapter1)
    book.toc = [epub.Link("chapter1.xhtml", "Chapter One", "ch1")]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter1]
    epub.write_epub(str(path), book)


# ##################################################################
# test normalize name
# verify name normalization works correctly
def test_normalize_name() -> None:
    assert normalize_name("Chapter One") == "chapter_one"
    assert normalize_name("The Don Salvara Game") == "the_don_salvara_game"
    assert normalize_name("Hello-World!") == "helloworld"
    assert normalize_name("Café Münster") == "cafe_munster"


# ##################################################################
# test html to text
# verify html conversion preserves paragraph breaks
def test_html_to_text() -> None:
    html = "<html><body><p>First paragraph.</p><p>Second paragraph.</p></body></html>"
    text = html_to_text(html)
    assert "First paragraph." in text
    assert "Second paragraph." in text
    assert "\n\n" in text


# ##################################################################
# test extract chapters
# verify chapter extraction from epub
def test_extract_chapters() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        epub_path = Path(tmpdir) / "test.epub"
        create_test_epub(epub_path)
        title, author, chapters = extract_chapters(epub_path)
        assert title == "Test Book"
        assert author == "Test Author"
        assert len(chapters) == 2
        assert chapters[0].title == "Chapter One"
        assert "Hello" in chapters[0].text
        assert "Mary" in chapters[0].text


# ##################################################################
# test extract epub
# verify full epub extraction to files
def test_extract_epub() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        epub_path = tmpdir / "test.epub"
        output_dir = tmpdir / "output"
        create_test_epub(epub_path)
        title, author, written = extract_epub(epub_path, output_dir)
        assert title == "Test Book"
        assert author == "Test Author"
        assert len(written) == 3
        intro = written[0]
        assert intro.name == "00-intro.txt"
        assert "Test Book" in intro.read_text()
        assert "Test Author" in intro.read_text()
        chapter1 = written[1]
        assert "chapter_one" in chapter1.name
        assert "Hello" in chapter1.read_text()


# ##################################################################
# test idempotent extraction
# verify extraction skips existing files
def test_idempotent_extraction() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        epub_path = tmpdir / "test.epub"
        output_dir = tmpdir / "output"
        create_test_epub(epub_path)
        extract_epub(epub_path, output_dir)
        intro = output_dir / "chapters" / "00-intro.txt"
        intro.write_text("MODIFIED")
        extract_epub(epub_path, output_dir)
        assert intro.read_text() == "MODIFIED"


# ##################################################################
# test extract cover image
# verify cover image is extracted from epub
def test_extract_cover_image() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cover_jpeg = tmpdir / "source_cover.jpeg"
        create_test_jpeg(cover_jpeg)
        epub_path = tmpdir / "test.epub"
        create_test_epub_with_cover(epub_path, cover_jpeg)
        output_dir = tmpdir / "output"
        book = epub.read_epub(str(epub_path))
        result = extract_cover_image(book, output_dir)
        assert result is not None
        assert result.exists()
        assert result.name == "cover.jpeg"
        assert result.stat().st_size > 0


# ##################################################################
# test extract cover image idempotent
# verify existing cover is not overwritten
def test_extract_cover_image_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cover_jpeg = tmpdir / "source_cover.jpeg"
        create_test_jpeg(cover_jpeg)
        epub_path = tmpdir / "test.epub"
        create_test_epub_with_cover(epub_path, cover_jpeg)
        output_dir = tmpdir / "output"
        output_dir.mkdir()
        existing = output_dir / "cover.jpeg"
        existing.write_bytes(b"PRESERVED")
        book = epub.read_epub(str(epub_path))
        result = extract_cover_image(book, output_dir)
        assert result is not None
        assert result.read_bytes() == b"PRESERVED"


# ##################################################################
# test extract cover no cover
# verify no cover returns none for epub without cover
def test_extract_cover_no_cover() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        epub_path = tmpdir / "test.epub"
        create_test_epub(epub_path)
        output_dir = tmpdir / "output"
        book = epub.read_epub(str(epub_path))
        result = extract_cover_image(book, output_dir)
        assert result is None


# ##################################################################
# test extract epub with cover
# verify full epub extraction includes cover
def test_extract_epub_with_cover() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cover_jpeg = tmpdir / "source_cover.jpeg"
        create_test_jpeg(cover_jpeg)
        epub_path = tmpdir / "test.epub"
        create_test_epub_with_cover(epub_path, cover_jpeg)
        output_dir = tmpdir / "output"
        extract_epub(epub_path, output_dir)
        cover_path = output_dir / "cover.jpeg"
        assert cover_path.exists()
        assert cover_path.stat().st_size > 0


# ##################################################################
# test is substantial item
# verify size-based filtering works
def test_is_substantial_item() -> None:
    assert not is_substantial_item("short text")
    assert not is_substantial_item("x " * 200)
    assert is_substantial_item("x " * 300)


# ##################################################################
# test classify is content
# verify haiku correctly classifies front matter vs real content
def test_classify_is_content() -> None:
    import asyncio
    synopsis_text = (
        "Synopsis\n\nIn this stunning debut, the author delivers a wonderfully thrilling tale "
        "of an audacious criminal and his band of confidence tricksters. Set in a fantastic city "
        "pulsing with the lives of decadent nobles and daring thieves, here is a story of "
        "adventure, loyalty, and survival.\n\nContents\n\nPrologue\nChapter One\nChapter Two"
    )
    assert not asyncio.run(classify_is_content(synopsis_text, "split_000.xhtml"))
    chapter_text = (
        "Chapter One\n\nJohn walked into the room slowly. He looked around nervously, taking "
        "in every detail of the old house. The floorboards creaked beneath his feet as he made "
        "his way toward the kitchen. 'Hello,' said Mary, emerging from the shadows."
    )
    assert asyncio.run(classify_is_content(chapter_text, "chapter1.xhtml"))
