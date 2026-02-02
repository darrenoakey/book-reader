import tempfile
from pathlib import Path

from ebooklib import epub

from src.epub_extract import extract_chapters, extract_epub, html_to_text, normalize_name


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
