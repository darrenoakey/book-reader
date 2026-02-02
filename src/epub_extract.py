import re
import unicodedata
import warnings
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# ##################################################################
# get metadata
# extract title and author from epub metadata
def get_metadata(book: epub.EpubBook) -> tuple[str, str]:
    title = book.get_metadata("DC", "title")
    title = title[0][0] if title else "Unknown Title"
    creator = book.get_metadata("DC", "creator")
    author = creator[0][0] if creator else "Unknown Author"
    return title, author


# ##################################################################
# normalize name
# convert a string to a safe filename-style identifier
def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name


# ##################################################################
# html to text
# convert html content to plain text preserving paragraph breaks
def html_to_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "lxml")
    for script in soup(["script", "style"]):
        script.decompose()
    paragraphs = []
    for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "div", "blockquote"]):
        text = element.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


# ##################################################################
# extract chapter title
# try to find a title from the chapter html content
def extract_chapter_title(html_content: str) -> str | None:
    soup = BeautifulSoup(html_content, "lxml")
    for tag in ["h1", "h2", "h3"]:
        heading = soup.find(tag)
        if heading:
            text = heading.get_text(strip=True)
            if text and len(text) < 200:
                return text
    return None


# ##################################################################
# is content chapter
# determine if an epub item is likely a content chapter vs front/back matter
def is_content_chapter(item: epub.EpubHtml, text: str) -> bool:
    if len(text.strip()) < 500:
        return False
    filename = item.get_name().lower()
    skip_patterns = ["cover", "title", "copyright", "toc", "contents", "dedication", "acknowledge", "about"]
    for pattern in skip_patterns:
        if pattern in filename:
            return False
    return True


# ##################################################################
# chapter info
# data class for chapter information
class ChapterInfo:
    def __init__(self, number: int, title: str, text: str, original_file: str):
        self.number = number
        self.title = title
        self.text = text
        self.original_file = original_file


# ##################################################################
# extract chapters
# parse epub and return list of chapter info objects
def extract_chapters(epub_path: Path) -> tuple[str, str, list[ChapterInfo]]:
    book = epub.read_epub(str(epub_path))
    title, author = get_metadata(book)
    chapters = []
    chapter_num = 0
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content().decode("utf-8", errors="ignore")
        text = html_to_text(content)
        if not is_content_chapter(item, text):
            continue
        chapter_num += 1
        chapter_title = extract_chapter_title(content)
        if not chapter_title:
            chapter_title = f"Chapter {chapter_num}"
        chapters.append(ChapterInfo(
            number=chapter_num,
            title=chapter_title,
            text=text,
            original_file=item.get_name(),
        ))
    return title, author, chapters


# ##################################################################
# write chapters
# write extracted chapters to output directory
def write_chapters(output_dir: Path, title: str, author: str, chapters: list[ChapterInfo]) -> list[Path]:
    chapters_dir = output_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    written = []
    intro_path = chapters_dir / "00-intro.txt"
    if not intro_path.exists():
        intro_text = f"{title} by {author}, narrated by Darren's Book Reader."
        intro_path.write_text(intro_text, encoding="utf-8")
    written.append(intro_path)
    for chapter in chapters:
        filename = f"{chapter.number:02d}-{normalize_name(chapter.title)[:40]}.txt"
        chapter_path = chapters_dir / filename
        if not chapter_path.exists():
            chapter_path.write_text(chapter.text, encoding="utf-8")
        written.append(chapter_path)
    return written


# ##################################################################
# get output dir
# create output directory based on epub filename
def get_output_dir(epub_path: Path) -> Path:
    book_name = epub_path.stem
    return epub_path.parent / "output" / book_name


# ##################################################################
# extract epub
# main entry point to extract an epub to chapters directory
def extract_epub(epub_path: Path, output_dir: Path) -> tuple[str, str, list[Path]]:
    title, author, chapters = extract_chapters(epub_path)
    written = write_chapters(output_dir, title, author, chapters)
    return title, author, written
