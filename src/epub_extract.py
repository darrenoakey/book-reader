import asyncio
import re
import unicodedata
import warnings
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query
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
# is substantial item
# check if an epub item has enough text to be a chapter
def is_substantial_item(text: str) -> bool:
    return len(text.strip()) >= 500


# ##################################################################
# query haiku
# send a prompt to claude haiku and get text response
async def query_haiku(prompt: str) -> str:
    response = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=[],
            permission_mode="bypassPermissions",
            model="haiku",
        )
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response += block.text
    return response.strip()


# ##################################################################
# classify is content
# use haiku to decide if text is book content or front/back matter
async def classify_is_content(text: str, filename: str) -> bool:
    preview = text[:1000]
    prompt = f"""You are classifying sections of an EPUB ebook. Given the filename and the first 1000 characters, decide if this is ACTUAL BOOK CONTENT (a chapter, prologue, epilogue, interlude, or other narrative/story section) or NON-CONTENT (synopsis, blurb, table of contents, copyright, dedication, acknowledgements, about the author, biography, advertisement, or other front/back matter).

Filename: {filename}
First 1000 characters:
{preview}

Reply with exactly one word: CONTENT or NONCONTENT"""
    response = await query_haiku(prompt)
    return "CONTENT" in response.upper() and "NONCONTENT" not in response.upper()


# ##################################################################
# trim non content
# scan from front and back using haiku to find where real content starts/ends
async def trim_non_content(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return candidates
    start = 0
    for i in range(len(candidates)):
        is_content = await classify_is_content(candidates[i]["text"], candidates[i]["filename"])
        if is_content:
            start = i
            break
    else:
        return []
    end = len(candidates) - 1
    for i in range(len(candidates) - 1, start - 1, -1):
        is_content = await classify_is_content(candidates[i]["text"], candidates[i]["filename"])
        if is_content:
            end = i
            break
    return candidates[start:end + 1]


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
# collect candidates
# gather all substantial epub items as candidates for chapter classification
def collect_candidates(book: epub.EpubBook) -> list[dict]:
    candidates = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html = item.get_content().decode("utf-8", errors="ignore")
        text = html_to_text(html)
        if not is_substantial_item(text):
            continue
        title = extract_chapter_title(html)
        candidates.append({
            "text": text,
            "html": html,
            "title": title,
            "filename": item.get_name(),
        })
    return candidates


# ##################################################################
# extract chapters
# parse epub and return list of chapter info objects
def extract_chapters(epub_path: Path, book: epub.EpubBook | None = None) -> tuple[str, str, list[ChapterInfo]]:
    if book is None:
        book = epub.read_epub(str(epub_path))
    title, author = get_metadata(book)
    candidates = collect_candidates(book)
    trimmed = asyncio.run(trim_non_content(candidates))
    chapters = []
    for i, cand in enumerate(trimmed, start=1):
        chapter_title = cand["title"] if cand["title"] else f"Chapter {i}"
        chapters.append(ChapterInfo(
            number=i,
            title=chapter_title,
            text=cand["text"],
            original_file=cand["filename"],
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
# extract cover image
# find and save the cover image from an epub
def extract_cover_image(book: epub.EpubBook, output_dir: Path) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("jpeg", "jpg", "png"):
        existing = output_dir / f"cover.{ext}"
        if existing.exists():
            return existing
    cover_item = None
    for item in book.get_items_of_type(ebooklib.ITEM_COVER):
        cover_item = item
        break
    if cover_item is None:
        cover_item = book.get_item_with_id("cover")
    if cover_item is None:
        cover_item = book.get_item_with_id("cover-image")
    if cover_item is None:
        for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            name = Path(item.get_name()).stem.lower()
            if "cover" in name:
                cover_item = item
                break
    if cover_item is None:
        return None
    content = cover_item.get_content()
    media_type = getattr(cover_item, "media_type", "") or ""
    if "png" in media_type:
        ext = "png"
    else:
        ext = "jpeg"
    cover_path = output_dir / f"cover.{ext}"
    cover_path.write_bytes(content)
    return cover_path


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
    book = epub.read_epub(str(epub_path))
    extract_cover_image(book, output_dir)
    title, author, chapters = extract_chapters(epub_path, book)
    written = write_chapters(output_dir, title, author, chapters)
    return title, author, written
