"""Resolve an assistant's "title / author" request to one readable local EPUB.

Only the user's own mounted library (``[audiobook] library_search_root``) is
searched. A request resolves to exactly one WORK (normalized title + author
from the EPUB's own OPF metadata); several files of the same work are
editions and one is chosen deterministically, with the rest reported. Two or
more distinct works matching the same request is genuine ambiguity and is
returned as such rather than guessed. DRM-protected EPUBs (encryption.xml
covering content, or rights.xml) are reported, never processed.

The file listing is cached as ``library-index.txt`` under the delivery
library root and refreshed when older than a day, so repeated requests are a
fast local scan rather than a walk of the network share.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import time
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree

INDEX_NAME = "library-index.txt"
INDEX_MAX_AGE_SECONDS = 24 * 3600
BOOK_SUFFIXES = (".epub",)
OTHER_FORMATS = (".mobi", ".azw", ".azw3", ".pdf", ".kfx")
TOKEN_MATCH = 0.84
_OPF_NS = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}


# ##################################################################
# normalize
# lowercase ascii word tokens; leading English articles dropped from titles
def tokens(text: str) -> list[str]:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z0-9]+", ascii_text)


def title_key(title: str) -> str:
    words = tokens(title)
    while words and words[0] in ("the", "a", "an"):
        words = words[1:]
    return " ".join(words)


def author_key(author: str) -> str:
    # "McCaffrey Anne", "Anne McCaffrey" and "McCaffrey, Anne" are one author
    return " ".join(sorted(word for word in tokens(author) if len(word) > 1))


# ##################################################################
# index
# every regular file path under the search root, cached on disk
def load_index(search_root: Path, cache_dir: Path, refresh: bool = False) -> list[str]:
    cache = cache_dir / INDEX_NAME
    fresh = cache.is_file() and time.time() - cache.stat().st_mtime < INDEX_MAX_AGE_SECONDS
    if fresh and not refresh:
        return cache.read_text(encoding="utf-8").splitlines()
    if not search_root.is_dir():
        raise FileNotFoundError(f"library search root {search_root} is not mounted")
    skip = cache_dir.resolve()
    paths: list[str] = []
    for directory, dirnames, filenames in os.walk(search_root):
        if Path(directory).resolve() == skip:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        paths.extend(str(Path(directory) / name) for name in sorted(filenames) if not name.startswith("."))
    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(".partial")
    partial.write_text("\n".join(paths) + "\n", encoding="utf-8")
    partial.replace(cache)
    return paths


# ##################################################################
# fuzzy token containment
# every wanted token appears (or nearly appears) among the haystack tokens
def _contains(wanted: list[str], haystack: list[str]) -> bool:
    for word in wanted:
        if word in haystack:
            continue
        if len(word) < 4 or not difflib.get_close_matches(word, haystack, n=1, cutoff=TOKEN_MATCH):
            return False
    return True


# ##################################################################
# epub metadata
# title and joined creators from the OPF package document
def epub_metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
        rootfile = next(el for el in container.iter() if el.tag.endswith("rootfile"))
        opf = ElementTree.fromstring(archive.read(rootfile.attrib["full-path"]))
        names = set(archive.namelist())
        encryption = archive.read("META-INF/encryption.xml").decode("utf-8", "ignore") if "META-INF/encryption.xml" in names else ""
    title = next((el.text or "" for el in opf.iter(f"{{{_OPF_NS['dc']}}}title")), "").strip()
    creators = [(el.text or "").strip() for el in opf.iter(f"{{{_OPF_NS['dc']}}}creator") if (el.text or "").strip()]
    return {
        "title": title,
        "authors": creators,
        "drm": drm_reason(names, encryption),
    }


# ##################################################################
# drm reason
# None for an unrestricted EPUB; a reason when content is encrypted.
# Font obfuscation (IDPF/Adobe font mangling) is not DRM and is allowed.
def drm_reason(names: set[str], encryption_xml: str) -> str | None:
    if "META-INF/rights.xml" in names:
        return "rights.xml present (Adobe ADEPT DRM)"
    if not encryption_xml:
        return None
    root = ElementTree.fromstring(encryption_xml)
    for data in (el for el in root.iter() if el.tag.endswith("EncryptedData")):
        method = next((el.attrib.get("Algorithm", "") for el in data.iter() if el.tag.endswith("EncryptionMethod")), "")
        uri = next((el.attrib.get("URI", "") for el in data.iter() if el.tag.endswith("CipherReference")), "")
        font = method in ("http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC") or uri.lower().endswith(
            (".ttf", ".otf", ".woff", ".woff2")
        )
        if not font:
            return f"encrypted content ({uri or 'unknown item'})"
    return None


# ##################################################################
# resolve
# {"status": "resolved"|"ambiguous"|"not_found"|"drm_restricted"|"unsupported_format", ...}
def resolve_book(title: str, author: str | None, index: list[str]) -> dict:
    want_title = tokens(title_key(title))
    want_author = tokens(author or "")
    if not want_title:
        raise ValueError("a title is required")
    by_name = []
    other_formats = []
    for raw in index:
        path = Path(raw)
        suffix = path.suffix.lower()
        if suffix not in BOOK_SUFFIXES + OTHER_FORMATS:
            continue
        stem_tokens = tokens(path.stem)
        if not _contains(want_title, stem_tokens) and "".join(want_title) not in "".join(stem_tokens):
            continue
        if suffix in BOOK_SUFFIXES:
            by_name.append(path)
        elif not want_author or _contains(want_author, tokens(raw)):
            other_formats.append(path)
    works: dict[tuple[str, str], list[dict]] = {}
    for path in by_name:
        try:
            meta = epub_metadata(path)
        except (KeyError, StopIteration, zipfile.BadZipFile, ElementTree.ParseError, OSError) as error:
            meta = {"title": "", "authors": [], "drm": None, "unreadable": str(error)}
        if meta.get("unreadable"):
            continue
        meta_title = title_key(meta["title"]) or title_key(path.stem)
        ratio = difflib.SequenceMatcher(None, " ".join(want_title), meta_title).ratio()
        compact = meta_title.replace(" ", "") == "".join(want_title)
        title_ok = compact or ratio >= 0.9 or _contains(meta_title.split(), want_title)
        if not title_ok:
            continue
        # the author is checked against the book's own metadata; the path only when metadata has none
        author_fields = meta["authors"] or [str(path)]
        if want_author and not any(_contains(want_author, tokens(a)) for a in author_fields):
            continue
        key = (meta_title, author_key(meta["authors"][0]) if meta["authors"] else author_key(author or ""))
        works.setdefault(key, []).append({"path": str(path), "exact": compact, **meta})
    # an exact title match outranks near-spellings: "Dragonsong" never becomes ambiguous with "The Dragonstone"
    if any(item["exact"] for items in works.values() for item in items):
        works = {key: items for key, items in works.items() if any(item["exact"] for item in items)}
    if not works:
        if other_formats:
            return {"status": "unsupported_format", "candidates": sorted(map(str, other_formats))}
        return {"status": "not_found", "query": {"title": title, "author": author}}
    if len(works) > 1:
        return {
            "status": "ambiguous",
            "works": [{"title": items[0]["title"], "authors": items[0]["authors"], "paths": [i["path"] for i in items]} for items in works.values()],
        }
    editions = sorted(next(iter(works.values())), key=lambda item: (item["drm"] is not None, item["path"]))
    chosen = editions[0]
    if chosen["drm"]:
        return {"status": "drm_restricted", "path": chosen["path"], "reason": chosen["drm"]}
    source = Path(chosen["path"])
    return {
        "status": "resolved",
        "source": str(source),
        "title": chosen["title"],
        "authors": chosen["authors"],
        "work_id": work_identity(chosen["title"], chosen["authors"]),
        "source_sha256": _sha256(source),
        "other_editions": [item["path"] for item in editions[1:]],
    }


# ##################################################################
# work identity
# stable id of the WORK (not the file): normalized title + first author
def work_identity(title: str, authors: list[str]) -> str:
    key = f"{title_key(title)}|{author_key(authors[0]) if authors else ''}"
    return "work-" + hashlib.sha256(key.encode()).hexdigest()[:16]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["author_key", "drm_reason", "epub_metadata", "load_index", "resolve_book", "title_key", "tokens", "work_identity"]
