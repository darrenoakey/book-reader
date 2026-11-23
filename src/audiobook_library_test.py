"""Real EPUB packages on disk exercise the title/author resolver end to end."""

import tempfile
import zipfile
from pathlib import Path

from src.audiobook_library import (
    author_key,
    drm_reason,
    epub_metadata,
    load_index,
    resolve_book,
    title_key,
    work_identity,
)

FONT_OBFUSCATION = """<?xml version="1.0"?>
<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" xmlns:enc="http://www.w3.org/2001/04/xmlenc#">
 <enc:EncryptedData><enc:EncryptionMethod Algorithm="http://www.idpf.org/2008/embedding"/>
  <enc:CipherData><enc:CipherReference URI="OEBPS/fonts/serif.otf"/></enc:CipherData></enc:EncryptedData>
</encryption>"""
CONTENT_ENCRYPTION = FONT_OBFUSCATION.replace("http://www.idpf.org/2008/embedding", "http://www.w3.org/2001/04/xmlenc#aes128-cbc").replace(
    "OEBPS/fonts/serif.otf", "OEBPS/chapter1.xhtml"
)


def make_epub(path: Path, title: str, author: str, extra: dict[str, str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
 <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{title}</dc:title><dc:creator>{author}</dc:creator></metadata>
 <manifest><item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/></manifest>
 <spine><itemref idref="c1"/></spine>
</package>"""
    container = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
 <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/chapter1.xhtml", f"<html><body><p>{title} opens.</p></body></html>")
        for name, text in (extra or {}).items():
            archive.writestr(name, text)
    return path


def library(root: Path) -> list[str]:
    make_epub(root / "SF" / "Anne McCaffrey - Pern 03 - Dragonsong.epub", "Dragonsong", "McCaffrey Anne")
    make_epub(root / "By Author" / "McCaffrey" / "Dragonsong.epub", "Dragonsong", "Anne McCaffrey")
    make_epub(root / "SF" / "Dennis McKiernan - The Dragonstone.epub", "The Dragonstone", "Dennis L. McKiernan")
    make_epub(root / "SF" / "Anne McCaffrey - Pern 04 - Dragonsinger.epub", "Dragonsinger", "McCaffrey Anne")
    make_epub(root / "Other" / "Jane Roe - Harbor.epub", "Harbor", "Jane Roe")
    make_epub(root / "Other" / "John Doe - Harbor.epub", "Harbor", "John Doe")
    make_epub(
        root / "Locked" / "Ann Locke - Sealed Vault.epub", "Sealed Vault", "Ann Locke", {"META-INF/encryption.xml": CONTENT_ENCRYPTION}
    )
    make_epub(
        root / "Fonts" / "Ida Font - Obfuscated Serif.epub", "Obfuscated Serif", "Ida Font", {"META-INF/encryption.xml": FONT_OBFUSCATION}
    )
    (root / "Kindle").mkdir()
    (root / "Kindle" / "Ann Other - Only On Kindle.mobi").write_bytes(b"BOOKMOBI")
    return load_index(root, root / "Audiobook Productions")


def test_exact_title_with_reordered_author_resolves_one_work_across_editions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        index = library(Path(tmp))
        found = resolve_book("Dragonsong", "Anne McCaffrey", index)
        assert found["status"] == "resolved"
        assert found["title"] == "Dragonsong"
        assert len(found["other_editions"]) == 1  # the second file is an edition of the same work
        assert found["work_id"] == work_identity("Dragonsong", ["Anne McCaffrey"]) == work_identity("Dragonsong", ["McCaffrey Anne"])
        again = resolve_book("dragonsong", None, index)
        assert again["source"] == found["source"]  # deterministic edition choice


def test_exact_title_outranks_near_spelling_and_typos_still_resolve() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        index = library(Path(tmp))
        assert resolve_book("Dragonsong", None, index)["title"] == "Dragonsong"  # not ambiguous with The Dragonstone
        assert resolve_book("Dragonsnog", "McCaffrey", index)["title"] == "Dragonsong"
        assert resolve_book("Dragon song", None, index)["title"] == "Dragonsong"
        assert resolve_book("The Dragonstone", None, index)["title"] == "The Dragonstone"
        assert resolve_book("Dragonsinger", "anne mccaffrey", index)["title"] == "Dragonsinger"


def test_distinct_works_sharing_a_title_are_ambiguous_until_author_given() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        index = library(Path(tmp))
        found = resolve_book("Harbor", None, index)
        assert found["status"] == "ambiguous"
        assert sorted(work["authors"][0] for work in found["works"]) == ["Jane Roe", "John Doe"]
        assert resolve_book("Harbor", "Roe", index)["authors"] == ["Jane Roe"]


def test_drm_formats_and_missing_books_are_reported_not_processed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        index = library(Path(tmp))
        assert resolve_book("Sealed Vault", None, index)["status"] == "drm_restricted"
        assert resolve_book("Obfuscated Serif", None, index)["status"] == "resolved"  # font mangling is not DRM
        assert resolve_book("Only On Kindle", None, index)["status"] == "unsupported_format"
        assert resolve_book("No Such Book", None, index)["status"] == "not_found"


def test_metadata_and_keys() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = make_epub(Path(tmp) / "a.epub", "The Harper Hall", "McCaffrey Anne")
        assert epub_metadata(path) == {"title": "The Harper Hall", "authors": ["McCaffrey Anne"], "drm": None}
    assert title_key("The Harper Hall") == "harper hall"
    assert author_key("McCaffrey, Anne") == author_key("Anne McCaffrey")
    assert drm_reason({"META-INF/rights.xml"}, "") is not None


def test_index_is_cached_and_excludes_the_delivery_root() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        index = library(root)
        assert not any("Audiobook Productions" in path for path in index)
        make_epub(root / "New" / "Late Arrival.epub", "Late Arrival", "A. Writer")
        assert load_index(root, root / "Audiobook Productions") == index  # cached
        refreshed = load_index(root, root / "Audiobook Productions", refresh=True)
        assert any(path.endswith("Late Arrival.epub") for path in refreshed)
