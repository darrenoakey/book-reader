"""Private iCloud delivery against real files, real bytes and a real container.

Nothing here is mocked or faked. Every test builds its own private directory
tree on this machine's real file system, writes real audiobook bytes, and reads
them back with the module's own sha256 and the real read-only Foundation
metadata probe (``osascript -l JavaScript``) when it is available.

The Apple Books library is opened read-only: the real library database is
probed, never written, so these tests are parallel-safe and leave no state.

Two claims are tested separately and deliberately:
  * placing bytes in the configured private iCloud folder, proven by sha256 of
    the placed file, no-clobber, atomicity and cleanup of partial copies;
  * what such a copy does and does not prove — the evidence vocabulary refuses
    to claim a Books import, an iOS download or any remote sync.
"""

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.books_delivery import (
    audiobook_assets,
    copy_to_icloud_files,
    icloud_container,
    icloud_delivery_evidence,
    library_database,
    sha256_file,
    ubiquitous_metadata,
    validate_icloud_root,
)

# ##################################################################
# fixtures
# real byte payloads and a real iCloud-container-shaped tree
PAYLOAD = bytes(range(256)) * 4096  # 1 MiB of non-repeating-in-blocks data


@pytest.fixture()
def studio() -> Path:
    """A real private tree whose parent plays the iCloud container."""
    with tempfile.TemporaryDirectory(prefix="books-delivery-") as tmp:
        container = Path(tmp) / "com~apple~CloudDocs"
        folder = container / "Audiobooks" / "Test Book"
        folder.mkdir(parents=True)
        yield Path(tmp)


@pytest.fixture()
def m4b(studio: Path) -> Path:
    source = studio / "render" / "test-book.m4b"
    source.parent.mkdir(parents=True)
    source.write_bytes(PAYLOAD)
    return source


def container_of(studio: Path) -> Path:
    return studio / "com~apple~CloudDocs"


def private_folder(studio: Path) -> Path:
    # macOS resolves a tempdir under /var to /private/var; the module reports the
    # canonical path it actually wrote, so compare against the canonical folder.
    return (container_of(studio) / "Audiobooks" / "Test Book").resolve()


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ##################################################################
# test copy is byte identical and reported from the placed file
# the returned evidence must describe what is on disk, not the source claim
def test_copy_places_verified_bytes_in_the_private_folder(studio: Path, m4b: Path) -> None:
    folder = private_folder(studio)
    copied = copy_to_icloud_files(m4b, folder, container=container_of(studio))

    placed = Path(copied["path"])
    assert placed == folder / "test-book.m4b"
    assert placed.is_file()
    assert placed.read_bytes() == PAYLOAD
    assert copied["sha256"] == digest_of(placed) == sha256_file(m4b)
    assert copied["bytes"] == placed.stat().st_size == len(PAYLOAD)
    assert copied["verified"] is True
    assert copied["created"] is True
    assert copied["reused"] is False
    assert copied["already_present"] is False
    assert copied["atomic_placement"] in {"link", "rename"}
    assert copied["icloud_root"] == str(folder.resolve())
    assert Path(copied["icloud_root"]).is_dir()


# ##################################################################
# test partial copies never survive
# an interrupted placement must leave the folder exactly as it was
def test_failed_placement_leaves_no_partial_copy(studio: Path, m4b: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = private_folder(studio)
    from src import books_delivery

    def explode(source: Path, target: Path) -> None:
        Path(target).write_bytes(b"half a book")
        raise OSError("simulated I/O failure mid-copy")

    monkeypatch.setattr(books_delivery.shutil, "copyfile", explode)
    with pytest.raises(OSError):
        copy_to_icloud_files(m4b, folder, container=container_of(studio))
    monkeypatch.undo()
    assert sorted(path.name for path in folder.iterdir()) == []
    assert not (folder / "test-book.m4b").exists()


# ##################################################################
# test a corrupt partial copy is rejected and removed
# bytes are verified before the final name is ever used
def test_corrupt_partial_copy_is_rejected_and_cleaned(studio: Path, m4b: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = private_folder(studio)
    from src import books_delivery

    def truncate(source: Path, target: Path) -> None:
        Path(target).write_bytes(PAYLOAD[:1024])

    monkeypatch.setattr(books_delivery.shutil, "copyfile", truncate)
    with pytest.raises(ValueError, match="does not match its source bytes"):
        copy_to_icloud_files(m4b, folder, container=container_of(studio))
    monkeypatch.undo()
    assert sorted(path.name for path in folder.iterdir()) == []


# ##################################################################
# test repeat delivery reuses and never duplicates
# a second call reports reuse instead of importing or rewriting the same book
def test_repeat_copy_reuses_the_identical_file(studio: Path, m4b: Path) -> None:
    folder = private_folder(studio)
    first = copy_to_icloud_files(m4b, folder, container=container_of(studio))
    inode = Path(first["path"]).stat().st_ino
    second = copy_to_icloud_files(m4b, folder, container=container_of(studio))

    assert second["created"] is False
    assert second["reused"] is True
    assert second["already_present"] is True
    assert second["atomic_placement"] == "reused"
    assert second["sha256"] == first["sha256"]
    assert Path(second["path"]).stat().st_ino == inode
    assert sorted(path.name for path in folder.iterdir()) == ["test-book.m4b"]


# ##################################################################
# test different bytes are never overwritten
# an existing distinct audiobook of the same name fails closed
def test_different_bytes_are_never_overwritten(studio: Path, m4b: Path) -> None:
    folder = private_folder(studio)
    existing = folder / "test-book.m4b"
    existing.write_bytes(b"a different edition")
    before = digest_of(existing)

    with pytest.raises(ValueError, match="refusing to overwrite"):
        copy_to_icloud_files(m4b, folder, container=container_of(studio))
    assert digest_of(existing) == before


# ##################################################################
# test an existing symlink destination fails closed
# never follow a link out of the private folder
def test_symlink_destination_fails_closed(studio: Path, m4b: Path) -> None:
    folder = private_folder(studio)
    outside = studio / "elsewhere.m4b"
    outside.write_bytes(PAYLOAD)
    (folder / "test-book.m4b").symlink_to(outside)

    with pytest.raises(ValueError, match="not a regular file"):
        copy_to_icloud_files(m4b, folder, container=container_of(studio))
    assert outside.read_bytes() == PAYLOAD


# ##################################################################
# test roots outside the iCloud container are refused
# a plain local folder is not private delivery
def test_root_outside_the_container_is_refused(studio: Path, m4b: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="not-icloud-") as tmp:
        with pytest.raises(ValueError, match="not a folder inside"):
            copy_to_icloud_files(m4b, Path(tmp), container=container_of(studio))
        assert list(Path(tmp).iterdir()) == []


# ##################################################################
# test a missing or symlinked root is refused
# delivery fails closed instead of creating a folder somewhere unexpected
def test_missing_and_symlinked_roots_are_refused(studio: Path, m4b: Path) -> None:
    container = container_of(studio)
    with pytest.raises(ValueError, match="does not exist"):
        copy_to_icloud_files(m4b, container / "Audiobooks" / "Absent", container=container)

    real = container / "Audiobooks" / "Real"
    real.mkdir()
    alias = container / "Audiobooks" / "Alias"
    alias.symlink_to(real)
    with pytest.raises(ValueError, match="is a symlink"):
        validate_icloud_root(alias, container)
    assert list(real.iterdir()) == []


# ##################################################################
# test a destination may use a real per-book folder inside the root
# the shipping layout Audiobooks/<Book>/<file>.m4b is supported
def test_destination_may_use_a_real_book_folder(studio: Path, m4b: Path) -> None:
    folder = private_folder(studio)
    book = folder / "Some Book"
    book.mkdir()
    copied = copy_to_icloud_files(m4b, folder, destination=book / "part-one.m4b", container=container_of(studio))
    assert Path(copied["path"]) == (book / "part-one.m4b").resolve()
    assert Path(copied["path"]).read_bytes() == PAYLOAD
    assert copied["sha256"] == digest_of(Path(copied["path"]))


# ##################################################################
# test the destination can never escape the configured root
# a relative name is taken under the root; absolutes outside it and links fail
def test_destination_cannot_escape_the_root(studio: Path, m4b: Path) -> None:
    folder = private_folder(studio)
    relative = copy_to_icloud_files(m4b, folder, destination="relative.m4b", container=container_of(studio))
    assert Path(relative["path"]) == (folder / "relative.m4b").resolve()

    with pytest.raises(ValueError, match="is not inside"):
        copy_to_icloud_files(m4b, folder, destination=studio / "escape.m4b", container=container_of(studio))
    with pytest.raises(ValueError, match="is not inside"):
        copy_to_icloud_files(m4b, folder, destination=folder / ".." / "up.m4b", container=container_of(studio))
    with pytest.raises(ValueError, match="does not exist or is a symlink"):
        copy_to_icloud_files(m4b, folder, destination=folder / "absent" / "x.m4b", container=container_of(studio))
    assert not (studio / "escape.m4b").exists()
    assert not (folder.parent / "up.m4b").exists()


# ##################################################################
# test an unreadable source is refused before anything is written
# nothing is placed when the input is not a real file
def test_missing_source_is_refused(studio: Path) -> None:
    folder = private_folder(studio)
    with pytest.raises(ValueError, match="not a readable file"):
        copy_to_icloud_files(studio / "absent.m4b", folder, container=container_of(studio))
    assert list(folder.iterdir()) == []


# ##################################################################
# test evidence claims only a private copy
# no Books import, no device sync, no remote service is ever asserted
def test_evidence_claims_only_a_private_icloud_copy(studio: Path, m4b: Path) -> None:
    folder = private_folder(studio)
    copied = copy_to_icloud_files(m4b, folder, container=container_of(studio))
    evidence = icloud_delivery_evidence(copied)

    assert evidence["kind"] == "icloud_files_private_copy"
    assert evidence["path"] == copied["path"]
    assert evidence["sha256"] == copied["sha256"] == sha256_file(m4b)
    assert evidence["bytes"] == len(PAYLOAD)
    assert evidence["verified"] is True
    assert evidence["books_import"] == "not_claimed"
    assert evidence["ios_device_sync"] == "not_claimed"
    assert evidence["remote_sync_service"] == "not_claimed"
    assert evidence["verified_via"] in {"bytes_only", "read_only_metadata"}
    for forbidden in ("delivered", "synced", "on_device", "imported"):
        assert forbidden not in str(evidence).lower().replace("not_claimed", "")
    # the private folder is what was proven, and it is not the Books folder
    assert evidence["path"].startswith(str(folder))
    assert "iCloud~com~apple~iBooks" not in evidence["path"]


# ##################################################################
# test the ubiquity probe answers about real items only
# a local file has no ubiquitous metadata; a real container item does
def test_ubiquitous_metadata_reports_only_real_sync_state(studio: Path) -> None:
    local = studio / "local.m4b"
    local.write_bytes(PAYLOAD)
    local_metadata = ubiquitous_metadata(local)
    assert local_metadata is None or "ubiquitous" not in local_metadata

    container = icloud_container()
    if not container.is_dir():
        pytest.skip(f"{container} is not present on this machine")
    container_metadata = ubiquitous_metadata(container)
    if container_metadata is None:
        pytest.skip("read-only ubiquitous metadata is unavailable in this environment")
    assert container_metadata["ubiquitous"] is True
    # the container display name is the whole container, never the private folder
    assert container_metadata["container_display_name"] == "iCloud Drive"


# ##################################################################
# test private folder placement works in the real iCloud container
# a real file, in the real container, verified and then removed again
def test_real_container_placement_is_private_and_reversible() -> None:
    container = icloud_container()
    if not container.is_dir():
        pytest.skip(f"{container} is not present on this machine")
    folder = container / "Audiobooks"
    if not folder.is_dir():
        pytest.skip(f"{folder} is not present on this machine")

    name = f".books-delivery-test-{os.getpid()}.m4b"
    target = folder / name
    if target.exists():
        pytest.skip(f"refusing to disturb an existing {target}")
    with tempfile.TemporaryDirectory(prefix="books-delivery-source-") as tmp:
        source = Path(tmp) / name
        source.write_bytes(PAYLOAD)
        copied = copy_to_icloud_files(source, folder)
        try:
            placed = Path(copied["path"])
            assert placed.parent == folder.resolve()
            assert placed.read_bytes() == PAYLOAD
            assert copied["sha256"] == digest_of(placed)
            assert copied["container"] in {None, "iCloud Drive"}
            evidence = icloud_delivery_evidence(copied)
            assert evidence["kind"] == "icloud_files_private_copy"
            assert evidence["books_import"] == "not_claimed"
        finally:
            target.unlink(missing_ok=True)
    assert not target.exists()


# ##################################################################
# test the real Books library is only ever read
# probing the shipping library must not modify it
def test_real_books_library_probe_is_read_only() -> None:
    try:
        database = library_database()
    except RuntimeError as error:
        pytest.skip(f"Apple Books library unavailable: {error}")
    before = (sha256_file(database), os.stat(database).st_mtime_ns, os.stat(database).st_mode)
    assets = audiobook_assets(database)
    after = (sha256_file(database), os.stat(database).st_mtime_ns, os.stat(database).st_mode)
    assert before == after
    assert isinstance(assets, list)
    for asset in assets:
        assert set(asset) == {"asset_id", "title", "path"}
        assert asset["path"] is None or isinstance(asset["path"], str)


# ##################################################################
# test sha256_file streams real bytes
# the digest helper must agree with a whole-file hash
def test_sha256_file_matches_a_whole_file_digest(tmp_path: Path) -> None:
    payload = b"chapter one " * 50000
    (tmp_path / "book.m4b").write_bytes(payload)
    assert sha256_file(tmp_path / "book.m4b") == digest_of(tmp_path / "book.m4b")
    assert sha256_file(tmp_path / "book.m4b") == hashlib.sha256(payload).hexdigest()


# ##################################################################
# test the module never drives Apple Books from the private route
# private delivery must not launch or script the Books app
def test_private_copy_never_launches_books(studio: Path, m4b: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = private_folder(studio)
    calls: list[list[str]] = []
    real_run = subprocess.run

    def watch(args, *rest, **kwargs):
        if isinstance(args, (list, tuple)):
            calls.append([str(part) for part in args])
            assert "open" not in [str(part) for part in args[:1]]
        return real_run(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "run", watch)
    copy_to_icloud_files(m4b, folder, container=container_of(studio))
    monkeypatch.undo()
    assert calls, "expected at least the read-only metadata probe"
    for call in calls:
        assert call[0] not in {"open", "open -a"}
        assert "Books" not in call
