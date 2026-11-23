"""Deliver a finished M4B off this Mac, and prove only what is provable.

Two independent, deliberately separated routes:

``copy_to_icloud_files`` — the private route. The M4B is placed in the owner's
own iCloud Drive folder, byte-verified after placement, and optionally re-read
through read-only Foundation metadata. This proves a private iCloud Drive copy
of exactly these bytes exists under the configured folder. It does **not** prove
an Apple Books import, an iPhone/iPad download, or any remote/device sync: the
metadata service reports the *system's* upload state for the container, and no
public interface exists for "this audiobook is on that device". Those three are
reported as explicitly not claimed rather than guessed.

``deliver_to_books`` — the local route. Apple Books has no scripting API for
importing audiobooks; the supported noninteractive route is handing the file to
the Books app (``open -a Books``), which copies it into its own container and
registers a library asset. This is only ever reported from evidence: a Books
library row (content type 6, audiobook, not hidden) whose stored file is
byte-identical (sha256) to the delivered M4B. Nothing is reported delivered on a
launch alone, and a Books library row says nothing about any other device.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

LIBRARY_GLOB = "~/Library/Containers/com.apple.iBooksX/Data/Documents/BKLibrary/BKLibrary-*.sqlite"
AUDIOBOOK_CONTENT_TYPE = 6
ICLOUD_CONTAINER = "com~apple~CloudDocs"
UBIQUITOUS_PROBE_TIMEOUT_SECONDS = 30

# Read-only Foundation metadata for one ubiquitous item. Nothing here writes,
# launches or drives the file system: it asks the sync service what IT thinks.
UBIQUITOUS_PROBE = """
function run(argv) {
  ObjC.import('Foundation');
  var url = $.NSURL.fileURLWithPath(argv[0]);
  function value(key) {
    var values = url.resourceValuesForKeysError($([key]), $());
    if (!values) { return null; }
    var raw = values.objectForKey(key);
    return raw ? ObjC.unwrap(raw) : null;
  }
  var result = {
    ubiquitous: value('NSURLIsUbiquitousItemKey'),
    uploaded: value('NSURLUbiquitousItemIsUploadedKey'),
    uploading: value('NSURLUbiquitousItemIsUploadingKey'),
    download_status: value('NSURLUbiquitousItemDownloadingStatusKey'),
    container_display_name: value('NSURLUbiquitousItemContainerDisplayNameKey')
  };
  var present = {};
  for (var name in result) { if (result[name] !== null) { present[name] = result[name]; } }
  return JSON.stringify(present);
}
"""


# ##################################################################
# library database
# the single Books library database; absent or ambiguous fails closed
def library_database() -> Path:
    matches = [Path(p) for p in glob.glob(str(Path(LIBRARY_GLOB).expanduser())) if not p.endswith(("-shm", "-wal"))]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one Apple Books library database, found {len(matches)}")
    return matches[0]


# ##################################################################
# audiobook assets
# every visible audiobook asset Books has registered: (asset_id, title, stored path)
def audiobook_assets(database: Path | None = None) -> list[dict]:
    database = database or library_database()
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=10)
    try:
        rows = connection.execute(
            "SELECT ZASSETID, ZTITLE, ZPATH FROM ZBKLIBRARYASSET "
            "WHERE ZCONTENTTYPE = ? AND COALESCE(ZISHIDDEN, 0) = 0 AND ZPATH IS NOT NULL",
            (AUDIOBOOK_CONTENT_TYPE,),
        ).fetchall()
    finally:
        connection.close()
    return [{"asset_id": asset_id, "title": title, "path": path} for asset_id, title, path in rows]


# ##################################################################
# find delivered asset
# the registered audiobook whose stored file is byte-identical to `m4b`
def find_delivered_asset(m4b: Path, database: Path | None = None) -> dict | None:
    size = m4b.stat().st_size
    digest = sha256_file(m4b)
    for asset in audiobook_assets(database):
        stored = Path(asset["path"])
        candidates = sorted(stored.glob("*.m4b")) if stored.is_dir() else [stored]
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size == size and sha256_file(candidate) == digest:
                return {**asset, "stored_file": str(candidate), "sha256": digest, "bytes": size}
    return None


# ##################################################################
# deliver to books
# idempotent: an already-registered identical file is never imported twice
def deliver_to_books(m4b: Path, timeout_seconds: float = 300, poll_seconds: float = 5) -> dict:
    database = library_database()
    found = find_delivered_asset(m4b, database)
    if found is not None:
        return {**found, "imported_now": False}
    subprocess.run(["open", "-g", "-a", "Books", str(m4b)], check=True, timeout=60)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        time.sleep(poll_seconds)
        found = find_delivered_asset(m4b, database)
        if found is not None:
            return {**found, "imported_now": True}
    raise RuntimeError(f"Apple Books did not register a byte-identical copy of {m4b.name} within {timeout_seconds:.0f}s")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ##################################################################
# private delivery target
# the configured folder must be a real directory inside the iCloud container
def icloud_container() -> Path:
    return Path("~/Library/Mobile Documents").expanduser() / ICLOUD_CONTAINER


def validate_icloud_root(icloud_root: Path, container: Path | None = None) -> Path:
    root = Path(icloud_root).expanduser()
    if root.is_symlink():
        raise ValueError(f"iCloud delivery root {root} is a symlink; configure the real folder")
    if not root.is_dir():
        raise ValueError(f"iCloud delivery root {root} does not exist or is not a directory")
    container = Path(container).expanduser().resolve() if container is not None else icloud_container()
    try:
        resolved = root.resolve()
    except OSError as error:
        raise ValueError(f"iCloud delivery root {root} cannot be resolved: {error}") from error
    if resolved == container or not resolved.is_relative_to(container):
        raise ValueError(f"iCloud delivery root {root} is not a folder inside {container}")
    return resolved


# ##################################################################
# private delivery evidence
# what a private iCloud copy proves, stated exactly and nothing more
def icloud_delivery_evidence(copied: dict) -> dict:
    return {
        "kind": "icloud_files_private_copy",
        "path": copied["path"],
        "sha256": copied["sha256"],
        "bytes": copied["bytes"],
        "container_display_name": copied.get("container"),
        "metadata": copied.get("metadata"),
        "books_import": "not_claimed",
        "ios_device_sync": "not_claimed",
        "remote_sync_service": "not_claimed",
        "verified_via": "read_only_metadata" if copied.get("metadata") else "bytes_only",
        "verified": copied["verified"],
    }


# ##################################################################
# ubiquitous metadata
# read-only sync state as the SYSTEM reports it; absent when it is unavailable
def ubiquitous_metadata(path: Path) -> dict | None:
    result = subprocess.run(
        ["osascript", "-l", "JavaScript", "-e", UBIQUITOUS_PROBE, str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=UBIQUITOUS_PROBE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        return None
    try:
        metadata = json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return None
    return metadata or None


# ##################################################################
# copy to icloud files
# place `m4b` in the private iCloud folder, no-clobber, then prove the bytes
def copy_to_icloud_files(
    m4b: Path, icloud_root: Path, *, destination: Path | None = None, container: Path | None = None
) -> dict:
    source = Path(m4b)
    if not source.is_file():
        raise ValueError(f"{source} is not a readable file")
    root = validate_icloud_root(icloud_root, container)
    target = _resolve_destination(root, source.name, destination)
    byte_count = source.stat().st_size
    digest = sha256_file(source)
    existent = False
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"iCloud destination {target} exists and is not a regular file")
        if sha256_file(target) != digest:
            raise ValueError(f"iCloud destination {target} already holds different bytes; refusing to overwrite")
        existent = True
        placed = "reused"
    else:
        placed = _place_atomically(source, target, digest)
    stored = sha256_file(target)
    if stored != digest or target.stat().st_size != byte_count:
        raise ValueError(f"iCloud copy at {target} does not match the delivered bytes")
    metadata = ubiquitous_metadata(target)
    return {
        "path": str(target),
        "sha256": stored,
        "bytes": target.stat().st_size,
        "created": not existent,
        "reused": existent,
        "already_present": existent,
        "atomic_placement": placed,
        "verified": True,
        "icloud_root": str(root),
        "container": (metadata or {}).get("container_display_name"),
        "metadata": metadata,
    }


# ##################################################################
# delivery destination
# `destination` may name a subfolder that already exists inside the root, so the
# real per-book layout (Audiobooks/<Book>/<file>.m4b) works; it may never escape
def _resolve_destination(root: Path, name: str, destination: Path | str | None) -> Path:
    if destination is None:
        return root / name
    candidate = Path(destination).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    target = candidate.resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"destination {target} is not inside {root}")
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"destination folder {parent} does not exist or is a symlink")
    if target.exists() and target.is_dir():
        raise ValueError(f"destination {target} is a directory, not a file")
    return target


# ##################################################################
# atomic no-clobber placement
# a partial copy is never visible under the final name and never overwrites
def _place_atomically(source: Path, target: Path, digest: str) -> str:
    partial = target.with_name(f".{target.name}.partial-{uuid.uuid4().hex}")
    try:
        shutil.copyfile(source, partial)
        if sha256_file(partial) != digest:
            raise ValueError(f"partial copy of {source.name} does not match its source bytes")
        try:
            os.link(partial, target)
            placed = "link"
        except FileExistsError as error:
            raise ValueError(f"iCloud destination {target} appeared during delivery; refusing to overwrite") from error
        except OSError:
            # A file system without hard links still gets an atomic rename, which
            # is only taken when the final name is still free.
            if target.exists():
                raise ValueError(f"iCloud destination {target} appeared during delivery; refusing to overwrite")
            os.replace(partial, target)
            placed = "rename"
    finally:
        partial.unlink(missing_ok=True)
    return placed


__all__ = [
    "audiobook_assets", "copy_to_icloud_files", "deliver_to_books", "find_delivered_asset",
    "icloud_container", "icloud_delivery_evidence", "library_database", "sha256_file", "ubiquitous_metadata",
    "validate_icloud_root",
]
