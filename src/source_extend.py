"""Append newly released story text to an hourly project's bound source.

A long serial (e.g. a web novel still being released) outgrows the source an
hourly project was bound to. ``extend_source`` appends a continuation without
disturbing anything already produced:

* the old source bytes stay an exact prefix of the new source;
* existing chapter files are never rewritten; the continuation is chunked with
  the same word-driven chunker and numbered after the last existing chapter;
* the hour ledger, lean cast summary and continuation EOF target are re-bound
  from the previous source SHA-256 to the new one, and the ledger records the
  extension (previous SHA, previous byte length, appended chapter range).

The operation is resumable: a pending record is written before the source is
replaced, so a crash between replacing the source and re-binding the project
is finished by re-running the same command. Any other mismatch fails closed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from src.epub_extract import normalize_name
from src.hour_runner import (
    atomic_json,
    chapter_order,
    source_chapters,
    source_fingerprint,
)
from src.pipeline import get_output_dir
from src.text_ingest import chunk_paragraphs

PENDING_NAME = "source_extension_pending.json"
LEAN_NAME = "lean_characters.json"
TARGET_NAME = "hour_continuation_target.json"


# ##################################################################
# sha256 bytes
# digest helper shared by the source and chapter bindings.
def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ##################################################################
# extended bytes
# the new source is the old bytes verbatim, a paragraph break, then the continuation.
def extended_bytes(old: bytes, addition: str) -> bytes:
    separator = b"" if old.endswith(b"\n\n") else (b"\n" if old.endswith(b"\n") else b"\n\n")
    return old + separator + addition.strip().encode("utf-8") + b"\n"


# ##################################################################
# planned chapters
# chunk the continuation and name each chunk after the project's last chapter, matching its prefix width.
def planned_chapters(chapters: list[Path], addition: str) -> list[tuple[str, str]]:
    last = chapters[-1]
    last_number = chapter_order(last)[0]
    width = len(last.name.partition("-")[0])
    planned = []
    for offset, chunk in enumerate(chunk_paragraphs(addition.strip()), start=1):
        number = last_number + offset
        name = f"{str(number).zfill(width)}-{normalize_name(f'Part {number}')[:40]}.txt"
        planned.append((name, chunk))
    return planned


# ##################################################################
# write chapters
# create each new chapter file once; an existing file with different bytes is a conflict, never overwritten.
def write_chapters(chapters_dir: Path, planned: list[tuple[str, str]]) -> None:
    for name, chunk in planned:
        path = chapters_dir / name
        if path.exists():
            if path.read_text(encoding="utf-8") != chunk:
                raise RuntimeError(f"chapter {name} already exists with different text; refusing to overwrite")
            continue
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(chunk, encoding="utf-8")
        temporary.replace(path)


# ##################################################################
# rebind
# move every source-bound project record from the previous fingerprint to the new one and log the extension.
def rebind(project: Path, pending: dict) -> dict:
    previous, current = pending["previous_sha256"], pending["sha256"]
    ledger_path = project / "hours.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("sha256") != previous:
        raise RuntimeError("hour ledger is not bound to the pending extension's previous source")
    lean_path = project / LEAN_NAME
    lean = json.loads(lean_path.read_text(encoding="utf-8")) if lean_path.exists() else None
    if lean is not None and lean.get("source_sha256") not in (previous, current):
        raise RuntimeError("lean character summary is bound to a different source")
    target_path = project / TARGET_NAME
    target = json.loads(target_path.read_text(encoding="utf-8")) if target_path.exists() else None
    if target is not None and target.get("source_sha256") not in (previous, current):
        raise RuntimeError("continuation target is bound to a different source")
    if lean is not None:
        lean["source_sha256"] = current
        atomic_json(lean_path, lean)
    if target is not None:
        target.update(
            {"source_sha256": current, "target_chapter": pending["chapter_count"], "target_piece": 0}
        )
        atomic_json(target_path, target)
    record = {key: pending[key] for key in ("previous_sha256", "previous_bytes", "sha256", "bytes", "first_chapter", "last_chapter")}
    record["at"] = datetime.now(UTC).isoformat()
    ledger["sha256"] = current
    ledger.setdefault("extensions", []).append(record)
    atomic_json(ledger_path, ledger)
    (project / PENDING_NAME).unlink()
    return record


# ##################################################################
# extend source
# append a continuation to an hourly project's source and re-bind the project; resumable and fail-closed.
def extend_source(source: Path, continuation: Path) -> dict:
    source = source.resolve()
    project = get_output_dir(source)
    ledger_path = project / "hours.json"
    if not ledger_path.exists():
        raise RuntimeError("no hour ledger: an unproduced source needs no extension, just edit it")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    fingerprint = source_fingerprint(source)
    pending_path = project / PENDING_NAME
    resume_from = None
    if pending_path.exists():
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        if fingerprint == pending["sha256"]:
            return rebind(project, pending)
        if fingerprint != pending["previous_sha256"]:
            raise RuntimeError("source matches neither side of the pending extension")
        # crashed before the source swap: chapters from the pending first chapter on are this extension's own.
        resume_from = chapter_order(Path(pending["first_chapter"]))[0]
    if ledger.get("sha256") != fingerprint:
        raise RuntimeError("hour ledger source SHA-256 does not match input; refusing to extend a different story")
    addition = continuation.read_text(encoding="utf-8")
    if not addition.strip():
        raise ValueError("continuation is empty")
    old = source.read_bytes()
    _, _, chapters = source_chapters(source, project)
    if resume_from is not None:
        chapters = [path for path in chapters if chapter_order(path)[0] < resume_from]
    planned = planned_chapters(chapters, addition)
    new = extended_bytes(old, addition)
    pending = {
        "previous_sha256": fingerprint,
        "previous_bytes": len(old),
        "sha256": sha256_bytes(new),
        "bytes": len(new),
        "first_chapter": planned[0][0],
        "last_chapter": planned[-1][0],
        "chapter_count": len(chapters) + len(planned),
    }
    atomic_json(pending_path, pending)
    write_chapters(project / "chapters", planned)
    temporary = source.with_name(source.name + ".extending")
    temporary.write_bytes(new)
    temporary.replace(source)
    return rebind(project, pending)
