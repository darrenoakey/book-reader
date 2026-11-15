"""Real-file tests for source_extend: an hourly project's bound source grows by a continuation."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.epub_extract import get_output_dir
from src.hour_runner import (
    load_lean_cast,
    load_ledger,
    source_chapters,
    source_fingerprint,
)
from src.source_extend import PENDING_NAME, extend_source

OLD_TEXT = "Ren met Luna at the gate.\n\nThe dragon slept.\n"


# ##################################################################
# write project
# an isolated produced project: extracted chapters, a ledger at EOF, a lean cast and an EOF target, all bound to the source.
def write_project(root: Path) -> tuple[Path, Path]:
    source = root / f"extend_story_{root.name}.txt"
    source.write_text(OLD_TEXT, encoding="utf-8")
    project = get_output_dir(source)
    chapters = project / "chapters"
    chapters.mkdir(parents=True)
    (chapters / "00-intro.txt").write_text("Story by Author, narrated by Narrator", encoding="utf-8")
    (chapters / "1-part_1.txt").write_text("Ren met Luna at the gate.", encoding="utf-8")
    (chapters / "2-part_2.txt").write_text("The dragon slept.", encoding="utf-8")
    fingerprint = source_fingerprint(source)
    ledger = {"source": str(source), "sha256": fingerprint, "hours": {"1": {"audio_complete": True, "next_chapter": 2, "next_piece": 0}}}
    (project / "hours.json").write_text(json.dumps(ledger), encoding="utf-8")
    lean = {"mode": "lean_existing_profiles_v1", "source_sha256": fingerprint, "actors": {"ren": {}}, "aliases": {"wren": "ren"}}
    (project / "lean_characters.json").write_text(json.dumps(lean), encoding="utf-8")
    target = {"source_sha256": fingerprint, "target_chapter": 2, "target_piece": 0}
    (project / "hour_continuation_target.json").write_text(json.dumps(target), encoding="utf-8")
    return source, project


# ##################################################################
# continuation text
# two paragraphs per chunk-sized block so the chunker must emit more than one new chapter.
def write_continuation(root: Path) -> Path:
    paragraph = " ".join(["The figure stepped into the throne room."] * 70)
    path = root / "continuation.txt"
    path.write_text("\n\n".join([paragraph] * 4), encoding="utf-8")
    return path


# ##################################################################
# test extend appends, numbers after last chapter and rebinds
# old bytes stay a prefix, old chapters are untouched, and every binding moves to the new fingerprint.
def test_extend_source_appends_and_rebinds() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, project = write_project(root)
        try:
            old_sha = source_fingerprint(source)
            record = extend_source(source, write_continuation(root))
            new_bytes = source.read_bytes()
            assert new_bytes.startswith(OLD_TEXT.encode("utf-8"))
            assert record["previous_sha256"] == old_sha
            assert record["previous_bytes"] == len(OLD_TEXT.encode("utf-8"))
            assert record["sha256"] == source_fingerprint(source) != old_sha
            assert record["first_chapter"] == "3-part_3.txt"
            _, _, chapters = source_chapters(source, project)
            names = [path.name for path in chapters]
            assert names[:2] == ["1-part_1.txt", "2-part_2.txt"]
            assert len(names) >= 4 and names[-1] == record["last_chapter"]
            assert (project / "chapters" / "2-part_2.txt").read_text(encoding="utf-8") == "The dragon slept."
            ledger = load_ledger(project, source)
            assert ledger["extensions"][-1]["sha256"] == ledger["sha256"]
            assert ledger["hours"]["1"]["next_chapter"] == 2
            cast, aliases = load_lean_cast(source, project)
            assert aliases == {"wren": "ren"} and "ren" in cast
            target = json.loads((project / "hour_continuation_target.json").read_text(encoding="utf-8"))
            assert target == {"source_sha256": ledger["sha256"], "target_chapter": len(names), "target_piece": 0}
            assert not (project / PENDING_NAME).exists()
        finally:
            shutil.rmtree(project, ignore_errors=True)


# ##################################################################
# test extend resumes after source replaced but before rebind
# a crash after the source swap is finished by re-running the same command.
def test_extend_source_resumes_pending_rebind() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, project = write_project(root)
        try:
            continuation = write_continuation(root)
            ledger_before = (project / "hours.json").read_text(encoding="utf-8")
            record = extend_source(source, continuation)
            # simulate the crash window: restore the pre-rebind ledger and pending record
            pending = {key: record[key] for key in ("previous_sha256", "previous_bytes", "sha256", "bytes", "first_chapter", "last_chapter")}
            pending["chapter_count"] = len(source_chapters(source, project)[2])
            (project / "hours.json").write_text(ledger_before, encoding="utf-8")
            (project / PENDING_NAME).write_text(json.dumps(pending), encoding="utf-8")
            with pytest.raises(RuntimeError):
                load_ledger(project, source)
            resumed = extend_source(source, continuation)
            assert resumed["sha256"] == source_fingerprint(source)
            assert load_ledger(project, source)["sha256"] == resumed["sha256"]
        finally:
            shutil.rmtree(project, ignore_errors=True)


# ##################################################################
# test extend refuses a different story
# a source that no longer matches the ledger is never extended.
def test_extend_source_rejects_mismatched_source() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, project = write_project(root)
        try:
            source.write_text("A different story entirely.\n", encoding="utf-8")
            with pytest.raises(RuntimeError):
                extend_source(source, write_continuation(root))
            assert not any(path.name.startswith("3-") for path in (project / "chapters").iterdir())
        finally:
            shutil.rmtree(project, ignore_errors=True)


# ##################################################################
# test extend resumes a crash before the source swap
# chapters already written by the interrupted run are reused when identical and refused when the continuation changed.
def test_extend_source_resumes_before_swap_and_refuses_changed_text() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, project = write_project(root)
        try:
            continuation = write_continuation(root)
            pending = {"previous_sha256": source_fingerprint(source), "sha256": "0" * 64, "first_chapter": "3-part_3.txt"}
            (project / PENDING_NAME).write_text(json.dumps(pending), encoding="utf-8")
            (project / "chapters" / "3-part_3.txt").write_text("interrupted different text", encoding="utf-8")
            with pytest.raises(RuntimeError):
                extend_source(source, continuation)
            assert source.read_text(encoding="utf-8") == OLD_TEXT
            (project / "chapters" / "3-part_3.txt").unlink()
            record = extend_source(source, continuation)
            assert record["first_chapter"] == "3-part_3.txt"
            assert load_ledger(project, source)["sha256"] == source_fingerprint(source)
        finally:
            shutil.rmtree(project, ignore_errors=True)
