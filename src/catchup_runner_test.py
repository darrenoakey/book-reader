"""Real filesystem tests for the single-chapter catch-up runner (namespaced worktree output)."""

import json
import shutil
import uuid
from pathlib import Path

import pytest

from src.catchup_runner import main, run_catchup, script_speakers
from src.epub_extract import get_output_dir
from src.hour_runner import atomic_json, source_fingerprint
from src.pipeline import acquire_lock, release_lock


@pytest.fixture()
def book(tmp_path: Path):
    source = tmp_path / f"catchup_test_{uuid.uuid4().hex[:10]}.txt"
    source.write_text("Chapter 1\n\nIt began.\n\nChapter 2\n\nIt went on.\n", encoding="utf-8")
    project = get_output_dir(source)
    yield source, project
    shutil.rmtree(project, ignore_errors=True)


def test_script_speakers_reads_only_valid_single_key_lines(tmp_path: Path) -> None:
    script = tmp_path / "s.jsonl"
    script.write_text('{"narrator": "a"}\n\nnot json\n{"ren": "b"}\n{"a": 1, "b": 2}\n', encoding="utf-8")
    assert script_speakers(script) == {"narrator", "ren"}


def test_unknown_chapter_is_refused(book) -> None:
    source, _ = book
    with pytest.raises(RuntimeError, match="not a unique source chapter"):
        run_catchup(source, 99, defer_images=True)


def test_render_without_saved_audio_is_refused(book) -> None:
    source, project = book
    with pytest.raises(RuntimeError, match="no saved audio timeline"):
        run_catchup(source, 1, render_images=True)
    assert not (project / ".pipeline.lock").exists()


def test_modes_are_exclusive(book) -> None:
    source, _ = book
    with pytest.raises(SystemExit):
        main([str(source), "--chapter", "1", "--defer-images", "--render-images"])


def test_held_pipeline_lock_blocks_and_is_untouched(book) -> None:
    source, project = book
    project.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(project)
    try:
        with pytest.raises(SystemExit):
            run_catchup(source, 1, defer_images=True)
        assert lock.exists()
    finally:
        release_lock(lock)


def test_ledger_for_other_chapter_content_is_refused_and_hours_untouched(book) -> None:
    source, project = book
    from src.hour_runner import source_chapters

    source_chapters(source, project)
    hours = {"source": str(source), "sha256": source_fingerprint(source), "hours": {"1": {"complete": True}},
             "catchups": {"chapter-1": {"source_sha256": "0" * 64, "chapter_sha256": "1" * 64}}}
    atomic_json(project / "hours.json", hours)
    with pytest.raises(RuntimeError, match="does not match"):
        run_catchup(source, 1, defer_images=True)
    assert json.loads((project / "hours.json").read_text())["hours"] == hours["hours"]


def test_refresh_adds_active_lean_actor_and_keeps_scene_seconds(book) -> None:
    from src.catchup_runner import refresh_local_metadata

    _source, project = book
    project.mkdir(parents=True, exist_ok=True)
    for name in ("characters.json", "voices.json", "breeze_voices.json", "appearances.json"):
        atomic_json(project / name, {"old": {"name": "Old"}})
    cast = {"old": {"name": "Old", "bio": "b", "look": "l"}, "newbie": {"name": "New", "bio": "nb", "look": "nl"}}
    atomic_json(project / "voices.json", {"old": {}, "newbie": {"description": "d"}})
    script = project / "s.jsonl"
    script.write_text('{"newbie": "hi"}\n', encoding="utf-8")
    directory = project / "catchup" / "chapter-1"
    refresh_local_metadata(project, directory, script, cast)
    local = json.loads((directory / "characters.json").read_text())
    assert local["newbie"] == {"name": "New", "bio": "nb", "look": "nl"} and "old" in local
    assert "newbie" in json.loads((directory / "voices.json").read_text())
    assert (directory / "scene_seconds.txt").read_text() == "1\n"
    assert json.loads((project / "characters.json").read_text()) == {"old": {"name": "Old"}}


def test_render_catchup_requires_audio_before_any_portrait_work(book) -> None:
    from src.catchup_runner import render_catchup

    _source, project = book
    directory = project / "catchup" / "chapter-1"
    directory.mkdir(parents=True)
    with pytest.raises((OSError, RuntimeError, ValueError)):
        render_catchup(directory, "t")
    assert not (directory / "refs").exists()
