"""Real filesystem tests for the single-step runner."""

import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from src.epub_extract import get_output_dir
from src.state import is_step_complete
from src.step_runner import get_book_info, main, run_step

STORY = (
    "The Lantern Keeper\n\nby Ada Tester\n\n"
    "The keeper climbed the stairs each dusk.\n\nShe lit the lamp and watched the sea.\n"
)


# ##################################################################
# story project
# a uniquely-named .txt story whose output directory lives under the real output root
@pytest.fixture
def story_project(tmp_path: Path):
    name = f"step-runner-test-{uuid.uuid4().hex[:12]}"
    story = tmp_path / f"{name}.txt"
    story.write_text(STORY, encoding="utf-8")
    output_dir = get_output_dir(story)
    try:
        yield story, output_dir
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


# ##################################################################
# write intro
def write_intro(output_dir: Path, text: str) -> None:
    (output_dir / "chapters").mkdir(parents=True, exist_ok=True)
    (output_dir / "chapters" / "00-intro.txt").write_text(text, encoding="utf-8")


# ##################################################################
# test book info parsing
# title/author split on the first " by " and the narrator suffix is dropped
def test_get_book_info(tmp_path: Path) -> None:
    write_intro(tmp_path, "Stand by Me by Jo Writer, narrated by Darren's Book Reader.")
    assert get_book_info(tmp_path) == ("Stand", "Me by Jo Writer")
    write_intro(tmp_path, "Gone Girl by Gillian Flynn, narrated by Darren's Book Reader.")
    assert get_book_info(tmp_path) == ("Gone Girl", "Gillian Flynn")


# ##################################################################
# test book info without author
def test_get_book_info_unknown_author(tmp_path: Path) -> None:
    write_intro(tmp_path, "Untitled Work\n")
    assert get_book_info(tmp_path) == ("Untitled Work", "Unknown")


# ##################################################################
# test extract step
# the extract step ingests a real text story, writes chapters, and records completion
def test_run_step_extract(story_project, capsys: pytest.CaptureFixture[str]) -> None:
    story, output_dir = story_project
    assert run_step("extract", story) == 0
    assert "Extracted 2 files" in capsys.readouterr().out
    chapters = sorted(p.name for p in (output_dir / "chapters").iterdir())
    assert chapters[0] == "00-intro.txt" and len(chapters) == 2
    assert get_book_info(output_dir) == ("The Lantern Keeper", "Ada Tester")
    assert is_step_complete(output_dir, "extract")
    assert not is_step_complete(output_dir, "characters")


# ##################################################################
# test unknown step
# an unrecognised step reports valid names, returns failure, and records nothing
def test_run_step_unknown(story_project, capsys: pytest.CaptureFixture[str]) -> None:
    story, output_dir = story_project
    assert run_step("bogus", story) == 1
    out = capsys.readouterr().out
    assert "Unknown step: bogus" in out and "Valid steps:" in out and "movie" in out
    assert not (output_dir / "state.jsonl").exists()


# ##################################################################
# test main cli
# the argparse entry point drives a real extract and rejects unsupported resolutions
def test_main_cli(story_project, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    story, output_dir = story_project
    monkeypatch.setattr(sys, "argv", ["step_runner", "extract", str(story)])
    assert main() == 0
    assert is_step_complete(output_dir, "extract")
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["step_runner", "movie", str(story), "--resolution", "999"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


# ##################################################################
# test state log
# a completed step is persisted as a JSON line the pipeline can resume from
def test_state_file_is_json_lines(story_project) -> None:
    story, output_dir = story_project
    run_step("extract", story)
    entries = [json.loads(line) for line in (output_dir / "state.jsonl").read_text().splitlines()]
    assert entries[-1]["step"] == "extract" and entries[-1]["detail"] == "complete"
