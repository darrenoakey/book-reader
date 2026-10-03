"""Real filesystem, state-log, and ffmpeg tests for pipeline orchestration."""

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from src.epub_extract import get_output_dir
from src.movie_resolution import movie_dimensions
from src.pipeline import (
    _time_step,
    acquire_lock,
    print_done,
    print_skip,
    print_step,
    release_lock,
    run_pipeline,
)
from src.state import mark_step_complete

ALL_STEPS = [
    "extract",
    "characters",
    "voices_desc",
    "voices_clone",
    "scripts",
    "audio",
    "m4b",
    "storyboard",
    "titlepage",
    "refimages",
    "sceneimages",
    "movie",
]


# ##################################################################
# dead pid
# obtain a genuinely dead process id by running and reaping a real child
def dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


# ##################################################################
# make movie
# render a real H.264 clip at the exact dimensions the pipeline expects
def make_movie(path: Path, resolution: int) -> None:
    width, height = movie_dimensions(resolution)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:d=0.2:r=5",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()


# ##################################################################
# finished project
# a uniquely-named project directory under the real output root with every step recorded complete
@pytest.fixture
def finished_project():
    name = f"pipeline-test-{uuid.uuid4().hex[:12]}"
    source = Path(f"/nonexistent-input/{name}.epub")
    output_dir = get_output_dir(source)
    chapters = output_dir / "chapters"
    chapters.mkdir(parents=True)
    (chapters / "00-intro.txt").write_text("A Tale of Two Tests by Ada Tester, narrated by Darren's Book Reader.")
    for step in ALL_STEPS:
        mark_step_complete(output_dir, step)
    try:
        yield source, output_dir
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


# ##################################################################
# test print helpers
# each helper emits its label text on stdout
def test_print_helpers(capsys: pytest.CaptureFixture[str]) -> None:
    print_step(7, "Assemble M4B")
    print_done("all good")
    print_skip("not needed")
    out = capsys.readouterr().out
    assert "[Step 7]" in out and "Assemble M4B" in out
    assert "✓" in out and "all good" in out
    assert "→" in out and "not needed" in out


# ##################################################################
# test lock lifecycle
# acquiring writes our pid and releasing removes the file, tolerating a repeat release
def test_lock_acquire_release(tmp_path: Path) -> None:
    lock = acquire_lock(tmp_path)
    assert lock == tmp_path / ".pipeline.lock"
    assert lock.read_text() == str(os.getpid())
    release_lock(lock)
    assert not lock.exists()
    release_lock(lock)


# ##################################################################
# test lock refuses live runner
# a lock owned by a running process blocks a second pipeline and is left intact
def test_lock_refuses_live_owner(tmp_path: Path) -> None:
    lock = acquire_lock(tmp_path)
    with pytest.raises(SystemExit) as exc:
        acquire_lock(tmp_path)
    assert str(os.getpid()) in str(exc.value)
    assert lock.read_text() == str(os.getpid())


# ##################################################################
# test lock takes over stale
# a lock left by a dead process or holding garbage is replaced by ours
def test_lock_replaces_stale_and_garbage(tmp_path: Path) -> None:
    lock = tmp_path / ".pipeline.lock"
    lock.write_text(str(dead_pid()))
    assert acquire_lock(tmp_path).read_text() == str(os.getpid())
    lock.write_text("not-a-pid")
    assert acquire_lock(tmp_path).read_text() == str(os.getpid())


# ##################################################################
# test time step records
# a completed step appends one timing record and prints the elapsed line
def test_time_step_records(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with _time_step(tmp_path, "extract"):
        pass
    records = [json.loads(line) for line in (tmp_path / "timings.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["step"] == "extract"
    assert records[0]["seconds"] >= 0
    assert records[0]["started"].endswith("+00:00")
    assert "extract" in capsys.readouterr().out


# ##################################################################
# test time step on failure
# timing is still recorded when the step body raises, and the error propagates
def test_time_step_records_on_failure(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"), _time_step(tmp_path, "audio"):
        raise RuntimeError("boom")
    records = [json.loads(line) for line in (tmp_path / "timings.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == ["audio"]


# ##################################################################
# test pipeline resumes fully complete project
# every step recorded complete (with a real 720p movie) is skipped: no generation, lock released, total timed
def test_run_pipeline_skips_completed_steps(finished_project, capsys: pytest.CaptureFixture[str]) -> None:
    source, output_dir = finished_project
    make_movie(output_dir / "movie" / "movie.mp4", 720)
    result = run_pipeline(source, resolution=720)
    assert result == output_dir / f"{output_dir.name}.m4b"
    out = capsys.readouterr().out
    assert out.count("[Step ") == 12
    assert out.count("→") == 12
    assert "Movie already assembled at 720p" in out
    assert "Complete!" in out
    assert not (output_dir / ".pipeline.lock").exists()
    records = [json.loads(line) for line in (output_dir / "timings.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == ["TOTAL"]


# ##################################################################
# test pipeline refuses concurrent run
# a live lock aborts the run before any step, leaving the other runner's lock in place
def test_run_pipeline_refuses_when_locked(finished_project, capsys: pytest.CaptureFixture[str]) -> None:
    source, output_dir = finished_project
    lock = output_dir / ".pipeline.lock"
    lock.write_text(str(os.getpid()))
    with pytest.raises(SystemExit, match="already running"):
        run_pipeline(source)
    assert lock.read_text() == str(os.getpid())
    assert "[Step" not in capsys.readouterr().out
    assert not (output_dir / "timings.jsonl").exists()
