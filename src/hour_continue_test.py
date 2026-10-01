import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.epub_extract import get_output_dir
from src.hour_continue import (
    PROGRESS_NAME,
    clear_continuation_blocker,
    continuation_status,
    qa_movie,
)
from src.hour_runner import source_fingerprint


# ##################################################################
# create real movie
# render a short 854x480 A/V movie so supervisor QA exercises ffprobe and actual start/middle/end ffmpeg frame decoding.
def create_movie(path: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=854x480:rate=30:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# ##################################################################
# write continuation project
# establish an isolated extracted source/ledger fixture with two completed hours and a known EOF cursor.
def write_project(root: Path, eof: bool) -> tuple[Path, Path]:
    source = root / f"story_{root.name}.txt"
    source.write_text("An isolated continuation source.", encoding="utf-8")
    project = get_output_dir(source)
    chapters = project / "chapters"
    chapters.mkdir(parents=True)
    (chapters / "00-intro.txt").write_text("Story by Author, narrated by Narrator", encoding="utf-8")
    (chapters / "01-part.txt").write_text("First source chapter.", encoding="utf-8")
    (chapters / "02-part.txt").write_text("Final source chapter.", encoding="utf-8")
    cursor = [2, 0] if eof else [1, 3]
    ledger = {
        "source": str(source),
        "sha256": source_fingerprint(source),
        "hours": {
            "1": {"complete": True, "movie": "hours/hour-001/movie.mp4", "next_chapter": cursor[0], "next_piece": cursor[1]},
        },
    }
    (project / "hours.json").write_text(json.dumps(ledger), encoding="utf-8")
    return source, project


# ##################################################################
# test status establishes immutable eof and stops cleanly
# a ledger cursor at the extracted final chapter produces EOF rather than scheduling an empty extra hour.
def test_continuation_status_establishes_eof_without_extra_hour() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source, project = write_project(Path(directory), eof=True)
        try:
            status = continuation_status(source, create_target=True)
            assert status["eof"] is True
            assert status["next_hour"] == 2
            target = json.loads((project / "hour_continuation_target.json").read_text(encoding="utf-8"))
            assert target["target_chapter"] == 2
            assert target["target_piece"] == 0
        finally:
            shutil.rmtree(project, ignore_errors=True)


# ##################################################################
# test status preserves next cursor
# a non-EOF cursor continues with precisely the next sequential hour rather than re-running an already committed one.
def test_continuation_status_uses_ledger_cursor() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source, project = write_project(Path(directory), eof=False)
        try:
            status = continuation_status(source, create_target=True)
            assert status["eof"] is False
            assert status["next_hour"] == 2
            assert status["cursor"] == {"chapter": 1, "piece": 3}
        finally:
            shutil.rmtree(project, ignore_errors=True)


# ##################################################################
# test explicit failure clear
# a permanent failure remains blocked until an operator explicitly records that its root cause was addressed.
def test_clear_blocker_requires_latched_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source, project = write_project(Path(directory), eof=False)
        try:
            continuation_status(source, create_target=True)
            (project / PROGRESS_NAME).write_text(json.dumps({"status": "failed", "error": "schema rejected"}), encoding="utf-8")
            cleared = clear_continuation_blocker(source)
            assert cleared["status"] == "ready"
            assert cleared["cleared_failure"] == "schema rejected"
            assert "continuation_failure_cleared" in (project / "hour_continuation_events.jsonl").read_text(encoding="utf-8")
        finally:
            shutil.rmtree(project, ignore_errors=True)


# ##################################################################
# test real movie qa
# reject container-only confidence by decoding actual video frames and requiring both A/V streams at final 480p dimensions.
def test_qa_movie_real_av_decode() -> None:
    with tempfile.TemporaryDirectory() as directory:
        movie = Path(directory) / "movie.mp4"
        create_movie(movie)
        qa = qa_movie(movie)
        assert qa["resolution"] == [854, 480]
        assert qa["frames"] == "start,middle,end"
        assert qa["av_decode"] == "ok"
        qa_dir = movie.parent / "qa"
        assert (qa_dir / "av_decode.log").is_file()
        assert all((qa_dir / f"{label}.png").is_file() for label in ("start", "middle", "end"))


# ##################################################################
# test malformed eof cursor fails closed
# an EOF chapter with a residual piece would duplicate or skip story text and must never be auto-advanced.
def test_continuation_status_rejects_invalid_eof_piece() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source, project = write_project(Path(directory), eof=True)
        try:
            ledger = json.loads((project / "hours.json").read_text(encoding="utf-8"))
            ledger["hours"]["1"]["next_piece"] = 1
            (project / "hours.json").write_text(json.dumps(ledger), encoding="utf-8")
            with pytest.raises(RuntimeError, match="EOF cursor"):
                continuation_status(source, create_target=True)
        finally:
            shutil.rmtree(project, ignore_errors=True)
