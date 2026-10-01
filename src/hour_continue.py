"""Idempotent supervised continuation of bounded hourly movie production."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from src.epub_extract import get_output_dir
from src.hour_runner import (
    HOUR_MAX_SECONDS,
    atomic_json,
    load_ledger,
    run_hour,
    source_chapters,
    source_fingerprint,
)
from src.movie_assemble import probe_duration, probe_resolution
from src.movie_resolution import movie_dimensions

MIN_FREE_BYTES = 20 * 1024 * 1024 * 1024
TARGET_NAME = "hour_continuation_target.json"
PROGRESS_NAME = "hour_continuation_progress.json"
EVENTS_NAME = "hour_continuation_events.jsonl"


# ##################################################################
# append event
# write a readable, append-only record so an auto-managed resume can be diagnosed without changing ledger state.
def append_event(project: Path, event: str, **details: object) -> None:
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **details}
    with (project / EVENTS_NAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")


# ##################################################################
# continuation target
# establish exactly one immutable source EOF target; restarts validate rather than advancing it again.
def continuation_target(project: Path, source: Path, chapter_count: int, create: bool) -> dict:
    target_path = project / TARGET_NAME
    expected = {"source_sha256": source_fingerprint(source), "target_chapter": chapter_count, "target_piece": 0}
    if not target_path.exists():
        if create:
            atomic_json(target_path, expected)
        return expected
    target = json.loads(target_path.read_text(encoding="utf-8"))
    if target != expected:
        raise RuntimeError("continuation target does not match the immutable source EOF")
    return target


# ##################################################################
# completed hours
# require a contiguous ledger prefix so an auto restart cannot skip a failed or missing production hour.
def completed_hours(ledger: dict) -> list[tuple[int, dict]]:
    hours = ledger.get("hours")
    if not isinstance(hours, dict):
        raise TypeError("hours ledger has no hours object")
    result: list[tuple[int, dict]] = []
    index = 1
    while str(index) in hours:
        entry = hours[str(index)]
        if not isinstance(entry, dict) or not entry.get("complete"):
            raise RuntimeError(f"hour {index} is incomplete; refusing to skip it")
        result.append((index, entry))
        index += 1
    if any(not str(key).isdigit() or int(key) >= index for key in hours):
        raise RuntimeError("hours ledger is not a contiguous completed sequence")
    return result


# ##################################################################
# current cursor
# derive the only allowable next source position from the final completed ledger entry.
def current_cursor(completed: list[tuple[int, dict]]) -> tuple[int, int]:
    if not completed:
        return 0, 0
    entry = completed[-1][1]
    try:
        return int(entry["next_chapter"]), int(entry["next_piece"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("completed hour ledger lacks a valid next source cursor") from error


# ##################################################################
# validate cursor
# reject out-of-range and impossible EOF positions before invoking the expensive hourly production pipeline.
def validate_cursor(cursor: tuple[int, int], chapter_count: int) -> None:
    chapter, piece = cursor
    if chapter < 0 or piece < 0 or chapter > chapter_count:
        raise RuntimeError("hour ledger cursor is outside extracted source chapters")
    if chapter == chapter_count and piece != 0:
        raise RuntimeError("EOF cursor must have piece index zero")


# ##################################################################
# wait for active project run
# leave a live parent-owned hourly pipeline untouched, waiting only for its lock to clear before the supervisor attempts the next hour.
def wait_for_active_project_run(project: Path) -> None:
    lock = project / ".pipeline.lock"
    while lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError):
            lock.unlink(missing_ok=True)
            return
        append_event(project, "waiting_for_active_pipeline", pid=pid)
        time.sleep(10)


# ##################################################################
# require free disk
# fail closed before beginning another hour; cache and completed production assets are never pruned or relocated.
def require_free_disk(project: Path) -> None:
    free = shutil.disk_usage(project).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"insufficient free disk space: {free} bytes available, need at least {MIN_FREE_BYTES}")


# ##################################################################
# extract nonblank frame
# decode a real movie frame and prove it contains visible pixels, rather than trusting container metadata alone.
def extract_nonblank_frame(movie: Path, seconds: float, destination: Path) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{seconds:.3f}", "-i", str(movie), "-frames:v", "1", str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"ffmpeg frame decode failed for {movie.name}: {result.stderr[-400:]}")
    with Image.open(destination) as frame:
        extrema = frame.convert("RGB").getextrema()
    if not any(high > 0 for _, high in extrema):
        raise RuntimeError(f"decoded frame is blank for {movie.name} at {seconds:.3f}s")


# ##################################################################
# qa movie
# prove real A/V decode, final 480p dimensions, legal duration, and visible beginning/middle/end frames for each completed hour.
def qa_movie(movie: Path) -> dict:
    if not movie.is_file():
        raise RuntimeError(f"completed hour movie is missing: {movie}")
    duration = probe_duration(movie)
    if not 0 < duration <= HOUR_MAX_SECONDS:
        raise RuntimeError(f"movie duration is outside bounds: {duration:.3f}s")
    if probe_resolution(movie) != movie_dimensions(480):
        raise RuntimeError(f"movie resolution is not 854x480: {probe_resolution(movie)}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(movie)],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode:
        raise RuntimeError(f"ffprobe A/V validation failed: {probe.stderr[-400:]}")
    kinds = {stream.get("codec_type") for stream in json.loads(probe.stdout).get("streams", [])}
    if not {"audio", "video"} <= kinds:
        raise RuntimeError("movie does not contain both audio and video streams")
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        for label, seconds in (("start", min(0.5, duration / 4)), ("middle", duration / 2), ("end", max(0.0, duration - 0.5))):
            extract_nonblank_frame(movie, seconds, scratch / f"{label}.png")
    return {"movie": str(movie), "duration_seconds": round(duration, 3), "resolution": [854, 480], "frames": "start,middle,end"}


# ##################################################################
# continuation status
# inspect the immutable source, target, and ledger without scheduling TTS, images, or an hourly render.
def continuation_status(source: Path, create_target: bool = False) -> dict:
    source = source.resolve()
    if not source.is_file():
        raise ValueError(f"input source is not a file: {source}")
    project = get_output_dir(source)
    ledger_path = project / "hours.json"
    if not ledger_path.is_file():
        raise RuntimeError("hours ledger does not exist; refusing to infer continuation state")
    _, _, chapters = source_chapters(source, project)
    target = continuation_target(project, source, len(chapters), create_target)
    ledger = load_ledger(project, source)
    completed = completed_hours(ledger)
    cursor = current_cursor(completed)
    validate_cursor(cursor, len(chapters))
    return {
        "project": str(project),
        "target": target,
        "completed_hours": len(completed),
        "next_hour": len(completed) + 1,
        "cursor": {"chapter": cursor[0], "piece": cursor[1]},
        "eof": cursor == (target["target_chapter"], target["target_piece"]),
    }


# ##################################################################
# continue to eof
# idempotently run precisely the next ledger hour until the one established EOF cursor is reached; no upload action is part of this supervisor.
def continue_to_eof(source: Path, verify_only: bool = False) -> dict:
    status = continuation_status(source, create_target=not verify_only)
    if verify_only:
        return status
    source = source.resolve()
    project = Path(status["project"])
    target = status["target"]
    append_event(project, "continuation_started", target=target)
    while True:
        status = continuation_status(source)
        cursor = (status["cursor"]["chapter"], status["cursor"]["piece"])
        if cursor == (target["target_chapter"], target["target_piece"]):
            atomic_json(project / PROGRESS_NAME, {**status, "status": "complete"})
            append_event(project, "continuation_complete", completed_hours=status["completed_hours"])
            return status
        require_free_disk(project)
        wait_for_active_project_run(project)
        status = continuation_status(source)
        cursor = (status["cursor"]["chapter"], status["cursor"]["piece"])
        if cursor == (target["target_chapter"], target["target_piece"]):
            continue
        hour = status["next_hour"]
        append_event(project, "hour_started", hour=hour, cursor={"chapter": cursor[0], "piece": cursor[1]})
        try:
            movie = run_hour(source, hour_index=hour)
        except SystemExit as error:
            if "pipeline already running" not in str(error):
                raise
            append_event(project, "hour_deferred_for_active_pipeline", hour=hour)
            continue
        qa = qa_movie(movie)
        after = continuation_status(source)
        after_cursor = (after["cursor"]["chapter"], after["cursor"]["piece"])
        if after_cursor <= cursor:
            raise RuntimeError("completed hour did not advance the source cursor")
        atomic_json(project / PROGRESS_NAME, {**after, "status": "running", "last_qa": qa})
        append_event(project, "hour_complete", hour=hour, qa=qa, cursor={"chapter": after_cursor[0], "piece": after_cursor[1]})
