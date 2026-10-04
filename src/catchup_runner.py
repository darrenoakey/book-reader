"""Source-boundary catch-up production: one chapter, audio first, images on demand.

Usage: python -m src.catchup_runner SOURCE --chapter 224 [--defer-images | --render-images]

Reuses the existing hour helpers for a single chapter with the lean existing-profile
cast; no whole-book audits and no invented hourly completion flags.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from src import tts_engine
from src.audio_synth import concat_wavs, wav_duration, write_timeline
from src.epub_extract import get_output_dir
from src.hour_runner import (
    AUDIO_MAX_SECONDS,
    HOUR_MAX_SECONDS,
    atomic_json,
    chapter_order,
    copy_chapter_context,
    extend_appearances,
    extend_voices,
    load_lean_cast,
    load_ledger,
    prepare_hour_directory,
    render_hour,
    script_for_chapter,
    source_chapters,
    source_fingerprint,
    synthesize_window,
)
from src.movie_assemble import probe_duration
from src.pipeline import acquire_lock, release_lock

CATCHUP_SCENE_SECONDS = 1.0


# ##################################################################
# file digest
# bind saved audio to the exact bytes later image rendering will use.
def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ##################################################################
# script speakers
# the speaker ids actually present in one chapter script.
def script_speakers(script: Path) -> set[str]:
    speakers: set[str] = set()
    for raw in script.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        if isinstance(entry, dict) and len(entry) == 1:
            speakers.add(next(iter(entry)))
    return speakers


# ##################################################################
# ensure active voices
# create only voices, appearances and reference voice wavs that active speakers genuinely lack.
def ensure_active_voices(project: Path, script: Path, cast: dict) -> list[str]:
    root_voices = json.loads((project / "voices.json").read_text(encoding="utf-8"))
    manifest_path = project / "breeze_voices.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    active = {name: cast[name] for name in script_speakers(script) if name in cast}
    missing = sorted(
        name for name in active
        if name not in root_voices or name not in manifest or not (project / "voices" / f"{name}.wav").exists()
    )
    if missing:
        subset = {name: active[name] for name in missing}
        asyncio.run(extend_voices(project, subset))
    extend_appearances(project, active)
    return missing


# ##################################################################
# refresh local metadata
# re-view root metadata after missing assets exist, add lean-registry profiles for active actors only, keep scene_seconds at 1.
def refresh_local_metadata(project: Path, directory: Path, script: Path, cast: dict) -> None:
    prepare_hour_directory(project, directory, cast)
    path = directory / "characters.json"
    local = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name in script_speakers(script):
        if name in cast and name not in local:
            local[name] = {key: cast[name][key] for key in ("name", "bio", "look") if key in cast[name]}
    atomic_json(path, local)
    (directory / "scene_seconds.txt").write_text(f"{CATCHUP_SCENE_SECONDS:g}\n", encoding="utf-8")


# ##################################################################
# catchup entry
# the durable state of one catch-up chapter, stored beside its output and mirrored into the ledger.
def write_state(project: Path, directory: Path, label: str, state: dict) -> None:
    atomic_json(directory / "catchup.json", state)
    ledger = load_ledger(project, Path(state["source"]))
    ledger.setdefault("catchups", {})[label] = state
    atomic_json(project / "hours.json", ledger)


# ##################################################################
# run catchup
# produce chapter audio (always) and the movie (unless images are deferred), under the project pipeline lock.
def run_catchup(source: Path, chapter_number: int, defer_images: bool = False, render_images: bool = False) -> Path | None:
    source = source.resolve()
    project = get_output_dir(source)
    project.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(project)
    started = time.monotonic()
    try:
        return _run_locked(source, project, chapter_number, defer_images, render_images)
    finally:
        with (project / "hour_timings.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"catchup": chapter_number, "seconds": round(time.monotonic() - started, 3)}) + "\n")
        release_lock(lock)


def _run_locked(source: Path, project: Path, chapter_number: int, defer_images: bool, render_images: bool) -> Path | None:
    if defer_images and render_images:
        raise ValueError("--defer-images and --render-images are mutually exclusive")
    label = f"chapter-{chapter_number}"
    directory = project / "catchup" / label
    ledger = load_ledger(project, source)
    prior = ledger.get("catchups", {}).get(label)
    sha = source_fingerprint(source)
    title, _, chapters = source_chapters(source, project)
    matches = [i for i, path in enumerate(chapters) if chapter_order(path)[0] == chapter_number]
    if len(matches) != 1:
        raise RuntimeError(f"chapter {chapter_number} is not a unique source chapter")
    index = matches[0]
    chapter = chapters[index]
    chapter_sha = hashlib.sha256(chapter.read_bytes()).hexdigest()
    if prior and (prior.get("source_sha256") != sha or prior.get("chapter_sha256") != chapter_sha):
        raise RuntimeError("catch-up ledger entry does not match this source/chapter; refusing to mix stories")
    if prior and prior.get("images_complete"):
        movie = project / prior["movie"]
        if movie.exists() and probe_duration(movie) <= HOUR_MAX_SECONDS:
            return movie
        raise RuntimeError("completed catch-up ledger points at a missing or oversized movie")
    if tts_engine.engine_name() != "breeze":
        raise RuntimeError("catch-up production requires Breeze shared voice references")
    wav = directory / "audio" / "hour-00000.wav"
    if not (prior and prior.get("audio_complete")):
        if render_images:
            raise RuntimeError("no saved audio timeline for this chapter; run with --defer-images first")
        cast, aliases = load_lean_cast(source, project)
        prepare_hour_directory(project, directory, cast)
        (directory / "scene_seconds.txt").write_text(f"{CATCHUP_SCENE_SECONDS:g}\n", encoding="utf-8")
        copy_chapter_context(directory, chapter)
        script = script_for_chapter(project, directory, chapter, cast, aliases, None)
        created = ensure_active_voices(project, script, cast)
        refresh_local_metadata(project, directory, script, cast)
        selected, consumed = synthesize_window(project, script, AUDIO_MAX_SECONDS, 0)
        from src.audio_synth import plan_chapter

        total = len(plan_chapter(script, project / "audio_cache", project / "voices", tts_engine.speaker_set(project))[3])
        if not selected or consumed != total:
            raise RuntimeError(f"chapter {chapter_number} audio did not fit completely ({consumed}/{total} pieces)")
        concat_wavs([item["path"] for item in selected], wav)
        write_timeline(wav, selected)
        prior = {
            "source": str(source),
            "source_sha256": sha,
            "chapter": str(chapter.relative_to(project)),
            "chapter_sha256": chapter_sha,
            "start_chapter": index,
            "start_piece": 0,
            "next_chapter": index + 1,
            "next_piece": 0,
            "audio_complete": True,
            "audio_sha256": file_digest(wav),
            "timeline_sha256": file_digest(wav.with_suffix(".timeline.json")),
            "audio_seconds": round(wav_duration(wav), 3),
            "created_voices": created,
            "images_complete": False,
            "dir": str(directory.relative_to(project)),
        }
        write_state(project, directory, label, prior)
    if defer_images:
        return None
    if file_digest(wav) != prior["audio_sha256"] or file_digest(wav.with_suffix(".timeline.json")) != prior["timeline_sha256"]:
        raise RuntimeError("saved catch-up audio changed since its timeline was recorded")
    movie = render_hour(directory, title)
    duration = probe_duration(movie)
    state = {**prior, "images_complete": True, "movie": str(movie.relative_to(project)), "duration_seconds": round(duration, 3)}
    write_state(project, directory, label, state)
    return movie


# ##################################################################
# main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--chapter", type=int, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--defer-images", action="store_true", help="audio and timeline only")
    mode.add_argument("--render-images", action="store_true", help="render the movie from the saved audio timeline")
    args = parser.parse_args(argv)
    movie = run_catchup(Path(args.source), args.chapter, args.defer_images, args.render_images)
    print(f"catch-up chapter {args.chapter}: {movie if movie else 'audio complete, images deferred'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
