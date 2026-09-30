"""Bounded, resumable production of one audiobook-movie hour.

The normal pipeline is intentionally book-wide.  This runner is for very long
books: it only sends enough source chapters to the language and TTS services to
produce one finished hour, while keeping cast, Breeze references, portraits,
and appearances in the project root for later hours.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from src import tts_engine
from src.audio_synth import concat_wavs, plan_chapter, wav_duration, write_timeline
from src.breeze_voices import prepare_breeze_voices
from src.character_analysis import analyze_chapter, create_narrator_entry, merge_character_info
from src.epub_extract import get_output_dir
from src.llm import ask_sync
from src.movie_assemble import assemble_movie, probe_duration
from src.movie_images import generate_character_refs, generate_scene_images
from src.movie_storyboard import build_storyboard
from src.pipeline import acquire_lock, release_lock
from src.script_generate import generate_script_for_file
from src.text_ingest import extract_any
from src.voice_description import _voice_description_for_one, parse_json_response

HOUR_MAX_SECONDS = 3600.0
# Leave one second for AAC container priming while retaining a near-full natural hour.
AUDIO_MAX_SECONDS = 3599.0
SCENE_SECONDS = 20.0
FINAL_RESOLUTION = 480


# ##################################################################
# chapter order
# use the numeric prefix rather than lexical ordering so chapter 100 follows 99
# in million-word text ingests.
def chapter_order(path: Path) -> tuple[int, str]:
    prefix, _, remainder = path.name.partition("-")
    return int(prefix), remainder


# ##################################################################
# atomic json
# make resume metadata durable: a power loss cannot leave a half-written cursor.
def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


# ##################################################################
# source fingerprint
# bind an hourly project to exactly one input book and fail closed on mismatch.
def source_fingerprint(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ##################################################################
# verify hour runtime
# prove the production interpreter can import the complete runner and bind this source without calling an LLM, TTS, or image service.
def verify_hour_runtime(source: Path) -> dict[str, str]:
    source = source.resolve()
    if not source.is_file():
        raise ValueError(f"input source is not a file: {source}")
    return {"source": str(source), "sha256": source_fingerprint(source), "output": str(get_output_dir(source))}


# ##################################################################
# hour ledger
# load the durable sequence ledger, rejecting a different source at the same output path.
def load_ledger(project: Path, source: Path) -> dict:
    path = project / "hours.json"
    fingerprint = source_fingerprint(source)
    if not path.exists():
        return {"source": str(source), "sha256": fingerprint, "hours": {}}
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("sha256") != fingerprint:
        raise RuntimeError("hour ledger source SHA-256 does not match input; refusing to mix stories")
    return ledger


# ##################################################################
# source chapters
# extract once, then return only narrative chapters in numeric source order.
def source_chapters(source: Path, project: Path) -> tuple[str, str, list[Path]]:
    chapters_dir = project / "chapters"
    if not chapters_dir.exists() or not list(chapters_dir.glob("*.txt")):
        extract_any(source, project)
    intro = chapters_dir / "00-intro.txt"
    parts = intro.read_text(encoding="utf-8").split(" by ", 1)
    title = parts[0].strip()
    author = parts[1].split(", narrated by", 1)[0].strip() if len(parts) > 1 else "Unknown"
    chapters = sorted((path for path in chapters_dir.glob("*.txt") if path != intro), key=chapter_order)
    if not chapters:
        raise RuntimeError("input extraction produced no narrative chapters")
    return title, author, chapters


# ##################################################################
# project cast
# read the monotonic shared cast; old identities are never replaced by later hours.
def load_cast(project: Path) -> dict:
    path = project / "characters.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ##################################################################
# cast aliases
# map only exact normalized display-name matches so spelling/case aliases keep one identity without merging distinct names such as Ren and Ron.
def canonical_character_id(project: Path, cast: dict, candidate_id: str, name: str) -> str:
    aliases_path = project / "character_aliases.json"
    aliases = json.loads(aliases_path.read_text(encoding="utf-8")) if aliases_path.exists() else {}
    if candidate_id in aliases:
        return aliases[candidate_id]
    normalized = "".join(char for char in name.casefold() if char.isalnum())
    for established_id, info in cast.items():
        established = "".join(char for char in str(info.get("name", "")).casefold() if char.isalnum())
        if normalized and normalized == established:
            aliases[candidate_id] = established_id
            atomic_json(aliases_path, aliases)
            return established_id
    return candidate_id


# ##################################################################
# extend cast
# analyze one upcoming chapter only, append its new information, and retain established identities.
async def extend_cast(project: Path, chapter: Path, chapter_number: int, title: str, author: str) -> dict:
    cast = load_cast(project)
    known = ", ".join(f"{cid}={info.get('name', cid)}" for cid, info in cast.items())
    found = merge_character_info([await analyze_chapter(chapter, chapter_number, known)])
    if "narrator" not in cast:
        cast["narrator"] = await create_narrator_entry(title, author, chapter.read_text(encoding="utf-8")[:3000])
    for char_id, info in found.items():
        char_id = canonical_character_id(project, cast, char_id, str(info.get("name", char_id)))
        if char_id not in cast:
            cast[char_id] = info
            continue
        for field in ("bio", "look"):
            incoming = (info.get(field) or "").strip()
            existing = (cast[char_id].get(field) or "").strip()
            if incoming and incoming not in existing:
                cast[char_id][field] = f"{existing} {incoming}".strip()
    atomic_json(project / "characters.json", cast)
    return cast


# ##################################################################
# extend voices
# create Breeze descriptions only for identities not already established in the shared voice map.
async def extend_voices(project: Path, cast: dict) -> None:
    path = project / "voices.json"
    voices = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [(char_id, info) for char_id, info in cast.items() if char_id not in voices]
    if missing:
        described = await asyncio.gather(*(_voice_description_for_one(char_id, info) for char_id, info in missing))
        for char_id, info in described:
            voices[char_id] = info
        atomic_json(path, voices)
    prepare_breeze_voices(project)


# ##################################################################
# extend appearances
# add a visual identity only once; established descriptions remain the portrait source for all later hours.
def extend_appearances(project: Path, cast: dict) -> None:
    path = project / "appearances.json"
    appearances = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = {char_id: info for char_id, info in cast.items() if char_id != "narrator" and char_id not in appearances}
    if not missing:
        return
    roster = "\n".join(
        f"- {char_id}: {(info.get('look') or info.get('bio') or '')[:700]}" for char_id, info in missing.items()
    )
    response = ask_sync(
        "Output JSON only. For each character, give a 50-80 word canonical visual description for an image generator. "
        "Use source facts exactly; include age, build, hair, face, skin, clothing, and species when known. "
        "Never describe plot or relationships. Characters:\n" + roster,
        max_tokens=2200,
    )
    generated = parse_json_response(response)
    for char_id, info in missing.items():
        description = generated.get(char_id)
        if isinstance(description, str) and description.strip():
            appearances[char_id] = description.strip()
        else:
            appearances[char_id] = (info.get("look") or "ordinary human, appearance unspecified").strip()
    atomic_json(path, appearances)


# ##################################################################
# link shared artifact
# expose one root-level immutable-ish production asset to an hour directory without copies.
def link_shared(hour_dir: Path, project: Path, name: str) -> None:
    target = hour_dir / name
    if target.exists() or target.is_symlink():
        return
    relative = os.path.relpath(project / name, hour_dir)
    target.symlink_to(relative, target_is_directory=(project / name).is_dir())


# ##################################################################
# prepare hour directory
# make the hour see the canonical cast and Breeze reference files while retaining hour-local script/audio/movie outputs.
def prepare_hour_directory(project: Path, hour_dir: Path) -> None:
    hour_dir.mkdir(parents=True, exist_ok=True)
    intro = project / "chapters" / "00-intro.txt"
    if intro.exists():
        hour_chapters = hour_dir / "chapters"
        hour_chapters.mkdir(exist_ok=True)
        intro_copy = hour_chapters / intro.name
        if not intro_copy.exists():
            shutil.copyfile(intro, intro_copy)
    (project / "refs").mkdir(exist_ok=True)
    for name in (
        "characters.json",
        "voices.json",
        "breeze_voices.json",
        "voices",
        "refs",
        "appearances.json",
        "world_bible.json",
        "locations.json",
    ):
        link_shared(hour_dir, project, name)
    for name in ("style.txt", "style-reference.png"):
        if (project / name).exists():
            link_shared(hour_dir, project, name)
    (hour_dir / "scene_seconds.txt").write_text(f"{SCENE_SECONDS:g}\n", encoding="utf-8")
    # Hourly productions deliberately use only the clean style asset and canonical cast portraits.
    (hour_dir / "image_reference_mode.txt").write_text("style-and-cast-only\n", encoding="utf-8")


# ##################################################################
# script for chapter
# create one chapter script against the current shared cast, avoiding any full-book script pass.
def script_for_chapter(project: Path, hour_dir: Path, chapter: Path, cast: dict) -> Path:
    shared_dir = project / "script_cache"
    shared_dir.mkdir(exist_ok=True)
    canonical = asyncio.run(generate_script_for_file(chapter, shared_dir, sorted(cast)))
    script_dir = hour_dir / "script"
    script_dir.mkdir(exist_ok=True)
    hour_script = script_dir / canonical.name
    if not hour_script.exists():
        hour_script.symlink_to(os.path.relpath(canonical, script_dir))
    return canonical


# ##################################################################
# copy chapter context
# retain only this hour's source chapters for world and location prompts, never the full novel.
def copy_chapter_context(hour_dir: Path, chapter: Path) -> None:
    chapters_dir = hour_dir / "chapters"
    chapters_dir.mkdir(exist_ok=True)
    destination = chapters_dir / chapter.name
    if not destination.exists():
        shutil.copyfile(chapter, destination)


# ##################################################################
# synthesize window
# synthesize only an upcoming chapter, then select complete spoken pieces that fit the strict hour cap.
def synthesize_window(project: Path, script: Path, remaining: float, start_piece: int) -> tuple[list[dict], int]:
    audio_dir = project / "audio_cache"
    audio_dir.mkdir(exist_ok=True)
    _, paths, jobs, metadata = plan_chapter(script, audio_dir, project / "voices", tts_engine.speaker_set(project))
    if jobs:
        tts_engine.synthesize_jobs(jobs, project)
    selected: list[dict] = []
    elapsed = 0.0
    for index, meta in enumerate(metadata[start_piece:], start=start_piece):
        duration = wav_duration(paths[index])
        if elapsed + duration > remaining + 0.0001:
            break
        selected.append({**meta, "duration": duration})
        elapsed += duration
    return selected, len(selected)


# ##################################################################
# render hour
# create scene images and the final movie only after its audio duration has been proven at or below one hour.
def render_hour(hour_dir: Path, title: str) -> Path:
    audio_dir = hour_dir / "audio"
    selected = sorted(audio_dir.glob("hour-*.wav"))
    if not selected:
        raise RuntimeError("no selected audio was produced for the requested hour")
    total = sum(wav_duration(path) for path in selected)
    if total > HOUR_MAX_SECONDS + 0.0001:
        raise RuntimeError(f"hour audio is {total:.3f}s, exceeding the hard 3600s cap")
    build_storyboard(hour_dir, title)
    generate_character_refs(hour_dir)
    generate_scene_images(hour_dir)
    movie = assemble_movie(hour_dir, title, resolution=FINAL_RESOLUTION)
    duration = probe_duration(movie)
    if duration > HOUR_MAX_SECONDS:
        raise RuntimeError(f"movie is {duration:.3f}s, exceeding the hard 3600s cap")
    return movie


# ##################################################################
# record hour timing
# append actual lifecycle outcomes so interrupted long production is observable and resumable.
def record_hour_timing(project: Path, hour_index: int, outcome: str, seconds: float) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hour": hour_index,
        "outcome": outcome,
        "seconds": round(seconds, 3),
    }
    with (project / "hour_timings.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")


# ##################################################################
# run hour
# hold the project-wide pipeline lock through the complete bounded production transaction.
def run_hour(source: Path, hour_index: int = 1) -> Path:
    source = source.resolve()
    project = get_output_dir(source)
    project.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(project)
    started = time.monotonic()
    record_hour_timing(project, hour_index, "started", 0.0)
    try:
        movie = _run_hour_locked(source, hour_index)
        record_hour_timing(project, hour_index, "complete", time.monotonic() - started)
        return movie
    except Exception:
        record_hour_timing(project, hour_index, "failed", time.monotonic() - started)
        raise
    finally:
        release_lock(lock)


# ##################################################################
# run locked hour
# produce one numbered hour and commit the exact next source cursor only after its movie succeeds.
def _run_hour_locked(source: Path, hour_index: int = 1) -> Path:
    if hour_index < 1:
        raise ValueError("hour index must be at least one")
    if tts_engine.engine_name() != "breeze":
        raise RuntimeError("hour production requires Breeze shared voice references")
    source = source.resolve()
    project = get_output_dir(source)
    project.mkdir(parents=True, exist_ok=True)
    ledger = load_ledger(project, source)
    key = str(hour_index)
    prior = ledger["hours"].get(key)
    if prior and prior.get("complete"):
        movie = project / prior["movie"]
        if movie.exists() and probe_duration(movie) <= HOUR_MAX_SECONDS:
            return movie
        raise RuntimeError("completed hour ledger points at a missing or oversized movie")
    if hour_index > 1:
        previous = ledger["hours"].get(str(hour_index - 1))
        if not previous or not previous.get("complete"):
            raise RuntimeError("previous hour is not complete; refusing to duplicate or skip story content")
        chapter_cursor = int(previous["next_chapter"])
        piece_cursor = int(previous["next_piece"])
    else:
        chapter_cursor, piece_cursor = 0, 0
    title, author, chapters = source_chapters(source, project)
    hour_dir = project / "hours" / f"hour-{hour_index:03d}"
    all_selected: list[dict] = []
    remaining = AUDIO_MAX_SECONDS
    next_chapter, next_piece = chapter_cursor, piece_cursor
    for chapter_index in range(chapter_cursor, len(chapters)):
        cast = asyncio.run(extend_cast(project, chapters[chapter_index], chapter_index + 1, title, author))
        asyncio.run(extend_voices(project, cast))
        extend_appearances(project, cast)
        prepare_hour_directory(project, hour_dir)
        copy_chapter_context(hour_dir, chapters[chapter_index])
        script = script_for_chapter(project, hour_dir, chapters[chapter_index], cast)
        start_piece = piece_cursor if chapter_index == chapter_cursor else 0
        candidates, _ = synthesize_window(project, script, remaining, start_piece)
        if not candidates:
            if all_selected:
                break
            raise RuntimeError("first natural spoken piece does not fit within 3600 seconds")
        elapsed = sum(item["duration"] for item in candidates)
        if elapsed > remaining + 0.0001:
            raise RuntimeError("internal selection exceeded hour budget")
        all_selected.extend(candidates)
        remaining -= elapsed
        all_metadata = plan_chapter(script, project / "audio_cache", project / "voices", tts_engine.speaker_set(project))[3]
        consumed = start_piece + len(candidates)
        if consumed < len(all_metadata):
            next_chapter, next_piece = chapter_index, consumed
            break
        next_chapter, next_piece = chapter_index + 1, 0
        if remaining <= 0.0001:
            break
    if not all_selected:
        raise RuntimeError("no source content selected for requested hour")
    audio_dir = hour_dir / "audio"
    output = audio_dir / "hour-00000.wav"
    concat_wavs([item["path"] for item in all_selected], output)
    write_timeline(output, all_selected)
    movie = render_hour(hour_dir, title)
    ledger["hours"][key] = {
        "complete": True,
        "movie": str(movie.relative_to(project)),
        "duration_seconds": round(probe_duration(movie), 3),
        "next_chapter": next_chapter,
        "next_piece": next_piece,
        "source_chapters": [str(path.relative_to(project)) for path in chapters[chapter_cursor : next_chapter + 1]],
    }
    atomic_json(project / "hours.json", ledger)
    return movie
