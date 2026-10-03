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
from src.character_analysis import (
    analyze_chapter,
    create_narrator_entry,
    merge_character_info,
)
from src.data_recovery import (
    DataIssue,
    OperationalError,
    RecoveryLedger,
    is_data_error,
    load_json_store,
)
from src.epub_extract import get_output_dir
from src.llm import ask_sync
from src.movie_assemble import assemble_movie, probe_duration
from src.movie_images import generate_character_refs, generate_scene_images
from src.movie_storyboard import build_storyboard
from src.pipeline import acquire_lock, release_lock
from src.text_ingest import extract_any
from src.voice_description import _voice_description_for_one

HOUR_MAX_SECONDS = 3600.0
# Leave one second for AAC container priming while retaining a near-full natural hour.
AUDIO_MAX_SECONDS = 3599.0
SCENE_SECONDS = 20.0
# Per-hour override: Part 3 gets one generated illustration per second of runtime.
HOUR_SCENE_SECONDS = {3: 1.0}
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
    return {
        "source": str(source),
        "sha256": source_fingerprint(source),
        "output": str(get_output_dir(source)),
    }


# ##################################################################
# hour ledger
# load the durable sequence ledger, rejecting a different source at the same output path.
def load_ledger(project: Path, source: Path) -> dict:
    path = project / "hours.json"
    fingerprint = source_fingerprint(source)
    if not path.exists():
        return {"source": str(source), "sha256": fingerprint, "hours": {}}
    ledger = load_json_store(path, "hour_ledger")
    if ledger.get("sha256") != fingerprint:
        raise RuntimeError(
            "hour ledger source SHA-256 does not match input; refusing to mix stories"
        )
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
    author = (
        parts[1].split(", narrated by", 1)[0].strip() if len(parts) > 1 else "Unknown"
    )
    chapters = sorted(
        (path for path in chapters_dir.glob("*.txt") if path != intro),
        key=chapter_order,
    )
    if not chapters:
        raise OperationalError(
            "source_no_chapters", "input extraction produced no narrative chapters"
        )
    return title, author, chapters


# ##################################################################
# project cast
# read the monotonic shared cast; old identities are never replaced by later hours.
def load_cast(project: Path) -> dict:
    path = project / "characters.json"
    return load_json_store(path, "cast", default={})


# ##################################################################
# cast aliases
# map only exact normalized display-name matches so spelling/case aliases keep one identity without merging distinct names such as Ren and Ron.
def canonical_character_id(
    project: Path, cast: dict, candidate_id: str, name: str
) -> str:
    aliases_path = project / "character_aliases.json"
    aliases = load_json_store(aliases_path, "character_aliases", default={})
    if candidate_id in aliases:
        return aliases[candidate_id]
    normalized = "".join(char for char in name.casefold() if char.isalnum())
    for established_id, info in cast.items():
        established = "".join(
            char for char in str(info.get("name", "")).casefold() if char.isalnum()
        )
        if normalized and normalized == established:
            aliases[candidate_id] = established_id
            atomic_json(aliases_path, aliases)
            return established_id
    return candidate_id


# ##################################################################
# extend cast
# analyze one upcoming chapter only, append its new information, and retain established identities.
async def extend_cast(
    project: Path, chapter: Path, chapter_number: int, title: str, author: str
) -> dict:
    cast = load_cast(project)
    known = ", ".join(f"{cid}={info.get('name', cid)}" for cid, info in cast.items())
    found = merge_character_info(
        [await analyze_chapter(chapter, chapter_number, known)], RecoveryLedger(project)
    )
    if "narrator" not in cast:
        cast["narrator"] = await create_narrator_entry(
            title, author, chapter.read_text(encoding="utf-8")[:3000]
        )
    for char_id, info in found.items():
        char_id = canonical_character_id(
            project, cast, char_id, str(info.get("name", char_id))
        )
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
    voices = load_json_store(path, "voices", default={})
    missing = [
        (char_id, info) for char_id, info in cast.items() if char_id not in voices
    ]
    if missing:
        described = await asyncio.gather(
            *(_voice_description_for_one(char_id, info) for char_id, info in missing)
        )
        for char_id, info in described:
            voices[char_id] = info
        atomic_json(path, voices)
    prepare_breeze_voices(project)


APPEARANCE_BATCH_SIZE = 3
APPEARANCE_ATTEMPTS = 2


# ##################################################################
# appearance schema
# constrain each native model response to exactly the requested stable identities and nonempty visual descriptions.
def appearance_schema(character_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            character_id: {"type": "string", "minLength": 1}
            for character_id in character_ids
        },
        "required": character_ids,
        "additionalProperties": False,
    }


# ##################################################################
# validate appearances
# reject any partial, renamed, blank, or non-string appearance response before production state can be changed.
def validate_appearances(value: object, character_ids: list[str]) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(character_ids):
        raise ValueError(
            "appearance response keys do not exactly match requested character IDs"
        )
    result: dict[str, str] = {}
    for character_id in character_ids:
        description = value[character_id]
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"appearance response for {character_id} is not a nonempty string"
            )
        result[character_id] = description.strip()
    return result


# ##################################################################
# generate appearance batch
# make a small schema-constrained native request with bounded format repairs, preserving failure context and never inventing a fallback.
def generate_appearance_batch(items: list[tuple[str, dict]]) -> dict[str, str]:
    character_ids = [character_id for character_id, _ in items]
    roster = "\n".join(
        f"- {character_id}: {(info.get('look') or info.get('bio') or '')[:700]}"
        for character_id, info in items
    )
    prompt = (
        "For each requested character, give a 50-80 word canonical visual description for an image generator. "
        "Use source facts exactly; include age, build, hair, face, skin, clothing, and species when known. "
        "Never describe plot or relationships. Return only the response-schema object. Characters:\n"
        + roster
    )
    last_error = "no response"
    for attempt in range(APPEARANCE_ATTEMPTS):
        response = ask_sync(
            prompt, max_tokens=1200, response_schema=appearance_schema(character_ids)
        )
        try:
            return validate_appearances(json.loads(response), character_ids)
        except (ValueError, json.JSONDecodeError) as error:
            last_error = f"{error}; response={response[:500]!r}"
    raise DataIssue(
        "appearance_generation_failed",
        f"appearance generation failed for {', '.join(character_ids)} after {APPEARANCE_ATTEMPTS} attempts: {last_error}",
        {"characters": character_ids},
    )


# ##################################################################
# extend appearances
# append only all-valid new visual identities in one atomic write; established descriptions are never regenerated or replaced.
def extend_appearances(project: Path, cast: dict) -> None:
    path = project / "appearances.json"
    try:
        appearances = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
    except (OSError, ValueError) as error:
        raise OperationalError(
            "appearances_unreadable", f"appearances cache is unreadable: {path}"
        ) from error
    if not isinstance(appearances, dict):
        raise OperationalError(
            "appearances_invalid", "appearances cache is not an object"
        )
    for character_id in cast:
        if character_id in appearances and (
            not isinstance(appearances[character_id], str)
            or not appearances[character_id].strip()
        ):
            raise OperationalError(
                "appearances_invalid",
                f"cached appearance for {character_id} is invalid",
            )
    missing = [
        (character_id, info)
        for character_id, info in cast.items()
        if character_id != "narrator" and character_id not in appearances
    ]
    if not missing:
        return
    generated: dict[str, str] = {}
    for start in range(0, len(missing), APPEARANCE_BATCH_SIZE):
        generated.update(
            generate_appearance_batch(missing[start : start + APPEARANCE_BATCH_SIZE])
        )
    appearances.update(generated)
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
def prepare_hour_directory(
    project: Path, hour_dir: Path, approved_cast: dict | None = None
) -> None:
    hour_dir.mkdir(parents=True, exist_ok=True)
    intro = project / "chapters" / "00-intro.txt"
    if intro.exists():
        hour_chapters = hour_dir / "chapters"
        hour_chapters.mkdir(exist_ok=True)
        intro_copy = hour_chapters / intro.name
        if not intro_copy.exists():
            shutil.copyfile(intro, intro_copy)
    (project / "refs").mkdir(exist_ok=True)
    frozen_names = set(approved_cast or {})
    if frozen_names:
        # Later hours consume a local active-only metadata view. Root files keep
        # historical aliases for cache preservation but must never steer scenes.
        for name in (
            "characters.json",
            "voices.json",
            "breeze_voices.json",
            "appearances.json",
        ):
            source = json.loads((project / name).read_text(encoding="utf-8"))
            atomic_json(
                hour_dir / name,
                {
                    actor_id: source[actor_id]
                    for actor_id in frozen_names
                    if actor_id in source
                },
            )
    for name in (
        "voices",
        "refs",
        "world_bible.json",
        "locations.json",
        "frozen_cast_manifest.json",
        "frozen_character_aliases.json",
    ):
        link_shared(hour_dir, project, name)
    if frozen_names:
        (hour_dir / "frozen_cast_active_only.txt").write_text(
            "true\n", encoding="utf-8"
        )
    for name in ("style.txt", "style-reference.png"):
        if (project / name).exists():
            link_shared(hour_dir, project, name)
    hour_number = (
        int(hour_dir.name.rsplit("-", 1)[-1])
        if hour_dir.name.startswith("hour-")
        else 0
    )
    scene_seconds = HOUR_SCENE_SECONDS.get(hour_number, SCENE_SECONDS)
    (hour_dir / "scene_seconds.txt").write_text(
        f"{scene_seconds:g}\n", encoding="utf-8"
    )
    # Hourly productions deliberately use only the clean style asset and canonical cast portraits.
    (hour_dir / "image_reference_mode.txt").write_text(
        "style-and-cast-only\n", encoding="utf-8"
    )


# ##################################################################
# script for chapter
# create one chapter script against the current shared cast, avoiding any full-book script pass.
def script_for_chapter(
    project: Path,
    hour_dir: Path,
    chapter: Path,
    cast: dict,
    aliases: dict[str, str] | None = None,
    scoped_references: list[dict] | None = None,
) -> Path:
    shared_dir = project / "script_cache"
    shared_dir.mkdir(exist_ok=True)
    canonical = shared_dir / f"{chapter.stem}.jsonl"
    source = chapter.read_text(encoding="utf-8")
    metadata = canonical.with_name(canonical.name + ".hour.meta.json")
    if canonical.exists():
        payload = canonical.read_bytes()
        try:
            lines = [
                json.loads(line)
                for line in payload.decode("utf-8").splitlines()
                if line.strip()
            ]
            meta_path = (
                metadata
                if metadata.exists()
                else canonical.with_name(canonical.name + ".meta.json")
            )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("sha256") != hashlib.sha256(payload).hexdigest():
                raise ValueError("payload hash mismatch")
            if metadata.exists():
                if (
                    meta.get("mode") != "immutable-spans"
                    or meta.get("source_sha256")
                    != hashlib.sha256(source.encode()).hexdigest()
                ):
                    raise ValueError("hourly source metadata mismatch")
                if "".join(next(iter(line.values())) for line in lines) != source:
                    raise ValueError("hourly immutable spans do not reconstruct source")
                unknown = [
                    next(iter(line)) for line in lines if next(iter(line)) not in cast
                ]
                remapped_indices = [
                    index
                    for index, line in enumerate(lines)
                    if next(iter(line)) not in cast
                ]
                if unknown:
                    if aliases is None or any(
                        speaker not in aliases or aliases[speaker] not in cast
                        for speaker in unknown
                    ):
                        raise ValueError(
                            "hourly script has an unknown speaker outside frozen aliases"
                        )
                    frozen_dir = project / "frozen_script_cache"
                    frozen_dir.mkdir(exist_ok=True)
                    frozen = frozen_dir / canonical.name
                    remapped = [
                        {
                            aliases.get(next(iter(line)), next(iter(line))): next(
                                iter(line.values())
                            )
                        }
                        for line in lines
                    ]
                    if (
                        "".join(next(iter(line.values())) for line in remapped)
                        != source
                    ):
                        raise ValueError("frozen alias remap changed source text")
                    payload = "".join(
                        json.dumps(line, ensure_ascii=False) + "\n" for line in remapped
                    ).encode("utf-8")
                    if not frozen.exists():
                        frozen.write_bytes(payload)
                    elif frozen.read_bytes() != payload:
                        raise ValueError(
                            "frozen alias script content differs from existing cache"
                        )
                    frozen_meta = frozen.with_name(frozen.name + ".hour.meta.json")
                    atomic_json(
                        frozen_meta,
                        {
                            "mode": "immutable-spans",
                            "source_sha256": hashlib.sha256(
                                source.encode()
                            ).hexdigest(),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "legacy_script": str(canonical.relative_to(project)),
                            "remapped_speakers": sorted(set(unknown)),
                            "remapped_line_indexes": remapped_indices,
                        },
                    )
                    return frozen
            else:
                from src.script_generate import validate_script_lines

                validate_script_lines(lines, source, sorted(cast))
            return canonical
        except (
            ValueError,
            StopIteration,
            TypeError,
            KeyError,
            AttributeError,
        ) as error:
            raise DataIssue(
                "canonical_script_invalid",
                f"existing canonical script {canonical.name} failed read-only validation: {error}",
                {"script": canonical.name},
            ) from error
        except OSError as error:
            raise OperationalError(
                "canonical_script_unreadable",
                f"existing canonical script {canonical.name} is unreadable",
            ) from error
    from src.hourly_spans import generate_hourly_script_sync

    generate_hourly_script_sync(
        chapter, canonical, sorted(cast), aliases, scoped_references
    )
    payload = canonical.read_bytes()
    atomic_json(
        metadata,
        {
            "mode": "immutable-spans",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            **(
                {
                    "scoped_references_sha256": hashlib.sha256(
                        json.dumps(scoped_references, sort_keys=True).encode()
                    ).hexdigest()
                }
                if scoped_references
                else {}
            ),
        },
    )
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
def synthesize_window(
    project: Path, script: Path, remaining: float, start_piece: int
) -> tuple[list[dict], int]:
    audio_dir = project / "audio_cache"
    audio_dir.mkdir(exist_ok=True)
    _, paths, jobs, metadata = plan_chapter(
        script, audio_dir, project / "voices", tts_engine.speaker_set(project)
    )
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
def record_hour_timing(
    project: Path, hour_index: int, outcome: str, seconds: float
) -> None:
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
        raise RuntimeError(
            "completed hour ledger points at a missing or oversized movie"
        )
    if hour_index > 1:
        previous = ledger["hours"].get(str(hour_index - 1))
        if not previous or not previous.get("complete"):
            raise RuntimeError(
                "previous hour is not complete; refusing to duplicate or skip story content"
            )
        chapter_cursor = int(previous["next_chapter"])
        piece_cursor = int(previous["next_piece"])
    else:
        chapter_cursor, piece_cursor = 0, 0
    title, author, chapters = source_chapters(source, project)
    # Completed Parts 1 and 2 are historical assets and may be returned above without
    # touching a registry. Every new production hour starts only after the whole-source
    # frozen manifest verifies the source and every voice/portrait anchor byte.
    if hour_index >= 3:
        from src.cast_freeze import (
            load_approved_cast,
            validated_scoped_references,
            verify_frozen_cast,
        )

        frozen = verify_frozen_cast(source, project)
        cast = load_approved_cast(source, project)
        aliases = frozen["approved_aliases"]
        scoped_references = validated_scoped_references(source, project, frozen)
    else:
        cast = {}
        aliases = None
        scoped_references = None
    if tts_engine.engine_name() != "breeze":
        raise RuntimeError("hour production requires Breeze shared voice references")
    hour_dir = project / "hours" / f"hour-{hour_index:03d}"
    all_selected: list[dict] = []
    remaining = AUDIO_MAX_SECONDS
    next_chapter, next_piece = chapter_cursor, piece_cursor
    recovery = RecoveryLedger(project)
    pending_chapters: list[str] = []
    for chapter_index in range(chapter_cursor, len(chapters)):
        try:
            if hour_index < 3:
                cast = asyncio.run(
                    extend_cast(
                        project,
                        chapters[chapter_index],
                        chapter_index + 1,
                        title,
                        author,
                    )
                )
                asyncio.run(extend_voices(project, cast))
                extend_appearances(project, cast)
            # New hours never extend cast, voice, appearance, or portrait state.
            prepare_hour_directory(project, hour_dir, cast if hour_index >= 3 else None)
            copy_chapter_context(hour_dir, chapters[chapter_index])
            script = script_for_chapter(
                project,
                hour_dir,
                chapters[chapter_index],
                cast,
                aliases,
                scoped_references,
            )
            start_piece = piece_cursor if chapter_index == chapter_cursor else 0
            candidates, _ = synthesize_window(project, script, remaining, start_piece)
            if not candidates:
                if all_selected:
                    break
                raise DataIssue(
                    "piece_exceeds_hour_cap",
                    "first natural spoken piece does not fit within 3600 seconds",
                    {"chapter": chapters[chapter_index].name},
                )
            elapsed = sum(item["duration"] for item in candidates)
            if elapsed > remaining + 0.0001:
                raise RuntimeError("internal selection exceeded hour budget")
            all_selected.extend(candidates)
            remaining -= elapsed
            all_metadata = plan_chapter(
                script,
                project / "audio_cache",
                project / "voices",
                tts_engine.speaker_set(project),
            )[3]
            consumed = start_piece + len(candidates)
            if consumed < len(all_metadata):
                next_chapter, next_piece = chapter_index, consumed
                break
            next_chapter, next_piece = chapter_index + 1, 0
            if remaining <= 0.0001:
                break
        except Exception as error:
            if not is_data_error(error):
                raise
            # Per-chapter recovery: this chapter is quarantined with exact evidence, the cursor moves
            # past it only (never skipping silently), and production continues with the next chapter.
            recovery.record_error(
                "hour",
                chapters[chapter_index].name,
                error,
                checkpoint={
                    "hour": hour_index,
                    "chapter_index": chapter_index,
                    "next_chapter": chapter_index + 1,
                    "next_piece": 0,
                },
            )
            pending_chapters.append(chapters[chapter_index].name)
            next_chapter, next_piece = chapter_index + 1, 0
            continue
    if pending_chapters:
        # Publication gate: scan artifacts are durable above; the hour is never committed or rendered with pending chapters.
        raise DataIssue(
            "hour_blocked_pending",
            f"hour {hour_index} not committed: pending chapters {pending_chapters} (see data_recovery.jsonl)",
            {"pending": pending_chapters},
        )
    if not all_selected:
        raise DataIssue(
            "hour_no_content",
            "no source content selected for requested hour; every remaining chapter was quarantined (see data_recovery.jsonl)",
            {
                "quarantined": [
                    r["item"] for r in recovery.entries() if r["stage"] == "hour"
                ]
            },
        )
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
        "source_chapters": [
            str(path.relative_to(project))
            for path in chapters[chapter_cursor : next_chapter + 1]
        ],
    }
    atomic_json(project / "hours.json", ledger)
    return movie
