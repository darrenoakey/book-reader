"""Direct import of a finished noveliser project as a pre-attributed book-reader script.

Noveliser generates every line already tagged with its speaker (audio_chapter_N.json
holds [{"speaker", "text"}]). Re-parsing the rendered prose would throw that
attribution away, so this importer maps each supplied label onto one canonical
book-reader identity and writes the normal create-pipeline layout (chapters/,
characters.json, script/*.jsonl) without any LLM attribution.

Contract:
- Labels are identity declarations. "narrator" is the narrator; an exact
  characters.json name is that character; any other well-formed label becomes its
  own canonical identity with demographics explicitly unknown and a profile made
  only of verbatim source excerpts. Labels that are case/spacing variants of, or
  share a name token with, another identity are ambiguous and fail closed unless the
  operator maps them explicitly with --speaker-alias.
- Script lines follow the immutable-span convention: line i carries SEPARATOR
  before the supplied text for i > 0, so "".join(line texts) == the chapter text
  byte for byte and every supplied text survives verbatim as a suffix.
- noveliser_import.json binds the project to the exact source bytes. A changed
  source fails closed until --reimport archives (never deletes) the old project;
  frozen voice artifacts carry forward only when the cast is byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from src.epub_extract import get_output_dir, normalize_name
from src.state import is_step_complete, mark_step_complete

IMPORTER_VERSION = 1
MANIFEST_NAME = "noveliser_import.json"
SOURCE_CONTEXT_NAME = "source_context.json"
NARRATOR_ID = "narrator"
SEPARATOR = "\n\n"
AUDIO_PATTERN = re.compile(r"^audio_chapter_([1-9][0-9]*)\.json$")
SECTION_PATTERN = re.compile(r"^chapter_([1-9][0-9]*)_section_([1-9][0-9]*)\.json$")
OUTLINE_PATTERN = re.compile(r"^into_([1-9][0-9]*)_chapters\.json$")
# Files a run may create before the import publishes (CLI overrides, the run lock).
PRE_IMPORT_FILES = {"style.txt", "scene_seconds.txt", ".pipeline.lock"}
# Cast-frozen voice artifacts: reused across a re-import only for a byte-identical cast.
FROZEN_VOICE_ARTIFACTS = ("voices.json", "breeze_voices.json", "kokoro_voices.json", "voices")
NAME_STOPWORDS = {"mr", "mrs", "ms", "miss", "dr", "the", "a", "an", "of", "sir", "lady", "man", "woman"}
PROFILE_FIELDS = ("biography", "role", "traits", "wound", "lie", "want", "need", "arc")
MAX_SPOKEN_EXCERPTS = 12
MAX_CONTEXT_EXCERPTS = 8
MAX_EXCERPT_CHARS = 400
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class NoveliserImportError(ValueError):
    """The noveliser project cannot be imported without guessing; nothing is published."""


# ##################################################################
# hashing and atomic writes
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _jsonl(lines: list[dict]) -> bytes:
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ##################################################################
# read source json
# every source document is read as bytes so its hash binds exactly what was parsed.
def _read_source(path: Path, what: str) -> tuple[object, bytes]:
    if not path.is_file():
        raise NoveliserImportError(f"missing {what}: {path}")
    data = path.read_bytes()
    try:
        return json.loads(data.decode("utf-8")), data
    except (UnicodeDecodeError, ValueError) as error:
        raise NoveliserImportError(f"{what} is not valid UTF-8 JSON: {path}: {error}") from error


# ##################################################################
# identity helpers
def character_id(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    identity = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    if not identity:
        raise NoveliserImportError(f"speaker name {name!r} has no usable identity characters")
    return identity


def _squash(name: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", name).casefold() if char.isalnum())


def name_tokens(name: str) -> set[str]:
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", name).casefold())
    return {word for word in words if word not in NAME_STOPWORDS and len(word) > 1}


# ##################################################################
# validate text
# malformed text fails closed; supplied text is never stripped or rewritten.
def _check_text(text: object, where: str) -> str:
    if not isinstance(text, str):
        raise NoveliserImportError(f"{where}: text is not a string")
    if not text.strip():
        raise NoveliserImportError(f"{where}: text is empty")
    if CONTROL_CHARS.search(text) or "\r" in text:
        raise NoveliserImportError(f"{where}: text contains control characters")
    if "\ufffd" in text:
        raise NoveliserImportError(f"{where}: text contains U+FFFD (decoding damage)")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise NoveliserImportError(f"{where}: text has unpaired surrogates") from error
    return text


def _check_label(label: object, where: str) -> str:
    if not isinstance(label, str) or not label:
        raise NoveliserImportError(f"{where}: speaker is missing or not a string")
    if label != label.strip() or "  " in label or CONTROL_CHARS.search(label) or "\n" in label or "\t" in label:
        raise NoveliserImportError(f"{where}: speaker label {label!r} is malformed (stray whitespace/control)")
    return label


# ##################################################################
# load chapters
# chapter identity is the filename number; the set must be exactly 1..num_chapters.
def _load_chapters(source: Path, expected: int) -> tuple[dict[int, list[dict]], dict[str, str]]:
    hashes: dict[str, str] = {}
    chapters: dict[int, list[dict]] = {}
    for path in sorted(source.glob("audio_chapter_*")):
        match = AUDIO_PATTERN.match(path.name)
        if not match:
            raise NoveliserImportError(f"audio chapter file has no exact chapter identity: {path.name}")
        number = int(match.group(1))
        value, data = _read_source(path, f"audio chapter {number}")
        hashes[path.name] = sha256_bytes(data)
        if not isinstance(value, list) or not value:
            raise NoveliserImportError(f"{path.name}: expected a non-empty JSON array of lines")
        lines = []
        for index, item in enumerate(value):
            where = f"{path.name} line {index}"
            if not isinstance(item, dict) or set(item) != {"speaker", "text"}:
                raise NoveliserImportError(f"{where}: line must be exactly {{speaker, text}}")
            lines.append(
                {"speaker": _check_label(item["speaker"], where), "text": _check_text(item["text"], where)}
            )
        chapters[number] = lines
    found = sorted(chapters)
    wanted = list(range(1, expected + 1))
    if found != wanted:
        missing = sorted(set(wanted) - set(found))
        extra = sorted(set(found) - set(wanted))
        raise NoveliserImportError(
            f"audio chapters do not match metadata num_chapters={expected}: missing {missing}, unexpected {extra}"
        )
    return chapters, hashes


# ##################################################################
# section checkpoints
# noveliser derives each audio chapter from its section checkpoints' tagged lines; when the
# checkpoints are present they must reproduce the audio chapter exactly (same derivation rule).
def _verify_sections(
    source: Path, chapters: dict[int, list[dict]], per_chapter: object
) -> tuple[dict[str, str], dict[int, int]]:
    hashes: dict[str, str] = {}
    counts: dict[int, int] = {}
    grouped: dict[int, dict[int, Path]] = {}
    for path in sorted(source.glob("chapter_*_section_*.json")):
        match = SECTION_PATTERN.match(path.name)
        if not match:
            raise NoveliserImportError(f"section checkpoint has no exact identity: {path.name}")
        grouped.setdefault(int(match.group(1)), {})[int(match.group(2))] = path
    unknown = sorted(set(grouped) - set(chapters))
    if unknown:
        raise NoveliserImportError(f"section checkpoints exist for chapters without audio: {unknown}")
    for number, sections in sorted(grouped.items()):
        if sorted(sections) != list(range(1, len(sections) + 1)):
            raise NoveliserImportError(f"chapter {number} section checkpoints have gaps: {sorted(sections)}")
        if isinstance(per_chapter, int) and len(sections) != per_chapter:
            raise NoveliserImportError(
                f"chapter {number} has {len(sections)} section checkpoints, metadata says {per_chapter}"
            )
        derived: list[dict] = []
        for section in sorted(sections):
            path = sections[section]
            value, data = _read_source(path, f"chapter {number} section {section}")
            hashes[path.name] = sha256_bytes(data)
            lines = value.get("lines") if isinstance(value, dict) else None
            if not isinstance(lines, list):
                raise NoveliserImportError(f"{path.name}: checkpoint has no tagged lines array")
            for item in lines:
                if not isinstance(item, dict):
                    raise NoveliserImportError(f"{path.name}: tagged line is not an object")
                speaker = str(item.get("speaker") or "").strip()
                text = str(item.get("text") or "").strip()
                if text:
                    derived.append({"speaker": NARRATOR_ID if speaker.upper() == "NARRATOR" else speaker, "text": text})
        if derived != chapters[number]:
            first = next(
                (i for i, pair in enumerate(zip(derived, chapters[number])) if pair[0] != pair[1]),
                min(len(derived), len(chapters[number])),
            )
            raise NoveliserImportError(
                f"audio_chapter_{number}.json differs from its section checkpoints at line {first}"
            )
        counts[number] = len(sections)
    return hashes, counts


# ##################################################################
# chapter titles
def _chapter_titles(source: Path, expected: int) -> tuple[dict[int, str], dict[str, str]]:
    outlines = [path for path in source.glob("into_*_chapters.json") if OUTLINE_PATTERN.match(path.name)]
    if not outlines:
        return {number: f"Chapter {number}" for number in range(1, expected + 1)}, {}
    if len(outlines) != 1:
        raise NoveliserImportError(f"ambiguous chapter outlines: {sorted(p.name for p in outlines)}")
    value, data = _read_source(outlines[0], "chapter outline")
    entries = value.get("chapters") if isinstance(value, dict) else None
    if not isinstance(entries, list):
        raise NoveliserImportError(f"{outlines[0].name}: no chapters array")
    titles: dict[int, str] = {}
    for entry in entries:
        number = entry.get("number") if isinstance(entry, dict) else None
        title = entry.get("title") if isinstance(entry, dict) else None
        if not isinstance(number, int) or not isinstance(title, str) or not title.strip() or number in titles:
            raise NoveliserImportError(f"{outlines[0].name}: malformed or duplicate chapter entry {entry!r:.200}")
        titles[number] = title.strip()
    if sorted(titles) != list(range(1, expected + 1)):
        raise NoveliserImportError(f"{outlines[0].name}: chapter numbers {sorted(titles)} != 1..{expected}")
    return titles, {outlines[0].name: sha256_bytes(data)}


# ##################################################################
# listed characters
def _listed_characters(value: object) -> list[dict]:
    entries = value.get("characters") if isinstance(value, dict) else None
    if not isinstance(entries, list):
        raise NoveliserImportError("characters.json has no characters array")
    listed = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise NoveliserImportError(f"characters.json entry {index} is not an object")
        name = _check_label(entry.get("name"), f"characters.json entry {index}")
        if not isinstance(entry.get("biography", ""), str):
            raise NoveliserImportError(f"characters.json {name}: biography is not text")
        listed.append(entry)
    return listed


def _excerpt(text: str) -> str:
    if len(text) <= MAX_EXCERPT_CHARS:
        return text
    return text[:MAX_EXCERPT_CHARS] + " [excerpt truncated]"


# ##################################################################
# textual evidence
# verbatim spoken lines plus narration that names the speaker or immediately surrounds
# their lines; casting reads this instead of any invented description.
def textual_evidence(chapters: dict[int, list[dict]], labels: set[str], mention_names: list[str]) -> dict:
    spoken: list[dict] = []
    context: list[dict] = []
    seen_context: set[tuple[int, int]] = set()
    patterns = [re.compile(rf"(?<!\w){re.escape(name)}(?!\w)") for name in mention_names if name]
    total = 0
    for number, lines in sorted(chapters.items()):
        for index, line in enumerate(lines):
            if line["speaker"] in labels:
                total += 1
                if len(spoken) < MAX_SPOKEN_EXCERPTS:
                    spoken.append({"chapter": number, "line": index, "text": _excerpt(line["text"])})
                for neighbour in (index - 1, index + 1):
                    if 0 <= neighbour < len(lines) and lines[neighbour]["speaker"] == NARRATOR_ID:
                        seen_context.add((number, neighbour))
            elif line["speaker"] == NARRATOR_ID and any(p.search(line["text"]) for p in patterns):
                seen_context.add((number, index))
    for number, index in sorted(seen_context)[:MAX_CONTEXT_EXCERPTS]:
        context.append({"chapter": number, "line": index, "text": _excerpt(chapters[number][index]["text"])})
    return {"spoken_line_count": total, "spoken": spoken, "narration": context}


def _evidence_prose(evidence: dict) -> str:
    spoken = " | ".join(f'"{item["text"]}"' for item in evidence["spoken"])
    narration = " | ".join(item["text"] for item in evidence["narration"])
    parts = [f"Speaks {evidence['spoken_line_count']} script lines."]
    if spoken:
        parts.append(f"Spoken lines (verbatim): {spoken}")
    if narration:
        parts.append(f"Narration naming or surrounding them (verbatim): {narration}")
    return " ".join(parts)


# ##################################################################
# resolve cast
# map every supplied label to exactly one canonical identity, or fail closed listing every problem.
def resolve_cast(
    listed: list[dict],
    chapters: dict[int, list[dict]],
    speaker_aliases: dict[str, str],
) -> tuple[dict[str, str], dict[str, dict]]:
    """Return (label -> canonical id, canonical id -> {name, origin, entry})."""
    problems: list[str] = []
    identities: dict[str, dict] = {NARRATOR_ID: {"name": "Narrator", "origin": "narrator", "entry": None}}
    label_map: dict[str, str] = {NARRATOR_ID: NARRATOR_ID}
    by_name: dict[str, str] = {}
    for entry in listed:
        name = entry["name"]
        identity = character_id(name)
        if identity in identities:
            problems.append(f"characters.json name {name!r} collides with identity {identity!r}")
            continue
        identities[identity] = {"name": name, "origin": "characters.json", "entry": entry}
        label_map[name] = identity
        by_name[name] = identity
    usage: dict[str, list[tuple[int, int]]] = {}
    for number, lines in sorted(chapters.items()):
        for index, line in enumerate(lines):
            usage.setdefault(line["speaker"], []).append((number, index))
    for label, target in sorted(speaker_aliases.items()):
        if label not in usage:
            problems.append(f"--speaker-alias {label!r} is never used as a speaker label")
        elif label in label_map:
            problems.append(f"--speaker-alias {label!r} is already an exact identity; aliasing it would rewrite it")
        elif target not in by_name and target != NARRATOR_ID:
            problems.append(f"--speaker-alias target {target!r} is not an exact characters.json name or narrator")
        else:
            label_map[label] = by_name.get(target, NARRATOR_ID)
    unlisted = sorted(label for label in usage if label not in label_map)
    known_names = {identity: info["name"] for identity, info in identities.items()}
    for label in unlisted:
        where = usage[label][0]
        variants = [
            known for known in known_names.values() if _squash(known) == _squash(label)
        ] + [other for other in unlisted if other != label and _squash(other) == _squash(label)]
        overlaps = [
            known
            for known in [*known_names.values(), *unlisted]
            if known != label and name_tokens(known) & name_tokens(label)
        ]
        if variants:
            problems.append(
                f"speaker {label!r} (chapter {where[0]} line {where[1]}) is an inexact variant of {sorted(variants)};"
                " map it explicitly with --speaker-alias"
            )
        elif overlaps:
            problems.append(
                f"speaker {label!r} (chapter {where[0]} line {where[1]}, {len(usage[label])} lines) is ambiguous:"
                f" shares a name with {sorted(overlaps)}; map it explicitly with --speaker-alias"
            )
        else:
            identity = character_id(label)
            if identity in identities:
                problems.append(f"speaker {label!r} collides with identity {identity!r}")
                continue
            identities[identity] = {"name": label, "origin": "script-label", "entry": None}
            label_map[label] = identity
    if problems:
        raise NoveliserImportError("speaker identity cannot be resolved:\n  - " + "\n  - ".join(problems))
    used = {label_map[label] for label in usage}
    return {label: identity for label, identity in label_map.items() if label in usage}, {
        identity: info for identity, info in identities.items() if identity in used or info["origin"] != "script-label"
    }


# ##################################################################
# cast profiles
# reader bios built only from supplied text: noveliser profile fields, the verbatim book
# premise, and verbatim source excerpts. Unlisted speakers get explicitly unknown demographics.
def build_characters(
    identities: dict[str, dict],
    label_map: dict[str, str],
    chapters: dict[int, list[dict]],
    title: str,
    author: str,
    premise: str,
) -> dict[str, dict]:
    labels_for: dict[str, set[str]] = {}
    for label, identity in label_map.items():
        labels_for.setdefault(identity, set()).add(label)
    premise_text = f"Book premise (noveliser metadata, verbatim): {premise}" if premise else ""
    cast: dict[str, dict] = {}
    for identity, info in identities.items():
        name = info["name"]
        if identity == NARRATOR_ID:
            bio = f"Audiobook narrator of {title!r} by {author}, carrying all narration. {premise_text}".strip()
            cast[identity] = {
                "name": "Narrator",
                "bio": bio,
                "look": "",
                "role": "narrator",
                "source": "noveliser",
                "origin": "narrator",
                "story_context": premise,
            }
            continue
        labels = labels_for.get(identity, set())
        mentions = [name] + ([name.split()[0]] if info["origin"] == "characters.json" and " " in name else [])
        evidence = textual_evidence(chapters, labels, mentions)
        if info["origin"] == "characters.json":
            entry = info["entry"]
            profile = {field: entry[field] for field in PROFILE_FIELDS if field in entry}
            traits = entry.get("traits") or []
            detail = " ".join(
                f"{label}: {entry[field]}"
                for field, label in (("wound", "Wound"), ("lie", "Lie"), ("want", "Want"), ("need", "Need"),
                                     ("arc", "Arc"))
                if isinstance(entry.get(field), str) and entry[field].strip()
            )
            trait_text = f"Traits: {', '.join(str(t) for t in traits)}." if isinstance(traits, list) and traits else ""
            role = entry.get("role") if isinstance(entry.get("role"), str) else ""
            bio_parts = [
                f"{name}{f' ({role})' if role else ''}.",
                entry.get("biography", "").strip(),
                trait_text,
                detail,
                _evidence_prose(evidence),
                premise_text,
            ]
        else:
            role = "minor (script label only)"
            profile = {
                "demographics": "unknown",
                "note": "noveliser named this speaker only in script labels and supplied no profile",
            }
            bio_parts = [
                (
                    f"{name}: minor character named only by the noveliser script's speaker labels."
                    " Noveliser supplied no profile: age, gender and background are unknown and must be"
                    " judged only from the verbatim evidence below."
                ),
                _evidence_prose(evidence),
                premise_text,
            ]
        cast[identity] = {
            "name": name,
            "bio": " ".join(part for part in bio_parts if part),
            "look": "",
            "role": role,
            "source": "noveliser",
            "origin": info["origin"],
            "labels": sorted(labels),
            "profile": profile,
            "evidence": evidence,
            "story_context": premise,
        }
    return cast


# ##################################################################
# plan import
# read and validate the complete source; returns every output file's bytes without writing.
def plan_import(source: Path, speaker_aliases: dict[str, str] | None = None) -> dict:
    source = source.resolve()
    if not source.is_dir():
        raise NoveliserImportError(f"noveliser project is not a directory: {source}")
    aliases = dict(speaker_aliases or {})
    metadata, metadata_bytes = _read_source(source / "metadata.json", "metadata.json")
    if not isinstance(metadata, dict):
        raise NoveliserImportError("metadata.json is not an object")
    if metadata.get("status") != "finished":
        raise NoveliserImportError(f"noveliser project is not finished (status={metadata.get('status')!r})")
    title = metadata.get("title")
    if not isinstance(title, str) or not title.strip():
        raise NoveliserImportError("metadata.json has no title")
    title = title.strip()
    source_hashes = {"metadata.json": sha256_bytes(metadata_bytes)}
    if (source / "a_title.json").exists():
        title_doc, title_bytes = _read_source(source / "a_title.json", "a_title.json")
        if not isinstance(title_doc, dict) or str(title_doc.get("title", "")).strip() != title:
            raise NoveliserImportError("a_title.json title disagrees with metadata.json title")
        source_hashes["a_title.json"] = sha256_bytes(title_bytes)
    author = metadata.get("author")
    author = author.strip() if isinstance(author, str) and author.strip() else "Unknown Author"
    premise = metadata.get("description") if isinstance(metadata.get("description"), str) else ""
    expected = metadata.get("num_chapters")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
        raise NoveliserImportError(f"metadata.json num_chapters is not a positive integer: {expected!r}")
    characters_doc, characters_bytes = _read_source(source / "characters.json", "characters.json")
    source_hashes["characters.json"] = sha256_bytes(characters_bytes)
    listed = _listed_characters(characters_doc)
    chapters, chapter_hashes = _load_chapters(source, expected)
    section_hashes, section_counts = _verify_sections(source, chapters, metadata.get("sections_per_chapter"))
    titles, outline_hashes = _chapter_titles(source, expected)
    source_hashes.update(chapter_hashes)
    source_hashes.update(section_hashes)
    source_hashes.update(outline_hashes)
    label_map, identities = resolve_cast(listed, chapters, aliases)
    cast = build_characters(identities, label_map, chapters, title, author, premise)
    cast_bytes = _canonical_json(cast)

    files: dict[str, bytes] = {}
    intro = f"{title} by {author}, narrated by Darren's Book Reader."
    files["chapters/00-intro.txt"] = intro.encode("utf-8")
    files["script/00-intro.jsonl"] = _jsonl([{NARRATOR_ID: intro}])
    chapter_records = []
    stems: set[str] = set()
    for number, lines in sorted(chapters.items()):
        stem = f"{number:02d}-{normalize_name(titles[number])[:40]}"
        if stem in stems:
            raise NoveliserImportError(f"chapter file name collision: {stem}")
        stems.add(stem)
        script = [
            {label_map[line["speaker"]]: (SEPARATOR if index else "") + line["text"]}
            for index, line in enumerate(lines)
        ]
        chapter_text = "".join(next(iter(entry.values())) for entry in script)
        files[f"chapters/{stem}.txt"] = chapter_text.encode("utf-8")
        files[f"script/{stem}.jsonl"] = _jsonl(script)
        speakers: dict[str, int] = {}
        for line in lines:
            speakers[line["speaker"]] = speakers.get(line["speaker"], 0) + 1
        chapter_records.append(
            {
                "number": number,
                "title": titles[number],
                "stem": stem,
                "source_file": f"audio_chapter_{number}.json",
                "source_sha256": chapter_hashes[f"audio_chapter_{number}.json"],
                "line_count": len(lines),
                "section_checkpoints": section_counts.get(number, 0),
                "speaker_labels": dict(sorted(speakers.items())),
            }
        )
    files["characters.json"] = cast_bytes
    files[SOURCE_CONTEXT_NAME] = _canonical_json(
        {
            "title": title,
            "author": author,
            "description": premise,
            "source_dir": str(source),
            "metadata_sha256": source_hashes["metadata.json"],
            "characters_sha256": source_hashes["characters.json"],
        }
    )
    for name in ("cover.jpg", "cover.jpeg", "cover.png"):
        if (source / name).is_file():
            cover = (source / name).read_bytes()
            source_hashes[name] = sha256_bytes(cover)
            files[name] = cover
            break
    fingerprint = sha256_bytes(_canonical_json(sorted(source_hashes.items())))
    identities_out = {
        identity: {
            "name": info["name"],
            "origin": info["origin"],
            "labels": sorted(label for label, target in label_map.items() if target == identity),
        }
        for identity, info in identities.items()
    }
    manifest = {
        "importer": "noveliser",
        "importer_version": IMPORTER_VERSION,
        "source_dir": str(source),
        "source_fingerprint": fingerprint,
        "source_files": dict(sorted(source_hashes.items())),
        "options": {"speaker_aliases": dict(sorted(aliases.items()))},
        "title": title,
        "author": author,
        "separator": SEPARATOR,
        "attribution": {"engine": "noveliser-supplied", "llm": False},
        "cast_sha256": sha256_bytes(cast_bytes),
        "identities": identities_out,
        "label_map": dict(sorted(label_map.items())),
        "chapters": chapter_records,
        "total_lines": sum(record["line_count"] for record in chapter_records),
        "outputs": {path: sha256_bytes(data) for path, data in sorted(files.items())},
    }
    return {"files": files, "manifest": manifest, "chapters": chapters, "label_map": label_map}


# ##################################################################
# verify script exactness
# each imported line is the supplied text verbatim (after the declared separator), every
# speaker is the label's canonical identity, and the lines concatenate to the chapter file.
def verify_scripts(output_dir: Path, plan: dict) -> None:
    manifest = plan["manifest"]
    for record in manifest["chapters"]:
        script_path = output_dir / "script" / f"{record['stem']}.jsonl"
        chapter_path = output_dir / "chapters" / f"{record['stem']}.txt"
        rows = [json.loads(raw) for raw in script_path.read_text(encoding="utf-8").splitlines() if raw]
        supplied = plan["chapters"][record["number"]]
        if len(rows) != len(supplied):
            raise NoveliserImportError(f"{script_path.name}: {len(rows)} lines, source has {len(supplied)}")
        for index, (row, line) in enumerate(zip(rows, supplied)):
            expected = {plan["label_map"][line["speaker"]]: (SEPARATOR if index else "") + line["text"]}
            if row != expected:
                raise NoveliserImportError(f"{script_path.name} line {index} differs from the supplied source line")
        if "".join(next(iter(row.values())) for row in rows) != chapter_path.read_text(encoding="utf-8"):
            raise NoveliserImportError(f"{script_path.name} does not concatenate to {chapter_path.name}")


def _verify_outputs(output_dir: Path, manifest: dict) -> None:
    for relative, digest in manifest["outputs"].items():
        path = output_dir / relative
        if not path.is_file():
            raise NoveliserImportError(f"imported output missing: {relative}")
        if sha256_bytes(path.read_bytes()) != digest:
            raise NoveliserImportError(f"imported output was modified after import: {relative}")


def read_manifest(output_dir: Path) -> dict | None:
    path = output_dir / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise NoveliserImportError(f"{MANIFEST_NAME} is unreadable: {path}") from error
    if not isinstance(manifest, dict) or manifest.get("importer") != "noveliser":
        raise NoveliserImportError(f"{MANIFEST_NAME} is not a noveliser import manifest: {path}")
    return manifest


# ##################################################################
# verify cast frozen
# once voices exist they must voice exactly the imported identities: casting may describe
# them but never add, drop, or rename a speaker.
def verify_cast_frozen(output_dir: Path, manifest: dict) -> None:
    voices_path = output_dir / "voices.json"
    if not voices_path.exists():
        return
    voices = json.loads(voices_path.read_text(encoding="utf-8"))
    expected = set(manifest["identities"])
    # "_"-prefixed keys (e.g. a cast-plan fingerprint) are casting metadata, not speakers.
    actual = {key for key in voices if not key.startswith("_")} if isinstance(voices, dict) else set()
    if not isinstance(voices, dict) or actual != expected:
        raise NoveliserImportError(
            f"voices.json identities differ from the imported cast: missing {sorted(expected - actual)},"
            f" extra {sorted(actual - expected)}"
        )


# ##################################################################
# verify import
# resume gate: the recorded source still has the same bytes and every output is unchanged.
def verify_import(output_dir: Path) -> dict:
    manifest = read_manifest(output_dir)
    if manifest is None:
        raise NoveliserImportError(f"not a noveliser import (no {MANIFEST_NAME}): {output_dir}")
    if manifest.get("importer_version") != IMPORTER_VERSION:
        raise NoveliserImportError(
            f"import was made by importer v{manifest.get('importer_version')}, this is v{IMPORTER_VERSION};"
            " run ./run noveliser <project> --reimport"
        )
    plan = plan_import(Path(manifest["source_dir"]), manifest["options"]["speaker_aliases"])
    if plan["manifest"]["source_fingerprint"] != manifest["source_fingerprint"]:
        raise NoveliserImportError(_change_summary(manifest, plan["manifest"]))
    _verify_outputs(output_dir, manifest)
    verify_scripts(output_dir, plan)
    verify_cast_frozen(output_dir, manifest)
    return manifest


def _change_summary(old: dict, new: dict) -> str:
    before, after = old.get("source_files", {}), new["source_files"]
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    return (
        f"noveliser source changed since import ({', '.join(changed) or 'options'}); refusing to mix"
        " versions. Run ./run noveliser <project> --reimport to archive this project and import afresh."
    )


# ##################################################################
# archive project
# a re-import never deletes produced work: the old project moves aside intact.
def _archive(output_dir: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archived = output_dir.with_name(f"{output_dir.name}.superseded-{stamp}")
    output_dir.rename(archived)
    return archived


def _publish(output_dir: Path, plan: dict) -> None:
    for relative, data in plan["files"].items():
        _atomic_write(output_dir / relative, data)
    _atomic_write(output_dir / MANIFEST_NAME, _canonical_json(plan["manifest"]))


def _mark_imported(output_dir: Path) -> None:
    for step in ("extract", "characters"):
        if not is_step_complete(output_dir, step):
            mark_step_complete(output_dir, step)


# ##################################################################
# import noveliser
# publish (or resume) the import. Options None reuses the recorded options on resume.
def import_noveliser(
    source: Path,
    output_dir: Path | None = None,
    speaker_aliases: dict[str, str] | None = None,
    reimport: bool = False,
) -> dict:
    source = source.resolve()
    output_dir = output_dir or get_output_dir(source)
    previous = read_manifest(output_dir) if output_dir.exists() else None
    if previous is not None and Path(previous["source_dir"]) != source:
        raise NoveliserImportError(
            f"{output_dir} was imported from {previous['source_dir']}, not {source}; refusing to mix projects"
        )
    aliases = speaker_aliases
    if aliases is None:
        aliases = previous["options"]["speaker_aliases"] if previous else {}
    plan = plan_import(source, aliases)
    if previous is not None:
        same = (
            previous.get("importer_version") == IMPORTER_VERSION
            and previous["source_fingerprint"] == plan["manifest"]["source_fingerprint"]
            and previous["options"] == plan["manifest"]["options"]
        )
        if same:
            _verify_outputs(output_dir, previous)
            verify_scripts(output_dir, plan)
            verify_cast_frozen(output_dir, previous)
            _mark_imported(output_dir)
            return previous
        if not reimport:
            raise NoveliserImportError(_change_summary(previous, plan["manifest"]))
        archived = _archive(output_dir)
        output_dir.mkdir(parents=True)
        if (archived / ".pipeline.lock").is_file():
            # the running pipeline's single-runner lock stays with the live project
            (archived / ".pipeline.lock").rename(output_dir / ".pipeline.lock")
        for name in ("style.txt", "scene_seconds.txt"):
            if (archived / name).is_file():
                shutil.copy2(archived / name, output_dir / name)
        if previous.get("cast_sha256") == plan["manifest"]["cast_sha256"]:
            for name in FROZEN_VOICE_ARTIFACTS:
                artifact = archived / name
                if artifact.is_dir():
                    shutil.copytree(artifact, output_dir / name)
                elif artifact.is_file():
                    shutil.copy2(artifact, output_dir / name)
        plan["manifest"]["superseded"] = str(archived)
    elif output_dir.exists():
        allowed = set(plan["files"]) | PRE_IMPORT_FILES | {"chapters", "script"}
        foreign = sorted(
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.relative_to(output_dir).as_posix() not in allowed
            and not path.name.endswith(".tmp")
        )
        if foreign:
            raise NoveliserImportError(
                f"{output_dir} already holds a non-noveliser project ({foreign[:5]}); refusing to overwrite"
            )
    _publish(output_dir, plan)
    verify_scripts(output_dir, plan)
    if (output_dir / "voices.json").exists():
        verify_cast_frozen(output_dir, plan["manifest"])
        if not is_step_complete(output_dir, "voices_desc"):
            mark_step_complete(output_dir, "voices_desc")
    _mark_imported(output_dir)
    return plan["manifest"]


# ##################################################################
# parse alias options
def parse_aliases(values: list[str] | None) -> dict[str, str] | None:
    if not values:
        return None
    aliases: dict[str, str] = {}
    for value in values:
        label, sep, target = value.partition("=")
        if not sep or not label or not target or label in aliases:
            raise NoveliserImportError(f"--speaker-alias must be unique LABEL=CHARACTER NAME, got {value!r}")
        aliases[label] = target
    return aliases


def summary(manifest: dict) -> str:
    identities = manifest["identities"]
    lines = [
        (
            f"Imported {manifest['title']!r} by {manifest['author']}: {len(manifest['chapters'])} chapters,"
            f" {manifest['total_lines']} lines, {len(identities)} speaker identities (no LLM attribution)"
        ),
    ]
    for identity, info in sorted(identities.items()):
        count = sum(
            count for record in manifest["chapters"] for label, count in record["speaker_labels"].items()
            if label in info["labels"]
        )
        lines.append(f"  {identity}: {info['name']} [{info['origin']}] {count} lines")
    return "\n".join(lines)
