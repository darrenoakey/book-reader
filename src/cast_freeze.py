"""Resumable, source-grounded full-book cast preparation and freeze verification."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

from src.breeze_voices import prepare_breeze_voices
from src.epub_extract import get_output_dir
from src.hour_runner import atomic_json, source_chapters, source_fingerprint
from src.hourly_spans import immutable_spans
from src.llm import LLM_STYLE, ask_sync
from src.movie_images import generate_missing_character_refs
from src.voice_description import _voice_description_for_one

MANIFEST_NAME = "frozen_cast_manifest.json"
PROGRESS_NAME = "cast_preparation_progress.json"
ALIASES_NAME = "frozen_character_aliases.json"
AUDIT_NAME = "cast_alias_audit.json"
DISCOVERIES_NAME = "cast_preparation_discoveries.jsonl"
BATCH_CHAPTERS = 6
MAX_BATCH_CHARACTERS = 24
VOICE_PROFILE_BATCH_SIZE = 4
# The router may use the primary's 32,768-token context on any request, not
# only the backup's 40,960-token context. Two characters/token is deliberately
# conservative for names, punctuation, and transcription artefacts; reserve
# structured output plus system/router framing before accepting source bytes.
NATIVE_MIN_CONTEXT_TOKENS = 32_768
NATIVE_RESERVED_OUTPUT_TOKENS = 3_500
NATIVE_RESERVED_ROUTER_TOKENS = 2_500
PREPARATION_PROMPT_MAX_CHARS = 2 * (
    NATIVE_MIN_CONTEXT_TOKENS - NATIVE_RESERVED_OUTPUT_TOKENS - NATIVE_RESERVED_ROUTER_TOKENS
)
ANCHOR_IDS = {
    "ceremony_master",
    "ron_blackfire",
    "ren",
    "k_goldest",
    "mother",
    "father",
    "patender",
    "luna_starwaver",
    "narrator",
}
IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,79}$")


# ##################################################################
# stable JSON digest
# makes profile and manifest checks independent of JSON whitespace while every media asset remains byte-addressed.
def json_digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ##################################################################
# file digest
# binds an immutable production asset to its actual bytes rather than its path or timestamp.
def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ##################################################################
# normalized ID
# accepts model-created identities only when the declared display name deterministically produces their identifier.
def normalized_id(name: str) -> str:
    value = "".join(char.lower() if char.isalnum() else "_" for char in name.strip())
    return re.sub(r"_+", "_", value).strip("_")


# ##################################################################
# load object
# reject malformed caches before they can become an authoritative frozen registry.
def load_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required {label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"required {label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"required {label} is not an object: {path}")
    return value


# ##################################################################
# immutable evidence units
# assigns each source sentence a stable batch-local identifier so a model selects evidence without ever reproducing source prose.
def immutable_evidence_units(chapters: list[Path]) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for chapter_index, chapter in enumerate(chapters):
        for sentence_index, quote in enumerate(immutable_spans(chapter.read_text(encoding="utf-8"))):
            units.append(
                {
                    "id": f"c{chapter_index:02d}s{sentence_index:05d}",
                    "chapter": chapter.name,
                    "quote": quote,
                }
            )
    if not units:
        raise ValueError("source batch has no immutable evidence units")
    return units


# ##################################################################
# source batch schema
# asks native Ollama to select only enumerated immutable source IDs; the program, never the model, materializes quotations.
def discovery_schema(known_ids: list[str], evidence_unit_ids: list[str] | None = None) -> dict:
    known = sorted(set(known_ids))
    evidence_property = (
        {"type": "array", "items": {"type": "string", "enum": evidence_unit_ids}, "minItems": 1, "maxItems": 8}
        if evidence_unit_ids is not None
        else {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1, "maxItems": 8}
    )
    evidence_key = "evidence_unit_ids" if evidence_unit_ids is not None else "evidence"
    return {
        "type": "object",
        "properties": {
            "characters": {
                "type": "array",
                "maxItems": MAX_BATCH_CHARACTERS,
                "items": {
                    "type": "object",
                    "properties": {
                        "canonical_id": {"type": "string", "enum": ["new", *known]},
                        "id": {"type": "string", "minLength": 1, "maxLength": 80},
                        "name": {"type": "string", "minLength": 1, "maxLength": 120},
                        "aliases": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 12},
                        "voice_facts": {"type": "string", "maxLength": 1000},
                        "look_facts": {"type": "string", "maxLength": 1000},
                        evidence_key: evidence_property,
                    },
                    "required": ["canonical_id", "id", "name", "aliases", "voice_facts", "look_facts", evidence_key],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["characters"],
        "additionalProperties": False,
    }


# ##################################################################
# discovery prompt
# sends numbered immutable source sentences so model output remains compact and citations can be reconstructed exactly and locally.
def discovery_prompt(chapters: list[Path], registry: dict, aliases: dict[str, str]) -> str:
    roster = "; ".join(f"{actor_id}={info.get('name', actor_id)}" for actor_id, info in sorted(registry.items()))
    approved_aliases = "; ".join(
        f"{alias}->{canonical}" for alias, canonical in sorted(aliases.items()) if alias != canonical
    )
    units = immutable_evidence_units(chapters)
    excerpts = "\n".join(f"[{unit['id']}] {unit['quote']}" for unit in units)
    return f"""Extract every legitimate named or consistently role-named person/creature who speaks, has internal monologue, or is directly depicted in these source chapters. Do not invent a character for an unnamed crowd, pronoun, title, or a mere mention. Use the response schema only.

Known canonical IDs (reuse a listed ID only when source evidence establishes it is the same identity): {roster or "(none)"}
Approved source-audited aliases (always use their canonical target, never create the alias as a new identity): {approved_aliases or "(none)"}

For a new identity, id must be lowercase underscore normalization of name. For an existing identity, canonical_id must be that exact known ID and id must repeat it. aliases are source spellings/titles for that same identity. Select evidence_unit_ids only from the numbered source units. Do not copy, paraphrase, or quote source text: local code creates the exact citations from your selected IDs. name and every alias must be visibly supported by a selected unit. voice_facts and look_facts may contain only facts supported by selected units; use an empty string when unstated. Never infer a merge from similarity alone.

SOURCE EVIDENCE UNITS:\n{excerpts}"""


# ##################################################################
# context-safe source batch
# takes consecutive complete chapters only while leaving native Ollama enough context for its structured response; it never truncates or skips source.
def context_safe_batch(chapters: list[Path], start: int, registry: dict, aliases: dict[str, str]) -> tuple[list[Path], str]:
    selected: list[Path] = []
    prompt = ""
    for chapter in chapters[start : start + BATCH_CHAPTERS]:
        candidate = [*selected, chapter]
        candidate_prompt = discovery_prompt(candidate, registry, aliases)
        if len(candidate_prompt) > PREPARATION_PROMPT_MAX_CHARS:
            if not selected:
                raise RuntimeError(
                    f"source chapter {chapter.name} exceeds safe native model context; refusing to truncate or skip it"
                )
            break
        selected, prompt = candidate, candidate_prompt
    if not selected:
        raise RuntimeError("no source chapters fit the native model context")
    return selected, prompt


# ##################################################################
# materialize evidence discovery
# turns schema-enumerated IDs into exact original citations before compatibility validation, preventing copied or paraphrased model quotations.
def materialize_evidence_discovery(value: object, units: list[dict[str, str]]) -> tuple[dict, list[list[str]]]:
    if not isinstance(value, dict) or set(value) != {"characters"} or not isinstance(value["characters"], list):
        raise ValueError("evidence discovery response is not the exact object schema")
    by_id = {unit["id"]: unit for unit in units}
    if len(by_id) != len(units):
        raise RuntimeError("immutable evidence unit IDs are not unique")
    materialized: list[dict] = []
    citations: list[list[str]] = []
    expected = {"canonical_id", "id", "name", "aliases", "voice_facts", "look_facts", "evidence_unit_ids"}
    for item in value["characters"]:
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError("evidence discovery contains an invalid character record")
        selected = item["evidence_unit_ids"]
        if not isinstance(selected, list) or not selected or len(selected) > 8 or any(
            not isinstance(unit_id, str) or unit_id not in by_id for unit_id in selected
        ):
            raise ValueError("evidence discovery selected an invalid immutable evidence ID")
        selected_units = [by_id[unit_id] for unit_id in selected]
        labels = [item.get("name"), *(item.get("aliases") if isinstance(item.get("aliases"), list) else [])]
        for label in labels:
            if not isinstance(label, str) or not label.strip() or not any(
                re.search(rf"(?<!\\w){re.escape(label.strip())}(?!\\w)", unit["quote"], re.IGNORECASE)
                for unit in selected_units
            ):
                raise ValueError(f"discovery name or alias lacks selected source evidence: {label!r}")
        materialized.append({key: item[key] for key in expected - {"evidence_unit_ids"}} | {"evidence": [unit["quote"] for unit in selected_units]})
        citations.append(selected)
    return {"characters": materialized}, citations


# ##################################################################
# validate discovery
# fail closed unless every model fact has exact local source evidence and every identity is schema-compatible with the current registry.
def validate_discovery(
    value: object, source_text: str, known_ids: set[str], ambiguous_new_ids: set[str] | None = None
) -> list[dict]:
    if not isinstance(value, dict) or set(value) != {"characters"} or not isinstance(value["characters"], list):
        raise ValueError("discovery response is not the exact object schema")
    found: list[dict] = []
    for item in value["characters"]:
        if not isinstance(item, dict) or set(item) != {
            "canonical_id",
            "id",
            "name",
            "aliases",
            "voice_facts",
            "look_facts",
            "evidence",
        }:
            raise ValueError("discovery contains an invalid character record")
        canonical = item["canonical_id"]
        actor_id = item["id"]
        name = item["name"]
        aliases = item["aliases"]
        evidence = item["evidence"]
        if not all(
            isinstance(value, str) for value in (canonical, actor_id, name, item["voice_facts"], item["look_facts"])
        ):
            raise ValueError("discovery character fields must be strings")
        if canonical != "new" and canonical not in known_ids:
            raise ValueError(f"discovery selected unknown canonical ID: {canonical}")
        if canonical == "new" and (not IDENTIFIER.fullmatch(actor_id) or actor_id != normalized_id(name)):
            raise ValueError(f"new discovery ID is not deterministic for {name!r}")
        if canonical == "new" and actor_id in (ambiguous_new_ids or set()):
            raise ValueError(f"new discovery {actor_id!r} is source-ambiguous and requires an audit decision")
        if canonical != "new" and actor_id != canonical:
            raise ValueError(f"existing discovery changed canonical ID: {actor_id}")
        if not isinstance(aliases, list) or not all(isinstance(alias, str) and alias.strip() for alias in aliases):
            raise ValueError("discovery aliases are invalid")
        if (
            not isinstance(evidence, list)
            or not evidence
            or not all(isinstance(quote, str) and quote in source_text for quote in evidence)
        ):
            raise ValueError(f"discovery has non-source evidence for {actor_id}")
        found.append({key: item[key].strip() if isinstance(item[key], str) else item[key] for key in item})
    return found


# ##################################################################
# append fact
# retains source-derived later facts outside immutable existing profile records.
def append_fact(target: dict, field: str, text: str) -> None:
    if text and text not in target[field]:
        target[field].append(text)


# ##################################################################
# apply discoveries
# updates only the resumable preparation registry; original profiles are not enriched or rewritten.
def apply_discoveries(registry: dict, aliases: dict, discoveries: list[dict]) -> None:
    for item in discoveries:
        actor_id = item["id"] if item["canonical_id"] == "new" else item["canonical_id"]
        if actor_id not in registry:
            registry[actor_id] = {
                "name": item["name"],
                "bio": "",
                "look": "",
                "origin": "prepared",
                "facts": {"voice": [], "look": []},
            }
        facts = registry[actor_id].setdefault("facts", {"voice": [], "look": []})
        append_fact(facts, "voice", item["voice_facts"])
        append_fact(facts, "look", item["look_facts"])
        for alias in [actor_id, item["name"], *item["aliases"]]:
            key = normalized_id(alias)
            if key:
                prior = aliases.get(key)
                if prior is not None and prior != actor_id:
                    raise RuntimeError(
                        f"source-backed alias is ambiguous: {alias!r} maps to both {prior} and {actor_id}"
                    )
                aliases[key] = actor_id


# ##################################################################
# established identity preference
# protects original anchors first, then the earliest existing portrait/voice asset, so a later duplicate name never replaces a live face or timbre.
def preferred_established_identity(project: Path, first: str, second: str) -> str:
    if first in ANCHOR_IDS:
        return first
    if second in ANCHOR_IDS:
        return second

    def origin(identity: str) -> tuple[float, float]:
        portrait = project / "refs" / f"{identity}.png"
        voice = project / "voices" / f"{identity}.wav"
        portrait_time = portrait.stat().st_mtime if portrait.is_file() else float("inf")
        voice_time = voice.stat().st_mtime if voice.is_file() else float("inf")
        return portrait_time, voice_time

    first_origin, second_origin = origin(first), origin(second)
    if first_origin == (float("inf"), float("inf")):
        return second
    if second_origin == (float("inf"), float("inf")):
        return first
    return first if first_origin <= second_origin else second


# ##################################################################
# alias audit
# permits only externally audited source-backed legacy mappings, leaving every legacy file and asset in place.
def apply_alias_audit(project: Path, source_text: str, registry: dict, aliases: dict) -> tuple[set[str], set[str]]:
    path = project / AUDIT_NAME
    if not path.exists():
        return set(), set()
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("alias audit is unreadable") from error
    records = report if isinstance(report, list) else report.get("records") if isinstance(report, dict) else None
    if not isinstance(records, list):
        raise TypeError("alias audit requires a records array")
    merges: dict[str, str] = {}
    ambiguous: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"alias", "canonical", "evidence", "decision"}:
            raise RuntimeError("alias audit record has an invalid schema")
        alias = record["alias"]
        canonical = record["canonical"]
        evidence = record["evidence"]
        decision = record["decision"]
        if not all(isinstance(value, str) for value in (alias, canonical, decision)) or not isinstance(evidence, list):
            raise RuntimeError("alias audit record has invalid values")
        if (
            canonical not in registry
            or not evidence
            or not all(isinstance(quote, str) and quote in source_text for quote in evidence)
        ):
            raise RuntimeError("alias audit lacks source-grounded canonical evidence")
        if decision == "distinct":
            continue
        if decision not in {"merge", "ambiguous"}:
            raise RuntimeError("alias audit decision must be merge, distinct, or ambiguous")
        if decision == "ambiguous":
            ambiguous.add(normalized_id(alias))
            continue
        if decision == "merge":
            alias_id = normalized_id(alias)
            canonical_id = normalized_id(canonical)
            preferred = preferred_established_identity(project, alias_id, canonical_id)
            merge_alias = canonical_id if preferred == alias_id else alias_id
            merge_target = alias_id if preferred == alias_id else canonical_id
            prior = merges.get(merge_alias)
            if prior is not None and prior != merge_target:
                raise RuntimeError(f"alias audit conflicts with prior alias: {alias}")
            merges[merge_alias] = merge_target

    def resolved(actor_id: str, trail: set[str] | None = None) -> str:
        trail = trail or set()
        if actor_id in trail:
            raise RuntimeError(f"alias audit contains a cycle at {actor_id}")
        target = merges.get(actor_id)
        if target is None:
            return actor_id
        return resolved(target, {*trail, actor_id})

    inactive: set[str] = set()
    for alias_id in merges:
        canonical = resolved(alias_id)
        if canonical not in registry:
            raise RuntimeError(f"alias audit resolves outside registry: {alias_id} -> {canonical}")
        aliases[alias_id] = canonical
        if alias_id != canonical:
            inactive.add(alias_id)
    # Flatten every historical alias too: gene->jean and jean->tiger_boy must
    # never leak one hop into a future script or create a third voice identity.
    for alias_id, canonical in list(aliases.items()):
        final = resolved(canonical)
        aliases[alias_id] = final
        if alias_id in registry and alias_id != final:
            inactive.add(alias_id)
    return inactive, ambiguous


# ##################################################################
# source profile
# turns preparation-only facts into a profile for a genuinely new actor without touching an established actor profile.
def prepared_profile(entry: dict) -> dict:
    facts = entry.get("facts", {})
    bio = " ".join(facts.get("voice", [])) or "Source does not state distinctive voice details."
    look = " ".join(facts.get("look", [])) or "no visual details given"
    return {"name": entry["name"], "bio": bio, "look": look}


# ##################################################################
# materialize profiles
# append only missing prepared actors, then make their Breeze descriptions, clips, appearances, and portrait references once before freeze.
def materialize_profiles(project: Path, registry: dict) -> dict:
    characters_path = project / "characters.json"
    characters = load_object(characters_path, "characters profile")
    added = False
    for actor_id, entry in registry.items():
        if actor_id not in characters:
            characters[actor_id] = prepared_profile(entry)
            added = True
    if added:
        atomic_json(characters_path, characters)

    voices_path = project / "voices.json"
    voices = load_object(voices_path, "voice profiles")
    missing = [(actor_id, characters[actor_id]) for actor_id in registry if actor_id not in voices]
    if missing:

        async def describe_and_checkpoint(batch: list[tuple[str, dict]]) -> None:
            tasks = [
                asyncio.create_task(_voice_description_for_one(actor_id, profile))
                for actor_id, profile in batch
            ]
            for completed in asyncio.as_completed(tasks):
                actor_id, description = await completed
                # Each completed real model response is durable before waiting
                # for another, so a restart never repeats an already-described actor.
                voices[actor_id] = description
                atomic_json(voices_path, voices)

        for start in range(0, len(missing), VOICE_PROFILE_BATCH_SIZE):
            asyncio.run(describe_and_checkpoint(missing[start : start + VOICE_PROFILE_BATCH_SIZE]))
    prepare_breeze_voices(project)

    from src.hour_runner import extend_appearances

    extend_appearances(project, {actor_id: characters[actor_id] for actor_id in registry})
    missing_refs = [
        actor_id
        for actor_id in registry
        if actor_id != "narrator" and not (project / "refs" / f"{actor_id}.png").is_file()
    ]
    if missing_refs:
        generate_missing_character_refs(project, missing_refs)
    return characters


# ##################################################################
# asset hashes
# records exactly the actor profiles, Breeze WAV references, and portrait references whose immutability later hours require.
def asset_hashes(project: Path, actor_ids: set[str]) -> dict:
    characters = load_object(project / "characters.json", "characters profile")
    voices = load_object(project / "voices.json", "voice profiles")
    breeze = load_object(project / "breeze_voices.json", "Breeze voices")
    appearances = load_object(project / "appearances.json", "appearances")
    profiles: dict[str, dict] = {}
    for actor_id in sorted(actor_ids):
        if actor_id not in characters or actor_id not in voices or actor_id not in breeze:
            raise RuntimeError(f"frozen actor lacks profile or Breeze voice: {actor_id}")
        if actor_id != "narrator" and (
            actor_id not in appearances or not (project / "refs" / f"{actor_id}.png").is_file()
        ):
            raise RuntimeError(f"frozen actor lacks appearance or portrait: {actor_id}")
        wav = project / str(breeze[actor_id].get("ref_wav", ""))
        if not wav.is_file():
            raise RuntimeError(f"frozen actor lacks reference WAV: {actor_id}")
        profiles[actor_id] = {
            "character_profile": json_digest(characters[actor_id]),
            "voice_profile": json_digest(voices[actor_id]),
            "breeze_profile": json_digest(breeze[actor_id]),
            "voice_wav": file_digest(wav),
            "appearance": json_digest(appearances[actor_id]) if actor_id != "narrator" else None,
            "portrait": file_digest(project / "refs" / f"{actor_id}.png") if actor_id != "narrator" else None,
        }
    return profiles


# ##################################################################
# verify freeze
# validates source binding, complete chapter coverage, approved alias targets, and every frozen profile byte before any later hour can start.
def verify_frozen_cast(source: Path, project: Path | None = None) -> dict:
    source = source.resolve()
    project = project or get_output_dir(source)
    manifest = load_object(project / MANIFEST_NAME, "frozen cast manifest")
    if manifest.get("version") != 1 or manifest.get("source_sha256") != source_fingerprint(source):
        raise RuntimeError("frozen cast manifest does not bind this exact source")
    _, _, chapters = source_chapters(source, project)
    expected = {path.name: file_digest(path) for path in chapters}
    if manifest.get("chapter_sha256") != expected:
        raise RuntimeError("frozen cast manifest lacks complete exact source chapter coverage")
    actors = manifest.get("actors")
    aliases = manifest.get("approved_aliases")
    if not isinstance(actors, dict) or not actors or not isinstance(aliases, dict):
        raise RuntimeError("frozen cast manifest has no approved registry")
    if not ANCHOR_IDS <= set(actors):
        raise RuntimeError("frozen cast manifest is missing original anchor identities")
    if any(not isinstance(value, str) or value not in actors for value in aliases.values()):
        raise RuntimeError("frozen cast manifest has alias outside approved registry")
    actual = asset_hashes(project, set(actors))
    if actual != manifest.get("asset_hashes"):
        raise RuntimeError("frozen cast profile, voice, WAV, appearance, or portrait bytes changed")
    return manifest


# ##################################################################
# approved cast
# yields only frozen canonical actors for new production, never legacy aliases or mutable cache discoveries.
def load_approved_cast(source: Path, project: Path | None = None) -> dict:
    project = project or get_output_dir(source)
    manifest = verify_frozen_cast(source, project)
    actors = manifest["actors"]
    return {
        actor_id: {"name": value["name"], "bio": value["bio"], "look": value["look"]}
        for actor_id, value in actors.items()
    }


# ##################################################################
# validate preparation coverage
# preserves every existing resumable buffer while rejecting duplicate, skipped, or source-drifted completed batches before a new request can append.
def validate_preparation_coverage(progress: dict, chapters: list[Path]) -> None:
    expected_start = 0
    seen: set[str] = set()
    for batch in progress.get("completed_batches", []):
        if not isinstance(batch, dict) or set(batch) != {"start", "end", "chapter_sha256"}:
            raise RuntimeError("cast preparation has an invalid completed batch record")
        start, end, hashes = batch["start"], batch["end"], batch["chapter_sha256"]
        if not isinstance(start, int) or not isinstance(end, int) or start != expected_start or end <= start or end > len(chapters):
            raise RuntimeError("cast preparation completed batches do not have unique consecutive chapter coverage")
        expected_hashes = {path.name: file_digest(path) for path in chapters[start:end]}
        if not isinstance(hashes, dict) or hashes != expected_hashes or seen.intersection(hashes):
            raise RuntimeError("cast preparation completed batches lack unique exact chapter hashes")
        seen.update(hashes)
        expected_start = end
    if progress.get("next_chapter") != expected_start:
        raise RuntimeError("cast preparation cursor does not match completed exact chapter coverage")


# ##################################################################
# prepare cast
# resumes each bounded native-Ollama batch from durable progress and atomically publishes a freeze only after all chapters and assets validate.
def prepare_cast(source: Path, verify_only: bool = False, max_batches: int | None = None) -> dict:
    source = source.resolve()
    if not source.is_file():
        raise ValueError(f"input source is not a file: {source}")
    if LLM_STYLE != "ollama" and not verify_only:
        raise RuntimeError("prepare-cast requires schema-constrained native Ollama configured in local/config.toml")
    project = get_output_dir(source)
    title, author, chapters = source_chapters(source, project)
    del title, author
    if (project / MANIFEST_NAME).exists():
        manifest = verify_frozen_cast(source, project)
        return {
            "status": "frozen",
            "chapters": len(chapters),
            "actors": len(manifest["actors"]),
            "manifest": str(project / MANIFEST_NAME),
        }
    if verify_only:
        raise RuntimeError("no frozen cast manifest exists")
    source_sha = source_fingerprint(source)
    source_text = source.read_text(encoding="utf-8")
    progress_path = project / PROGRESS_NAME
    if progress_path.exists():
        progress = load_object(progress_path, "cast preparation progress")
        if progress.get("source_sha256") != source_sha or progress.get("chapter_count") != len(chapters):
            raise RuntimeError("cast preparation progress belongs to a different source")
        if progress.get("version") not in {1, 2}:
            raise RuntimeError("cast preparation progress has an unsupported version")
        if progress.get("version") == 1:
            # Version 2 changes only future discovery transport to immutable IDs;
            # every prior registry, cursor, and completed batch stays intact.
            progress["version"] = 2
            atomic_json(progress_path, progress)
    else:
        base = load_object(project / "characters.json", "characters profile")
        registry = {
            actor_id: {
                "name": info.get("name", actor_id),
                "bio": info.get("bio", ""),
                "look": info.get("look", ""),
                "origin": "existing",
                "facts": {"voice": [], "look": []},
            }
            for actor_id, info in base.items()
        }
        if not ANCHOR_IDS <= set(registry):
            raise RuntimeError("existing project is missing original Part 1 canonical anchors")
        progress = {
            "version": 2,
            "source_sha256": source_sha,
            "chapter_count": len(chapters),
            "next_chapter": 0,
            "registry": registry,
            "aliases": {actor_id: actor_id for actor_id in registry},
            "completed_batches": [],
        }
        atomic_json(progress_path, progress)
    validate_preparation_coverage(progress, chapters)
    if not progress.get("audit_applied"):
        inactive, ambiguous = apply_alias_audit(project, source_text, progress["registry"], progress["aliases"])
        progress["registry"] = {
            actor_id: entry for actor_id, entry in progress["registry"].items() if actor_id not in inactive
        }
        progress["inactive_legacy_ids"] = sorted(inactive)
        progress["ambiguous_new_ids"] = sorted(ambiguous)
        progress["audit_applied"] = True
        atomic_json(progress_path, progress)
    inactive = set(progress.get("inactive_legacy_ids", []))
    ambiguous = set(progress.get("ambiguous_new_ids", []))
    batches = 0
    while int(progress["next_chapter"]) < len(chapters):
        start = int(progress["next_chapter"])
        batch, prompt = context_safe_batch(chapters, start, progress["registry"], progress["aliases"])
        batch_units = immutable_evidence_units(batch)
        batch_text = "\n".join(path.read_text(encoding="utf-8") for path in batch)
        response = ask_sync(
            prompt,
            max_tokens=3500,
            response_schema=discovery_schema(list(progress["registry"]), [unit["id"] for unit in batch_units]),
        )
        try:
            materialized, evidence_unit_ids = materialize_evidence_discovery(json.loads(response), batch_units)
            discoveries = validate_discovery(materialized, batch_text, set(progress["registry"]), ambiguous)
        except (ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"cast preparation batch {start}-{start + len(batch) - 1} rejected: {error}; response={response[:500]!r}"
            ) from error
        apply_discoveries(progress["registry"], progress["aliases"], discoveries)
        with (project / DISCOVERIES_NAME).open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "start_chapter": start,
                        "chapters": [path.name for path in batch],
                        "evidence_units": [
                            {"id": unit["id"], "chapter": unit["chapter"]} for unit in batch_units
                        ],
                        "evidence_unit_ids": evidence_unit_ids,
                        "discoveries": discoveries,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        progress["next_chapter"] = start + len(batch)
        progress["completed_batches"].append(
            {
                "start": start,
                "end": start + len(batch),
                "chapter_sha256": {path.name: file_digest(path) for path in batch},
            }
        )
        atomic_json(progress_path, progress)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            return {
                "status": "preparing",
                "chapters": len(chapters),
                "next_chapter": progress["next_chapter"],
                "actors": len(progress["registry"]),
            }
    # Reapply the audited transitive map against retained legacy profiles before
    # publication; no inactive identifier can escape into the approved registry.
    legacy_registry = {**load_object(project / "characters.json", "characters profile"), **progress["registry"]}
    final_inactive, final_ambiguous = apply_alias_audit(project, source_text, legacy_registry, progress["aliases"])
    if final_ambiguous != ambiguous:
        raise RuntimeError("alias audit ambiguity changed during cast preparation")
    inactive.update(final_inactive)
    active = {actor_id for actor_id in progress["registry"] if actor_id not in inactive}
    characters = materialize_profiles(project, {actor_id: progress["registry"][actor_id] for actor_id in active})
    actors = {
        actor_id: {
            "name": characters[actor_id].get("name", actor_id),
            "bio": characters[actor_id].get("bio", ""),
            "look": characters[actor_id].get("look", ""),
        }
        for actor_id in active
    }
    aliases = {alias: canonical for alias, canonical in progress["aliases"].items() if canonical in actors}
    alias_payload = {"version": 1, "approved": aliases, "inactive_legacy_ids": sorted(inactive)}
    atomic_json(project / ALIASES_NAME, alias_payload)
    manifest = {
        "version": 1,
        "source_sha256": source_sha,
        "chapter_sha256": {path.name: file_digest(path) for path in chapters},
        "actors": actors,
        "approved_aliases": aliases,
        "inactive_legacy_ids": sorted(inactive),
        "asset_hashes": asset_hashes(project, set(actors)),
    }
    atomic_json(project / MANIFEST_NAME, manifest)
    verify_frozen_cast(source, project)
    return {
        "status": "frozen",
        "chapters": len(chapters),
        "actors": len(actors),
        "manifest": str(project / MANIFEST_NAME),
    }
