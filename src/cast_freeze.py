"""Resumable, source-grounded full-book cast preparation and freeze verification."""

from __future__ import annotations

import asyncio
import collections
import copy
import difflib
import hashlib
import json
import os
import re
from pathlib import Path

from src.breeze_voices import prepare_breeze_voices
from src.data_recovery import (
    REVALIDATION_ATTEMPT_LIMIT,
    DataIssue,
    OperationalError,
    RecoveryLedger,
    bounded,
    payload_hash,
    row_key,
    row_scope,
    safe_evidence,
)
from src.epub_extract import get_output_dir
from src.hour_runner import atomic_json, source_chapters, source_fingerprint
from src.hourly_spans import immutable_spans
from src.llm import LLM_STYLE, ask_sync, telemetry_scope
from src.movie_images import generate_missing_character_refs
from src.voice_description import _voice_description_for_one

MANIFEST_NAME = "frozen_cast_manifest.json"
PROGRESS_NAME = "cast_preparation_progress.json"
ALIASES_NAME = "frozen_character_aliases.json"
AUDIT_NAME = "cast_alias_audit.json"
DISCOVERIES_NAME = "cast_preparation_discoveries.jsonl"
REJECTIONS_NAME = "cast_preparation_rejections.jsonl"
BATCH_CHAPTERS = 6
MAX_BATCH_CHARACTERS = 24
EVIDENCE_REPAIR_ATTEMPTS = 2
SEMANTIC_COVERAGE_VERSION = 1
SEMANTIC_OPTIONAL_KEYS = frozenset({"quarantined", "empty_chapters", "revalidated"})
QUARANTINE_REVALIDATION_LIMIT = 25
CLASSIFICATION_CHUNK_SIZE = 16
GENERIC_PRONOUN_ALIASES = frozenset(
    {
        "i",
        "me",
        "my",
        "mine",
        "we",
        "us",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
    }
)
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
NON_NAME_COMPOUND_PREFIXES = frozenset(
    {
        "A",
        "An",
        "The",
        "All",
        "Each",
        "Every",
        "Some",
        "Any",
        "No",
        "Not",
        "Only",
        "As",
        "If",
        "When",
        "While",
        "After",
        "Before",
        "Because",
        "Although",
        "Though",
        "Since",
        "Unless",
        "And",
        "But",
        "Or",
        "Nor",
        "So",
        "Yet",
        "Then",
        "Also",
        "However",
        "Therefore",
        "In",
        "On",
        "At",
        "By",
        "From",
        "With",
        "Without",
        "For",
        "To",
        "Of",
        "Into",
        "Out",
        "Up",
        "Down",
        "Over",
        "Under",
        "Around",
        "Through",
        "Across",
        "During",
        "Beyond",
        "Within",
        "Against",
        "Between",
        "Among",
        "About",
    }
)
NON_ENTITY_LABELS = frozenset(
    {
        "Someone",
        "Anyone",
        "Everyone",
        "Nobody",
        "Nothing",
        "Something",
        "He",
        "She",
        "Him",
        "Her",
        "His",
        "Hers",
        "They",
        "Them",
        "Their",
        "Theirs",
        "It",
        "Its",
        "We",
        "Us",
        "Our",
        "Ours",
        "I",
        "Me",
        "My",
        "Mine",
        "You",
        "Your",
        "Yours",
    }
)
TITLE_WORDS = frozenset({"Professor", "Master", "Doctor", "Captain", "Commander"})
NARRATIVE_ATTRIBUTION_VERBS = frozenset(
    {
        "added",
        "announced",
        "asked",
        "called",
        "continued",
        "intervened",
        "murmured",
        "ordered",
        "replied",
        "said",
        "shouted",
        "spoke",
        "whispered",
    }
)


class CastDataIssue(DataIssue):
    """Recoverable cast-evidence problem (model or source content); quarantined per batch/chapter, never fatal."""

    def __init__(self, message: str, evidence: dict | None = None, code: str = "cast_evidence_rejected"):
        super().__init__(code, message, evidence)


class CastValidationError(CastDataIssue, ValueError):
    """Content or model-response validation failure at a targeted parse boundary (a DataIssue that is also a ValueError)."""


def model_object(raw: object, what: str, required: tuple[str, ...] = ()) -> dict:
    """Parse one native model reply into an object with the required keys; any shape problem is a typed DataIssue."""
    try:
        value = json.loads(raw)
    except (ValueError, TypeError, RecursionError) as error:
        raise CastValidationError(f"{what} is not valid JSON", {"reason": bounded(str(error))}) from error
    if not isinstance(value, dict) or not set(required) <= set(value):
        raise CastValidationError(f"{what} is not an object with the required fields")
    return value


class UnicodeNameWord:
    """Unicode-aware equivalent of ^[A-Z][a-z]*(?:-[A-Za-z]+)?$ so non-ASCII names are scanned, never silently dropped."""

    def fullmatch(self, value: str) -> bool:
        head, sep, tail = value.partition("-")
        if (
            not head
            or not head[0].isupper()
            or not head[0].isalpha()
            or any(not (c.isalpha() and c.islower()) for c in head[1:])
        ):
            return False
        return not sep or (tail != "" and tail.isalpha())


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
        raise OperationalError("store_missing", f"required {label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise OperationalError("store_unreadable", f"required {label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise OperationalError("store_invalid", f"required {label} is not an object: {path}")
    return value


# ##################################################################
# immutable evidence units
# assigns each source sentence a stable batch-local identifier so a model selects evidence without ever reproducing source prose.
def undecodable_issue(chapter: Path, error: UnicodeDecodeError) -> CastDataIssue:
    """Typed issue carrying the exact byte-level evidence of an invalid-UTF-8 chapter."""
    raw = chapter.read_bytes()
    return CastDataIssue(
        f"source chapter {chapter.name} is not valid UTF-8",
        {
            "chapter": chapter.name,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_length": len(raw),
            "error_start": error.start,
            "error_end": error.end,
            "invalid_bytes_hex": raw[error.start : error.end].hex(),
            "reason": bounded(str(error)),
        },
        "source_not_utf8",
    )


def empty_chapter_attestations(chapters: list[Path]) -> dict[str, str]:
    """Exact raw-byte hash of every valid-UTF-8 chapter that has no immutable span (zero-coverage, non-character)."""
    attested: dict[str, str] = {}
    for chapter in chapters:
        try:
            immutable_spans(chapter.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        except ValueError:
            attested[chapter.name] = file_digest(chapter)
    return attested


def immutable_evidence_units(chapters: list[Path]) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for chapter_index, chapter in enumerate(chapters):
        try:
            chapter_text = chapter.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise undecodable_issue(chapter, error) from error
        chapter_hash = hashlib.sha256(chapter_text.encode("utf-8")).hexdigest()
        try:
            spans = immutable_spans(chapter_text)
        except ValueError:
            # No characters to find: zero units; empty_chapter_attestations records the exact hash.
            continue
        for sentence_index, quote in enumerate(spans):
            units.append(
                {
                    "id": f"c{chapter_index:02d}s{sentence_index:05d}",
                    "chapter": chapter.name,
                    "chapter_sha256": chapter_hash,
                    "quote": quote,
                }
            )
    return units


# ##################################################################
# immutable name references
# enumerates only exact contiguous capitalized lexical spans; source text is never copied or offset-calculated by a model.
def immutable_name_references(units: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    references: dict[str, dict[str, str]] = {}
    word = re.compile(r"[^\W\d_](?:[^\W\d_]|['-])*")
    name_word = UnicodeNameWord()
    for unit in units:
        tokens = list(word.finditer(unit["quote"]))
        index = 0
        for start, token in enumerate(tokens):
            if not name_word.fullmatch(token.group().removesuffix("'s")) or token.group() in NON_ENTITY_LABELS:
                continue
            end = start
            while end < len(tokens) and end < start + 4:
                # A title followed by a capitalized narrative subject and then an
                # attribution verb is a malformed punctuation boundary, not a full name
                # (e.g. "Professor Taro intervened").  Retain the title and subject as
                # independent source references for semantic classification.
                if (
                    end == start + 1
                    and token.group() in TITLE_WORDS
                    and end + 1 < len(tokens)
                    and tokens[end + 1].group().casefold() in NARRATIVE_ATTRIBUTION_VERBS
                ):
                    break
                current = tokens[end]
                possessive = current.group().endswith("'s")
                current_label = current.group()[:-2] if possessive else current.group()
                if (
                    end > start and unit["quote"][tokens[end - 1].end() : current.start()].strip()
                ) or not name_word.fullmatch(current_label):
                    break
                label_end = current.end() - 2 if possessive else current.end()
                label = unit["quote"][token.start() : label_end]
                if end > start and token.group() in NON_NAME_COMPOUND_PREFIXES:
                    break
                references[f"{unit['id']}n{index:03d}"] = {
                    "unit_id": unit["id"],
                    "label": label,
                    "start": token.start(),
                    "suffix": start > 0
                    and not unit["quote"][tokens[start - 1].end() : token.start()].strip()
                    and bool(name_word.fullmatch(tokens[start - 1].group()))
                    and tokens[start - 1].group() not in NON_NAME_COMPOUND_PREFIXES,
                }
                index += 1
                if possessive:
                    break
                end += 1
    return references


# ##################################################################
# mention-scoped source audit
# validates generic audit records bound to one exact mention (actual chapter content hash, quote hash, label, span offset) and indexes them by that scope.
SCOPED_AUDIT_FIELDS = ("chapter_sha256", "quote_sha256", "label", "canonical", "decision", "confidence", "reason")


def mention_scope(unit: dict, reference: dict, label: str | None = None) -> tuple[str, str, str, int]:
    return (unit["chapter_sha256"], text_digest(unit["quote"]), label or reference["label"], reference["start"])


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mention_scoped_audit_index(records: object) -> dict[tuple[str, str, str, int], dict]:
    if not isinstance(records, list):
        raise TypeError("mention-scoped audit records must be a list")
    index: dict[tuple[str, str, str, int], dict] = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or any(
                not isinstance(record.get(field), str) or not record[field]
                for field in SCOPED_AUDIT_FIELDS
                if field != "confidence"
            )
            or not isinstance(record.get("confidence"), (int, float))
            or type(record.get("span_start")) is not int
            or record["span_start"] < 0
        ):
            raise ValueError("mention-scoped audit record is incomplete")
        if "owners" in record and (
            not isinstance(record["owners"], list) or not all(isinstance(owner, str) for owner in record["owners"])
        ):
            raise ValueError("mention-scoped audit record has invalid offered owners")
        if "history" in record and (
            not isinstance(record["history"], list)
            or not all(
                isinstance(item, dict) and isinstance(item.get("decision"), str) and isinstance(item.get("reason"), str)
                for item in record["history"]
            )
        ):
            raise ValueError("mention-scoped audit record has invalid history")
        scope = (record["chapter_sha256"], record["quote_sha256"], record["label"], record["span_start"])
        if scope in index and index[scope] != record:
            raise ValueError("mention-scoped audit has conflicting records for one mention")
        index[scope] = record
    return index


NON_NAME_COMPOUND_WORDS = frozenset(word.casefold() for word in NON_NAME_COMPOUND_PREFIXES)
ADJUDICATION_SNIPPET_CHARS = 80


def label_components(text: str) -> set[str]:
    """Normalized name words of a label, without articles/prepositions that are not part of a name."""
    return {
        normalized_id(word)
        for word in re.split(r"[\s_]+", text)
        if len(word) > 1 and word.casefold() not in NON_NAME_COMPOUND_WORDS
    }


def owner_relevance(label: str, actor_id: str, entry: dict, aliases: dict[str, str]) -> int:
    """Rank one established actor only from independently inspectable identity evidence; zero means no owner option exists."""
    label_id = normalized_id(label)
    name = str(entry.get("name", actor_id))
    aliases_for_owner = [alias for alias, target in aliases.items() if target == actor_id]
    literal_ids = {normalized_id(actor_id), normalized_id(name), *(normalized_id(alias) for alias in aliases_for_owner)}
    if label_id and label_id in literal_ids:
        return 4
    label_words = label_components(label)
    owner_words = label_components(name) | set(actor_id.split("_"))
    owner_words.update(word for alias in aliases_for_owner for word in label_components(alias))
    if label_words & owner_words:
        return 3
    pattern = re.compile(rf"(?<!\w){re.escape(label.strip())}(?!\w)", re.IGNORECASE)
    if label.strip() and any(pattern.search(text) for _, text in extract_owner_profile_facts(entry)):
        return 2
    return 0


# ##################################################################
# adjudication owners
# ranks every established actor with literal canonical/alias, name-component, or exact profile/source-fact evidence; a model proposal and roster position are never evidence, so no-relevance produces no owners and preserves an open-world pending outcome.
def adjudication_owners(
    label: str, registry: dict, aliases: dict | None = None, proposed: str | None = None
) -> list[str]:
    del proposed
    safe_aliases = aliases or {}
    ranked = [
        (owner_relevance(label, actor_id, entry, safe_aliases), actor_id)
        for actor_id, entry in registry.items()
        if actor_id != "narrator" and isinstance(entry, dict)
    ]
    return [actor_id for score, actor_id in sorted(ranked, key=lambda pair: (-pair[0], pair[1])) if score]


def readjudication_due(record: dict, owners: list[str]) -> bool:
    """A cached ambiguous decision is stale when it never saw (or was not offered) every owner now plausible for the label."""
    if record["decision"] != "ambiguous" or not owners:
        return False
    offered = record.get("owners")
    return not (isinstance(offered, list) and set(owners) <= set(offered))


def scope_final(candidate: dict) -> bool:
    """True when the candidate's cached scoped decision is binding rather than awaiting re-adjudication."""
    return bool(candidate.get("scoped_audit")) and not candidate.get("scoped_stale")


def scoped_alias_approved(candidate: dict, canonical: str) -> bool:
    scoped = candidate.get("scoped_audit")
    return (
        bool(scoped)
        and scoped["decision"] == "alias"
        and scoped["canonical"] == canonical
        and not candidate.get("scoped_unproven")
    )


# ##################################################################
# candidate coverage ledger
# groups exact lexical labels while retaining immutable witness IDs, so every possible named span receives a durable explicit decision.
def candidate_coverage_ledger(
    units: list[dict[str, str]],
    registry: dict[str, dict],
    aliases: dict[str, str],
    scoped_audit: list[dict] | None = None,
) -> list[dict]:
    scoped = mention_scoped_audit_index(scoped_audit or [])
    grouped: dict[str, dict] = {}
    units_by_id = {unit["id"]: unit for unit in units}
    order = [unit["id"] for unit in units]
    references = immutable_name_references(units)
    for ref_id, reference in references.items():
        label = reference["label"]
        key = normalized_id(label)
        if not key:
            continue
        unit = units_by_id[reference["unit_id"]]
        record = scoped.get(mention_scope(unit, reference)) if scoped else None
        if record:
            key = f"{key}@{record['decision']}:{record['canonical']}"
        candidate = grouped.setdefault(
            key,
            {
                "label": label,
                "ref_ids": [],
                "has_standalone": False,
                "scoped": record,
                "records": [],
                "unproven": False,
            },
        )
        candidate["ref_ids"].append(ref_id)
        if record:
            candidate["records"].append(record)
            if (
                record["decision"] == "alias"
                and (
                    scoped_alias_proof(
                        label,
                        record["canonical"],
                        unit,
                        scene_units_at(units_by_id, order, order.index(unit["id"])),
                        registry,
                        aliases,
                    )[1]
                )
            ):
                # a cached alias is only as good as the mechanical proof of its exact mention; an unproven one is re-decided, never trusted
                candidate["unproven"] = True
        candidate["has_standalone"] = candidate["has_standalone"] or not reference["suffix"]
    for candidate in grouped.values():
        label_id = normalized_id(candidate["label"])
        direct = aliases.get(label_id)
        names = [
            actor_id
            for actor_id, entry in registry.items()
            if label_id in {normalized_id(actor_id), normalized_id(str(entry.get("name", actor_id)))}
        ]
        candidate["known_owner"] = direct or (names[0] if len(names) == 1 else None)
        candidate["nonentity"] = False
        candidate["stale"] = False
        if candidate["scoped"]:
            ledger_owners = adjudication_owners(candidate["label"], registry, aliases)
            candidate["stale"] = any(readjudication_due(item, ledger_owners) for item in candidate["records"])
            candidate["stale"] = candidate["stale"] or candidate["unproven"]
            candidate["known_owner"] = (
                candidate["scoped"]["canonical"]
                if candidate["scoped"]["decision"] == "alias" and not candidate["unproven"]
                else None
            )
            if candidate["scoped"]["decision"] != "alias":
                candidate["nonentity"] = candidate["scoped"]["decision"] == "non_character"
        for ref_id in candidate["ref_ids"]:
            unit = units_by_id[references[ref_id]["unit_id"]]
            label = re.escape(candidate["label"])
            if re.search(
                rf"(?i)\b(?:impact of|attack of) {label}(?:\s+and\s+[A-Z][a-z]+)?\b", unit["quote"]
            ) or re.search(rf"(?i)\b{label}\s+(?:air|team|group|clan|family|house)\b", unit["quote"]):
                candidate["nonentity"] = True
                break
    lowercase_source = "\n".join(unit["quote"] for unit in units)

    def has_lowercase_occurrence(label: str) -> bool:
        return bool(re.search(rf"(?<!\w){re.escape(label.casefold())}(?!\w)", lowercase_source))

    qualified_components = {
        normalized_id(word)
        for value in grouped.values()
        if " " in value["label"] and any(not has_lowercase_occurrence(word) for word in value["label"].split())
        for word in value["label"].split()
    }
    retained = [
        (key, value)
        for key, value in grouped.items()
        if (" " in value["label"] or value["has_standalone"])
        and (
            " " in value["label"]
            or normalized_id(value["label"]) in qualified_components
            or not has_lowercase_occurrence(value["label"])
        )
    ]
    retained.sort(key=lambda item: (-len(item[1]["label"].split()), item[0]))
    return [
        {
            "id": f"p{index:04d}",
            "label": value["label"],
            "ref_ids": value["ref_ids"],
            "known_owner": value["known_owner"],
            "nonentity": value["nonentity"],
            **({"scoped_audit": value["scoped"]} if value["scoped"] else {}),
            **({"scoped_stale": True} if value["stale"] else {}),
            **({"scoped_unproven": True} if value["unproven"] else {}),
        }
        for index, (_, value) in enumerate(retained)
    ]


# ##################################################################
# classification schema
# forces one bounded native decision for every local candidate; all labels, IDs and witness bytes remain program-derived.
def discovery_schema(
    known_ids: list[str], candidates: list[dict], identity_candidates: list[dict] | None = None, allow_new: bool = True
) -> dict:
    if not candidates:
        raise CastValidationError("classification schema requires lexical candidates")
    # candidates with an established owner are never legal identity targets: their canonical ID is already legal
    identity_ids = [
        candidate["id"] for candidate in (identity_candidates or candidates) if not candidate.get("known_owner")
    ]
    unit_ids = sorted({ref_id.rsplit("n", 1)[0] for candidate in candidates for ref_id in candidate["ref_ids"]})

    def record_schema(candidate: dict) -> dict:
        evidence = {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
            "items": {"type": "string", "enum": unit_ids},
        }
        fixed_owner = candidate.get("known_owner")
        scoped = candidate.get("scoped_audit")
        audited_target = candidate.get("audited_target")
        base = {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string", "enum": [candidate["id"]]},
                "evidence_unit_ids": evidence,
            },
            "required": ["candidate_id", "status", "identity", "evidence_unit_ids"],
            "additionalProperties": False,
        }
        if scope_final(candidate) and scoped["decision"] != "alias":
            return {
                **base,
                "properties": {
                    **base["properties"],
                    "status": {"type": "string", "enum": [scoped["decision"]]},
                    "identity": {"type": "string", "enum": ["none"]},
                },
            }
        if fixed_owner:
            return {
                **base,
                "properties": {
                    **base["properties"],
                    "status": {"type": "string", "enum": ["known"]},
                    "identity": {"type": "string", "enum": [fixed_owner]},
                },
            }
        if audited_target:
            return {
                **base,
                "properties": {
                    **base["properties"],
                    "status": {"type": "string", "enum": ["known"]},
                    "identity": {"type": "string", "enum": [audited_target]},
                },
            }
        known_targets = sorted(set(known_ids + [identity for identity in identity_ids if identity != candidate["id"]]))
        branches = [
            {
                **base,
                "properties": {
                    **base["properties"],
                    "status": {"type": "string", "enum": ["known"]},
                    "identity": {"type": "string", "enum": known_targets},
                },
            },
            {
                **base,
                "properties": {
                    **base["properties"],
                    "status": {"type": "string", "enum": ["non_character", "ambiguous"]},
                    "identity": {"type": "string", "enum": ["none"]},
                },
            },
        ]
        if allow_new:
            branches.insert(
                0,
                {
                    **base,
                    "properties": {
                        **base["properties"],
                        "status": {"type": "string", "enum": ["new"]},
                        "identity": {"type": "string", "enum": [candidate["id"]]},
                    },
                },
            )
        return {"oneOf": branches}

    return {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "minItems": len(candidates),
                "maxItems": len(candidates),
                "uniqueItems": True,
                "items": {"oneOf": [record_schema(candidate) for candidate in candidates]},
            }
        },
        "required": ["classifications"],
        "additionalProperties": False,
    }


# ##################################################################
# classification prompt
# sends compact candidate-owned witnesses rather than an open-ended prose scan and requires exhaustive classifications in schema order.
def discovery_prompt(
    chapters: list[Path],
    registry: dict,
    aliases: dict[str, str],
    candidates: list[dict] | None = None,
    all_candidates: list[dict] | None = None,
    allow_new: bool = True,
    units: list[dict[str, str]] | None = None,
) -> str:
    units = units if units is not None else immutable_evidence_units(chapters)
    candidates = candidates or candidate_coverage_ledger(units, registry, aliases)
    all_candidates = all_candidates or candidates
    if not candidates:
        return "No lexical candidates exist; return the schema response."
    by_id = {unit["id"]: unit["quote"] for unit in units}
    roster = "; ".join(f"{actor_id}={entry.get('name', actor_id)}" for actor_id, entry in sorted(registry.items()))
    audited = "; ".join(f"{alias}->{target}" for alias, target in sorted(aliases.items()) if alias != target)
    rows = []
    for candidate in candidates:
        witness_ids = [candidate["ref_ids"][0].rsplit("n", 1)[0]]
        contexts = " | ".join(f"[{unit_id}] {by_id[unit_id]}" for unit_id in witness_ids)
        fixed = f" FIXED_KNOWN_OWNER={candidate['known_owner']}" if candidate.get("known_owner") else ""
        audited_target = (
            f" FIXED_AUDITED_TARGET={candidate['audited_target']}" if candidate.get("audited_target") else ""
        )
        rows.append(f"{candidate['id']} label={candidate['label']!r}{fixed}{audited_target} witnesses: {contexts}")
    global_ids = "; ".join(f"{candidate['id']}={candidate['label']!r}" for candidate in all_candidates)
    full_narrative = "\n".join(f"[{unit['id']}] {unit['quote']}" for unit in units)
    return (
        f"""Classify EVERY candidate exactly once using only the response schema. Candidate labels and source witnesses are immutable local evidence; never copy a name, quote, offset, or invented ID into JSON.

status=new means this candidate is a distinct named living person/creature and identity MUST equal its own candidate_id. A named weapon, equipment item, attack, skill, species, group, or action is non_character even when capitalized; require source behavior/description proving a living entity before new. When both a source-qualified full name and a shorter component occur, make the full name the new identity and map the shorter label only when source evidence proves it is that identity. status=known means it is the same identity as an approved canonical ID or another new candidate in the global ledger; identity MUST name that target. An alias of a new full-name owner MUST be status=known targeting that owner, never status=new with a different identity. status=non_character means the lexical capitalisation is not a person/creature. status=ambiguous means source evidence cannot safely decide; identity MUST be none. For known mappings select source units proving identity; co-occurrence in one sentence alone is NOT proof. Classify the exact label/span only: a neighboring capitalized name followed by a speech/action verb may be narrative attribution, not part of this label. For a title vocative, use direct response and bounded scene continuity to identify the addressed existing owner; never fabricate a title-plus-attribution full name. Do not merge spelling variants on similarity. A kinship/role label (Mom, Dad) or a partly matching full name may be proposed status=known to an approved canonical ID when its witness context supports that person; a separate per-mention validation then decides it, so prefer a contextual known proposal over ambiguous when the cast plausibly holds the owner. Bare Xiao is ambiguous unless a source witness identifies it. Indefinite sentence words Someone, Anyone, Everyone, Nobody, Nothing, and Something are non_character, never unresolved people. A bare surname or title fragment such as Crest is non_character unless it is an approved alias or its own selected witness explicitly identifies the same person; sharing a longer name is not identity proof. House, clan, family, place, group, team, species, and organization labels are non_character even when they mention or surround a known person; classify the exact label, never merge a house or clan into its member. Existing identities may only use their canonical name or a pre-approved alias below. A new identity may have zero aliases. Every evidence_unit_ids list must include a witness for its candidate. For an alias-to-new-identity link, include distinct witnesses for both spellings; a shared co-occurrence sentence alone is invalid.

Known canonical IDs: {roster or "(none)"}. Every ledger row carrying FIXED_KNOWN_OWNER MUST be status=known with exactly that identity; never create a new actor for it.
Approved aliases: {audited or "(none)"}
Global candidate identities (for cross-chunk links only): {global_ids}

CANDIDATE LEDGER:\n"""
        + "\n".join(rows)
        + "\n\nFULL BOUNDED SOURCE NARRATIVE (use it to resolve source-proven variant groups; never copy its text into JSON):\n"
        + full_narrative
    )


# ##################################################################
# approved known label
# accepts audited aliases and a unique literal component of an established full name only when the selected source witness contains that full name.
def approved_known_label(
    label: str, canonical: str, evidence_unit_ids: list[str], units: list[dict[str, str]], registry: dict, aliases: dict
) -> bool:
    label_id = normalized_id(label)
    entry = registry[canonical]
    full_name = str(entry.get("name", canonical))
    if aliases.get(label_id) == canonical or label_id in {normalized_id(canonical), normalized_id(full_name)}:
        return True
    if " " in label.strip() or not label_id:
        return False
    owners = [
        actor_id
        for actor_id, info in registry.items()
        if label_id in {normalized_id(word) for word in str(info.get("name", actor_id)).split()}
    ]
    selected = [unit for unit in units if unit["id"] in evidence_unit_ids]
    return owners == [canonical] and any(source_label_present(full_name, [unit]) for unit in selected)


# ##################################################################
# materialize classifications
# validates exhaustive schema transport and creates discoveries only from source-derived candidate labels and links.
def materialize_classifications(
    value: object, units: list[dict[str, str]], candidates: list[dict], registry: dict, aliases: dict
) -> tuple[list[dict], list[dict]]:
    if (
        not isinstance(value, dict)
        or set(value) != {"classifications"}
        or not isinstance(value["classifications"], list)
    ):
        raise CastValidationError("classification response is not the exact object schema")
    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    unit_ids = {unit["id"] for unit in units}
    classifications = value["classifications"]
    if len(classifications) != len(candidates):
        raise CastValidationError("classification response omitted or duplicated lexical candidates")
    seen: set[str] = set()
    validated: dict[str, dict] = {}
    for item in classifications:
        if not isinstance(item, dict) or set(item) != {"candidate_id", "status", "identity", "evidence_unit_ids"}:
            raise CastValidationError("classification record has invalid fields")
        candidate_id, status, identity, evidence = (
            item.get("candidate_id"),
            item.get("status"),
            item.get("identity"),
            item.get("evidence_unit_ids"),
        )
        if (
            candidate_id not in candidate_by_id
            or candidate_id in seen
            or status not in {"known", "new", "non_character", "ambiguous"}
            or not isinstance(identity, str)
            or not isinstance(evidence, list)
            or not evidence
            or len(set(evidence)) != len(evidence)
            or not set(evidence) <= unit_ids
        ):
            raise CastValidationError("classification record has invalid candidate or source evidence")
        candidate_units = {ref_id.rsplit("n", 1)[0] for ref_id in candidate_by_id[candidate_id]["ref_ids"]}
        if not candidate_units.intersection(evidence):
            raise CastValidationError(f"classification lacks a source witness for {candidate_id}")
        if status in {"non_character", "ambiguous"} and identity != "none":
            raise CastValidationError(f"non-identity classification has a target for {candidate_id}")
        if status == "new" and identity != candidate_id:
            raise CastValidationError(f"new classification must own its candidate ID: {candidate_id}")
        if status == "known" and identity not in registry and identity not in candidate_by_id:
            raise CastValidationError(f"known classification has unknown identity target: {identity}")
        seen.add(candidate_id)
        validated[candidate_id] = item
    if seen != set(candidate_by_id):
        raise CastValidationError("classification response did not cover the complete candidate ledger")
    discoveries: list[dict] = []
    for candidate in candidates:
        item = validated[candidate["id"]]
        if item["status"] != "new":
            continue
        label = candidate["label"]
        actor_id = normalized_id(label)
        if not IDENTIFIER.fullmatch(actor_id):
            raise CastDataIssue(
                f"candidate cannot form a canonical ID: {label!r}", {"label": bounded(label), "code": "non_ascii_id"}
            )
        if actor_id in registry or aliases.get(actor_id) not in {None, actor_id}:
            raise CastDataIssue(
                f"new candidate conflicts with established identity: {label!r}",
                {"label": bounded(label), "code": "id_collision"},
            )
        source_quotes = source_context_quotes(
            units, [*item["evidence_unit_ids"], *(ref_id.rsplit("n", 1)[0] for ref_id in candidate["ref_ids"])]
        )
        source_facts = " ".join(source_quotes)
        discoveries.append(
            {
                "canonical_id": "new",
                "id": actor_id,
                "name": label,
                "aliases": [],
                "voice_facts": source_facts,
                "look_facts": source_facts,
                "evidence": source_quotes,
            }
        )
    unresolved = [candidate for candidate in candidates if validated[candidate["id"]]["status"] == "ambiguous"]
    if unresolved:
        # Pending mention-scoped references are preserved verbatim as evidence; no identity is chosen or merged.
        raise CastDataIssue(
            f"semantic classification remains unresolved for {unresolved[0]['label']!r}",
            {
                "unresolved": [
                    {"label": c["label"], "candidate_id": c["id"], "ref_ids": list(c.get("ref_ids", []))}
                    for c in unresolved
                ]
            },
        )
    for candidate in candidates:
        item = validated[candidate["id"]]
        if item["status"] != "known":
            continue
        target = item["identity"]
        label_id = normalized_id(candidate["label"])
        if target in registry:
            if not scoped_alias_approved(candidate, target) and not approved_known_label(
                candidate["label"], target, item["evidence_unit_ids"], units, registry, aliases
            ):
                raise CastValidationError(f"known classification reassigns an unapproved alias: {candidate['label']!r}")
        else:
            owner = validated[target]
            if owner["status"] != "new":
                raise CastValidationError(f"candidate identity target is not a new identity: {target}")
            owner_label = candidate_by_id[target]["label"]
            owner_id = normalized_id(owner_label)
            if owner_id == label_id:
                continue
            candidate_units = {ref_id.rsplit("n", 1)[0] for ref_id in candidate["ref_ids"]}
            target_units = {ref_id.rsplit("n", 1)[0] for ref_id in candidate_by_id[target]["ref_ids"]}
            link_units = set(item["evidence_unit_ids"])
            full_name_component = normalized_id(candidate["label"]) in {
                normalized_id(word) for word in candidate_by_id[target]["label"].split()
            }
            if (
                not candidate.get("audited_target")
                and not full_name_component
                and (
                    not candidate_units.intersection(link_units)
                    or not target_units.intersection(link_units)
                    or candidate_units.intersection(target_units).intersection(link_units) == link_units
                )
            ):
                raise CastValidationError(
                    f"candidate link lacks distinct source identity evidence: {candidate['label']!r}"
                )
            alias_unit_ids = list(
                dict.fromkeys(
                    [*item["evidence_unit_ids"], *(ref_id.rsplit("n", 1)[0] for ref_id in candidate["ref_ids"])]
                )
            )
            alias_quotes = [
                next(unit["quote"] for unit in units if unit["id"] == evidence_id) for evidence_id in alias_unit_ids
            ]
            for discovery in discoveries:
                if discovery["id"] == owner_id:
                    discovery["aliases"].append(candidate["label"])
                    discovery["evidence"] = list(dict.fromkeys([*discovery["evidence"], *alias_quotes]))
                    merged_facts = " ".join(discovery["evidence"])
                    discovery["voice_facts"] = merged_facts
                    discovery["look_facts"] = merged_facts
                    break
    return discoveries, [validated[candidate["id"]] for candidate in candidates]


# ##################################################################
# exact decision memo
# Reuses only a syntactically valid response to byte-identical offered prompt/schema/options
# inside one discovery transaction. Every caller still performs its normal source-anchor and
# schema validation; malformed replies are deliberately never memoized.
def memoized_model_ask(ask):
    cache: dict[str, dict] = {}

    def cached(prompt: str, **kwargs) -> str:
        schema = kwargs.get("response_schema")
        key = json_digest({"prompt": prompt, "max_tokens": kwargs.get("max_tokens"), "schema": schema})
        prior = cache.get(key)
        if prior is not None:
            return prior["response"]
        response = ask(prompt, **kwargs)
        try:
            parsed = json.loads(response)
        except (TypeError, ValueError):
            return response
        if isinstance(parsed, dict):
            cache[key] = {
                "response": response,
                "request_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "schema_sha256": json_digest(schema),
            }
        return response

    return cached


# ##################################################################
# partition classification chunk
# Keeps a well-formed response's independently valid candidate rows when one row has unusable
# citation/semantic evidence. The rejected row becomes typed pending evidence with a
# program-derived own witness; malformed envelopes and unidentifiable rows still require the
# bounded whole-chunk repair path because there is no safe exact-row attribution.
def partition_classification_chunk(
    value: object,
    candidates: list[dict],
    registry: dict,
    aliases: dict,
    all_candidates: list[dict],
    units: list[dict],
    pending: list[dict],
) -> tuple[list[dict], list[dict]]:
    if (
        not isinstance(value, dict)
        or set(value) != {"classifications"}
        or not isinstance(value["classifications"], list)
    ):
        raise CastValidationError("classification chunk is not the exact object schema")
    by_id = {candidate["id"]: candidate for candidate in candidates}
    all_ids = {candidate["id"] for candidate in all_candidates}
    supplied: dict[str, list[dict]] = {candidate_id: [] for candidate_id in by_id}
    for record in value["classifications"]:
        # A row for another known chunk candidate is attributable cardinality drift: retain
        # the rows that belong here and hold the omitted expected row pending. An unknown ID
        # or an object with no ID remains an unusable envelope requiring bounded repair.
        if not isinstance(record, dict) or record.get("candidate_id") not in all_ids:
            raise CastValidationError("classification chunk has an unidentifiable record")
        if record["candidate_id"] in by_id:
            supplied[record["candidate_id"]].append(record)
    accepted: list[dict] = []
    rejected: list[dict] = []
    for candidate in candidates:
        rows = supplied[candidate["id"]]
        if len(rows) != 1:
            reason = "classification omitted this candidate" if not rows else "classification duplicated this candidate"
            rejected.append({"candidate": candidate, "reason": reason})
        else:
            local_pending: list[dict] = []
            try:
                accepted.extend(
                    validate_classification_chunk(
                        {"classifications": rows},
                        [candidate],
                        registry,
                        aliases,
                        all_candidates,
                        units,
                        local_pending,
                    )
                )
                pending.extend(local_pending)
                continue
            except CastValidationError as error:
                rejected.append({"candidate": candidate, "reason": str(error)})
        # This is not a model decision: it is a conservative transport of the candidate's
        # immutable first witness into the pending lane. It keeps ledger exact-once coverage
        # while preventing a bad row from invalidating unrelated rows in the same response.
        accepted.append(
            {
                "candidate_id": candidate["id"],
                "status": "ambiguous",
                "identity": "none",
                "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]],
            }
        )
    return accepted, rejected


# ##################################################################
# classification decision cache
# Maps model-produced records through immutable candidate scopes, never transient pNNNN IDs.
# An entry can only be reused during this batch when its complete offered identities/facts and
# immutable scene digest are identical, then it is fed back through normal validation.
def candidate_scope_fingerprint(candidate: dict) -> str:
    return json_digest(
        {
            "label": candidate["label"],
            "ref_ids": candidate["ref_ids"],
            "known_owner": candidate.get("known_owner"),
            "nonentity": candidate.get("nonentity", False),
            "audited_target": candidate.get("audited_target"),
            "scoped_audit": candidate.get("scoped_audit"),
            "scoped_stale": candidate.get("scoped_stale", False),
        }
    )


def classification_offer_fingerprint(
    chunk: list[dict], candidates: list[dict], registry: dict, aliases: dict, units: list[dict]
) -> tuple[str, dict]:
    scopes = {candidate["id"]: candidate_scope_fingerprint(candidate) for candidate in candidates}
    provenance = {
        "candidate_scope_fingerprints": [scopes[candidate["id"]] for candidate in chunk],
        "offered_options_fingerprint": json_digest(
            sorted(
                [
                    {
                        "scope": scopes[candidate["id"]],
                        "known_owner": candidate.get("known_owner"),
                        "audited_target": candidate.get("audited_target"),
                        "scoped_audit": candidate.get("scoped_audit"),
                    }
                    for candidate in candidates
                ],
                key=lambda item: item["scope"],
            )
        ),
        "offered_facts_fingerprint": json_digest({"registry": registry, "aliases": aliases}),
        "scene_fingerprint": json_digest(
            [
                {
                    "id": unit["id"],
                    "chapter_sha256": unit["chapter_sha256"],
                    "quote_sha256": hashlib.sha256(unit["quote"].encode("utf-8")).hexdigest(),
                }
                for unit in units
            ]
        ),
    }
    return json_digest(provenance), {**provenance, "candidate_id_to_scope": scopes}


def cache_model_records(records: list[dict], provenance: dict) -> list[dict]:
    scopes = provenance["candidate_id_to_scope"]
    return [
        {
            "candidate_scope": scopes[record["candidate_id"]],
            "status": record["status"],
            "identity": (
                {"candidate_scope": scopes[record["identity"]]}
                if record["identity"] in scopes
                else {"canonical": record["identity"]}
            ),
            "evidence_unit_ids": record["evidence_unit_ids"],
        }
        for record in records
    ]


def restore_cached_model_records(records: list[dict], provenance: dict) -> list[dict] | None:
    current = {scope: candidate_id for candidate_id, scope in provenance["candidate_id_to_scope"].items()}
    restored: list[dict] = []
    for record in records:
        candidate_id = current.get(record.get("candidate_scope"))
        identity = record.get("identity")
        if candidate_id is None or not isinstance(identity, dict):
            return None
        if "candidate_scope" in identity:
            target = current.get(identity["candidate_scope"])
            if target is None:
                return None
        elif isinstance(identity.get("canonical"), str):
            target = identity["canonical"]
        else:
            return None
        restored.append(
            {
                "candidate_id": candidate_id,
                "status": record.get("status"),
                "identity": target,
                "evidence_unit_ids": record.get("evidence_unit_ids"),
            }
        )
    return restored


# ##################################################################
# source-audited variant targets
# converts only read-only audit groups whose full owner is present in the current ledger into schema targets, preserving source-audited rather than guessed identity links.
def source_audited_variant_targets(
    project: Path, candidates: list[dict], source_text: str, registry: dict
) -> dict[str, str]:
    labels = {candidate["label"]: candidate["id"] for candidate in candidates}
    targets: dict[str, str] = {}
    for filename, key in (
        ("qa-klein-team-alias-proposal.json", "decisions"),
        ("qa-batch61-62-semantic-audit.json", "batch61_62_expected_new_named"),
    ):
        path = project / filename
        if not path.is_file():
            continue
        payload = load_object(path, "source-audited variant context")
        records = payload.get(key)
        if not isinstance(records, list):
            raise TypeError("source-audited variant context has no records")
        for record in records:
            if not isinstance(record, dict):
                continue
            aliases = record.get("alias") if key == "decisions" else record.get("name")
            if isinstance(aliases, str):
                aliases = [
                    part.strip() for part in re.sub(r"[()]", "", aliases).replace("also written", ",").split(",")
                ]
            if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
                continue
            owner_label = next(
                (alias for alias in aliases if alias in labels and " " in alias),
                next((alias for alias in aliases if alias in labels), None),
            )
            if owner_label is None:
                continue
            evidence = record.get("evidence")
            if (
                not isinstance(evidence, list)
                or not evidence
                or not all(
                    isinstance(item, dict) and isinstance(item.get("excerpt"), str) and item["excerpt"] in source_text
                    for item in evidence
                )
            ):
                raise RuntimeError("source-audited variant evidence is absent from immutable source")
            owner_candidate = next(candidate for candidate in candidates if candidate["id"] == labels[owner_label])
            audited_canonical = record.get("canonical")
            owner = (
                audited_canonical
                if isinstance(audited_canonical, str) and audited_canonical in registry
                else owner_candidate.get("known_owner") or owner_candidate["id"]
            )
            for alias in aliases:
                if alias in labels and labels[alias] != owner_candidate["id"]:
                    targets[labels[alias]] = owner
    return targets


# ##################################################################
# source-audited variant context
# reads parent-maintained, source-audited identity groups as bounded evidence hints without creating identities or mutating production audit data.
def source_audited_variant_context(project: Path, candidates: list[dict]) -> str:
    labels = {candidate["label"] for candidate in candidates}
    path = project / "qa-klein-team-alias-proposal.json"
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("source-audited variant context is unreadable") from error
    decisions = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(decisions, list):
        raise TypeError("source-audited variant context has no decisions")
    groups = []
    for decision in decisions:
        aliases = decision.get("alias") if isinstance(decision, dict) else None
        note = decision.get("note", "") if isinstance(decision, dict) else ""
        if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
            continue
        present = [alias for alias in aliases if alias in labels]
        if len(present) < 2:
            continue
        full = [alias for alias in aliases if " " in alias and alias in labels]
        if full:
            groups.append(
                f"SOURCE-AUDITED VARIANT GROUP: {', '.join(present)}; use full source label {full[0]!r} as the one new owner and classify other listed labels known to it only with their source witnesses. {note}"
            )
    semantic_path = project / "qa-batch61-62-semantic-audit.json"
    if semantic_path.is_file():
        try:
            semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("source-audited semantic context is unreadable") from error
        expected = semantic.get("batch61_62_expected_new_named") if isinstance(semantic, dict) else None
        if not isinstance(expected, list):
            raise TypeError("source-audited semantic context has no expected identities")
        for entry in expected:
            name = entry.get("name") if isinstance(entry, dict) else None
            note = entry.get("note", "") if isinstance(entry, dict) else ""
            if not isinstance(name, str) or not isinstance(note, str):
                continue
            aliases = [part.strip() for part in re.sub(r"[()]", "", name).replace("also written", ",").split(",")]
            present = [alias for alias in aliases if alias in labels]
            if len(present) >= 2:
                full = next((alias for alias in present if " " in alias), present[0])
                groups.append(
                    f"SOURCE-AUDITED VARIANT GROUP: {', '.join(present)}; use full source label {full!r} as the one new owner and classify other listed labels known to it only with their source witnesses. {note}"
                )
    return "\n".join(groups)


# ##################################################################
# classification chunks
# partitions only native response cardinality while retaining the complete batch ledger as legal identity targets for every chunk.
def classification_chunks(candidates: list[dict]) -> list[list[dict]]:
    return [
        candidates[index : index + CLASSIFICATION_CHUNK_SIZE]
        for index in range(0, len(candidates), CLASSIFICATION_CHUNK_SIZE)
    ]


# ##################################################################
# validate classification chunk
# rejects malformed or incomplete chunk transport before its response can be composed into the batch-wide exact-once ledger.
def validate_classification_chunk(
    value: object,
    candidates: list[dict],
    registry: dict,
    aliases: dict,
    all_candidates: list[dict],
    units: list[dict],
    pending: list[dict] | None = None,
) -> list[dict]:
    if (
        not isinstance(value, dict)
        or set(value) != {"classifications"}
        or not isinstance(value["classifications"], list)
    ):
        raise CastValidationError("classification chunk is not the exact object schema")
    expected = {candidate["id"] for candidate in candidates}
    records = value["classifications"]
    actual = [record.get("candidate_id") for record in records if isinstance(record, dict)]
    identities = {candidate["id"] for candidate in all_candidates} | set(registry) | {"none"}
    if len(records) != len(candidates) or set(actual) != expected or len(actual) != len(set(actual)):
        raise CastValidationError("classification chunk omitted or duplicated lexical candidates")
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"candidate_id", "status", "identity", "evidence_unit_ids"}
            or record["status"] not in {"known", "new", "non_character", "ambiguous"}
            or record["identity"] not in identities
        ):
            raise CastValidationError("classification chunk has invalid record")
        candidate = next(candidate for candidate in candidates if candidate["id"] == record["candidate_id"])
        own_units = {ref_id.rsplit("n", 1)[0] for ref_id in candidate["ref_ids"]}
        evidence = record["evidence_unit_ids"]
        if (
            not isinstance(evidence, list)
            or not evidence
            or len(evidence) != len(set(evidence))
            or not set(evidence) <= {unit["id"] for unit in units}
            or not own_units.intersection(evidence)
        ):
            raise CastValidationError(f"classification lacks a unique own source witness: {record['candidate_id']}")
        if record["status"] == "new" and record["identity"] != record["candidate_id"]:
            raise CastValidationError(f"new classification must own its candidate ID: {record['candidate_id']}")
        if record["status"] in {"non_character", "ambiguous"} and record["identity"] != "none":
            raise CastValidationError(f"non-identity classification has a target: {record['candidate_id']}")
        if record["status"] == "known" and record["identity"] == "none":
            raise CastValidationError(f"known classification lacks an identity target: {record['candidate_id']}")
        if record["status"] == "known" and record["identity"] == record["candidate_id"]:
            raise CastValidationError(f"known classification must target another identity: {record['candidate_id']}")
        target_candidate = next(
            (candidate for candidate in all_candidates if candidate["id"] == record["identity"]), None
        )
        if (
            record["status"] == "known"
            and target_candidate is not None
            and target_candidate.get("known_owner") in registry
        ):
            # chain through a known-owner candidate resolves to its established canonical before any new-alias evidence guard; the mention still needs contextual approval below
            record["identity"] = target_candidate["known_owner"]
            target_candidate = None
            if record["identity"] == record["candidate_id"]:
                raise CastValidationError(
                    f"known classification must target another identity: {record['candidate_id']}"
                )
        if record["status"] == "known" and record["identity"] in registry:
            canonical = record["identity"]
            if scoped_alias_approved(candidate, canonical):
                pass
            elif (
                pending is not None
                and not scope_final(candidate)
                and not approved_known_label(
                    candidate["label"], canonical, record["evidence_unit_ids"], units, registry, aliases
                )
            ):
                pending.append({"candidate": candidate, "proposed": canonical})
            elif not approved_known_label(
                candidate["label"], canonical, record["evidence_unit_ids"], units, registry, aliases
            ):
                raise CastValidationError(
                    f"known classification assigns prose or cross-owner label to {canonical}: {candidate['label']!r}"
                )
        if record["status"] == "ambiguous" and pending is not None and not scope_final(candidate):
            pending.append({"candidate": candidate, "proposed": None})
        if (
            candidate.get("scoped_stale")
            and pending is not None
            and not any(item["candidate"] is candidate for item in pending)
        ):
            # a cached ambiguous decision is re-judged per mention whatever the primary says; it must not be silently replaced by a label-level answer
            pending.append(
                {"candidate": candidate, "proposed": record["identity"] if record["identity"] in registry else None}
            )
        if record["status"] == "known" and target_candidate is not None and not candidate.get("audited_target"):
            # A lexical candidate may itself resolve to an established canonical only after
            # its own bounded classification is composed.  Do not demand impossible
            # same-quote evidence for that provisional hop: send this exact mention to
            # scoped contextual adjudication, which may approve only a canonical owner.
            if pending is not None:
                pending.append({"candidate": candidate, "proposed": None})
            else:
                candidate_units = {ref_id.rsplit("n", 1)[0] for ref_id in candidate["ref_ids"]}
                target_units = {ref_id.rsplit("n", 1)[0] for ref_id in target_candidate["ref_ids"]}
                full_name_component = normalized_id(candidate["label"]) in {
                    normalized_id(word) for word in target_candidate["label"].split()
                }
                evidence = set(record["evidence_unit_ids"])
                if not full_name_component and (
                    not candidate_units.intersection(evidence)
                    or not target_units.intersection(evidence)
                    or candidate_units.intersection(target_units).intersection(evidence) == evidence
                ):
                    raise CastValidationError(
                        f"known candidate link lacks distinct source identity evidence: {candidate['label']!r}"
                    )
    return records


# ##################################################################
# verify proposed living entities
# performs a second native, schema-bound source review only for proposed new identities, refusing equipment, attacks, groups, and labels without a living-agent witness.
def verify_proposed_living_entities(
    batch_units: list[dict], discoveries: list[dict], ask, warnings: list[dict] | None = None
) -> set[str]:
    if not discoveries:
        return set()
    unit_ids = [unit["id"] for unit in batch_units]
    identities = [item["id"] for item in discoveries]
    roster = "; ".join(f"{item['id']} name={item['name']!r} aliases={item['aliases']!r}" for item in discoveries)
    schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "minItems": len(identities),
                "maxItems": len(identities),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": identities},
                        "eligibility": {"type": "string", "enum": ["living", "nonliving", "uncertain"]},
                        "evidence_unit_ids": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "uniqueItems": True,
                            "items": {"type": "string", "enum": unit_ids},
                        },
                    },
                    "required": ["id", "eligibility", "evidence_unit_ids"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["entities"],
        "additionalProperties": False,
    }
    narrative = "\n".join(f"[{unit['id']}] {unit['quote']}" for unit in batch_units)
    prompt = f"For each proposed identity decide only living, nonliving, or uncertain. A living result needs a cited witness naming this exact name or alias and proving a named living individual/creature. Equipment, weapons, attacks, skills, groups, places, and captions are nonliving. Use uncertain when source does not prove either result; uncertain fails closed. Output every ID exactly once.\nProposed identities: {roster}\nSOURCE:\n{narrative}"
    response = ask(prompt, max_tokens=1200, max_attempts=1, response_schema=schema)
    warn = warnings if warnings is not None else []

    def flag(identity: str, code: str, raw: object) -> None:
        warn.append({"id": identity, "code": code, "raw": bounded(raw, 1500)})

    try:
        value = json.loads(response)
    except (ValueError, RecursionError):
        for identity in identities:
            flag(identity, "verifier_malformed_json", response)
        return set()
    records = value.get("entities") if isinstance(value, dict) else None
    if not isinstance(records, list):
        for identity in identities:
            flag(identity, "verifier_malformed_json", response)
        return set()
    units_by_id = {unit["id"]: unit for unit in batch_units}
    by_id = {item["id"]: item for item in discoveries}
    counts = collections.Counter(record.get("id") for record in records if isinstance(record, dict))
    approved: set[str] = set()
    seen: set[str] = set()
    for record in records:
        rid = record.get("id") if isinstance(record, dict) else None
        if rid not in by_id:
            flag(str(rid), "verifier_bad_evidence", record)
            continue
        if counts[rid] > 1:
            flag(rid, "verifier_duplicate_id", record)
            seen.add(rid)
            continue
        seen.add(rid)
        evidence = record.get("evidence_unit_ids")
        if (
            record.get("eligibility") not in {"living", "nonliving", "uncertain"}
            or not isinstance(evidence, list)
            or not evidence
            or not all(isinstance(e, str) and e in units_by_id for e in evidence)
        ):
            flag(rid, "verifier_bad_evidence", record)
            continue
        identity = by_id[rid]
        labels = [identity["name"], *identity["aliases"]]
        if not any(
            source_label_present(label, [units_by_id[evidence_id]]) for label in labels for evidence_id in evidence
        ):
            flag(rid, "verifier_missing_witness", record)
            continue
        if record["eligibility"] == "uncertain":
            flag(rid, "verifier_uncertain", record)
            continue
        if record["eligibility"] == "living":
            approved.add(rid)
    for identity in identities:
        if identity not in seen:
            flag(identity, "verifier_bad_evidence", "identity omitted from verifier response")
    return approved


# ##################################################################
# proposed-new identity review
# before any proposed new identity is accepted, a native schema-bound review of each exact mention against the full bounded scene and the registry's prior facts decides existing:<id>, distinct_living_identity, nonidentity_fragment or uncertain. The raw verdict, witnesses and scope are persisted; existing/fragment verdicts become mention-scoped audit decisions; uncertain fails closed.
NEW_IDENTITY_AUDIT_NAME = "new-identity-review-audit.json"
NEW_IDENTITY_MENTIONS_PER_CALL = 4
DISTINCT_VERDICT = "distinct_living_identity"
FRAGMENT_VERDICT = "nonidentity_fragment"
SAME_PROVISIONAL = "same_provisional:"


def load_new_identity_audit(project: Path) -> list[dict]:
    path = project / NEW_IDENTITY_AUDIT_NAME
    return load_object(path, "new-identity review audit").get("records", []) if path.is_file() else []


NEW_IDENTITY_REASKS = 2
ROLE_WORDS = {
    "mom",
    "mum",
    "dad",
    "father",
    "mother",
    "uncle",
    "aunt",
    "sister",
    "brother",
    "boss",
    "master",
    "teacher",
    "captain",
    "king",
    "queen",
    "lord",
    "lady",
    "sir",
    "madam",
    "doctor",
    "guard",
    "elder",
    "boy",
    "girl",
    "man",
    "woman",
}

COMMON_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "another",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "even",
        "every",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

CONTENT_ANCHOR_STOP_WORDS = frozenset(
    {
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "among",
        "and",
        "another",
        "any",
        "are",
        "back",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "came",
        "come",
        "could",
        "did",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "even",
        "every",
        "first",
        "from",
        "good",
        "great",
        "had",
        "has",
        "have",
        "having",
        "here",
        "into",
        "just",
        "know",
        "like",
        "made",
        "make",
        "many",
        "more",
        "most",
        "much",
        "must",
        "never",
        "only",
        "other",
        "over",
        "said",
        "same",
        "should",
        "some",
        "still",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "time",
        "under",
        "very",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "will",
        "with",
        "would",
        "your",
        # generic descriptive words in character profiles
        "look",
        "voice",
        "male",
        "female",
        "young",
        "adult",
        "companion",
        "character",
        "person",
        "facts",
        "prior",
        "general",
        "name",
        "none",
        "true",
        "false",
    }
)


def provisional_plausible(label: str, other: str) -> bool:
    """A provisional same-person link is only offered between lexically continuous names (shared non-stop token, near spelling) or when both labels are role/kinship forms; unrelated names (Han vs Sora) or ordinary stop words (than vs Han) are never options."""
    a, b = label.casefold().strip(), other.casefold().strip()
    if a in COMMON_STOP_WORDS or b in COMMON_STOP_WORDS:
        return False
    tokens_a = {t for t in re.findall(r"\w+", a) if t not in COMMON_STOP_WORDS}
    tokens_b = {t for t in re.findall(r"\w+", b) if t not in COMMON_STOP_WORDS}
    if tokens_a & tokens_b or (tokens_a & ROLE_WORDS and tokens_b & ROLE_WORDS):
        return True
    # A high typo threshold admits Jun/June and Lou/Lu but not coincidental
    # multi-word overlap or substring matching.
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


NEW_IDENTITY_VARIANT_REFS = 6
NEW_IDENTITY_VARIANT_SCENE_CHARS = 600


def provisional_variant_refs(
    candidates: list[dict], classifications: list[dict], references: dict, units_by_id: dict
) -> dict[str, list[dict]]:
    """Per new-root candidate id, the other candidates the primary classification (or a source audit) already maps known->that root, each with its literally verified mention units. Registry-owned labels are never provisional; a chain is followed to its root with a visited set so a cycle or a chain that does not end at a root yields nothing; a label that does not literally occur at its recorded span is dropped."""
    by_id = {candidate["id"]: candidate for candidate in candidates}
    record_by_id = {item["candidate_id"]: item for item in classifications}

    def direct_target(candidate: dict) -> str | None:
        item = record_by_id.get(candidate["id"])
        target = item["identity"] if item and item["status"] == "known" else candidate.get("audited_target")
        return target if target in by_id and target != candidate["id"] else None

    variants: dict[str, list[dict]] = {}
    for candidate in candidates:
        if candidate.get("known_owner") or normalized_id(candidate["label"]) == "":
            continue
        seen, root = {candidate["id"]}, direct_target(candidate)
        while root is not None and direct_target(by_id[root]) is not None and root not in seen:
            seen.add(root)
            root = direct_target(by_id[root])
        root_item = record_by_id.get(root) if root else None
        if (
            root is None
            or root in seen
            or by_id[root].get("known_owner")
            or normalized_id(by_id[root]["label"]) == normalized_id(candidate["label"])
            or (root_item and root_item["status"] != "new")
        ):
            continue
        mention_units = []
        for ref_id in candidate["ref_ids"]:
            reference = references[ref_id]
            unit = units_by_id[reference["unit_id"]]
            if unit["quote"][reference["start"] : reference["start"] + len(reference["label"])] == reference["label"]:
                mention_units.append((unit, reference))
        if mention_units:
            variants.setdefault(root, []).append({"candidate": candidate, "mentions": mention_units})
    return variants


def extract_owner_profile_facts(entry: dict) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    if "bio" in entry and isinstance(entry["bio"], str):
        fields.append(("bio", entry["bio"]))
    if "look" in entry and isinstance(entry["look"], str):
        fields.append(("look", entry["look"]))
    facts = entry.get("facts")
    if isinstance(facts, dict):
        for k, v in sorted(facts.items()):
            if isinstance(v, list):
                fields.append((f"facts.{k}", " ".join(str(x) for x in v)))
            elif isinstance(v, str):
                fields.append((f"facts.{k}", v))
    elif isinstance(facts, list):
        fields.append(("facts", " ".join(str(x) for x in facts)))
    source_facts = entry.get("source_facts")
    if isinstance(source_facts, str):
        fields.append(("source_facts", source_facts))
    elif isinstance(source_facts, list):
        fields.append(("source_facts", " ".join(str(x) for x in source_facts)))
    return fields


def registry_name_words(registry: dict, aliases: dict | None = None) -> set[str]:
    names = set(ROLE_WORDS)
    for owner_id, entry in registry.items():
        names.update(re.findall(r"[a-z]+", owner_id.casefold()))
        if isinstance(entry, dict) and "name" in entry:
            names.update(re.findall(r"[a-z]+", str(entry["name"]).casefold()))
    if aliases:
        for alias, target in aliases.items():
            names.update(re.findall(r"[a-z]+", alias.casefold()))
            names.update(re.findall(r"[a-z]+", target.casefold()))
    return names


def registry_anchor_counts(registry: dict, name_words: set[str]) -> dict[str, int]:
    counts: dict[str, int] = collections.Counter()
    for entry in registry.values():
        if not isinstance(entry, dict):
            continue
        owner_words = set()
        for _, text in extract_owner_profile_facts(entry):
            words = {
                w.casefold()
                for w in re.findall(r"\b[a-zA-Z]{4,}\b", text)
                if w.casefold() not in CONTENT_ANCHOR_STOP_WORDS and w.casefold() not in name_words
            }
            owner_words.update(words)
        for w in owner_words:
            counts[w] += 1
    return counts


def find_existing_owner_support(
    label: str, owner: str, witness_units: list[dict], registry: dict, aliases: dict
) -> tuple[bool, dict | None]:
    names = [
        registry[owner].get("name", owner),
        *(alias.replace("_", " ") for alias, target in aliases.items() if target == owner),
    ]
    for unit in witness_units:
        for name in names:
            if source_label_present(name, [unit]):
                field = "name" if name == registry[owner].get("name") else "alias"
                return True, {
                    "type": "literal_witness",
                    "owner_name": name,
                    "owner_profile_field": field,
                    "witness_unit_id": unit["id"],
                }

    name_words = registry_name_words(registry, aliases) | set(re.findall(r"[a-z]+", label.casefold()))
    counts = registry_anchor_counts(registry, name_words)
    fields = extract_owner_profile_facts(registry[owner])
    best_candidate: tuple[int, int, str, str, str] | None = None

    for unit in witness_units:
        unit_words = {
            w.casefold()
            for w in re.findall(r"\b[a-zA-Z]{4,}\b", unit["quote"])
            if w.casefold() not in CONTENT_ANCHOR_STOP_WORDS and w.casefold() not in name_words
        }
        for field_name, field_text in fields:
            field_words = {
                w.casefold()
                for w in re.findall(r"\b[a-zA-Z]{4,}\b", field_text)
                if w.casefold() not in CONTENT_ANCHOR_STOP_WORDS and w.casefold() not in name_words
            }
            shared = unit_words & field_words
            for anchor in shared:
                owner_count = counts.get(anchor, 1)
                if owner_count <= 2:
                    cand = (owner_count, -len(anchor), anchor, field_name, unit["id"])
                    if best_candidate is None or cand < best_candidate:
                        best_candidate = cand

    if best_candidate is not None:
        owner_count, _neg_len, anchor, field_name, unit_id = best_candidate
        return True, {
            "type": "content_anchor",
            "anchor": anchor,
            "owner_profile_field": field_name,
            "witness_unit_id": unit_id,
            "owner_count": owner_count,
        }
    return False, None


def is_apposition_mention(quote: str, label: str, name: str) -> bool:
    """True when label and name appear in apposition (parenthetical, comma alias, or adjacent title/name)."""
    l_esc = re.escape(label.strip())
    n_esc = re.escape(name.strip())
    patterns = [
        rf"\b{l_esc}\b\s*\(\s*{n_esc}\b",
        rf"\b{n_esc}\b\s*\(\s*{l_esc}\b",
        rf"\b{l_esc}\b,\s*(?:also known as|known as|called|alias|aka|namely)\s+{n_esc}\b",
        rf"\b{n_esc}\b,\s*(?:also known as|known as|called|alias|aka|namely)\s+{l_esc}\b",
        rf"\b{l_esc}\b\s+(?:that is|aka|also known as|known as|called|alias)\s+{n_esc}\b",
        rf"\b{n_esc}\b\s+(?:that is|aka|also known as|known as|called|alias)\s+{l_esc}\b",
        rf"\b{l_esc}\b,\s*{n_esc}\b(?!\s+(?:and|or|nor)\b)",
        rf"\b{n_esc}\b,\s*{l_esc}\b(?!\s+(?:and|or|nor)\b)",
        rf"\b{l_esc}\s+{n_esc}\b",
        rf"\b{n_esc}\s+{l_esc}\b",
    ]
    return any(re.search(pat, quote, re.IGNORECASE) for pat in patterns)


def mention_unit_has_separate_owner_token(quote: str, label: str, name: str) -> bool:
    """True if quote has own label and a separate owner token outside that label occurrence."""
    label_matches = list(re.finditer(rf"(?<!\w){re.escape(label.strip())}(?!\w)", quote, re.IGNORECASE))
    if not label_matches:
        return False
    m = label_matches[0]
    masked = quote[: m.start()] + " " * (m.end() - m.start()) + quote[m.end() :]
    return bool(re.search(rf"(?<!\w){re.escape(name.strip())}(?!\w)", masked, re.IGNORECASE))


def are_enumerated_distinct_actors(quote: str, label_a: str, label_b: str) -> bool:
    """True when two labels appear enumerated as distinct actors (e.g. coordinated with and/or/comma list)."""
    a = re.escape(label_a.strip())
    b = re.escape(label_b.strip())
    patterns = [
        rf"\b{a}\b\s+(?:and|or|nor|&)\s+\b{b}\b",
        rf"\b{b}\b\s+(?:and|or|nor|&)\s+\b{a}\b",
        rf"\b(?:both|either|between)\s+{a}\s+(?:and|or)\s+{b}\b",
        rf"\b(?:both|either|between)\s+{b}\s+(?:and|or)\s+{a}\b",
        rf"\b{a}\b\s+(?:as well as|along with|alongside|together with)\s+\b{b}\b",
        rf"\b{b}\b\s+(?:as well as|along with|alongside|together with)\s+\b{a}\b",
        rf"\b{a}\b\s*,\s*(?:and\s+|or\s+)?\b{b}\b",
        rf"\b{b}\b\s*,\s*(?:and\s+|or\s+)?\b{a}\b",
    ]
    return any(re.search(pat, quote, re.IGNORECASE) for pat in patterns)


def owner_support(
    label: str, owner: str, witness_units: list[dict], episode_units: list[dict] | None, registry: dict, aliases: dict
) -> tuple[bool, dict | None]:
    """Source/registry support for an existing owner: the cited units first, then (only when a continuous episode was reviewed) the mention's own episode. Provenance is exactly what find_existing_owner_support found, naming the real unit it came from."""
    supported, provenance = find_existing_owner_support(label, owner, witness_units, registry, aliases)
    if not supported and episode_units:
        supported, provenance = find_existing_owner_support(label, owner, episode_units, registry, aliases)
    return supported, provenance


# ##################################################################
# scoped alias proof
# a contextual owner proposal, a cached model verdict, or a roster position never proves identity. An alias to a canonical owner is accepted only when program-derived source facts show (1) an exact literal tie between the label and the owner (approved name, shared name word, or the label written in the owner's own profile), (2) continuity between the owner and the label's own bounded scene that does not come from the label's mention itself, and (3) no contradiction: a distinct participant or enumerated actor in the mention, a name word belonging to another cast member, or another owner equally carrying the label named in the scene. Anything else stays unresolved (pending); no owner is forced.
TITLE_ROLE_TOKENS = frozenset({word.casefold() for word in TITLE_WORDS} | ROLE_WORDS)


def owner_name_forms(owner: str, registry: dict, aliases: dict) -> list[str]:
    forms = [str(registry[owner].get("name", owner))]
    forms.extend(alias.replace("_", " ") for alias, target in aliases.items() if target == owner and alias != owner)
    return list(dict.fromkeys(form for form in forms if form.strip()))


def owner_name_tokens(owner: str, registry: dict, aliases: dict) -> set[str]:
    tokens = set(owner.split("_"))
    for form in owner_name_forms(owner, registry, aliases):
        tokens |= label_components(form)
    return tokens


def literal_pattern(text: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(text.strip())}(?!\w)", re.IGNORECASE)


def mask_label(quote: str, label: str, protected: list[str]) -> str:
    """The quote with every occurrence of the label blanked, except where the occurrence sits inside a longer protected owner name."""
    shielded = [
        match.span()
        for name in protected
        if len(name.strip()) > len(label.strip())
        for match in literal_pattern(name).finditer(quote)
    ]
    out = quote
    for match in literal_pattern(label).finditer(quote):
        if not any(low <= match.start() and match.end() <= high for low, high in shielded):
            out = out[: match.start()] + " " * (match.end() - match.start()) + out[match.end() :]
    return out


BRIDGE_DETERMINERS = r"(?:the|our|their|his|her|my)"
BRIDGE_COPULA = r"(?:is|was|became|remained|served as|is known as|was known as|is called|was called)"


def source_bridge_predicate(
    label: str,
    owner: str,
    scene_units: list[dict],
    registry: dict,
    aliases: dict,
    common_noun_only: bool = False,
) -> dict | None:
    """Explicit source evidence that a role or title label is this owner, from the cited immutable scene units alone.

    Accepted only as one continuous literal predicate inside a single unit: role+known-name ("Master Chen"), name apposition
    ("Chen, the Master" / "Chen (the Master)" / "Master (Chen)" / "Master, Chen"), or a copula ("Chen was the Master").
    Co-occurrence of the label and the name, a bare title, a second participant of the same role (another cast member
    carrying the label, or the label followed by a different name) and a name shared with another cast member all yield None.
    common_noun_only (a label that is not a known title word) additionally demands the determiner-introduced forms, so a bare name never binds."""
    lab = re.escape(label.strip())
    if not label.strip():
        return None
    others = [other for other in registry if other not in {owner, "narrator"}]
    other_tokens = (
        set().union(*(owner_name_tokens(other, registry, aliases) for other in others))
        if others
        else set()
    )
    own_tokens = owner_name_tokens(owner, registry, aliases)
    names = set(owner_name_forms(owner, registry, aliases))
    names |= {
        word
        for form in list(names)
        for word in re.findall(r"[A-Za-z]+", form)
        if len(word) > 2 and normalized_id(word) not in TITLE_ROLE_TOKENS
    }
    names = {
        name
        for name in names
        if not label_components(name) & other_tokens
        or label_components(name) >= own_tokens
    }
    other_forms = [form for other in others for form in owner_name_forms(other, registry, aliases)]
    for unit in scene_units:
        quote = unit["quote"]
        if not re.search(lab, quote, re.IGNORECASE):
            continue
        for form in other_forms:
            if is_apposition_mention(quote, label, form) or re.search(
                rf"(?<!\w){re.escape(form)}\s*,?\s*(?:{BRIDGE_DETERMINERS}\s+)?{lab}\b",
                quote,
                re.IGNORECASE,
            ):
                return None
        for match in re.finditer(rf"(?<!\w){lab}\s+((?-i:[A-Z][A-Za-z]+))", quote, re.IGNORECASE):
            if normalized_id(match.group(1)) not in own_tokens:
                return None
    for unit in scene_units:
        quote = unit["quote"]
        if not re.search(lab, quote, re.IGNORECASE):
            continue
        for name in sorted(names, key=len, reverse=True):
            if are_enumerated_distinct_actors(quote, label, name):
                continue
            nm = re.escape(name)
            for kind, pattern in (
                ("role_plus_name", rf"(?<!\w){lab}\s+{nm}(?!\w)"),
                (
                    "name_apposition",
                    rf"(?<!\w){nm}\s*,\s*{BRIDGE_DETERMINERS}\s+{lab}(?!\w)",
                ),
                (
                    "name_apposition",
                    rf"(?<!\w){nm}\s*\(\s*(?:{BRIDGE_DETERMINERS}\s+)?{lab}\s*\)",
                ),
                ("name_apposition", rf"(?<!\w){lab}\s*\(\s*{nm}\s*\)"),
                (
                    "name_apposition",
                    rf"(?<!\w){lab}\s*,\s*{nm}(?!\w)(?!\s+(?:and|or|nor)\b)",
                ),
                (
                    "copula",
                    rf"(?<!\w){nm}\s+{BRIDGE_COPULA}\s+{BRIDGE_DETERMINERS}\s+{lab}(?!\w)",
                ),
                (
                    "copula",
                    rf"(?<!\w){BRIDGE_DETERMINERS}\s+{lab}\s+{BRIDGE_COPULA}\s+{nm}(?!\w)",
                ),
            ):
                if common_noun_only and BRIDGE_DETERMINERS not in pattern:
                    continue
                if re.search(pattern, quote, re.IGNORECASE):
                    return {
                        "type": "source_bridge",
                        "predicate": kind,
                        "owner_name": name,
                        "witness_unit_id": unit["id"],
                    }
    return None


def scoped_alias_proof(
    label: str,
    owner: str,
    mention_unit: dict,
    scene_units: list[dict],
    registry: dict,
    aliases: dict,
) -> tuple[dict | None, str | None]:
    """(provenance, None) when the alias of this exact mention to the owner is mechanically proven, else (None, why not)."""
    if owner not in registry or owner == "narrator":
        return None, f"owner {owner!r} is outside the cast"
    forms = owner_name_forms(owner, registry, aliases)
    label_id = normalized_id(label)
    if label_id in {normalized_id(form) for form in forms} | {normalized_id(owner)} or aliases.get(label_id) == owner:
        return {"literal": "approved_name"}, None
    owner_tokens = owner_name_tokens(owner, registry, aliases)
    label_tokens = label_components(label)
    name_tokens = label_tokens - TITLE_ROLE_TOKENS
    profile = [text for _field, text in extract_owner_profile_facts(registry[owner])]
    pattern = literal_pattern(label)
    in_profile = any(pattern.search(text) for text in profile)
    role_only = not name_tokens
    bridge = None
    if (
        name_tokens
        and not name_tokens & owner_tokens
        and not in_profile
        and len(label.split()) == 1
        and not any(
            name_tokens & owner_name_tokens(other, registry, aliases) for other in registry if other not in {owner, "narrator"}
        )
    ):
        bridge = source_bridge_predicate(label, owner, scene_units, registry, aliases, common_noun_only=True)
    if bridge is not None:
        literal = "source_bridge"
    elif name_tokens:
        if not name_tokens & owner_tokens and not in_profile:
            return None, f"label {label!r} shares no name word with {owner!r} and is not written in its profile"
        extra = name_tokens - owner_tokens
        if extra and not in_profile:
            for other in registry:
                if other not in {owner, "narrator"} and extra & owner_name_tokens(other, registry, aliases):
                    return None, f"label {label!r} carries the name of another cast member {other!r}"
        literal = "shared_name_word" if name_tokens & owner_tokens else "owner_profile_literal"
    elif not in_profile:
        bridge = source_bridge_predicate(label, owner, scene_units, registry, aliases)
        if bridge is None:
            return None, f"role or title label {label!r} is not written in the profile of {owner!r}"
        literal = "source_bridge"
    else:
        literal = "owner_profile_literal"
    quote = mention_unit["quote"]
    for name in forms:
        if are_enumerated_distinct_actors(quote, label, name):
            return None, f"{label!r} and {name!r} are enumerated as distinct actors in the mention"
        if (
            not label_components(name) <= label_tokens
            and mention_unit_has_separate_owner_token(quote, label, name)
            and not is_apposition_mention(quote, label, name)
        ):
            return None, f"{name!r} is a distinct participant of the same mention as {label!r}"
    protected = [*forms, *(registry[other].get("name", other) for other in registry if other != owner)]
    masked = [
        {**unit, "quote": mask_label(unit["quote"], label, [str(name) for name in protected])} for unit in scene_units
    ]
    for other in sorted(registry):
        if other in {owner, "narrator"} or not label_tokens <= owner_name_tokens(other, registry, aliases):
            continue
        if any(source_label_present(form, masked) for form in owner_name_forms(other, registry, aliases)):
            return None, f"the scene names {other!r}, which carries {label!r} equally"
    supported, continuity = find_existing_owner_support(label, owner, masked, registry, aliases)
    profile_relation = None
    if in_profile:
        for text in profile:
            for sentence in re.split(r"[.;!?\n]", text):
                if not pattern.search(sentence):
                    continue
                for other in registry:
                    if other in {owner, "narrator"}:
                        continue
                    other_forms = owner_name_forms(other, registry, aliases)
                    if any(literal_pattern(form).search(sentence) for form in other_forms) and any(
                        source_label_present(form, masked) for form in other_forms
                    ):
                        profile_relation = {"type": "profile_relation", "related_owner": other}
                        break
                if profile_relation:
                    break
            if profile_relation:
                break
    if literal == "source_bridge":
        supported, continuity = True, bridge
    elif role_only:
        # A title cannot bind by authority, role similarity, or a broad profile anchor. It needs an
        # explicit owner name in the scene, or a profile relation whose other participant is actually named.
        if continuity is not None and continuity.get("type") != "literal_witness":
            supported, continuity = False, None
        if not supported and profile_relation:
            supported, continuity = True, profile_relation
    elif not supported and profile_relation:
        supported, continuity = True, profile_relation
    if not supported:
        return None, f"no source continuity ties {owner!r} to the scene of {label!r} independent of the mention itself"
    return {"literal": literal, "continuity": continuity}, None


def scene_units_at(units_by_id: dict, order: list[str], position: int) -> list[dict]:
    low, high = bounded_scene_range(units_by_id, order, position)
    return [units_by_id[order[index]] for index in range(low, high + 1)]


def review_verdict_error(
    label: str,
    verdict: str,
    witnesses: list[str],
    candidate: dict,
    plausible: dict,
    registry: dict,
    aliases: dict,
    units_by_id: dict,
    mention_unit: dict,
    episode_units: list[dict] | None = None,
) -> str | None:
    """Why a review verdict is invalid, or None. Witnesses must carry the label's own mention; an existing target needs its own source/registry support (cited units, or the mention's own reviewed episode) and may not be a distinct participant of the same quote."""
    witness_units = [units_by_id[w] for w in witnesses]
    # Every verdict—including nonidentity_fragment—must cite this exact label's
    # immutable span. A shorter adjacent form cannot turn a source-introduced
    # full name into prose merely by being the sole witness.
    if not source_label_present(label, witness_units):
        return "new-identity review lacks own-label witness"
    if verdict.startswith(SAME_PROVISIONAL):
        target = plausible[verdict[len(SAME_PROVISIONAL) :]]
        own_units = {ref_id.rsplit("n", 1)[0] for ref_id in candidate["ref_ids"]}
        if (
            not source_label_present(label, witness_units)
            or not source_label_present(target["label"], witness_units)
            or not own_units.intersection(witnesses)
        ):
            return f"new-identity review lacks literal witnesses for same as {target['label']!r}"
        if target.get("known_owner") or target.get("audited_target"):
            return f"new-identity review cannot chain provisional into existing {target['label']!r}"
        relevant_units = [*witness_units, mention_unit]
        if any(are_enumerated_distinct_actors(unit["quote"], label, target["label"]) for unit in relevant_units):
            return f"new-identity review links {label!r} to {target['label']!r}, enumerated distinct actors of the same scene"
    if verdict.startswith("existing:"):
        owner = verdict.split(":", 1)[1]
        names = [
            registry[owner].get("name", owner),
            *(alias.replace("_", " ") for alias, target in aliases.items() if target == owner),
        ]
        if not source_label_present(label, witness_units):
            return f"new-identity review lacks own-mention witness for existing {owner!r}"
        supported, _ = owner_support(label, owner, witness_units, episode_units, registry, aliases)
        if not supported:
            return f"new-identity review lacks source/registry support for existing {owner!r}"
        _proof, why = scoped_alias_proof(label, owner, mention_unit, episode_units or witness_units, registry, aliases)
        if why:
            return f"new-identity review lacks independent owner proof for existing {owner!r}: {why}"
        for name in names:
            if mention_unit_has_separate_owner_token(mention_unit["quote"], label, name):
                is_approved = (
                    aliases.get(normalized_id(label)) == owner
                    or aliases.get(label.casefold()) == owner
                    or normalized_id(label) == normalized_id(registry[owner].get("name", owner))
                )
                if not is_approved and not is_apposition_mention(mention_unit["quote"], label, name):
                    return f"new-identity review maps to {owner!r}, a distinct participant of the same scene"
    return None


# ##################################################################
# typed pending new-identity review lane
# a named span the source leaves unresolved is never forced onto an existing owner and never fatal to its chapter: the candidate is held out of the registry and aliases (a non_character classification, so no narrator or global alias can form), and a typed recoverable pending row preserves every literal mention, witness and nearby fact for replay or later review.
PENDING_IDENTITY_TYPES = (
    "new_identity",
    "garbled_variant",
    "unapproved_alias",
    "uncertain_living",
    "invalid_classification",
)
PENDING_IDENTITY_CODE_PREFIX = "pending_"
PERSON_KINDS = frozenset({"individual_name", "specific_role"})


def source_context_quotes(units: list[dict], unit_ids: list[str]) -> list[str]:
    """Literal quotes of each cited unit with its immediate neighbours (one before, two after), each once, in first-cited order."""
    order = [unit["id"] for unit in units]
    quotes_by_id = {unit["id"]: unit["quote"] for unit in units}
    context: list[str] = []
    for unit_id in unit_ids:
        position = order.index(unit_id)
        for nearby in order[max(0, position - 1) : position + 3]:
            if nearby not in context:
                context.append(nearby)
    return [quotes_by_id[unit_id] for unit_id in context]


def pending_identity_entry(
    candidate: dict,
    kind: str,
    reason: str,
    units: list[dict],
    references: dict,
    decisions: dict | None = None,
) -> dict:
    """Typed pending entry holding every exact mention of the candidate (with the review decision and cited witnesses where one exists) and the literal source facts around them."""
    if kind not in PENDING_IDENTITY_TYPES:
        raise ValueError(f"unknown pending identity type: {kind!r}")
    units_by_id = {unit["id"]: unit for unit in units}
    mentions: list[dict] = []
    for ref_id in candidate["ref_ids"]:
        reference = references[ref_id]
        unit = units_by_id[reference["unit_id"]]
        scope = mention_scope(unit, reference)
        decision = (decisions or {}).get(scope) or {}
        witnesses = [wid for wid in decision.get("witness_unit_ids", []) if wid in units_by_id]
        mentions.append(
            {
                "chapter": unit["chapter"],
                "chapter_sha256": scope[0],
                "quote_sha256": scope[1],
                "unit_id": unit["id"],
                "label": scope[2],
                "span_start": scope[3],
                "quote": unit["quote"],
                "verdict": decision.get("verdict"),
                "raw_review": decision.get("raw_review"),
                "witnesses": [{"unit_id": wid, "quote": units_by_id[wid]["quote"]} for wid in witnesses],
            }
        )
    return {
        "type": kind,
        "id": normalized_id(candidate["label"]),
        "candidate_id": candidate["id"],
        "label": candidate["label"],
        "reason": bounded(reason, 300),
        "mentions": mentions,
        "facts": source_context_quotes(units, [mention["unit_id"] for mention in mentions]),
    }


def pending_review_result(result: dict, rejected: set[str], reason: object) -> dict | None:
    """Tag a review result with the pending type of the invalid targets the model proposed (an unsupported existing owner is an unapproved alias; an unsupported provisional link is a garbled variant), or None when no target was invalid."""
    kind = (
        "unapproved_alias"
        if any(verdict.startswith("existing:") for verdict in rejected)
        else "garbled_variant"
        if any(verdict.startswith(SAME_PROVISIONAL) for verdict in rejected)
        else None
    )
    if kind is None:
        return None
    return {
        **result,
        "pending_type": kind,
        "invalid_reason": bounded(reason, 300),
        "raw_confidence": result["confidence"],
    }


def ambiguous_pending_kind(candidate: dict) -> str:
    """new_identity when an exact-mention adjudication found a named living individual or role that no offered owner proves; otherwise uncertain_living."""
    record = candidate.get("scoped_audit") or {}
    raw = record.get("raw_adjudication") or {}
    if (
        record.get("decision") == "ambiguous"
        and raw.get("refers_to_person") == "yes"
        and raw.get("candidate_kind") in PERSON_KINDS
    ):
        return "new_identity"
    return "uncertain_living"


def defer_ambiguous_candidates(
    project: Path, records: list[dict], candidates: list[dict], units: list[dict], deferred: list[dict]
) -> None:
    """Move every still-ambiguous candidate into the pending lane."""
    references = immutable_name_references(units)
    units_by_id = {unit["id"]: unit for unit in units}
    scoped = mention_scoped_audit_index(load_scoped_audit(project))
    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    already = {entry["candidate_id"] for entry in deferred}
    for item in records:
        if item["status"] == "ambiguous" and item["candidate_id"] not in already:
            candidate = candidate_by_id[item["candidate_id"]]
            # The exact-mention review just wrote its audit record. Read that binding directly instead of re-running
            # the complete classification merely to repopulate candidate_coverage_ledger with the same scope.
            audit = next(
                (
                    scoped.get(mention_scope(units_by_id[references[ref_id]["unit_id"]], references[ref_id]))
                    for ref_id in candidate["ref_ids"]
                    if scoped.get(mention_scope(units_by_id[references[ref_id]["unit_id"]], references[ref_id]))
                ),
                candidate.get("scoped_audit"),
            )
            candidate_with_audit = {**candidate, "scoped_audit": audit} if audit else candidate
            deferred.append(
                pending_identity_entry(
                    candidate_with_audit,
                    ambiguous_pending_kind(candidate_with_audit),
                    "source leaves this named span unresolved after exact-mention adjudication",
                    units,
                    references,
                )
            )


def demote_deferred(records: list[dict], candidates: list[dict], units: list[dict], deferred: list[dict]) -> None:
    """Hold every deferred candidate out of identity mapping; a candidate linked to a deferred one is itself a pending variant (cascades to a fixed point). Nothing is aliased or merged."""
    references = immutable_name_references(units)
    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    held = {entry["candidate_id"] for entry in deferred}
    grew = True
    while grew:
        grew = False
        for item in records:
            if item["candidate_id"] not in held and item["status"] == "known" and item["identity"] in held:
                deferred.append(
                    pending_identity_entry(
                        candidate_by_id[item["candidate_id"]],
                        "garbled_variant",
                        f"variant of pending identity {candidate_by_id[item['identity']]['label']!r}",
                        units,
                        references,
                    )
                )
                held.add(item["candidate_id"])
                grew = True
    for item in records:
        if item["candidate_id"] in held:
            item["status"], item["identity"] = "non_character", "none"


def review_proposed_identities(
    project: Path,
    units: list[dict],
    classifications: list[dict],
    candidates: list[dict],
    progress: dict,
    ask,
    deferred: list[dict] | None = None,
) -> bool:
    """Review every candidate classified new, mention by mention. Returns True when scoped audit decisions were written and classification must be redone.
    With a deferred list, a candidate whose review stays uncertain or invalid after bounded correction (and a provisional link that cannot stand) is appended there as a typed pending entry instead of failing the chapter."""
    registry, aliases = progress["registry"], progress["aliases"]
    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    proposed = [candidate_by_id[item["candidate_id"]] for item in classifications if item["status"] == "new"]
    if not proposed:
        return False
    references = immutable_name_references(units)
    units_by_id = {unit["id"]: unit for unit in units}
    order = [unit["id"] for unit in units]
    audit = load_new_identity_audit(project)
    by_scope = {(r["chapter_sha256"], r["quote_sha256"], r["label"], r["span_start"]): r for r in audit}
    scoped_records = load_scoped_audit(project)
    wrote_scoped = False
    final_verdicts: dict[str, set[str]] = {}
    variants_by_root = provisional_variant_refs(candidates, classifications, references, units_by_id)
    for candidate in proposed:
        label = candidate["label"]
        provisional = [other for other in proposed if other is not candidate]
        provisional_ids = {other["id"]: other for other in provisional}
        # Do not offer the full roster merely because generic adjudication found it: a
        # new label can map to an established owner only when its spelling/role form
        # has a plausible lexical-continuity path to that owner's canonical name or
        # approved aliases. Scene evidence then proves (rather than creates) it.
        owners = [
            owner
            for owner in adjudication_owners(label, registry, aliases)
            if any(
                provisional_plausible(label, name)
                for name in [
                    registry[owner].get("name", owner),
                    *(alias.replace("_", " ") for alias, target in aliases.items() if target == owner),
                ]
            )
        ]
        # Each mention is scoped by its own immutable reference label (several spellings can share one
        # normalized candidate key); the candidate label only seeds roster/provisional plausibility.
        mentions: dict[tuple[str, str, str, int], dict] = {}
        for ref_id in candidate["ref_ids"]:
            unit = units_by_id[references[ref_id]["unit_id"]]
            mentions.setdefault(mention_scope(unit, references[ref_id]), unit)
        verdicts: dict[tuple[str, str, str, int], str] = {}
        todo = []
        for scope in mentions:
            cached = by_scope.get(scope)
            if (
                cached
                and cached.get("owners") == owners
                and (
                    cached["verdict"] == DISTINCT_VERDICT
                    or (
                        cached["verdict"].startswith(SAME_PROVISIONAL)
                        and cached["verdict"][len(SAME_PROVISIONAL) :] in provisional_ids
                    )
                )
            ):
                verdicts[scope] = cached["verdict"]
            else:
                todo.append(scope)
        # A root is also a plausible target when one of its source-provisional variant refs (a short form the primary
        # classification already mapped to it) is lexically continuous with this label; the variant only widens context,
        # never the target (links always end at the independently introduced root).
        plausible = {
            other_id: other
            for other_id, other in provisional_ids.items()
            if provisional_plausible(label, other["label"])
            or any(
                provisional_plausible(label, variant["candidate"]["label"])
                for variant in variants_by_root.get(other_id, [])
            )
        }
        # This root's own variant refs: same-participant context nearest its mentions, bounded.
        own_positions = [order.index(unit["id"]) for unit in mentions.values()]
        variant_refs = sorted(
            (
                (unit, reference, variant["candidate"])
                for variant in variants_by_root.get(candidate["id"], [])
                for unit, reference in variant["mentions"]
            ),
            key=lambda row: (
                min(abs(order.index(row[0]["id"]) - position) for position in own_positions),
                order.index(row[0]["id"]),
                row[1]["start"],
            ),
        )[:NEW_IDENTITY_VARIANT_REFS]
        base_options = [
            *(f"existing:{owner}" for owner in owners),
            *(f"{SAME_PROVISIONAL}{other_id}" for other_id in plausible),
            DISTINCT_VERDICT,
            FRAGMENT_VERDICT,
            "uncertain",
        ]

        def build_review(
            chunk,
            options,
            note="",
            scene_chars=None,
            label=label,
            owners=owners,
            plausible=plausible,
            mentions=mentions,
            variant_refs=variant_refs,
        ):
            ids = [f"m{index}" for index in range(len(chunk))]
            item = {
                "type": "object",
                "properties": {
                    "mention_id": {"type": "string", "enum": ids},
                    "verdict": {"type": "string", "enum": options},
                    "witness_unit_ids": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 3,
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": order},
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string", "minLength": 1, "maxLength": 300},
                },
                "required": ["mention_id", "verdict", "witness_unit_ids", "confidence", "reason"],
                "additionalProperties": False,
            }
            schema = {
                "type": "object",
                "properties": {
                    "mentions": {"type": "array", "minItems": len(chunk), "maxItems": len(chunk), "items": item}
                },
                "required": ["mentions"],
                "additionalProperties": False,
            }
            compact = scene_chars is not None
            # Compact mode renders every mention's scene AND every source-provisional variant ref's scene as merged
            # continuous episodes (each unit once); the first pass keeps per-mention scenes and gives variant refs their own compact episodes.
            mention_positions = [order.index(mentions[scope]["id"]) for scope in chunk] if compact else []
            variant_positions = [order.index(unit["id"]) for unit, _, _ in variant_refs]
            episode_ranges, episode_of = scene_episode_ranges(
                units_by_id,
                order,
                [*mention_positions, *variant_positions],
                scene_chars if compact else NEW_IDENTITY_VARIANT_SCENE_CHARS,
            )
            episodes = [
                " ".join(f"[{order[index]}] {units_by_id[order[index]]['quote']}" for index in range(low, high + 1))
                for low, high in episode_ranges
            ]
            if compact:
                rows = [
                    f"{mention_id} [{mentions[scope]['chapter']}] unit {mentions[scope]['id']} label {scope[2]!r} at char {scope[3]} mention: {mentions[scope]['quote']}\n   scene: episode E{episode_of[index] + 1}"
                    for index, (mention_id, scope) in enumerate(zip(ids, chunk, strict=True))
                ]
            else:
                rows = [
                    f"{mention_id} [{mentions[scope]['chapter']}] unit {mentions[scope]['id']} mention: {mentions[scope]['quote']}\n   bounded scene (unit ids in brackets): {bounded_scene_with_ids(units_by_id, order, order.index(mentions[scope]['id']))}"
                    for mention_id, scope in zip(ids, chunk, strict=True)
                ]
            scene_block = (
                "CONTINUOUS SOURCE EPISODES (unit ids in brackets; each unit appears once):\n"
                + "\n".join(f"E{index + 1}: {text}" for index, text in enumerate(episodes))
                + "\n"
                if episodes
                else ""
            )
            if variant_refs:
                scene_block += (
                    f"SOURCE-PROVISIONAL VARIANT REFS (other labels the primary classification already mapped to {label!r}; same-participant CONTEXT only, never proof of a verdict and never a target):\n"
                    + "\n".join(
                        f"V{index + 1} label {reference['label']!r} ({variant['id']}) unit {unit['id']} at char {reference['start']} mention: {unit['quote']}\n   scene: episode E{episode_of[len(mention_positions) + index] + 1}"
                        for index, (unit, reference, variant) in enumerate(variant_refs)
                    )
                    + "\n"
                )
            episode_units = (
                {
                    scope: [
                        units_by_id[order[index]]
                        for index in range(
                            episode_ranges[episode_of[position_index]][0],
                            episode_ranges[episode_of[position_index]][1] + 1,
                        )
                    ]
                    for position_index, scope in enumerate(chunk)
                }
                if compact
                else {}
            )
            live_owners = [owner for owner in owners if f"existing:{owner}" in options]
            live_provisional = [other for other in plausible.values() if f"{SAME_PROVISIONAL}{other['id']}" in options]
            episode_clause = (
                " In a CONTINUOUS SOURCE EPISODES review the owner's name or profile anchor may also come from any unit of the mention's own episode (cite it when you can). "
                if compact
                else ""
            )
            prompt = (
                f"The label {label!r} was proposed as a NEW character identity. For each exact mention decide, from its full bounded scene and the registry's prior facts only (never from spelling or sound similarity, never generalising across mentions): existing:<id> when this exact span names or addresses that already-registered character (including a misspelling, nickname, title form, or a name merged with an adjacent word or hesitation); distinct_living_identity when it names a living individual or creature that is a different person from every registered character; nonidentity_fragment when the span is a prose fragment, a sentence-initial/hesitation/verb-bearing run of words, an equipment/skill/place/group name or otherwise not a stable actor name; same_provisional:<candidate> when this exact span is the same living individual as another proposed-new candidate listed below (a spelling variant, typo or short form of that candidate's own name proven by the scene, never by similarity alone) and not any registered character; No existing match is NOT the same as nonliving or uncertain: when the source introduces a named living individual (it acts, speaks, is addressed or is described as a person or creature) and no registered candidate credibly owns the span, that supports distinct_living_identity; nonidentity_fragment is only for spans that are not a living actor's name; uncertain only when the source leaves genuinely open whether the span is a living individual or which owner it belongs to. Each mention's own unit is attached automatically as the span's own-source witness. Cite 0-3 ADDITIONAL witness unit IDs only where they prove the verdict; existing:<id> and same_provisional:<candidate> must cite at least one unit, and the owner's name/facts (existing) or the target candidate's label literally (same_provisional) must appear in the units cited together with the mention's own unit. {episode_clause}Give honest confidence.\nOther proposed-new candidates: {'; '.join(f'{other['id']}={other['label']!r}' for other in live_provisional) or '(none)'}\nRegistered candidates: {'; '.join(f'{owner}={registry[owner].get('name', owner)!r}' for owner in live_owners) or '(none)'}\nCanonical prior facts:\n{owner_prior_facts(registry, live_owners, label)}\n{note}{scene_block}MENTIONS:\n"
                + "\n".join(rows)
            )
            return prompt, schema, ids, episode_units

        def review_chunk(
            chunk,
            options,
            note="",
            scene_chars=None,
            label=label,
            candidate=candidate,
            mentions=mentions,
            plausible=plausible,
        ):
            prompt, schema, ids, episode_units = build_review(chunk, options, note, scene_chars)
            if len(prompt) > PREPARATION_PROMPT_MAX_CHARS:
                raise CastDataIssue(f"new-identity review prompt exceeds native context budget for {label!r}")
            raw = ask(
                prompt,
                max_tokens=min(
                    NATIVE_RESERVED_OUTPUT_TOKENS, max(1500, 200 + REVIEW_OUTPUT_TOKENS_PER_MENTION * len(chunk))
                ),
                max_attempts=1,
                response_schema=schema,
            )
            value = model_object(raw, f"new-identity review for {label!r}")
            returned = value.get("mentions")
            if (
                not isinstance(returned, list)
                or len(returned) != len(ids)
                or {r.get("mention_id") for r in returned if isinstance(r, dict)} != set(ids)
                or not all({"verdict", "witness_unit_ids"} <= set(r) for r in returned)
            ):
                raise CastDataIssue(f"new-identity review omitted or duplicated mentions for {label!r}")
            out = []
            for result in returned:
                scope = chunk[ids.index(result["mention_id"])]
                verdict, selected = result["verdict"], result["witness_unit_ids"]
                if verdict not in options or not isinstance(selected, list) or not set(selected) <= set(order):
                    raise CastDataIssue(f"new-identity review returned an invalid verdict for {label!r}")
                # The mention's own unit is deterministic input, never a model choice: bind it to the exact
                # immutable reference (label bytes at the recorded span) and attach it as a witness.
                own = mentions[scope]
                if own["quote"][scope[3] : scope[3] + len(scope[2])] != scope[2]:
                    raise CastDataIssue(
                        f"immutable candidate reference for {scope[2]!r} does not match its source span"
                    )
                witnesses = [own["id"], *dict.fromkeys(w for w in selected if w != own["id"])]
                error = review_verdict_error(
                    scope[2],
                    verdict,
                    witnesses,
                    candidate,
                    plausible,
                    registry,
                    aliases,
                    units_by_id,
                    own,
                    episode_units.get(scope),
                )
                episode_support = None
                if not error and verdict.startswith("existing:"):
                    # support found only in the mention's own reviewed episode keeps its exact unit as a recorded witness
                    _, prov = owner_support(
                        scope[2],
                        verdict.split(":", 1)[1],
                        [units_by_id[w] for w in witnesses],
                        episode_units.get(scope),
                        registry,
                        aliases,
                    )
                    if prov and prov["witness_unit_id"] not in witnesses:
                        episode_support = prov["witness_unit_id"]
                        witnesses = [*witnesses, episode_support]
                if (
                    not error
                    and verdict.startswith((SAME_PROVISIONAL, "existing:"))
                    and not selected
                    and episode_support is None
                    and not (verdict.startswith("existing:") and episode_units.get(scope))
                ):
                    error = f"new-identity review cites no support for {verdict}"
                result = {
                    **result,
                    "witness_unit_ids": witnesses,
                    "selected_witness_unit_ids": selected,
                    "episode_support_unit_id": episode_support,
                }
                out.append((scope, result, error))
            return out

        def record_result(scope, result, owners=owners, verdicts=verdicts):
            verdict, witnesses = result["verdict"], result["witness_unit_ids"]
            if result["confidence"] < ADJUDICATION_MIN_CONFIDENCE and verdict != "uncertain":
                verdict = "uncertain"
            record = {
                "chapter_sha256": scope[0],
                "quote_sha256": scope[1],
                "label": scope[2],
                "span_start": scope[3],
                "verdict": verdict,
                "witness_unit_ids": witnesses,
                "own_source_witness": {
                    "unit_id": witnesses[0],
                    "provenance": "immutable_candidate_reference",
                    "label": scope[2],
                    "span_start": scope[3],
                },
                "confidence": float(result["confidence"]),
                "reason": result["reason"],
                "owners": list(owners),
                "raw_review": {
                    "verdict": result["verdict"],
                    "witness_unit_ids": result["selected_witness_unit_ids"],
                    "confidence": result["confidence"],
                    "reason": result["reason"],
                },
            }
            if result.get("episode_support_unit_id"):
                record["episode_support_unit_id"] = result["episode_support_unit_id"]
            if result.get("pending_type"):
                record["pending_type"] = result["pending_type"]
                record["invalid_reason"] = result["invalid_reason"]
                record["raw_review"]["confidence"] = result["raw_confidence"]
            if verdict.startswith("existing:"):
                owner = verdict.split(":", 1)[1]
                witness_units = [units_by_id[w] for w in witnesses]
                _, prov = find_existing_owner_support(scope[2], owner, witness_units, registry, aliases)
                if prov:
                    record["provenance"] = prov
                    record["witness"] = prov["witness_unit_id"]
                    record["owner_profile_field"] = prov["owner_profile_field"]
                    if prov.get("anchor"):
                        record["anchor"] = prov["anchor"]
            prior = by_scope.get(scope)
            if prior:
                record["history"] = [
                    *prior.get("history", []),
                    {
                        key: prior[key]
                        for key in (
                            "verdict",
                            "confidence",
                            "reason",
                            "owners",
                            "provenance",
                            "witness",
                            "owner_profile_field",
                            "anchor",
                        )
                        if key in prior
                    },
                ]
                audit[:] = [r for r in audit if r is not prior]
            audit.append(record)
            by_scope[scope] = record
            verdicts[scope] = verdict

        def correct_mention(
            scope,
            removed,
            error,
            context_note="",
            scene_chars=None,
            label=label,
            base_options=base_options,
            review_chunk=review_chunk,
        ):
            """Bounded correction shared by the ordinary and the consistency review: re-ask this one mention with every invalid target removed from the options (never supply aliases by hand); still invalid after the bounded reasks fails closed."""
            result = None
            first_error = error
            for _ in range(NEW_IDENTITY_REASKS):
                options = [option for option in base_options if option not in removed]
                ((_, result, error),) = review_chunk(
                    [scope],
                    options,
                    f"{context_note}A previous answer for this exact mention was invalid ({error}); the mention's own unit is already attached.\n",
                    scene_chars,
                )
                if not error:
                    # a model that withdraws to uncertain after an invalid target leaves a typed pending scope, not an anonymous one
                    pending = pending_review_result(result, removed, first_error) if deferred is not None else None
                    return pending if pending and result["verdict"] == "uncertain" else result
                removed.add(result["verdict"])
            # never choose a target by hand: the invalid verdict is kept as raw evidence and the scope becomes pending
            pending = pending_review_result(result, removed, error) if deferred is not None else None
            if pending is None:
                raise CastDataIssue(f"{error} for {label!r} after bounded correction")
            return {**pending, "confidence": 0.0}

        for offset in range(0, len(todo), NEW_IDENTITY_MENTIONS_PER_CALL):
            chunk = todo[offset : offset + NEW_IDENTITY_MENTIONS_PER_CALL]
            invalid: list[tuple] = []
            for scope, result, error in review_chunk(chunk, base_options):
                if error:
                    invalid.append((scope, {result["verdict"]}, error))
                else:
                    record_result(scope, result)
            for scope, removed, error in invalid:
                record_result(scope, correct_mention(scope, removed, error))
            atomic_json(project / NEW_IDENTITY_AUDIT_NAME, {"records": audit})

        # Bounded normalization of a per-scope conflict: one label whose exact scopes got different verdict kinds.
        # Source-validated existing / same_provisional scopes bind exactly (never collapsed onto each other); a
        # distinct verdict among them is an independent identity only if a single contextual re-review that
        # sees its siblings' verdicts reaffirms it with valid witnesses. Anything else fails closed.
        def verdict_kind(verdict):
            return (
                "existing"
                if verdict.startswith("existing:")
                else "same_provisional"
                if verdict.startswith(SAME_PROVISIONAL)
                else verdict
            )

        kinds = {verdict_kind(v) for v in verdicts.values()}
        if len(kinds - {FRAGMENT_VERDICT}) > 1 and "uncertain" not in kinds:
            recheck = [
                scope
                for scope, v in verdicts.items()
                if v != FRAGMENT_VERDICT and not by_scope[scope].get("reconciled")
            ]
            if recheck:
                summary = "; ".join(
                    f"unit {mentions[scope]['id']} char {scope[3]}: {verdicts[scope]}" for scope in recheck
                )
                note = f"CONSISTENCY REVIEW of conflicting verdicts for this one label across its exact mentions ({summary}). Decide every mention below once, on its own quote, direct owner anchors and continuous scene, never by majority and never by global alias. A mention may be existing:<id> or same_provisional only if its own scene proves that individual; distinct_living_identity only if its own scene proves a different living individual from every other verdict here. Genuine contextual splits between different people sharing a label are allowed when each scope is separately proven; otherwise choose uncertain.\n"
                output_cap = max(1, (NATIVE_RESERVED_OUTPUT_TOKENS - 200) // REVIEW_OUTPUT_TOKENS_PER_MENTION)

                def fits(group, width, output_cap=output_cap, base_options=base_options, note=note):
                    return (
                        len(group) <= output_cap
                        and len(build_review(group, base_options, note, width)[0]) <= PREPARATION_PROMPT_MAX_CHARS
                    )

                # One logical review: all conflicts in one call over compacted continuous episodes, narrowing every
                # scene uniformly until it fits. Only if even own-unit-only evidence (or the output budget) cannot hold
                # every scope are scopes split across calls; each call still carries the full conflict summary and
                # every scope is decided exactly once.
                width = next((w for w in CONSISTENCY_SCENE_WIDTHS if fits(recheck, w)), None)
                groups: list[list] = [recheck]
                if width is None:
                    width = ADJUDICATION_SCENE_CHARS
                    groups, current = [], []
                    for scope in recheck:
                        if current and not fits([*current, scope], width):
                            groups.append(current)
                            current = []
                        current.append(scope)
                    groups.append(current)
                for group in groups:
                    for scope, result, error in review_chunk(group, base_options, note, width):
                        if error:
                            # an invalid verdict (e.g. same_provisional without literal dual-label witnesses) gets the same bounded correction as an ordinary review, keeping the consistency context
                            result = correct_mention(scope, {result["verdict"]}, error, note, width)
                        record_result(scope, result)
                        by_scope[scope]["reconciled"] = True
                atomic_json(project / NEW_IDENTITY_AUDIT_NAME, {"records": audit})
        if "uncertain" in verdicts.values():
            if deferred is None:
                raise CastDataIssue(
                    f"proposed new identity {label!r} is uncertain for at least one mention: {sorted(set(verdicts.values()))}"
                )
            kinds = {
                by_scope[scope].get("pending_type") for scope, verdict in verdicts.items() if verdict == "uncertain"
            }
            deferred.append(
                pending_identity_entry(
                    candidate,
                    next(
                        (kind for kind in ("unapproved_alias", "garbled_variant") if kind in kinds), "uncertain_living"
                    ),
                    f"proposed new identity is uncertain for at least one mention: {sorted(set(verdicts.values()))}",
                    units,
                    references,
                    by_scope,
                )
            )
            continue
        final_verdicts[candidate["id"]] = set(verdicts.values())
        for scope in mentions:
            verdict = verdicts[scope]
            if verdict == DISTINCT_VERDICT or verdict.startswith(SAME_PROVISIONAL):
                continue
            record = by_scope[scope]
            existing = verdict.startswith("existing:")
            scoped = {
                "chapter_sha256": scope[0],
                "quote_sha256": scope[1],
                "label": scope[2],
                "span_start": scope[3],
                "canonical": verdict.split(":", 1)[1] if existing else "none",
                "decision": "alias" if existing else "non_character",
                "confidence": record["confidence"],
                "reason": f"[new-identity review {record['witness_unit_ids']}] {record['reason']}"[:300],
                "owners": list(owners),
                "raw_adjudication": {
                    "source": "new_identity_review",
                    "verdict": record["raw_review"]["verdict"],
                    "confidence": record["raw_review"]["confidence"],
                    "reason": record["raw_review"]["reason"],
                },
            }
            if "provenance" in record:
                scoped["provenance"] = record["provenance"]
                if "anchor" in record:
                    scoped["anchor"] = record["anchor"]
                if "owner_profile_field" in record:
                    scoped["owner_profile_field"] = record["owner_profile_field"]
            prior = next(
                (
                    r
                    for r in scoped_records
                    if (r["chapter_sha256"], r["quote_sha256"], r["label"], r["span_start"]) == scope
                ),
                None,
            )
            if prior:
                scoped["history"] = [
                    *prior.get("history", []),
                    {
                        key: prior[key]
                        for key in ("decision", "canonical", "confidence", "reason", "owners")
                        if key in prior
                    },
                ]
                scoped_records[:] = [r for r in scoped_records if r is not prior]
            scoped_records.append(scoped)
            wrote_scoped = True
        mention_scoped_audit_index(scoped_records)
        atomic_json(project / SCOPED_AUDIT_NAME, {"records": scoped_records})
    link_provisional_identities(
        proposed, classifications, final_verdicts, by_scope, references, units_by_id, units, deferred
    )
    return wrote_scoped


def resolve_provisional_chains_before_materialize(records: list[dict], candidates: list[dict], registry: dict) -> None:
    """Resolve candidate chains before materialization. If a candidate targets another candidate that resolved to an established registry owner, re-route to that canonical owner; if it targets a candidate by normalized label, map to the candidate ID; if the target is invalid or not a valid discovery, fail closed."""
    candidate_by_id = {c["id"]: c for c in candidates}
    candidate_by_label_id = {normalized_id(c["label"]): c["id"] for c in candidates}
    record_by_id = {item["candidate_id"]: item for item in records}
    for item in records:
        if item.get("status") != "known":
            continue
        identity = item.get("identity")
        if identity in registry:
            continue
        if identity not in candidate_by_id and identity in candidate_by_label_id:
            identity = candidate_by_label_id[identity]
            item["identity"] = identity
        if identity in candidate_by_id:
            target_record = record_by_id.get(identity)
            if target_record is None:
                raise CastDataIssue(f"candidate {item['candidate_id']!r} targets unrecorded candidate {identity!r}")
            if target_record["status"] == "known" and target_record["identity"] in registry:
                item["identity"] = target_record["identity"]
            elif target_record["status"] == "new":
                pass
            else:
                raise CastDataIssue(
                    f"candidate {item['candidate_id']!r} targets candidate {identity!r} which is not a living discovery ({target_record['status']})"
                )
        else:
            raise CastDataIssue(f"known classification has unknown identity target: {identity!r}")


def link_provisional_identities(
    proposed: list[dict],
    classifications: list[dict],
    final_verdicts: dict[str, set[str]],
    by_scope: dict,
    references: dict,
    units_by_id: dict,
    units: list[dict] | None = None,
    deferred: list[dict] | None = None,
) -> None:
    """Turn same_provisional verdicts into known->new links. The target must be independently introduced (its own mentions are reviewed distinct_living_identity), never itself a link (no chains or cycles), and every surviving mention of the linked label must agree.
    A link that cannot stand is a garbled_variant pending entry when a deferred list is given (candidates already deferred are skipped), otherwise it fails closed."""
    record_by_id = {item["candidate_id"]: item for item in classifications}

    def refuse(candidate: dict, message: str) -> None:
        if deferred is None:
            raise CastDataIssue(message)
        deferred.append(pending_identity_entry(candidate, "garbled_variant", message, units, references, by_scope))

    for candidate in proposed:
        if candidate["id"] not in final_verdicts:
            continue
        links = {verdict for verdict in final_verdicts[candidate["id"]] if verdict.startswith(SAME_PROVISIONAL)}
        if not links:
            continue
        if len(links) > 1 or DISTINCT_VERDICT in final_verdicts[candidate["id"]]:
            refuse(
                candidate,
                f"proposed new identity {candidate['label']!r} has conflicting provisional verdicts: {sorted(final_verdicts[candidate['id']])}",
            )
            continue
        target_id = next(iter(links))[len(SAME_PROVISIONAL) :]
        target_verdicts = final_verdicts.get(target_id, set())
        if (
            DISTINCT_VERDICT not in target_verdicts
            or any(verdict.startswith(SAME_PROVISIONAL) for verdict in target_verdicts)
            or any(verdict.startswith("existing:") for verdict in target_verdicts)
        ):
            refuse(
                candidate,
                f"provisional target {target_id!r} for {candidate['label']!r} is not an independently introduced identity",
            )
            continue
        candidate_scopes = {
            mention_scope(units_by_id[references[ref_id]["unit_id"]], references[ref_id])
            for ref_id in candidate["ref_ids"]
        }
        witnesses = sorted(
            {
                w
                for scope in candidate_scopes
                if by_scope[scope]["verdict"] == next(iter(links))
                for w in by_scope[scope]["witness_unit_ids"]
            }
        )
        item = record_by_id[candidate["id"]]
        item["status"], item["identity"], item["evidence_unit_ids"] = ("known", target_id, witnesses)
        candidate["audited_target"] = target_id


# ##################################################################
# scoped native adjudication
# decides each pending mention separately (exact chapter/quote scope, never a global label alias), and persists the interpretive record before any mapping uses it.
SCOPED_AUDIT_NAME = "mention-scoped-audit.json"
ADJUDICATION_MENTIONS_PER_CALL = 12
ADJUDICATION_MAX_ROUNDS = 4
ADJUDICATION_MIN_CONFIDENCE = 0.7
ADJUDICATION_SCENE_CHARS = 2400
CONSISTENCY_SCENE_WIDTHS = (ADJUDICATION_SCENE_CHARS, 1200, 600, 300, 0)
REVIEW_OUTPUT_TOKENS_PER_MENTION = 140
ADJUDICATION_OWNER_FACT_CHARS = 700
ADJUDICATION_FACTS_TOTAL_CHARS = 6000


def bounded_scene_range(
    units_by_id: dict, order: list[str], position: int, chars: int = ADJUDICATION_SCENE_CHARS
) -> tuple[int, int]:
    """Inclusive order positions of the contiguous same-chapter source around one mention, grown alternately both ways within a character budget; the mention's own unit is always whole."""
    chapter = units_by_id[order[position]]["chapter_sha256"]
    low = high = position
    used = len(units_by_id[order[position]]["quote"])
    grew = True
    while grew:
        grew = False
        for step in (-1, 1):
            edge = (low if step < 0 else high) + step
            if 0 <= edge < len(order) and units_by_id[order[edge]]["chapter_sha256"] == chapter:
                size = len(units_by_id[order[edge]]["quote"]) + 1
                if used + size <= chars:
                    used += size
                    low, high = (edge, high) if step < 0 else (low, edge)
                    grew = True
    return low, high


def bounded_scene(units_by_id: dict, order: list[str], position: int, with_ids: bool = False) -> str:
    """Contiguous same-chapter source around one mention, grown alternately both ways within a fixed character budget; the mention's own unit is always whole."""
    low, high = bounded_scene_range(units_by_id, order, position)
    return " ".join(
        (f"[{order[index]}] " if with_ids else "") + units_by_id[order[index]]["quote"]
        for index in range(low, high + 1)
    )


def scene_episode_ranges(
    units_by_id: dict, order: list[str], positions: list[int], chars: int
) -> tuple[list[tuple[int, int]], list[int]]:
    """Merge every position's bounded scene into maximal continuous same-chapter episodes (inclusive order ranges). Returns the ranges and, per input position, its episode index (no position is dropped)."""
    ranges = [bounded_scene_range(units_by_id, order, position, chars) for position in positions]
    merged: list[list[int]] = []
    for low, high in sorted(set(ranges)):
        if (
            merged
            and low <= merged[-1][1] + 1
            and units_by_id[order[low]]["chapter_sha256"] == units_by_id[order[merged[-1][0]]]["chapter_sha256"]
        ):
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return [(low, high) for low, high in merged], [
        next(i for i, (low, high) in enumerate(merged) if low <= r_low and r_high <= high) for r_low, r_high in ranges
    ]


def scene_episodes(
    units_by_id: dict, order: list[str], positions: list[int], chars: int
) -> tuple[list[str], list[int]]:
    """Episode texts (each unit rendered once with its ID) and, per input position, its episode index."""
    merged, episode_of = scene_episode_ranges(units_by_id, order, positions, chars)
    return [
        " ".join(f"[{order[index]}] {units_by_id[order[index]]['quote']}" for index in range(low, high + 1))
        for low, high in merged
    ], episode_of


def bounded_scene_with_ids(units_by_id: dict, order: list[str], position: int) -> str:
    return bounded_scene(units_by_id, order, position, True)


def owner_prior_facts(registry: dict, owners: list[str], label: str | None = None) -> str:
    """Canonical prior profile and source-derived facts for each possible owner, each bounded, and the whole block bounded.
    When a label is given, a short excerpt of the owner's own prior facts around that label leads the entry so it survives the per-owner bound."""
    lines: list[str] = []
    remaining = ADJUDICATION_FACTS_TOTAL_CHARS
    per_owner = min(ADJUDICATION_OWNER_FACT_CHARS, ADJUDICATION_FACTS_TOTAL_CHARS // max(len(owners), 1))
    for owner in owners:
        entry = registry[owner]
        facts = entry.get("facts") if isinstance(entry.get("facts"), dict) else {}
        parts = [f"name={entry.get('name', owner)!r}"]
        flat = json.dumps(entry, ensure_ascii=False)
        hit = flat.casefold().find(label.casefold()) if label else -1
        if hit >= 0:
            parts.append(
                f"prior facts mentioning {label!r}: ...{flat[max(0, hit - ADJUDICATION_SNIPPET_CHARS) : hit + len(label) + ADJUDICATION_SNIPPET_CHARS]}..."
            )
        for field in ("bio", "look"):
            if entry.get(field):
                parts.append(f"{field}={str(entry[field])!r}")
        for field in ("voice", "look"):
            if facts.get(field):
                parts.append(f"prior {field} facts={' | '.join(str(item) for item in facts[field])!r}")
        text = f"{owner}: " + "; ".join(parts)
        text = text[: min(per_owner, max(remaining, 0))]
        if text:
            lines.append(text)
            remaining -= len(text)
    return "\n".join(lines) or "(none)"


def load_scoped_audit(project: Path) -> list[dict]:
    path = project / SCOPED_AUDIT_NAME
    if not path.is_file():
        return []
    records = load_object(path, "mention-scoped audit").get("records")
    mention_scoped_audit_index(records)
    return records


SCOPED_REFERENCE_FIELDS = ("chapter_sha256", "quote_sha256", "label", "span_start", "canonical")


def scoped_audit_attestation(records: list[dict]) -> dict:
    """Order-independent digest binding the exact validated set of mention-scoped audit records to the freeze."""
    ordered = sorted(
        mention_scoped_audit_index(records).values(),
        key=lambda r: (r["chapter_sha256"], r["quote_sha256"], r["span_start"], r["label"]),
    )
    return {"version": 1, "count": len(ordered), "sha256": json_digest(ordered)}


def supersede_scoped_record(records: list[dict], known: dict, scope: tuple, record: dict) -> None:
    """Install the record for one exact mention, keeping every superseded decision and its reason as append-only history."""
    prior = known.get(scope)
    if prior:
        record["history"] = [
            *prior.get("history", []),
            {key: prior[key] for key in ("decision", "canonical", "confidence", "reason", "owners") if key in prior},
        ]
        records[:] = [item for item in records if item is not prior]
    records.append(record)
    known[scope] = record


def adjudicate_pending_mentions(
    project: Path,
    pending: list[dict],
    units: list[dict],
    registry: dict,
    ask,
    aliases: dict | None = None,
    *,
    binding_progress_only: bool = False,
) -> int | bool:
    """Adjudicate every pending mention. Normal callers receive the audit-write count; a classifier retry asks only whether a new source-proven alias/non-character binding was produced, because an ambiguous record is terminal pending evidence rather than new classification context."""
    written = binding_written = 0
    references = immutable_name_references(units)
    units_by_id = {unit["id"]: unit for unit in units}
    order = [unit["id"] for unit in units]
    records = load_scoped_audit(project)
    known = mention_scoped_audit_index(records)
    seen_candidates: set[str] = set()
    for entry in pending:
        candidate = entry["candidate"]
        if candidate["id"] in seen_candidates:
            continue
        seen_candidates.add(candidate["id"])
        owners = adjudication_owners(candidate["label"], registry, aliases, entry["proposed"])
        mentions: dict[tuple[str, str, str, int], dict] = {}
        for ref_id in candidate["ref_ids"]:
            unit = units_by_id[references[ref_id]["unit_id"]]
            scope = mention_scope(unit, references[ref_id], candidate["label"])
            prior = known.get(scope)
            if prior and prior["decision"] == "alias":
                _, why = scoped_alias_proof(
                    candidate["label"],
                    prior["canonical"],
                    unit,
                    scene_units_at(units_by_id, order, order.index(unit["id"])),
                    registry,
                    aliases or {},
                )
                if why:
                    # no model call can add proof the source lacks: the legacy alias is withdrawn mechanically and the mention stays unresolved
                    supersede_scoped_record(
                        records,
                        known,
                        scope,
                        {
                            "chapter_sha256": scope[0],
                            "quote_sha256": scope[1],
                            "label": candidate["label"],
                            "span_start": scope[3],
                            "canonical": "none",
                            "decision": "ambiguous",
                            "confidence": float(prior["confidence"]),
                            "reason": f"[alias proof rejected: {why}] {prior['reason']}"[:300],
                            "owners": list(owners),
                            "raw_adjudication": prior.get("raw_adjudication")
                            or {"decision": "alias", "canonical": prior["canonical"], "reason": prior["reason"]},
                        },
                    )
                    written += 1
                    mention_scoped_audit_index(records)
                    atomic_json(project / SCOPED_AUDIT_NAME, {"records": records})
                    continue
            if scope not in known or readjudication_due(known[scope], owners):
                mentions.setdefault(scope, unit)
        scopes = list(mentions)
        for offset in range(0, len(scopes), ADJUDICATION_MENTIONS_PER_CALL):
            chunk = scopes[offset : offset + ADJUDICATION_MENTIONS_PER_CALL]
            ids = [f"m{index}" for index in range(len(chunk))]
            decisions = (
                ["alias", "non_character", "ambiguous"] if owners else ["non_character", "ambiguous", "owner_absent"]
            )
            item = {
                "type": "object",
                "properties": {
                    "mention_id": {"type": "string", "enum": ids},
                    "refers_to_person": {"type": "string", "enum": ["yes", "no", "unclear"]},
                    "candidate_kind": {
                        "type": "string",
                        "enum": [
                            "individual_name",
                            "specific_role",
                            "endearment",
                            "prose_fragment",
                            "nonliving",
                            "unclear",
                        ],
                    },
                    "decision": {"type": "string", "enum": decisions},
                    "canonical": {"type": "string", "enum": [*owners, "none"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string", "minLength": 1, "maxLength": 300},
                },
                "required": [
                    "mention_id",
                    "refers_to_person",
                    "candidate_kind",
                    "decision",
                    "canonical",
                    "confidence",
                    "reason",
                ],
                "additionalProperties": False,
                "allOf": [
                    {
                        "if": {"properties": {"decision": {"const": "non_character"}}},
                        "then": {
                            "properties": {"candidate_kind": {"enum": ["endearment", "prose_fragment", "nonliving"]}}
                        },
                    },
                    {
                        "if": {"properties": {"candidate_kind": {"enum": ["individual_name", "specific_role"]}}},
                        "then": {"properties": {"decision": {"enum": ["alias", "ambiguous"]}}},
                    },
                ],
            }
            schema = {
                "type": "object",
                "properties": {
                    "mentions": {"type": "array", "minItems": len(chunk), "maxItems": len(chunk), "items": item}
                },
                "required": ["mentions"],
                "additionalProperties": False,
            }
            roster = (
                "; ".join(f"{owner}={registry[owner].get('name', owner)!r}" for owner in owners)
                or "(no candidate owner)"
            )
            rows = []
            for mention_id, scope in zip(ids, chunk, strict=True):
                position = order.index(mentions[scope]["id"])
                rows.append(
                    f"{mention_id} [{mentions[scope]['chapter']}] mention: {mentions[scope]['quote']}\n   bounded scene: {bounded_scene(units_by_id, order, position)}"
                )
                if scope in known:
                    prior = known[scope]
                    rows.append(
                        f"   previous decision (made when only {prior.get('owners', 'unrecorded owners')} were offered): {prior['decision']} confidence={prior['confidence']} because: {prior['reason']}"
                    )
            prompt = (
                f"Decide separately, for each exact mention of the label {candidate['label']!r}, whether THIS exact span refers to an existing character. Do not expand it with an adjacent capitalized narrative subject followed by a speech/action verb; for a title vocative, use direct response and scene continuity to identify its addressee. Never decide by spelling or sound similarity, and never generalise from one mention to another. alias requires the mention or its bounded scene to prove identity with the named owner, consistent with that owner's canonical prior facts; canonical must be that owner. The possible owners are only candidates (a shared name component, or a bounded roster of the whole cast when the label matches no name): a kinship or role label such as Mom or Dad may be a vocative or reference for a cast member whose prior facts say they are that person's parent, and a full name may extend a shorter canonical name; accept such a link only when the scene and prior facts support it. A previous decision is shown only where one exists; reconsider it with the fuller owner information. First answer refers_to_person for this mention: yes when it names or addresses a person/creature in the story (a vocative such as Mom, or a full name, counts), no only when it is a place, object, group, title word or other non-living thing, unclear when you cannot tell. Also classify candidate_kind: individual_name is a stable actor identity; specific_role can map only when the scene identifies its owner; endearment and prose_fragment address/describes a person but are not actor names; nonliving is not a person. Then decide: alias only for individual_name or source-owned specific_role with a supported owner. non_character is correct for nonliving, endearment, or prose_fragment even when refers_to_person=yes. ambiguous only when a potential individual_name/specific_role owner cannot be decided, which includes a named living person or creature that none of the possible owners is shown to be: never force an owner onto it, and never call it non_character merely because no owner fits. Give honest confidence and a short source-based reason.\nPossible owners: {roster}\nCanonical prior facts for the possible owners:\n{owner_prior_facts(registry, owners, candidate['label'])}\nMENTIONS:\n"
                + "\n".join(rows)
            )
            if len(prompt) > PREPARATION_PROMPT_MAX_CHARS:
                raise CastDataIssue(
                    f"scoped adjudication prompt exceeds native context budget for {candidate['label']!r}"
                )
            value = model_object(
                ask(prompt, max_tokens=1500, max_attempts=1, response_schema=schema),
                f"scoped adjudication for {candidate['label']!r}",
            )
            returned = value.get("mentions")
            if (
                not isinstance(returned, list)
                or {r.get("mention_id") for r in returned if isinstance(r, dict)} != set(ids)
                or len(returned) != len(ids)
                or not all(
                    {"decision", "canonical", "refers_to_person", "reason", "candidate_kind"} <= set(r)
                    for r in returned
                )
            ):
                raise CastDataIssue(f"scoped adjudication omitted or duplicated mentions for {candidate['label']!r}")
            for result in returned:
                scope = chunk[ids.index(result["mention_id"])]
                decision, canonical = result["decision"], result["canonical"]
                person = result["refers_to_person"]
                reason = str(result["reason"])
                kind = result["candidate_kind"]
                raw_decision = decision
                if decision == "owner_absent":
                    decision, canonical = "ambiguous", "none"
                if decision == "alias" and (
                    canonical not in owners
                    or result["confidence"] < ADJUDICATION_MIN_CONFIDENCE
                    or person != "yes"
                    or kind not in {"individual_name", "specific_role"}
                    or (kind == "specific_role" and not (label_components(candidate["label"]) - TITLE_ROLE_TOKENS))
                    or (
                        raw_decision == "non_character"
                        and kind != "individual_name"
                        and not (kind == "specific_role" and (label_components(candidate["label"]) - TITLE_ROLE_TOKENS))
                    )
                ):
                    decision, canonical = "ambiguous", "none"
                elif decision in {"non_character", "ambiguous"} and person == "yes" and result["canonical"] in owners:
                    relationship_schema = {
                        "type": "object",
                        "properties": {
                            "relationship": {
                                "type": "string",
                                "enum": ["same_owner", "distinct", "not_identity", "unclear"],
                            },
                            "witness_unit_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 3,
                                "uniqueItems": True,
                                "items": {"type": "string", "enum": [unit["id"] for unit in units]},
                            },
                            "reason": {"type": "string", "minLength": 1, "maxLength": 300},
                        },
                        "required": ["relationship", "witness_unit_ids", "reason"],
                        "additionalProperties": False,
                    }
                    relationship_prompt = f"For this one exact mention only, determine its relationship to proposed canonical {result['canonical']!r}: same_owner only with source scene continuity; distinct/not_identity/unclear otherwise. Do not infer from spelling. Cite witness IDs. Mention [{mentions[scope]['id']}]: {mentions[scope]['quote']!r}. Bounded scene: {bounded_scene(units_by_id, order, order.index(mentions[scope]['id']))}. Raw rationale: {reason!r}"
                    relationship = model_object(
                        ask(relationship_prompt, max_tokens=500, max_attempts=1, response_schema=relationship_schema),
                        "relationship review",
                        ("relationship", "witness_unit_ids", "reason"),
                    )
                    if relationship["relationship"] == "same_owner":
                        decision, canonical, reason = (
                            "alias",
                            result["canonical"],
                            f"[relationship review {relationship['witness_unit_ids']}] {relationship['reason']}"[:300],
                        )
                    else:
                        decision, canonical, reason = (
                            "ambiguous",
                            "none",
                            f"[relationship review={relationship['relationship']}] {relationship['reason']}"[:300],
                        )
                elif decision == "non_character" and person != "no" and kind not in {"endearment", "prose_fragment"}:
                    # One bounded native tiebreak distinguishes a bad coupled transport from a genuinely unresolved identity.
                    # It sees only this immutable mention/scene and the raw rationale; it cannot create a global alias.
                    review_schema = {
                        "type": "object",
                        "properties": {
                            "semantic_type": {
                                "type": "string",
                                "enum": [
                                    "individual_identity",
                                    "actor_reference",
                                    "endearment",
                                    "prose_fragment",
                                    "nonliving",
                                    "unclear",
                                ],
                            },
                            "witness_unit_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 3,
                                "uniqueItems": True,
                                "items": {"type": "string", "enum": [unit["id"] for unit in units]},
                            },
                            "reason": {"type": "string", "minLength": 1, "maxLength": 300},
                        },
                        "required": ["semantic_type", "witness_unit_ids", "reason"],
                        "additionalProperties": False,
                    }
                    review_prompt = f"Classify ONE exact source span using exactly one semantic_type and cite immutable witness IDs. individual_identity is a stable actor name; actor_reference is a known person's name/reference but not a new name; endearment/prose_fragment/nonliving are not stable actor identities. Do not output a decision, canonical ID, or person flag; the caller derives those mechanically. Raw inconsistent result: person={person!r}, kind={kind!r}, decision={decision!r}, reason={reason!r}. Mention [{mentions[scope]['id']}]: {mentions[scope]['quote']!r}. Bounded scene: {bounded_scene(units_by_id, order, order.index(mentions[scope]['id']))}"
                    reviewed = model_object(
                        ask(review_prompt, max_tokens=500, max_attempts=1, response_schema=review_schema),
                        "semantic-type review",
                        ("semantic_type", "witness_unit_ids", "reason"),
                    )
                    review_type, review_reason = (reviewed["semantic_type"], reviewed["reason"])
                    if review_type in {"endearment", "prose_fragment", "nonliving"}:
                        kind, decision, canonical, reason = (
                            review_type,
                            "non_character",
                            "none",
                            f"[semantic-type review {reviewed['witness_unit_ids']}] {review_reason}"[:300],
                        )
                    else:
                        decision, canonical, reason = (
                            "ambiguous",
                            "none",
                            f"[inconsistent non_character with refers_to_person={person}; semantic-type review={review_type}] {review_reason}"[
                                :300
                            ],
                        )
                proof = None
                if decision == "alias":
                    mention_unit = mentions[scope]
                    proof, why = scoped_alias_proof(
                        candidate["label"],
                        canonical,
                        mention_unit,
                        scene_units_at(units_by_id, order, order.index(mention_unit["id"])),
                        registry,
                        aliases or {},
                    )
                    if why:
                        decision, canonical, reason = (
                            "ambiguous",
                            "none",
                            f"[alias proof rejected: {why}] {reason}"[:300],
                        )
                if decision != "alias":
                    canonical = "none"
                record = {
                    "chapter_sha256": scope[0],
                    "quote_sha256": scope[1],
                    "label": candidate["label"],
                    "span_start": scope[3],
                    "canonical": canonical,
                    "decision": decision,
                    "confidence": float(result["confidence"]),
                    "reason": reason,
                    "owners": list(owners),
                    **({"proof": proof} if proof else {}),
                    "raw_adjudication": {
                        "decision": result["decision"],
                        "canonical": result["canonical"],
                        "refers_to_person": person,
                        "candidate_kind": kind,
                        "confidence": result["confidence"],
                        "reason": result["reason"],
                    },
                }
                supersede_scoped_record(records, known, scope, record)
                written += 1
                if decision in {"alias", "non_character"}:
                    binding_written += 1
            mention_scoped_audit_index(records)
            atomic_json(project / SCOPED_AUDIT_NAME, {"records": records})
    return bool(binding_written) if binding_progress_only else written


# ##################################################################
# discover batch
# executes bounded schema calls for ledger chunks, then materializes their composed complete classification once so omissions cannot be hidden between calls.
def discover_batch(
    project: Path,
    start: int,
    batch: list[Path],
    batch_units: list[dict[str, str]],
    batch_text: str,
    progress: dict,
    ambiguous: set[str],
    prompt: str | None = None,
    ask=None,
) -> tuple[list[dict], list[dict]]:
    del ambiguous, prompt
    ask = memoized_model_ask(ask or ask_sync)
    for review_round in range(2):
        records, candidates, rejected_rows = collect_classifications(
            project, start, batch, batch_units, batch_text, progress, ask
        )
        # The proposed-new review runs on the raw classification BEFORE materialization. A span it cannot resolve joins
        # the typed pending lane (never an owner, alias or fatal error); it is recorded below with literal evidence.
        deferred: list[dict] = []
        if not review_proposed_identities(project, batch_units, records, candidates, progress, ask, deferred):
            break
        if review_round == 1:
            raise CastDataIssue("proposed-new identity review did not converge")
    # A response row with bad citation/cardinality data is preserved exactly once in the raw
    # rejection archive and held pending per candidate. It must not consume a whole-chunk
    # repair or make valid neighbours re-run classification/adjudication.
    references = immutable_name_references(batch_units)
    for rejected in rejected_rows:
        entry = pending_identity_entry(
            rejected["candidate"],
            "invalid_classification",
            rejected["reason"],
            batch_units,
            references,
        )
        entry["rejection_archive"] = rejected["archive"]
        deferred.append(entry)
    defer_ambiguous_candidates(project, records, candidates, batch_units, deferred)
    demote_deferred(records, candidates, batch_units, deferred)
    resolve_provisional_chains_before_materialize(records, candidates, progress["registry"])
    discoveries, classifications = materialize_classifications(
        {"classifications": records}, batch_units, candidates, progress["registry"], progress["aliases"]
    )
    verifier_warnings: list[dict] = []
    approved = verify_proposed_living_entities(batch_units, discoveries, ask, verifier_warnings)
    proposals = {item["id"]: item for item in discoveries}
    pending_rows = [
        {
            "id": warning["id"],
            "code": warning["code"],
            "message": f"proposed identity {warning['id']} is pending",
            "evidence": warning,
            "proposal": proposals.get(warning["id"]),
        }
        for warning in verifier_warnings
    ] + [
        {
            "id": entry["id"],
            "code": f"{PENDING_IDENTITY_CODE_PREFIX}{entry['type']}",
            "message": f"{entry['type']} review of {entry['label']!r} is pending",
            "evidence": {key: entry[key] for key in ("type", "label", "reason", "mentions")},
            "proposal": entry,
        }
        for entry in deferred
    ]
    if pending_rows:
        recovery = RecoveryLedger(project)
        scope_names = [path.name for path in batch]
        scope_hashes = {path.name: file_digest(path) for path in batch}
        for row in pending_rows:
            proposal_sha = save_pending_proposal(project, row["proposal"]) if row["proposal"] is not None else None
            # Pending: kept out of registry/aliases (demoted), exact raw evidence persisted at the chapter checkpoint.
            recovery.record(
                "cast",
                f"chapters {start}-{start + len(batch) - 1}:{row['id']}",
                row["code"],
                row["message"],
                severity="pending",
                evidence={
                    **row["evidence"],
                    "source": scope_names,
                    "source_hash": scope_hashes,
                    **({"proposal_sha256": proposal_sha} if proposal_sha else {}),
                },
                checkpoint={"start_chapter": start, "next_chapter": start + len(batch)},
            )
    rejected = {item["id"] for item in discoveries} - approved
    if rejected:
        for record in classifications:
            candidate = next(item for item in candidates if item["id"] == record["candidate_id"])
            candidate_id = normalized_id(candidate["label"])
            if candidate_id in rejected or record["identity"] in {
                next(item["id"] for item in discoveries if item["id"] == rejected_id) for rejected_id in rejected
            }:
                record["status"], record["identity"] = "non_character", "none"
        discoveries = [item for item in discoveries if item["id"] in approved]
    return discoveries, classifications


def collect_classifications(
    project: Path,
    start: int,
    batch: list[Path],
    batch_units: list[dict[str, str]],
    batch_text: str,
    progress: dict,
    ask,
) -> tuple[list[dict], list[dict], list[dict]]:
    rejected_rows: list[dict] = []
    decision_cache: dict[str, dict] = {}
    for adjudication_round in range(ADJUDICATION_MAX_ROUNDS):
        candidates = candidate_coverage_ledger(
            batch_units, progress["registry"], progress["aliases"], load_scoped_audit(project)
        )
        if not candidates:
            return [], [], rejected_rows
        pending: list[dict] = []
        # Binding mention-scoped decisions are already source-validated exact references.
        # Materialize them deterministically; asking the provider to reclassify them can
        # overwrite a valid Lou→Lu scope with an unrelated lexical candidate target.
        bound = [
            candidate
            for candidate in candidates
            if candidate.get("scoped_audit", {}).get("decision") in {"alias", "non_character"}
            and not candidate.get("scoped_stale")
        ]
        records: list[dict] = [
            {
                "candidate_id": candidate["id"],
                "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]],
                "status": "known" if candidate["scoped_audit"]["decision"] == "alias" else "non_character",
                "identity": candidate["scoped_audit"]["canonical"]
                if candidate["scoped_audit"]["decision"] == "alias"
                else "none",
            }
            for candidate in bound
        ]
        unresolved = [candidate for candidate in candidates if candidate not in bound]
        allow_new = True
        variant_targets = source_audited_variant_targets(project, candidates, batch_text, progress["registry"])
        for candidate in candidates:
            candidate["audited_target"] = variant_targets.get(candidate["id"])
        audited_context = source_audited_variant_context(project, candidates)
        for chunk_index, chunk in enumerate(classification_chunks(unresolved)):
            chunk_prompt = discovery_prompt(
                batch, progress["registry"], progress["aliases"], chunk, candidates, allow_new, batch_units
            )
            if audited_context:
                chunk_prompt += "\n\n" + audited_context
            schema = discovery_schema(list(progress["registry"]), chunk, candidates, allow_new)
            offer_key, provenance = classification_offer_fingerprint(
                chunk, candidates, progress["registry"], progress["aliases"], batch_units
            )
            cached = decision_cache.get(offer_key)
            restored = restore_cached_model_records(cached["records"], provenance) if cached else None
            response = (
                json.dumps({"classifications": restored})
                if restored is not None
                else ask(chunk_prompt, max_tokens=1800, max_attempts=1, response_schema=schema)
            )
            for attempt in range(EVIDENCE_REPAIR_ATTEMPTS + 1):
                try:
                    accepted, rejected = partition_classification_chunk(
                        json.loads(response),
                        chunk,
                        progress["registry"],
                        progress["aliases"],
                        candidates,
                        batch_units,
                        pending,
                    )
                    # Cache only fully accepted provider decisions. The cache is process-local and
                    # has no authority: every hit is remapped by immutable scope and revalidated.
                    if not rejected and restored is None:
                        decision_cache[offer_key] = {
                            "records": cache_model_records(accepted, provenance),
                            "provenance": {
                                key: value for key, value in provenance.items() if key != "candidate_id_to_scope"
                            },
                            "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                        }
                    records.extend(accepted)
                    if rejected:
                        reasons = "; ".join(f"{item['candidate']['id']}: {item['reason']}" for item in rejected)
                        archive = record_rejected_discovery(
                            project,
                            start,
                            batch,
                            batch_units,
                            response,
                            CastValidationError(
                                f"chunk {chunk_index} has rejected candidate rows: {reasons}",
                                code="cast_record_rejected",
                            ),
                            attempt,
                        )
                        rejected_rows.extend({**item, "archive": archive} for item in rejected)
                    break
                except (ValueError, json.JSONDecodeError) as error:
                    record_rejected_discovery(
                        project,
                        start,
                        batch,
                        batch_units,
                        response,
                        RuntimeError(f"chunk {chunk_index}: {error}"),
                        attempt,
                    )
                    if attempt == EVIDENCE_REPAIR_ATTEMPTS:
                        raise CastDataIssue(
                            f"cast semantic classification {start}-{start + len(batch) - 1} chunk {chunk_index} rejected after bounded repairs; archived evidence: {error}"
                        ) from error
                    response = ask(
                        f"Your chunk classification was rejected: {error}. Return the complete schema object for this chunk only.\n\n{chunk_prompt}",
                        max_tokens=1800,
                        max_attempts=1,
                        response_schema=schema,
                    )
        if not pending:
            break
        # A typed ambiguous/owner-absent record is terminal evidence for this source scope: it must be deferred below, not treated as fresh context that re-asks the whole classification chunk. Only a newly source-proven alias/non-character binding can make the next classification round materially different.
        if not adjudicate_pending_mentions(
            project,
            pending,
            batch_units,
            progress["registry"],
            ask,
            progress["aliases"],
            binding_progress_only=True,
        ):
            break
        if adjudication_round == ADJUDICATION_MAX_ROUNDS - 1:
            raise CastDataIssue(
                f"scoped adjudication did not converge within {ADJUDICATION_MAX_ROUNDS} classification rounds"
            )
    return records, candidates, rejected_rows


# ##################################################################
# context-safe source batch
# takes consecutive complete chapters only while leaving native Ollama enough context for its structured response; it never truncates or skips source.
def context_safe_batch(
    chapters: list[Path], start: int, registry: dict, aliases: dict[str, str]
) -> tuple[list[Path], str]:
    selected: list[Path] = []
    prompt = ""
    for chapter in chapters[start : start + BATCH_CHAPTERS]:
        candidate = [*selected, chapter]
        # Reserve the complete original chapter bytes as semantic context budget even though the wire prompt transports compact candidate witnesses.
        # The source-only lower bound is checked first so an oversized chapter never pays for prompt/ledger construction.
        try:
            source_bytes = sum(len(path.read_text(encoding="utf-8")) for path in candidate)
        except UnicodeDecodeError as error:
            # An undecodable chapter ends the batch before it; when it is first it is quarantined by the caller.
            if not selected:
                raise undecodable_issue(chapter, error) from error
            break
        candidate_prompt = (
            "" if source_bytes * 2 > PREPARATION_PROMPT_MAX_CHARS else discovery_prompt(candidate, registry, aliases)
        )
        conservative_size = len(candidate_prompt) + (source_bytes * 2)
        if conservative_size > PREPARATION_PROMPT_MAX_CHARS:
            if not selected:
                # Oversized single chapter: the caller classifies it through immutable-unit context windows.
                return [chapter], ""
            break
        selected, prompt = candidate, candidate_prompt
    if not selected:
        raise CastDataIssue("no source chapters fit the native model context")
    return selected, prompt


# ##################################################################
# source label present
# requires a whole source label rather than accepting a coincidental substring in a selected immutable sentence.
def source_label_present(label: str, units: list[dict[str, str]]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(label.strip())}(?!\w)", unit["quote"], re.IGNORECASE) for unit in units)


# ##################################################################
# record rejected discovery
# retains every complete native response and immutable unit citations so each original or repair failure is auditable without a lossy exception snippet.
def record_rejected_discovery(
    project: Path,
    start: int,
    batch: list[Path],
    units: list[dict[str, str]],
    response: str,
    error: Exception,
    attempt: int,
) -> dict:
    payload = {
        "start_chapter": start,
        "chapters": [path.name for path in batch],
        "attempt": attempt,
        "code": getattr(error, "code", "cast_evidence_rejected"),
        "error": str(error),
        "evidence_units": units,
        "response": response,
    }
    path = project / REJECTIONS_NAME
    line = json.dumps(payload, ensure_ascii=False)
    line_number = sum(1 for _ in path.open(encoding="utf-8")) if path.is_file() else 0
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"line": line_number, "sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(), "code": payload["code"]}


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
                    raise CastDataIssue(
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
        raise OperationalError("cast_integrity", "alias audit is unreadable") from error
    records = report if isinstance(report, list) else report.get("records") if isinstance(report, dict) else None
    if not isinstance(records, list):
        raise TypeError("alias audit requires a records array")
    merges: dict[str, str] = {}
    ambiguous: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"alias", "canonical", "evidence", "decision"}:
            raise OperationalError("cast_integrity", "alias audit record has an invalid schema")
        alias = record["alias"]
        canonical = record["canonical"]
        evidence = record["evidence"]
        decision = record["decision"]
        if not all(isinstance(value, str) for value in (alias, canonical, decision)) or not isinstance(evidence, list):
            raise OperationalError("cast_integrity", "alias audit record has invalid values")
        if (
            canonical not in registry
            or not evidence
            or not all(isinstance(quote, str) and quote in source_text for quote in evidence)
        ):
            raise OperationalError("cast_integrity", "alias audit lacks source-grounded canonical evidence")
        if decision == "distinct":
            continue
        if decision not in {"merge", "ambiguous"}:
            raise OperationalError("cast_integrity", "alias audit decision must be merge, distinct, or ambiguous")
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
                raise OperationalError("cast_integrity", f"alias audit conflicts with prior alias: {alias}")
            merges[merge_alias] = merge_target

    def resolved(actor_id: str, trail: set[str] | None = None) -> str:
        trail = trail or set()
        if actor_id in trail:
            raise OperationalError("cast_integrity", f"alias audit contains a cycle at {actor_id}")
        target = merges.get(actor_id)
        if target is None:
            return actor_id
        return resolved(target, {*trail, actor_id})

    inactive: set[str] = set()
    for alias_id in merges:
        canonical = resolved(alias_id)
        if canonical not in registry:
            raise OperationalError(
                "cast_integrity", f"alias audit resolves outside registry: {alias_id} -> {canonical}"
            )
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
# refresh audited aliases
# reapplies verified audit records before each discovery without moving the cursor or generating media, retaining every earlier inactive identity and ambiguity.
def refresh_alias_audit(project: Path, source_text: str, progress: dict) -> tuple[set[str], set[str]]:
    legacy_registry = {**load_object(project / "characters.json", "characters profile"), **progress["registry"]}
    inactive, ambiguous = apply_alias_audit(project, source_text, legacy_registry, progress["aliases"])
    inactive.update(progress.get("inactive_legacy_ids", []))
    ambiguous.update(progress.get("ambiguous_new_ids", []))
    if inactive:
        progress["registry"] = {
            actor_id: entry for actor_id, entry in progress["registry"].items() if actor_id not in inactive
        }
    progress["inactive_legacy_ids"] = sorted(inactive)
    progress["ambiguous_new_ids"] = sorted(ambiguous)
    progress["audit_applied"] = True
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
            tasks = [asyncio.create_task(_voice_description_for_one(actor_id, profile)) for actor_id, profile in batch]
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
            raise OperationalError("cast_integrity", f"frozen actor lacks profile or Breeze voice: {actor_id}")
        if actor_id != "narrator" and (
            actor_id not in appearances or not (project / "refs" / f"{actor_id}.png").is_file()
        ):
            raise OperationalError("cast_integrity", f"frozen actor lacks appearance or portrait: {actor_id}")
        wav = project / str(breeze[actor_id].get("ref_wav", ""))
        if not wav.is_file():
            raise OperationalError("cast_integrity", f"frozen actor lacks reference WAV: {actor_id}")
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
        raise OperationalError("cast_integrity", "frozen cast manifest does not bind this exact source")
    _, _, chapters = source_chapters(source, project)
    expected = {path.name: file_digest(path) for path in chapters}
    if manifest.get("chapter_sha256") != expected:
        raise OperationalError("cast_integrity", "frozen cast manifest lacks complete exact source chapter coverage")
    actors = manifest.get("actors")
    aliases = manifest.get("approved_aliases")
    if not isinstance(actors, dict) or not actors or not isinstance(aliases, dict):
        raise OperationalError("cast_integrity", "frozen cast manifest has no approved registry")
    if not ANCHOR_IDS <= set(actors):
        raise OperationalError("cast_integrity", "frozen cast manifest is missing original anchor identities")
    if any(not isinstance(value, str) or value not in actors for value in aliases.values()):
        raise OperationalError("cast_integrity", "frozen cast manifest has alias outside approved registry")
    actual = asset_hashes(project, set(actors))
    if actual != manifest.get("asset_hashes"):
        raise OperationalError(
            "cast_integrity", "frozen cast profile, voice, WAV, appearance, or portrait bytes changed"
        )
    # A manifest frozen before scoped audit existed attests an empty record set; any record appearing later is unattested.
    attested = manifest.get("scoped_audit", scoped_audit_attestation([]))
    if attested != scoped_audit_attestation(load_scoped_audit(project)):
        raise OperationalError(
            "cast_integrity", "frozen cast manifest does not attest the current mention-scoped audit records"
        )
    return manifest


# ##################################################################
# scoped references
# yields each attested alias record as one exact mention reference, proven against the actual source spans and approved actors; never a global alias.
def validated_scoped_references(source: Path, project: Path | None = None, manifest: dict | None = None) -> list[dict]:
    source = source.resolve()
    project = project or get_output_dir(source)
    manifest = manifest or verify_frozen_cast(source, project)
    _, _, chapters = source_chapters(source, project)
    units = {(unit["chapter_sha256"], text_digest(unit["quote"])): unit for unit in immutable_evidence_units(chapters)}
    references: list[dict] = []
    for record in sorted(
        load_scoped_audit(project), key=lambda r: (r["chapter_sha256"], r["quote_sha256"], r["span_start"], r["label"])
    ):
        if record["decision"] != "alias":
            continue
        unit = units.get((record["chapter_sha256"], record["quote_sha256"]))
        if unit is None:
            raise OperationalError("cast_integrity", "mention-scoped alias is bound to source text that does not exist")
        if not any(
            ref["label"] == record["label"] and ref["start"] == record["span_start"]
            for ref in immutable_name_references([unit]).values()
        ):
            raise OperationalError(
                "cast_integrity", "mention-scoped alias does not name an exact source mention at its offset"
            )
        if record["canonical"] not in manifest["actors"]:
            raise OperationalError("cast_integrity", "mention-scoped alias targets an actor outside the frozen cast")
        if record["confidence"] < ADJUDICATION_MIN_CONFIDENCE:
            raise OperationalError("cast_integrity", "mention-scoped alias is below the adjudication confidence floor")
        references.append({field: record[field] for field in SCOPED_REFERENCE_FIELDS})
    return references


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
        if not isinstance(batch, dict) or set(batch) - {"quarantined"} != {"start", "end", "chapter_sha256"}:
            raise OperationalError("cast_integrity", "cast preparation has an invalid completed batch record")
        start, end, hashes = batch["start"], batch["end"], batch["chapter_sha256"]
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start != expected_start
            or end <= start
            or end > len(chapters)
        ):
            raise OperationalError(
                "cast_integrity", "cast preparation completed batches do not have unique consecutive chapter coverage"
            )
        expected_hashes = {path.name: file_digest(path) for path in chapters[start:end]}
        if not isinstance(hashes, dict) or hashes != expected_hashes or seen.intersection(hashes):
            raise OperationalError(
                "cast_integrity", "cast preparation completed batches lack unique exact chapter hashes"
            )
        seen.update(hashes)
        expected_start = end
    if progress.get("next_chapter") != expected_start:
        raise OperationalError(
            "cast_integrity", "cast preparation cursor does not match completed exact chapter coverage"
        )


# ##################################################################
# semantic coverage state
# migrates historical structural batches into a separate source-bound semantic ledger without moving the established production cursor.
def semantic_coverage(progress: dict) -> dict:
    coverage = progress.get("semantic_coverage")
    if coverage is None:
        coverage = {"version": SEMANTIC_COVERAGE_VERSION, "next_chapter": 0, "completed_batches": []}
        progress["semantic_coverage"] = coverage
    if not isinstance(coverage, dict) or coverage.get("version") != SEMANTIC_COVERAGE_VERSION:
        raise OperationalError("cast_integrity", "cast preparation has an unsupported semantic coverage ledger")
    return coverage


# ##################################################################
# validate semantic coverage
# proves every semantic ledger entry is consecutive, source-hashed, and bound to the exact deterministic candidate ledger used for its native classification.
def validate_semantic_coverage(progress: dict, chapters: list[Path]) -> None:
    coverage = semantic_coverage(progress)
    expected_start = 0
    for batch in coverage["completed_batches"]:
        if not isinstance(batch, dict) or set(batch) - SEMANTIC_OPTIONAL_KEYS != {
            "start",
            "end",
            "chapter_sha256",
            "ledger_sha256",
        }:
            raise OperationalError("cast_integrity", "semantic coverage has an invalid completed batch record")
        start, end = batch["start"], batch["end"]
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start != expected_start
            or end <= start
            or end > len(chapters)
        ):
            raise OperationalError(
                "cast_integrity", "semantic coverage batches are not unique consecutive source coverage"
            )
        if batch["chapter_sha256"] != {path.name: file_digest(path) for path in chapters[start:end]} or not isinstance(
            batch["ledger_sha256"], str
        ):
            raise OperationalError(
                "cast_integrity", "semantic coverage batch is not bound to exact source and candidate ledger"
            )
        for name, digest in (batch.get("empty_chapters") or {}).items():
            if batch["chapter_sha256"].get(name) != digest or empty_chapter_attestations(
                [path for path in chapters[start:end] if path.name == name]
            ) != {name: digest}:
                raise OperationalError(
                    "cast_integrity", "semantic coverage empty-chapter attestation does not match the exact source"
                )
        expected_start = end
    if coverage.get("next_chapter") != expected_start:
        raise OperationalError("cast_integrity", "semantic coverage cursor does not match its completed batches")


# ##################################################################
# write semantic record
# appends the complete local ledger and native classifications for one unit (never rewrites) and returns its source-hashed coverage entry; a chapter with no immutable span is attested as zero-coverage non-character with its exact hash.
def write_semantic_record(
    project: Path,
    start: int,
    batch: list[Path],
    units: list[dict],
    classifications: list[dict],
    revalidation: bool,
    quarantined: bool = False,
    supersedes_quarantine: bool = False,
) -> dict:
    ledger = [] if quarantined else candidate_coverage_ledger(units, {}, {})
    ledger_sha256 = json_digest(ledger)
    empty = {} if quarantined else empty_chapter_attestations(batch)
    payload = {
        "start_chapter": start,
        "chapters": [path.name for path in batch],
        "semantic_revalidation": revalidation,
        "candidate_ledger": ledger,
        "classifications": classifications,
        **({"quarantined": True} if quarantined else {}),
        **({"empty_chapters": empty, "attestation": "zero_coverage_non_character"} if empty else {}),
        **({"supersedes_quarantine": True} if supersedes_quarantine else {}),
    }
    with (project / DISCOVERIES_NAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {
        "start": start,
        "end": start + len(batch),
        "chapter_sha256": {path.name: file_digest(path) for path in batch},
        "ledger_sha256": ledger_sha256,
        **({"quarantined": True} if quarantined else {}),
        **({"empty_chapters": empty} if empty else {}),
        **({"revalidated": True} if supersedes_quarantine else {}),
    }


# ##################################################################
# record semantic batch
# writes the complete local ledger and native classifications before any cursor advances, making omitted candidates auditable across restarts.
def record_semantic_batch(
    project: Path,
    start: int,
    batch: list[Path],
    units: list[dict],
    classifications: list[dict],
    coverage: dict,
    revalidation: bool,
    quarantined: bool = False,
) -> None:
    coverage["completed_batches"].append(
        write_semantic_record(project, start, batch, units, classifications, revalidation, quarantined)
    )
    coverage["next_chapter"] = start + len(batch)


# ##################################################################
# recoverable batches
# runs one source batch; a data problem (malformed/unknown/ambiguous/overlarge/empty/odd-unicode source or bad model JSON) isolates the failing chapter(s), records exact evidence and a checkpoint, and continues. Registry and aliases only change on a fully valid result, so a quarantined unit can never leave a partial or global alias. Infrastructure errors propagate untouched.
def recoverable_batches(
    project: Path,
    pool: list[Path],
    start: int,
    progress: dict,
    ambiguous: set[str],
    source_text: str,
    recovery: RecoveryLedger,
    ask,
    stop_at: int | None,
):
    def attempt(paths: list[Path], prompt: str | None) -> dict:
        units = immutable_evidence_units(paths)
        work = {"registry": copy.deepcopy(progress["registry"]), "aliases": dict(progress["aliases"])}
        classifications: list[dict] = []
        # Each window is an immutable-unit slice that fits the native context; discoveries accumulate on the working copy only.
        for window_index, window in enumerate(unit_windows(units, work["registry"], work["aliases"])):
            with telemetry_scope("cast_discovery", start, start + len(paths), window_index):
                discoveries, window_classes = discover_batch(
                    project, start, paths, window, source_text, work, ambiguous, prompt, ask
                )
            apply_discoveries(work["registry"], work["aliases"], discoveries)
            classifications.extend(window_classes)
        return {
            "start": start,
            "batch": paths,
            "units": units,
            "classifications": classifications,
            "registry": work["registry"],
            "aliases": work["aliases"],
            "quarantine": None,
        }

    def quarantine(paths: list[Path], error: DataIssue) -> dict:
        names = [path.name for path in paths]
        archive = archive_quarantine(project, start, paths, error)
        recovery.record_error(
            "cast",
            f"chapters {start}-{start + len(paths) - 1}",
            error,
            evidence={
                "chapters": names,
                "rejection_archive": REJECTIONS_NAME,
                **archive,
                "source": names,
                "source_hash": archive["raw_chapter_sha256"],
            },
            checkpoint={"start_chapter": start, "end_chapter": start + len(paths), "next_chapter": start + len(paths)},
        )
        return {
            "start": start,
            "batch": paths,
            "units": [],
            "classifications": [],
            "registry": None,
            "aliases": None,
            "quarantine": names,
        }

    # Only a typed DataIssue is recoverable; programming, integrity, store, source-hash and alias-cycle errors propagate.
    try:
        batch, prompt = context_safe_batch(pool, start, progress["registry"], progress["aliases"])
    except DataIssue as error:
        yield quarantine(pool[start : start + 1], error)
        return
    try:
        yield attempt(batch, prompt)
        return
    except DataIssue as error:
        first_error = error
    if len(batch) == 1:
        yield quarantine(batch, first_error)
        return
    # Isolate: retry each chapter of the failed batch alone so one bad chapter never costs its neighbours.
    position = start
    for chapter in batch:
        start = position
        try:
            yield attempt([chapter], None)
        except DataIssue as error:
            yield quarantine([chapter], error)
        position += 1


# ##################################################################
# unit windows
# partitions a batch's immutable evidence units into consecutive windows that each fit the native context, so an oversized chapter is classified completely instead of quarantined.
def unit_windows(units: list[dict[str, str]], registry: dict, aliases: dict[str, str]) -> list[list[dict[str, str]]]:
    def fits(window: list[dict[str, str]]) -> bool:
        source = sum(len(unit["quote"]) for unit in window)
        if source * 2 > PREPARATION_PROMPT_MAX_CHARS:
            return False
        return len(discovery_prompt([], registry, aliases, units=window)) + source * 2 <= PREPARATION_PROMPT_MAX_CHARS

    windows: list[list[dict[str, str]]] = []
    index = 0
    while index < len(units):
        if fits(units[index:]):
            windows.append(units[index:])
            break
        if not fits(units[index : index + 1]):
            raise CastDataIssue(
                f"immutable unit {units[index]['id']} exceeds safe native model context",
                {
                    "unit_id": units[index]["id"],
                    "chapter": units[index]["chapter"],
                    "unit_chars": len(units[index]["quote"]),
                },
                "source_unit_oversized",
            )
        low, high = index + 1, len(units)  # low fits, high does not
        while high - low > 1:
            mid = (low + high) // 2
            if fits(units[index:mid]):
                low = mid
            else:
                high = mid
        windows.append(units[index:low])
        index = low
    return windows


# ##################################################################
# archive quarantine
# appends the durable rejection record for a quarantined unit: exact raw source bytes hashes and a reference to every prior raw model rejection for the same scope.
def archive_quarantine(project: Path, start: int, paths: list[Path], error: DataIssue) -> dict:
    names = [path.name for path in paths]
    raw = {path.name: path.read_bytes() for path in paths}
    archive = project / REJECTIONS_NAME
    prior: list[dict] = []
    line_count = 0
    if archive.exists():
        for number, line in enumerate(archive.read_text(encoding="utf-8").splitlines()):
            line_count = number + 1
            record = json.loads(line)
            if record.get("start_chapter") == start and record.get("chapters") == names:
                prior.append({"line": number, "sha256": hashlib.sha256(line.encode("utf-8")).hexdigest()})
    payload = {
        "kind": "quarantine",
        "start_chapter": start,
        "chapters": names,
        "code": getattr(error, "code", type(error).__name__),
        "error": str(error),
        "evidence": safe_evidence(getattr(error, "evidence", {})),
        "raw_chapter_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()},
        "raw_chapter_bytes": {name: len(data) for name, data in raw.items()},
        "model_rejections": prior,
    }
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with archive.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "archive_line": line_count,
        "archive_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
        "raw_chapter_sha256": payload["raw_chapter_sha256"],
        "model_rejections": prior,
    }


PENDING_REPLAY_LIMIT = 3


# ##################################################################
# replay model bindings
# lists every mention-scoped alias now bound inside the replayed source (exact scope, canonical, confidence, mechanical proof) so a resolved pending row states which model bindings the replay relied on.
def replay_model_bindings(project: Path, units: list[dict]) -> list[dict]:
    hashes = {unit["chapter_sha256"] for unit in units}
    return [
        {
            **{key: record[key] for key in ("chapter_sha256", "quote_sha256", "label", "span_start", "canonical")},
            "confidence": record["confidence"],
            "proof": record.get("proof"),
        }
        for record in load_scoped_audit(project)
        if record["decision"] == "alias" and record["chapter_sha256"] in hashes
    ]


PROPOSALS_NAME = "cast_pending_proposals.jsonl"
CONTEXT_PROPOSALS_NAME = "cast_context_resolution_proposals.json"
CONTEXT_PROPOSALS_VERSION = 1
QUALITY_KINDS = frozenset({"country", "garble", "duplicate_actor"})


# ##################################################################
# caretaker quality proposal input
# The caretaker supplies one atomically-renamed snapshot. It is input only: ingestion may
# create/resolve typed recovery rows, but never changes registry or aliases. A proposal is
# therefore incapable of approving a country/garble/duplicate actor or misrouting it to narrator.
def _quality_source_scope(
    value: object, chapters: dict[str, Path], units_cache: dict[str, dict[str, dict]]
) -> tuple[dict, Path]:
    if not isinstance(value, dict) or set(value) != {
        "chapter",
        "chapter_sha256",
        "unit_id",
        "quote",
        "quote_sha256",
        "label",
        "span_start",
    }:
        raise OperationalError("cast_integrity", "quality proposal scope has an invalid schema")
    chapter, chapter_sha, unit_id, quote, quote_sha, label, span = (
        value["chapter"],
        value["chapter_sha256"],
        value["unit_id"],
        value["quote"],
        value["quote_sha256"],
        value["label"],
        value["span_start"],
    )
    if (
        not all(isinstance(item, str) and item for item in (chapter, chapter_sha, unit_id, quote, quote_sha, label))
        or type(span) is not int
        or span < 0
    ):
        raise OperationalError("cast_integrity", "quality proposal scope has invalid values")
    path = chapters.get(chapter)
    if path is None or file_digest(path) != chapter_sha:
        raise OperationalError("cast_integrity", "quality proposal chapter is not the exact current source")
    by_id = units_cache.setdefault(chapter, {unit["id"]: unit for unit in immutable_evidence_units([path])})
    unit = by_id.get(unit_id)
    if (
        unit is None
        or unit["quote"] != quote
        or text_digest(quote) != quote_sha
        or quote[span : span + len(label)] != label
    ):
        raise OperationalError("cast_integrity", "quality proposal main mention is not an exact immutable scope")
    return value, path


def _quality_witnesses(
    value: object, chapters: dict[str, Path], units_cache: dict[str, dict[str, dict]]
) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise OperationalError("cast_integrity", "quality proposal requires source witnesses")
    witnesses: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for witness in value:
        if not isinstance(witness, dict) or set(witness) != {
            "chapter",
            "chapter_sha256",
            "unit_id",
            "quote",
            "quote_sha256",
        }:
            raise OperationalError("cast_integrity", "quality proposal witness has an invalid schema")
        chapter, chapter_sha, unit_id, quote, quote_sha = (
            witness["chapter"],
            witness["chapter_sha256"],
            witness["unit_id"],
            witness["quote"],
            witness["quote_sha256"],
        )
        if not all(isinstance(item, str) and item for item in (chapter, chapter_sha, unit_id, quote, quote_sha)):
            raise OperationalError("cast_integrity", "quality proposal witness has invalid values")
        path = chapters.get(chapter)
        if path is None or file_digest(path) != chapter_sha:
            raise OperationalError("cast_integrity", "quality proposal witness is not the exact current source")
        unit = units_cache.setdefault(chapter, {item["id"]: item for item in immutable_evidence_units([path])}).get(unit_id)
        if unit is None or unit["quote"] != quote or text_digest(quote) != quote_sha:
            raise OperationalError("cast_integrity", "quality proposal witness is not an exact immutable unit")
        key = (chapter, unit_id)
        if key in seen:
            raise OperationalError("cast_integrity", "quality proposal repeats a source witness")
        seen.add(key)
        witnesses.append(witness)
    return witnesses


def active_quality_pending(recovery: RecoveryLedger, registry: dict) -> list[dict]:
    return [
        row
        for row in recovery.open_pending("cast_quality")
        if (
            row.get("code") in {f"quality_{kind}" for kind in QUALITY_KINDS}
            and row.get("evidence", {}).get("registry_id") in registry
        )
        or row.get("code") == QUALITY_UNKNOWN_ACTOR
    ]


QUALITY_UNKNOWN_ACTOR = "quality_unknown_actor"


def _quality_inactive_basis(project: Path, progress: dict | None, registry: dict, registry_id: str) -> dict | None:
    """Source-proven reason a non-registry ID is gone: a recorded retirement, or a verified alias-audit merge.

    Returns None when nothing proves it, so the caller treats the ID as unknown (typed pending, never a crash).
    """
    if not isinstance(progress, dict):
        return None
    for entry in progress.get(SOURCE_RETIREMENT_HISTORY, []):
        if isinstance(entry, dict) and entry.get("actor_id") == registry_id and entry.get("entry_sha256"):
            return {"basis": "source_reviewed_retirement", "entry_sha256": entry["entry_sha256"]}
    target = progress.get("aliases", {}).get(registry_id)
    if registry_id in progress.get("inactive_legacy_ids", []) and target in registry and target != registry_id:
        for record in _alias_audit_records(project):
            if (
                isinstance(record, dict)
                and record.get("decision") == "merge"
                and registry_id in {normalized_id(str(record.get("alias", ""))), normalized_id(str(record.get("canonical", "")))}
            ):
                return {"basis": "verified_alias_audit_merge", "canonical": target}
    return None


def ingest_context_quality_proposals(
    project: Path,
    source_sha: str,
    chapters: list[Path],
    registry: dict,
    recovery: RecoveryLedger,
    progress: dict | None = None,
) -> list[dict]:
    """Atomically merge caretaker quality input at a producer batch boundary.

    Every proposal is source-validated before a typed pending/resolution history row is appended.
    The immutable material digest excludes only its requested resolution, so a changed proposal
    cannot silently replace prior evidence under the same proposal ID.
    """
    path = project / CONTEXT_PROPOSALS_NAME
    if not path.is_file():
        return active_quality_pending(recovery, registry)
    payload = load_object(path, "caretaker context proposals")
    if set(payload) != {"version", "source_sha256", "proposals"} or payload["version"] != CONTEXT_PROPOSALS_VERSION:
        raise OperationalError("cast_integrity", "caretaker context proposals have an invalid schema")
    if payload["source_sha256"] != source_sha or not isinstance(payload["proposals"], list):
        raise OperationalError("cast_integrity", "caretaker context proposals belong to a different source")
    by_chapter = {chapter.name: chapter for chapter in chapters}
    units_cache: dict[str, dict[str, dict]] = {}
    prior_rows = [row for row in recovery.entries() if row.get("stage") == "cast_quality"]
    prior_by_id: dict[str, str] = {}
    open_by_id = {row.get("evidence", {}).get("proposal_id"): row for row in active_quality_pending(recovery, registry)}
    seen_ids: set[str] = set()
    seen_scopes: set[tuple[str, str, str, int]] = set()
    input_sha = payload_hash(payload)
    for proposal in payload["proposals"]:
        if not isinstance(proposal, dict) or set(proposal) != {
            "proposal_id",
            "registry_id",
            "kind",
            "status",
            "scope",
            "witnesses",
            "note",
            "resolution",
        }:
            raise OperationalError("cast_integrity", "caretaker quality proposal has an invalid schema")
        proposal_id, registry_id, kind, status, note = (
            proposal["proposal_id"], proposal["registry_id"], proposal["kind"], proposal["status"], proposal["note"]
        )
        if (
            not isinstance(proposal_id, str)
            or not proposal_id
            or proposal_id in seen_ids
            or not isinstance(registry_id, str)
            or registry_id == "narrator"
            or kind not in QUALITY_KINDS
            or status not in {"pending", "resolved"}
            or not isinstance(note, str)
        ):
            raise OperationalError("cast_integrity", "caretaker quality proposal has invalid values")
        seen_ids.add(proposal_id)
        scope, _ = _quality_source_scope(proposal["scope"], by_chapter, units_cache)
        witnesses = _quality_witnesses(proposal["witnesses"], by_chapter, units_cache)
        scope_key = (scope["chapter_sha256"], scope["quote_sha256"], scope["label"], scope["span_start"])
        if scope_key in seen_scopes:
            raise OperationalError("cast_integrity", "caretaker quality proposals duplicate one exact mention scope")
        seen_scopes.add(scope_key)
        material = {key: proposal[key] for key in proposal if key not in {"status", "resolution"}}
        material_sha = payload_hash(material)
        for row in prior_rows:
            evidence = row.get("evidence", {})
            if evidence.get("proposal_id") == proposal_id:
                previous = evidence.get("proposal_sha256")
                if previous and previous != material_sha:
                    raise OperationalError("cast_integrity", "caretaker quality proposal changed immutable evidence under one ID")
                prior_by_id[proposal_id] = material_sha
        item = f"quality:{registry_id}:{proposal_id}"
        evidence = {
            "source": [scope["chapter"]],
            "source_hash": {scope["chapter"]: scope["chapter_sha256"]},
            "proposal_id": proposal_id,
            "proposal_sha256": material_sha,
            "input_sha256": input_sha,
            "registry_id": registry_id,
            "kind": kind,
            "scope": scope,
            "witnesses": witnesses,
        }
        if registry_id not in registry:
            basis = _quality_inactive_basis(project, progress, registry, registry_id)
            mine = [r for r in prior_rows if r.get("evidence", {}).get("proposal_id") == proposal_id]
            if basis is None:
                recovery.record(
                    "cast_quality", item, QUALITY_UNKNOWN_ACTOR,
                    f"quality proposal names unknown registry ID {registry_id!r}", severity="pending", evidence=evidence,
                )
                continue
            if not mine:
                recovery.record("cast_quality", item, f"quality_{kind}", note, severity="pending", evidence=evidence)
            for row in recovery.open_pending("cast_quality"):
                if row.get("evidence", {}).get("proposal_id") == proposal_id and row["code"] != QUALITY_UNKNOWN_ACTOR:
                    recovery.resolve(row, "actor_inactive_source_proven", {"proposal_id": proposal_id, "registry_id": registry_id, **basis})
            continue
        for row in recovery.open_pending("cast_quality"):
            if row["code"] == QUALITY_UNKNOWN_ACTOR and row.get("evidence", {}).get("proposal_id") == proposal_id:
                recovery.resolve(row, "actor_now_registered", {"proposal_id": proposal_id, "registry_id": registry_id})
        if status == "pending":
            if proposal["resolution"] is not None:
                raise OperationalError("cast_integrity", "pending quality proposal has a resolution")
            recovery.record("cast_quality", item, f"quality_{kind}", note, severity="pending", evidence=evidence)
        else:
            resolution = proposal["resolution"]
            if not isinstance(resolution, dict) or set(resolution) != {"reason"} or not isinstance(resolution["reason"], str):
                raise OperationalError("cast_integrity", "resolved quality proposal lacks a typed resolution reason")
            row = open_by_id.get(proposal_id)
            if row is None:
                if proposal_id not in prior_by_id:
                    raise OperationalError("cast_integrity", "quality resolution has no prior pending proposal")
                continue
            recovery.resolve(
                row,
                "caretaker_resolved_quality_flag",
                {"proposal_id": proposal_id, "proposal_sha256": material_sha, "input_sha256": input_sha, "reason": resolution["reason"]},
            )
    return active_quality_pending(recovery, registry)


# ##################################################################
# caretaker context resolution v2 adapter
# The v2 file carries per-mention decisions. Source identity is fail-closed (book SHA, exact immutable main unit,
# quote-relative span, every witness resolved globally by chapter hash), but a decision is applied only when the
# existing guards prove it (scoped_alias_proof for an alias; no plausible owner at all for a non_character);
# new_actor, hold and every unproven alias/non_character become typed pending rows that block final freeze.
CONTEXT_RESOLUTION_V2_NAME = "cast_context_resolution_proposals_v2.json"
CONTEXT_RESOLUTION_V2_CONTRACT = "cast_context_resolution_proposal_v2"
CONTEXT_V2_STAGE = "cast_context"
CONTEXT_V2_DECISIONS = frozenset({"alias", "new_actor", "non_character", "hold"})
CONTEXT_V2_ROW_FIELDS = frozenset(
    {
        "chapter_file",
        "chapter_sha256",
        "main_unit_id",
        "main_unit_quote",
        "main_unit_quote_sha256",
        "label",
        "span_start",
        "label_matches_span",
        "witnesses",
        "decision",
        "canonical_target",
        "target_registered",
        "reason",
    }
)
CONTEXT_V2_WITNESS_FIELDS = frozenset(
    {"chapter_file", "chapter_sha256", "unit_id", "unit_quote", "unit_quote_sha256"}
)


def _context_v2_invalid(message: str) -> OperationalError:
    return OperationalError(
        "cast_integrity", f"caretaker context resolution v2: {message}"
    )


class _ChapterSource:
    """Hash-addressed view of every source chapter: a witness may live in any chapter, not just the main mention's."""

    def __init__(self, chapters: list[Path]) -> None:
        self.by_hash: dict[str, list[Path]] = {}
        for chapter in chapters:
            self.by_hash.setdefault(file_digest(chapter), []).append(chapter)
        self._units: dict[Path, tuple[list[dict], dict[str, dict]]] = {}

    def resolve(self, name: object, sha: object, what: str) -> Path:
        if not isinstance(name, str) or not isinstance(sha, str) or not name or not sha:
            raise _context_v2_invalid(f"{what} chapter identity is invalid")
        matches = [path for path in self.by_hash.get(sha, []) if path.name == name]
        if len(matches) != 1:
            raise _context_v2_invalid(
                f"{what} chapter {name!r} is not the exact current source by hash"
            )
        return matches[0]

    def units(self, path: Path) -> tuple[list[dict], dict[str, dict]]:
        if path not in self._units:
            # Unit ids are single-chapter (c00sNNNNN), exactly as the caretaker computed them.
            ordered = immutable_evidence_units([path])
            self._units[path] = (ordered, {unit["id"]: unit for unit in ordered})
        return self._units[path]

    def exact_unit(
        self,
        name: object,
        sha: object,
        unit_id: object,
        quote: object,
        quote_sha: object,
        what: str,
    ) -> dict:
        path = self.resolve(name, sha, what)
        if not all(
            isinstance(item, str) and item for item in (unit_id, quote, quote_sha)
        ):
            raise _context_v2_invalid(f"{what} has invalid unit values")
        unit = self.units(path)[1].get(unit_id)
        if (
            unit is None
            or unit["quote"] != quote
            or text_digest(quote) != quote_sha
            or unit["chapter_sha256"] != sha
        ):
            raise _context_v2_invalid(f"{what} is not an exact immutable unit")
        return unit


def _context_pending(base: dict, code: str, why: str) -> dict:
    return {**base, "code": code, "why": why}


def adapt_context_resolution_v2(
    payload: object,
    source_sha: str,
    chapters: list[Path],
    registry: dict,
    aliases: dict,
    scoped_records: list[dict],
) -> dict[str, list]:
    """Validate a v2 caretaker file and split it into guard-proven scoped audit records and typed pending rows.

    Source violations raise OperationalError (fail closed, nothing applied). A decision the guards cannot prove is
    never an error: it is returned as pending with the reason and every validated source witness.
    """
    if (
        not isinstance(payload, dict)
        or payload.get("contract") != CONTEXT_RESOLUTION_V2_CONTRACT
    ):
        raise _context_v2_invalid("unsupported contract")
    if (
        payload.get("source_sha256") != source_sha
        or payload.get("source_book_sha256") != source_sha
    ):
        raise _context_v2_invalid("file belongs to a different source")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise _context_v2_invalid("rows must be a list")
    source = _ChapterSource(chapters)
    prior = mention_scoped_audit_index(scoped_records)
    seen: set[tuple[str, str, str, int]] = set()
    applied: list[dict] = []
    pending: list[dict] = []
    proven: list[str] = []
    new_actor_rows: list[dict] = []
    for row in rows:
        if not isinstance(row, dict) or not CONTEXT_V2_ROW_FIELDS <= set(row):
            raise _context_v2_invalid("row has an invalid schema")
        decision, label, span, target = (
            row["decision"],
            row["label"],
            row["span_start"],
            row["canonical_target"],
        )
        if (
            decision not in CONTEXT_V2_DECISIONS
            or not isinstance(label, str)
            or not label
            or type(span) is not int
            or span < 0
            or not isinstance(target, str)
            or not isinstance(row["reason"], str)
            or row["label_matches_span"] is not True
        ):
            raise _context_v2_invalid("row has invalid values")
        unit = source.exact_unit(
            row["chapter_file"],
            row["chapter_sha256"],
            row["main_unit_id"],
            row["main_unit_quote"],
            row["main_unit_quote_sha256"],
            "main mention",
        )
        if unit["quote"][span : span + len(label)] != label:
            raise _context_v2_invalid(
                "main mention span is not the exact quote-relative label"
            )
        scope = mention_scope(unit, {"start": span}, label)
        if scope in seen:
            raise _context_v2_invalid("duplicate exact mention scope")
        seen.add(scope)
        witnesses = row["witnesses"]
        if not isinstance(witnesses, list) or not witnesses:
            raise _context_v2_invalid("row requires source witnesses")
        checked: list[dict] = []
        witness_keys: set[tuple[str, str]] = set()
        for witness in witnesses:
            if not isinstance(witness, dict) or not CONTEXT_V2_WITNESS_FIELDS <= set(
                witness
            ):
                raise _context_v2_invalid("witness has an invalid schema")
            exact = source.exact_unit(
                witness["chapter_file"],
                witness["chapter_sha256"],
                witness["unit_id"],
                witness["unit_quote"],
                witness["unit_quote_sha256"],
                "witness",
            )
            key = (exact["chapter_sha256"], exact["id"])
            if key in witness_keys:
                # The same exact unit may serve several roles (e.g. keyword and identity fact); it is one witness.
                continue
            witness_keys.add(key)
            checked.append(
                {
                    field: witness[field]
                    for field in sorted(witness)
                    if field in CONTEXT_V2_WITNESS_FIELDS | {"role"}
                }
            )
        evidence_names = {
            row["chapter_file"]: row["chapter_sha256"],
            **{w["chapter_file"]: w["chapter_sha256"] for w in checked},
        }
        base = {
            "item": f"context:{scope[0][:16]}:{scope[1][:16]}:{span}:{label}",
            "source": sorted(evidence_names),
            "source_hash": dict(sorted(evidence_names.items())),
            "scope": {
                "chapter": row["chapter_file"],
                "chapter_sha256": scope[0],
                "unit_id": row["main_unit_id"],
                "quote_sha256": scope[1],
                "label": label,
                "span_start": span,
            },
            "decision": decision,
            "canonical_target": target,
            "reason": row["reason"],
            "witnesses": checked,
        }

        if decision == "new_actor":
            new_actor_rows.append(
                {
                    **base,
                    "main_quote": unit["quote"],
                    "provenance": row["provenance"] if isinstance(row.get("provenance"), dict) else {},
                }
            )
        bound = prior.get(scope) if decision == "new_actor" else None
        if (
            bound
            and (bound["decision"], bound["canonical"]) == ("alias", target)
            and isinstance(bound.get("approved_root"), dict)
            and bound["approved_root"].get("actor_id") == target
            and _materialized_root_sha(registry, target) == bound["approved_root"].get("proposal_sha256")
        ):
            # already bound to the actor registered from its approved root: closed, not a pending new_actor
            proven.append(base["item"])
            continue
        if decision in {"new_actor", "hold"}:
            pending.append(
                _context_pending(
                    base,
                    f"context_{decision}",
                    "caretaker has not resolved this mention to an approved registered actor",
                )
            )
            continue
        units_in_chapter, units_by_id = source.units(
            source.resolve(row["chapter_file"], row["chapter_sha256"], "main mention")
        )
        reference = next(
            (
                ref
                for ref in immutable_name_references([unit]).values()
                if ref["label"] == label and ref["start"] == span
            ),
            None,
        )
        if reference is None:
            pending.append(
                _context_pending(
                    base,
                    f"context_{decision}_unproven",
                    "the span is not an exact immutable name reference",
                )
            )
            continue
        existing = prior.get(scope)
        if decision == "alias":
            if (
                target not in registry
                or target == "narrator"
                or row["target_registered"] is not True
            ):
                pending.append(
                    _context_pending(
                        base,
                        "context_alias_unproven",
                        f"owner {target!r} is not an approved registered non-narrator actor",
                    )
                )
                continue
            order = [item["id"] for item in units_in_chapter]
            proof, why = scoped_alias_proof(
                label,
                target,
                unit,
                scene_units_at(units_by_id, order, order.index(unit["id"])),
                registry,
                aliases,
            )
            if why:
                pending.append(_context_pending(base, "context_alias_unproven", why))
                continue
            binding = {"decision": "alias", "canonical": target, "proof": proof}
        else:
            if target != "none" or row["target_registered"] is not False:
                raise _context_v2_invalid("non_character row names an actor")
            owners = adjudication_owners(label, registry, aliases)
            if owners:
                pending.append(
                    _context_pending(
                        base,
                        "context_non_character_unproven",
                        f"label {label!r} still has plausible owners {owners}",
                    )
                )
                continue
            binding = {"decision": "non_character", "canonical": "none"}
        if existing and (existing["decision"], existing["canonical"]) == (
            binding["decision"],
            binding["canonical"],
        ):
            proven.append(base["item"])
            continue
        if existing and existing["decision"] in {"alias", "non_character"}:
            pending.append(
                _context_pending(
                    base,
                    "context_conflict",
                    f"an existing scoped decision {existing['decision']}:{existing['canonical']} disagrees",
                )
            )
            continue
        proven.append(base["item"])
        applied.append(
            {
                "chapter_sha256": scope[0],
                "quote_sha256": scope[1],
                "label": label,
                "span_start": span,
                "confidence": 1.0,
                "reason": f"[caretaker context resolution v2] {row['reason']}"[:300],
                "provenance": row.get("provenance")
                if isinstance(row.get("provenance"), dict)
                else {},
                "witnesses": [
                    {"chapter_sha256": w["chapter_sha256"], "unit_id": w["unit_id"]}
                    for w in checked
                ],
                **binding,
            }
        )
    roots, draft_pending = stage_variant_drafts(
        payload.get("new_actor_drafts", []), new_actor_rows, source, registry, aliases
    )
    return {
        "applied": applied,
        "pending": [*pending, *draft_pending],
        "proven": proven,
        "roots": roots,
        "new_actor_targets": sorted({row["canonical_target"] for row in new_actor_rows}),
    }


# ##################################################################
# variant draft pending roots
# A v2 file may carry new_actor drafts (a proposed identity with bio/look) beside many per-mention new_actor rows whose
# labels are spelling/title variants of that identity. Generic and open-world: nothing here names an actor and a
# spelling or a confidence is never evidence. Rows are grouped under their draft by canonical_target; a mention joins the
# root only through validated linkage to the draft's own source facts: it shares an immutable unit with them that
# literally carries the mention's label or the draft's name, or the draft carries an explicit source-validated
# kinship / continuous-participant link for exactly that mention. Unlinked mentions are excluded and listed, never
# merged. The draft becomes ONE pending proposed root only when every bio/look claim has source-validated citations,
# at least one mention is linked, and it duplicates nothing (no registered/aliased actor, no other draft, no label
# claimed by two drafts). Anything else is a typed pending gap. A root is a proposal only: registry, aliases and
# the scoped audit are never written from it, and materialization is a separate gated step
# (variant_root_approval_gate) that this module never performs.
VARIANT_DRAFT_ITEM = "variant_draft:"
VARIANT_DRAFT_ROOT = "context_variant_draft_root"
VARIANT_DRAFT_INCOMPLETE = "context_variant_draft_incomplete"
VARIANT_DRAFT_DUPLICATE = "context_variant_draft_duplicate"
VARIANT_DRAFT_NO_MENTIONS = "context_variant_draft_no_mentions"
VARIANT_DRAFT_UNLINKED = "context_variant_draft_unlinked"
VARIANT_DRAFT_CODES = frozenset(
    {
        VARIANT_DRAFT_ROOT,
        VARIANT_DRAFT_INCOMPLETE,
        VARIANT_DRAFT_DUPLICATE,
        VARIANT_DRAFT_NO_MENTIONS,
        VARIANT_DRAFT_UNLINKED,
    }
)
VARIANT_DRAFT_FIELDS = frozenset({"actor_id", "name", "kind", "bio", "look", "status", "witnesses"})
VARIANT_CITATION_FIELDS = ("bio", "look")
VARIANT_LINK_KINDS = frozenset({"kinship", "continuous_participant"})
VARIANT_LINK_FIELDS = frozenset(
    {"kind", "chapter_file", "chapter_sha256", "main_unit_id", "label", "span_start", "witnesses"}
)
VARIANT_LIVING_KINDS = frozenset({"person"})
NO_VISUAL_DETAILS = "no visual details given"


def _exact_cited_witnesses(source: _ChapterSource, value: object, what: str) -> list[dict]:
    """Source-validate a list of witnesses (each an exact immutable unit); a repeated unit is one witness."""
    if not isinstance(value, list):
        raise _context_v2_invalid(f"{what} must be a list")
    seen: set[tuple[str, str]] = set()
    checked: list[dict] = []
    for witness in value:
        if not isinstance(witness, dict) or not CONTEXT_V2_WITNESS_FIELDS <= set(witness):
            raise _context_v2_invalid(f"{what} has an invalid witness schema")
        exact = source.exact_unit(
            witness["chapter_file"],
            witness["chapter_sha256"],
            witness["unit_id"],
            witness["unit_quote"],
            witness["unit_quote_sha256"],
            what,
        )
        key = (exact["chapter_sha256"], exact["id"])
        if key not in seen:
            seen.add(key)
            checked.append({field: witness[field] for field in sorted(CONTEXT_V2_WITNESS_FIELDS)})
    return checked


def _unit_key(witness: dict) -> tuple[str, str]:
    return witness["chapter_sha256"], witness["unit_id"]


def _scope_key(scope: dict) -> tuple[str, str, str, int]:
    return scope["chapter_sha256"], scope["unit_id"], scope["label"], scope["span_start"]


def _names_word(word: str, text: str) -> bool:
    return bool(word.strip()) and re.search(rf"(?<!\w){re.escape(word.strip())}(?!\w)", text, re.IGNORECASE) is not None


def _name_words(name: str) -> list[str]:
    """The words of a draft name that can identify it: no titles, roles or grammar words."""
    return [
        word
        for word in re.findall(r"\w+", name)
        if len(word) > 1 and word.casefold() not in TITLE_ROLE_TOKENS and word.casefold() not in NON_NAME_COMPOUND_WORDS
    ]


def _validated_variant_links(source: _ChapterSource, draft: dict) -> list[dict]:
    links = draft.get("variant_links", [])
    if not isinstance(links, list):
        raise _context_v2_invalid("new_actor draft variant_links must be a list")
    checked = []
    for link in links:
        if not isinstance(link, dict) or set(link) != VARIANT_LINK_FIELDS:
            raise _context_v2_invalid("new_actor draft link has an invalid schema")
        if (
            link["kind"] not in VARIANT_LINK_KINDS
            or not isinstance(link["label"], str)
            or not link["label"]
            or type(link["span_start"]) is not int
        ):
            raise _context_v2_invalid("new_actor draft link has invalid values")
        path = source.resolve(link["chapter_file"], link["chapter_sha256"], "link mention")
        unit = source.units(path)[1].get(link["main_unit_id"])
        label, span = link["label"], link["span_start"]
        if unit is None or unit["quote"][span : span + len(label)] != label or span < 0:
            raise _context_v2_invalid("new_actor draft link mention is not an exact immutable span")
        witnesses = _exact_cited_witnesses(source, link["witnesses"], "draft link witness")
        if not witnesses:
            raise _context_v2_invalid("new_actor draft link requires source witnesses")
        checked.append(
            {
                "kind": link["kind"],
                "scope": (unit["chapter_sha256"], unit["id"], label, span),
                "chapter_sha256": unit["chapter_sha256"],
                "witnesses": witnesses,
            }
        )
    return checked


def _validated_variant_draft(source: _ChapterSource, draft: object) -> dict:
    if not isinstance(draft, dict) or not VARIANT_DRAFT_FIELDS <= set(draft):
        raise _context_v2_invalid("new_actor draft has an invalid schema")
    actor_id, name, kind, bio, look, status = (draft[key] for key in ("actor_id", "name", "kind", "bio", "look", "status"))
    if (
        not all(isinstance(item, str) for item in (actor_id, name, kind, bio, look, status))
        or not IDENTIFIER.match(actor_id)
        or not name.strip()
        or not kind.strip()
        or normalized_id(name) == ""
        or not status.casefold().startswith("draft")
    ):
        raise _context_v2_invalid("new_actor draft has invalid values or claims approval")
    citations = draft.get("citations", {})
    if not isinstance(citations, dict) or set(citations) - set(VARIANT_CITATION_FIELDS):
        raise _context_v2_invalid("new_actor draft citations have an invalid schema")
    return {
        "actor_id": actor_id,
        "name": name,
        "kind": kind,
        "bio": bio,
        "look": look,
        "status": status,
        "witnesses": _exact_cited_witnesses(source, draft["witnesses"], f"draft {actor_id} witness"),
        "citations": {
            field: _exact_cited_witnesses(source, citations.get(field, []), f"draft {actor_id} {field} citation")
            for field in VARIANT_CITATION_FIELDS
        },
        "links": _validated_variant_links(source, draft),
    }


def _citation_gaps(draft: dict) -> list[str]:
    """Every bio/look claim needs its own citations; only the explicit no-visual-details sentinel needs none."""
    gaps = []
    if not draft["bio"].strip():
        gaps.append("bio text is empty")
    elif not draft["citations"]["bio"]:
        gaps.append("bio has no source citations")
    look = draft["look"].strip()
    if not look:
        gaps.append("look text is empty")
    elif look.rstrip(".").casefold() != NO_VISUAL_DETAILS and not draft["citations"]["look"]:
        gaps.append("look has no source citations")
    return gaps


def _identity_duplicates(actor_id: str, name: str, registry: dict, aliases: dict) -> list[str]:
    """Why a proposed identity would duplicate an actor that already exists: same id/name, an alias, or a shared name word."""
    duplicates = []
    known_ids = {normalized_id(key): value for key, value in aliases.items()}
    for candidate in (actor_id, name):
        key = normalized_id(candidate)
        if key in registry and key != "narrator":
            duplicates.append(f"{candidate!r} is already registered as {key}")
        elif key in known_ids:
            duplicates.append(f"{candidate!r} is already an alias of {known_ids[key]}")
    for owner in adjudication_owners(name, registry, aliases):
        if _shares_identifying_name(name, owner, registry[owner], aliases):
            duplicates.append(f"name {name!r} shares a name with registered actor {owner}")
    return duplicates


def _shares_identifying_name(name: str, owner: str, entry: dict, aliases: dict) -> bool:
    """True when a proposed name is an owner's exact name/id/alias or shares a non-title name word with it.

    A shared title or role word alone (Lord, Professor, Captain...) names a class of people, not one actor, so it never
    makes two different full names the same identity."""
    if owner_relevance(name, owner, entry, aliases) >= 4:
        return True
    owner_forms = [str(entry.get("name", owner)), owner, *(alias for alias, target in aliases.items() if target == owner)]
    owner_words = set().union(*(label_components(form) for form in owner_forms)) - TITLE_ROLE_TOKENS
    return bool((label_components(name) - TITLE_ROLE_TOKENS) & owner_words)


def _mention_link(row: dict, draft: dict, source: _ChapterSource, fact_units: dict[tuple[str, str], dict]) -> tuple[dict | None, str]:
    """Validated linkage of one mention to the draft's source facts, or (None, why not)."""
    scope = row["scope"]
    label, words = scope["label"], _name_words(draft["name"])
    mention_units = {(scope["chapter_sha256"], scope["unit_id"]): row["main_quote"]}
    mention_units.update({_unit_key(w): w["unit_quote"] for w in row["witnesses"]})
    shared = sorted(
        key
        for key, quote in mention_units.items()
        if key in fact_units and (_names_word(label, quote) or any(_names_word(word, quote) for word in words))
    )
    if shared:
        return {"kind": "shared_immutable_unit", "units": [{"chapter_sha256": c, "unit_id": u} for c, u in shared]}, ""
    link = next((item for item in draft["links"] if item["scope"] == _scope_key(scope)), None)
    if link is None:
        return None, "no shared immutable unit with the draft facts and no explicit link"
    if not _scene_link_proven(link["kind"], link["witnesses"], scope, label, words, source):
        return None, f"explicit {link['kind']} link is not proven by its witnesses"
    return {"kind": link["kind"], "witnesses": link["witnesses"]}, ""


def _scene_link_proven(kind: str, witnesses: list[dict], scope: dict, label: str, words: list[str], source: _ChapterSource) -> bool:
    """Both explicit link kinds are proven only from witnesses inside the mention's own bounded scene, never from elsewhere in the book."""
    path = source.resolve(scope["chapter"], scope["chapter_sha256"], "link mention")
    ordered, by_id = source.units(path)
    order = [item["id"] for item in ordered]
    scene = {item["id"] for item in scene_units_at(by_id, order, order.index(scope["unit_id"]))}
    within = [w["unit_quote"] for w in witnesses if w["chapter_sha256"] == scope["chapter_sha256"] and w["unit_id"] in scene]
    names_label = [_names_word(label, quote) for quote in within]
    names_draft = [any(_names_word(word, quote) for word in words) for quote in within]
    if kind == "kinship":
        # one source sentence ties this label to the draft identity (e.g. a relation stated in the scene)
        return any(a and b for a, b in zip(names_label, names_draft, strict=True))
    return any(names_label) and any(names_draft)


def _root_variants(linked: list[tuple[dict, dict]]) -> list[dict]:
    by_label: dict[str, list[dict]] = {}
    for row, link in sorted(linked, key=lambda pair: (pair[0]["scope"]["chapter"], pair[0]["scope"]["unit_id"], pair[0]["scope"]["span_start"])):
        by_label.setdefault(row["scope"]["label"], []).append(
            {
                "scope": row["scope"],
                "decision": row["decision"],
                "reason": row["reason"],
                "provenance": row["provenance"],
                "witnesses": row["witnesses"],
                "link": link,
            }
        )
    return [{"label": label, "scopes": by_label[label]} for label in sorted(by_label)]


def _pending_material_item(actor_id: str, material: object) -> str:
    return f"{VARIANT_DRAFT_ITEM}{actor_id}:{payload_hash(material)[:16]}"


def _materialized_root_sha(registry: dict, actor_id: str) -> str | None:
    """The proposal sha an actor was registered from by the root approval step, or None for any other actor."""
    entry = registry.get(actor_id)
    approved = entry.get("approved_root") if isinstance(entry, dict) else None
    sha = approved.get("proposal_sha256") if isinstance(approved, dict) else None
    return sha if isinstance(sha, str) and sha else None


def _registry_without_actor(registry: dict, aliases: dict, actor_id: str) -> tuple[dict, dict]:
    return (
        {key: value for key, value in registry.items() if key != actor_id},
        {alias: target for alias, target in aliases.items() if target != actor_id},
    )


def variant_draft_actor(item: str) -> str:
    return item.removeprefix(VARIANT_DRAFT_ITEM).split(":", 1)[0]


def stage_variant_drafts(
    drafts: object, new_actor_rows: list[dict], source: _ChapterSource, registry: dict, aliases: dict
) -> tuple[list[dict], list[dict]]:
    """Return (pending proposed roots, typed pending rows): one pending row per draft, a root only when complete, linked and duplicate-free."""
    if not isinstance(drafts, list):
        raise _context_v2_invalid("new_actor_drafts must be a list")
    validated = [_validated_variant_draft(source, draft) for draft in drafts]
    ids = [draft["actor_id"] for draft in validated]
    if len(set(ids)) != len(ids):
        raise _context_v2_invalid("duplicate new_actor draft actor_id")
    rows_by_target: dict[str, list[dict]] = {}
    for row in new_actor_rows:
        rows_by_target.setdefault(row["canonical_target"], []).append(row)
    for draft in validated:
        mentions = {_scope_key(row["scope"]) for row in rows_by_target.get(draft["actor_id"], [])}
        if any(link["scope"] not in mentions for link in draft["links"]):
            raise _context_v2_invalid("new_actor draft link names a mention that is not one of its own")
    # A label (exact spelling) offered as a variant by two different drafts cannot be minted for either.
    labels_by_draft = {
        draft["actor_id"]: {row["scope"]["label"].casefold() for row in rows_by_target.get(draft["actor_id"], [])}
        for draft in validated
    }
    roots: list[dict] = []
    pending: list[dict] = []
    for draft in validated:
        actor_id = draft["actor_id"]
        rows = rows_by_target.get(actor_id, [])
        facts = [*draft["witnesses"], *draft["citations"]["bio"], *draft["citations"]["look"]]
        fact_units = {_unit_key(w): w for w in facts}
        materialized = _materialized_root_sha(registry, actor_id)
        view_registry, view_aliases = (
            _registry_without_actor(registry, aliases, actor_id) if materialized else (registry, aliases)
        )
        duplicates = _identity_duplicates(actor_id, draft["name"], view_registry, view_aliases)
        for other in validated:
            if other["actor_id"] == actor_id:
                continue
            if normalized_id(other["name"]) == normalized_id(draft["name"]):
                duplicates.append(f"draft {other['actor_id']} has the same name")
            shared = sorted(labels_by_draft[actor_id] & labels_by_draft[other["actor_id"]])
            if shared:
                duplicates.append(f"variants {shared} are also claimed by draft {other['actor_id']}")
        linked: list[tuple[dict, dict]] = []
        unlinked: list[dict] = []
        for row in rows:
            link, why = _mention_link(row, draft, source, fact_units)
            if link:
                linked.append((row, link))
            else:
                unlinked.append({"scope": row["scope"], "why": why})
        evidence_names: dict[str, str] = {w["chapter_file"]: w["chapter_sha256"] for w in facts}
        for row, _link in linked:
            evidence_names.update(row["source_hash"])
        gaps = _citation_gaps(draft)
        base = {
            "source": sorted(evidence_names),
            "source_hash": dict(sorted(evidence_names.items())),
            "draft": {key: draft[key] for key in ("actor_id", "name", "kind", "status")},
        }

        def gap(code: str, why: str, detail: dict, base=base, actor_id=actor_id) -> dict:
            material = {"code": code, "draft": base["draft"], "source_hash": base["source_hash"], **detail}
            return _context_pending({**base, "item": _pending_material_item(actor_id, material)}, code, why) | detail

        if duplicates:
            found = sorted(set(duplicates))
            pending.append(gap(VARIANT_DRAFT_DUPLICATE, "; ".join(found), {"duplicates": found}))
        elif gaps:
            pending.append(gap(VARIANT_DRAFT_INCOMPLETE, "; ".join(gaps), {"missing": gaps}))
        elif not rows:
            pending.append(
                gap(VARIANT_DRAFT_NO_MENTIONS, "no new_actor mention names this draft, so no per-scope provenance exists", {})
            )
        elif not linked:
            pending.append(
                gap(
                    VARIANT_DRAFT_UNLINKED,
                    "no mention is linked to the draft's source facts by a shared immutable unit or an explicit link",
                    {"unlinked": unlinked},
                )
            )
        else:
            root = {
                "type": "variant_draft_root",
                "status": "pending_proposal_not_approved",
                "actor_id": actor_id,
                "name": draft["name"],
                "kind": draft["kind"],
                "bio": draft["bio"],
                "look": draft["look"],
                "citations": draft["citations"],
                "source_facts": draft["witnesses"],
                "variants": _root_variants(linked),
            }
            sha = proposal_sha(root)
            if materialized == sha:
                # exactly this proposal was already approved and registered by the approval step: nothing is pending
                continue
            if materialized:
                found = [f"actor {actor_id} is already registered from approved root {materialized[:16]}; this is a different proposal"]
                pending.append(gap(VARIANT_DRAFT_DUPLICATE, "; ".join(found), {"duplicates": found}))
                continue
            roots.append(root)
            pending.append(
                _context_pending({**base, "item": f"{VARIANT_DRAFT_ITEM}{actor_id}:{sha[:16]}"}, VARIANT_DRAFT_ROOT,
                    "complete, linked, duplicate-free draft held as a pending proposed root; never approved or registered here")
                | {
                    "proposal_sha256": sha,
                    "proposal_file": PROPOSALS_NAME,
                    "variant_labels": [variant["label"] for variant in root["variants"]],
                    "scope_count": len(linked),
                    "unlinked_excluded": unlinked,
                }
            )
    return roots, pending


# ##################################################################
# variant root approval gate
# The only way a pending root may ever become a registered actor is a SEPARATE, explicit approval step that passes this
# gate. It is pure: it returns the blockers and never touches registry, aliases or any file, and it admits nothing on
# spelling or confidence. A root qualifies only if (a) it is still an unapproved root of a living-capable kind, (b) it
# still duplicates nothing in the CURRENT registry/aliases, (c) its bio is cited by a source unit that literally names
# the identity, its look is cited (or explicitly absent), and (d) every included mention has an exact-scope
# distinct_living_identity verdict in the new-identity review audit whose own-source witness is that very mention.
def variant_root_approval_gate(root: dict, registry: dict, aliases: dict, review_records: list[dict]) -> list[str]:
    blockers: list[str] = []
    if not isinstance(root, dict) or root.get("type") != "variant_draft_root":
        return ["not a variant draft root"]
    if root.get("status") != "pending_proposal_not_approved":
        blockers.append("root is not a pending unapproved proposal")
    if root.get("kind") not in VARIANT_LIVING_KINDS:
        blockers.append(f"kind {root.get('kind')!r} is not a living-capable kind")
    actor_id, name = str(root.get("actor_id", "")), str(root.get("name", ""))
    blockers.extend(_identity_duplicates(actor_id, name, registry, aliases))
    variants = root.get("variants") if isinstance(root.get("variants"), list) else []
    labels = [str(variant.get("label", "")) for variant in variants if isinstance(variant, dict)]
    citations = root.get("citations") if isinstance(root.get("citations"), dict) else {}
    bio_quotes = [str(c.get("unit_quote", "")) for c in citations.get("bio", []) if isinstance(c, dict)]
    if not any(_names_word(word, quote) for quote in bio_quotes for word in [*_name_words(name), *labels]):
        blockers.append("no bio citation literally names the identity")
    look = str(root.get("look", "")).strip()
    if not look or (look.rstrip(".").casefold() != NO_VISUAL_DETAILS and not citations.get("look")):
        blockers.append("look is not cited")
    reviews = {
        (r.get("chapter_sha256"), r.get("quote_sha256"), r.get("label"), r.get("span_start")): r
        for r in review_records
        if isinstance(r, dict)
    }
    mentions = [scope for variant in variants for scope in variant.get("scopes", [])]
    if not mentions:
        blockers.append("root has no mentions")
    for item in mentions:
        scope = item["scope"]
        record = reviews.get((scope["chapter_sha256"], scope["quote_sha256"], scope["label"], scope["span_start"]))
        witness = record.get("own_source_witness") if record else None
        if (
            not record
            or record.get("verdict") != DISTINCT_VERDICT
            or not isinstance(witness, dict)
            or witness.get("unit_id") != scope["unit_id"]
            or witness.get("label") != scope["label"]
            or witness.get("span_start") != scope["span_start"]
        ):
            blockers.append(f"mention {scope['label']!r}@{scope['span_start']} lacks an exact-scope distinct living-identity review")
    return blockers


def variant_draft_resolution(
    row: dict, current: dict[str, tuple[str, str]], mention_targets: set[str], registry: dict
) -> str | None:
    """Outcome that durably closes an open draft-level row, or None to keep it open (a draft is never resolved by omission alone).

    A draft row's item carries a digest of its exact material, so any change in the draft, its mentions, its gaps or its
    stored proposal is a new row and the older one is superseded; an unchanged state stays open until the identity is
    registered through the registry or no new_actor mention names it any more.
    """
    actor_id = variant_draft_actor(row["item"])
    now = current.get(actor_id)
    if now is not None:
        return "variant_draft_state_superseded" if now != (row["item"], row["code"]) else None
    if row["code"] == VARIANT_DRAFT_ROOT and actor_id in registry:
        return "variant_draft_registered_through_registry"
    if actor_id not in mention_targets:
        # No new_actor mention names it any more: each of those mentions was resolved or re-decided on its own row.
        return "variant_draft_no_longer_asserted"
    return None


def ingest_context_resolution_v2(
    project: Path,
    source_sha: str,
    chapters: list[Path],
    registry: dict,
    aliases: dict,
    recovery: RecoveryLedger,
) -> list[dict]:
    """Apply guard-proven v2 decisions to the mention-scoped audit and record the rest as typed pending rows.

    The input file is never modified. A pending mention that later becomes proven is resolved by an appended
    history row. Returns the open context pending rows.
    """
    path = project / CONTEXT_RESOLUTION_V2_NAME
    if not path.is_file():
        return recovery.open_pending(CONTEXT_V2_STAGE)
    records = load_scoped_audit(project)
    payload = load_object(path, "caretaker context resolution v2")
    result = adapt_context_resolution_v2(
        payload, source_sha, chapters, registry, aliases, records
    )
    input_sha = payload_hash(payload)
    # The exact root proposals (full bio/look citations, per-scope provenance) are durable before any row refers to them.
    for root in result["roots"]:
        save_pending_proposal(project, root)
    for item in result["pending"]:
        recovery.record(
            CONTEXT_V2_STAGE,
            item["item"],
            item["code"],
            item["why"],
            severity="pending",
            evidence={
                key: value for key, value in item.items() if key not in {"item", "code"}
            }
            | {"input_sha256": input_sha},
        )
    if result["applied"]:
        known = mention_scoped_audit_index(records)
        for record in result["applied"]:
            supersede_scoped_record(
                records,
                known,
                (
                    record["chapter_sha256"],
                    record["quote_sha256"],
                    record["label"],
                    record["span_start"],
                ),
                record,
            )
        mention_scoped_audit_index(records)
        atomic_json(project / SCOPED_AUDIT_NAME, {"records": records})
    current_draft_codes = {
        variant_draft_actor(item["item"]): (item["item"], item["code"])
        for item in result["pending"]
        if item["code"] in VARIANT_DRAFT_CODES
    }
    for row in recovery.open_pending(CONTEXT_V2_STAGE):
        if row["item"] in result["proven"]:
            recovery.resolve(
                row, "caretaker_context_resolution_proven", {"input_sha256": input_sha}
            )
        elif row["item"].startswith(VARIANT_DRAFT_ITEM):
            outcome = variant_draft_resolution(
                row, current_draft_codes, set(result["new_actor_targets"]), registry
            )
            if outcome:
                recovery.resolve(row, outcome, {"input_sha256": input_sha})
    return recovery.open_pending(CONTEXT_V2_STAGE)


# ##################################################################
# save pending proposal
# persists the exact demoted identity proposal so a pending row's evidence is reproducible; the same exact proposal (same scope, mentions and evidence bytes) is stored once however often a replay or round re-presents it.
def proposal_line(proposal: dict) -> str:
    return json.dumps(proposal, ensure_ascii=False, sort_keys=True)


def proposal_sha(proposal: dict) -> str:
    """The identity a pending row cites for its stored proposal: sha256 of the exact stored line."""
    return hashlib.sha256(proposal_line(proposal).encode("utf-8")).hexdigest()


def save_pending_proposal(project: Path, proposal: dict) -> str:
    line = proposal_line(proposal)
    path = project / PROPOSALS_NAME
    if path.is_file() and line in path.read_text(encoding="utf-8").splitlines():
        return proposal_sha(proposal)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return proposal_sha(proposal)


# ##################################################################
# variant root approval and materialization
# The ONLY way a pending variant_draft_root becomes a registered actor. Input is one caretaker file (documented in
# docs/CAST_ROOT_APPROVALS.md) that names, per root, the exact pending proposal sha and one review per included mention.
# Nothing here is inferred: a root that is not named in the input is never touched. Every claim is re-validated against the
# exact current source, then variant_root_approval_gate plus independent duplicate and distinct-participant gates must all
# pass before anything is written. Only then: one new registry actor (profile facts from the cited bio/look), the exact
# actor-id/name aliases, one mention-scoped alias per included mention, and resolution of the matching pending rows. Original
# characters/voices files are never opened for writing, and no variant spelling becomes a global alias.
ROOT_APPROVAL_NAME = "cast_root_approvals.json"
ROOT_APPROVAL_CONTRACT = "cast_variant_root_approval"
ROOT_APPROVAL_VERSION = 1
ROOT_APPROVAL_STAGE = "cast_root_approval"
ROOT_APPROVAL_BLOCKED = "root_approval_blocked"
ROOT_APPROVAL_MALFORMED = "root_approval_malformed"
UNCERTAIN_VERDICT = "uncertain"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
ROOT_APPROVE_ACTION = "approve_root"
NATIVE_REVIEW = "native_review_audit"
SOURCE_REVIEW = "source_reviewed_trusted_role"
REVIEW_PROVENANCES = frozenset({NATIVE_REVIEW, SOURCE_REVIEW})
TRUSTED_REVIEW_ROLES = frozenset({"caretaker"})
ROOT_FIELDS = frozenset(
    {"type", "status", "actor_id", "name", "kind", "bio", "look", "citations", "source_facts", "variants"}
)
ROOT_APPROVAL_FIELDS = frozenset({"action", "actor_id", "proposal_sha256", "reviews", "note"})
ROOT_REVIEW_FIELDS = frozenset(
    {"proposal_sha256", "scope", "provenance", "verdict", "reviewer_role", "factual_witnesses", "factual_basis"}
)
ROOT_SCOPE_FIELDS = frozenset({"chapter", "chapter_sha256", "unit_id", "quote_sha256", "label", "span_start"})
HUMAN_REVIEW_CLAIM = re.compile(r"human[\s_-]*(?:review|verif|approv|check|audit|source)", re.IGNORECASE)
KIN_LABEL_WORDS = frozenset(
    {
        "mom", "mum", "mommy", "mama", "mother", "dad", "daddy", "papa", "father", "brother", "sister", "sibling",
        "aunt", "auntie", "aunty", "uncle", "cousin", "grandmother", "grandfather", "grandma", "grandpa", "granny",
        "son", "daughter", "wife", "husband", "nephew", "niece",
    }
)
KIN_MODIFIER_WORDS = frozenset({"little", "big", "older", "younger", "elder", "dear", "great", "my", "our", "your"})


def _approval_invalid(message: str) -> OperationalError:
    return OperationalError("cast_integrity", f"caretaker root approval: {message}")


class _MalformedApproval(Exception):
    """One approval entry is invalid on its own (schema, values, review structure): held as a typed pending row, never a crash."""


def _checked_approval_entry(approval: object) -> tuple[str, str, str]:
    """(actor_id, proposal sha, note) of a structurally valid approval entry, else _MalformedApproval."""
    if not isinstance(approval, dict) or set(approval) != ROOT_APPROVAL_FIELDS:
        raise _MalformedApproval("approval entry is not an object with exactly the documented fields")
    if approval["action"] != ROOT_APPROVE_ACTION:
        raise _MalformedApproval(f"only the {ROOT_APPROVE_ACTION!r} action exists")
    actor_id, sha, note = approval["actor_id"], approval["proposal_sha256"], approval["note"]
    if not all(isinstance(item, str) for item in (actor_id, sha, note)) or not SHA256_HEX.fullmatch(sha):
        raise _MalformedApproval("approval actor_id, note and a 64-hex proposal_sha256 must be text")
    if HUMAN_REVIEW_CLAIM.search(json.dumps(approval)):
        raise _MalformedApproval("approval claims a human review the system cannot attest")
    return actor_id, sha, note


def _stored_pending_roots(project: Path) -> dict[str, dict]:
    """sha -> root for every canonical variant_draft_root line of the pending proposal file."""
    path = project / PROPOSALS_NAME
    found: dict[str, dict] = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            proposal = json.loads(line)
        except ValueError as error:
            raise OperationalError("cast_integrity", f"pending proposal file has an unreadable line: {path}") from error
        if isinstance(proposal, dict) and proposal.get("type") == "variant_draft_root" and proposal_line(proposal) == line:
            found[proposal_sha(proposal)] = proposal
    return found


def _scope_tuple(scope: dict) -> tuple[str, str, str, int]:
    return scope["chapter_sha256"], scope["quote_sha256"], scope["label"], scope["span_start"]


def _revalidated_root(source: _ChapterSource, root: dict) -> list[dict]:
    """Re-prove every claim of a stored root against the exact current source; any drift or forgery fails the file closed.

    Returns one dict per included mention: its scope, exact unit, stored item and the mention's validated witnesses.
    """
    if not isinstance(root, dict) or set(root) != ROOT_FIELDS:
        raise _approval_invalid("stored root has an invalid schema")
    actor_id, name, bio, look = root["actor_id"], root["name"], root["bio"], root["look"]
    if (
        not all(isinstance(item, str) for item in (actor_id, name, root["kind"], bio, look))
        or not IDENTIFIER.match(actor_id)
        or not name.strip()
    ):
        raise _approval_invalid("stored root has invalid values")
    facts = _exact_cited_witnesses(source, root["source_facts"], "root source fact")
    citations = root["citations"]
    if not facts or facts != root["source_facts"] or not isinstance(citations, dict) or set(citations) != set(VARIANT_CITATION_FIELDS):
        raise _approval_invalid("stored root source facts or citations are not exact")
    fact_units = {_unit_key(w) for w in facts}
    for field in VARIANT_CITATION_FIELDS:
        checked = _exact_cited_witnesses(source, citations[field], f"root {field} citation")
        if checked != citations[field]:
            raise _approval_invalid(f"stored root {field} citations are not exact")
        fact_units |= {_unit_key(w) for w in checked}
    words = _name_words(name)
    variants = root["variants"]
    if not isinstance(variants, list) or not variants:
        raise _approval_invalid("stored root has no variants")
    mentions: list[dict] = []
    seen: set[tuple[str, str, str, int]] = set()
    for variant in variants:
        if not isinstance(variant, dict) or set(variant) != {"label", "scopes"} or not isinstance(variant["scopes"], list):
            raise _approval_invalid("stored root variant has an invalid schema")
        for item in variant["scopes"]:
            if not isinstance(item, dict) or not {"scope", "decision", "reason", "provenance", "witnesses", "link"} <= set(item):
                raise _approval_invalid("stored root mention has an invalid schema")
            scope = item["scope"]
            if not isinstance(scope, dict) or set(scope) != ROOT_SCOPE_FIELDS or scope["label"] != variant["label"]:
                raise _approval_invalid("stored root mention scope is invalid")
            span, label = scope["span_start"], scope["label"]
            if type(span) is not int or span < 0 or not isinstance(label, str) or not label or item["decision"] != "new_actor":
                raise _approval_invalid("stored root mention has invalid values")
            path = source.resolve(scope["chapter"], scope["chapter_sha256"], "root mention")
            unit = source.units(path)[1].get(scope["unit_id"])
            if (
                unit is None
                or text_digest(unit["quote"]) != scope["quote_sha256"]
                or unit["quote"][span : span + len(label)] != label
            ):
                raise _approval_invalid("stored root mention is not an exact immutable span of the current source")
            if _scope_tuple(scope) in seen:
                raise _approval_invalid("stored root repeats one exact mention scope")
            seen.add(_scope_tuple(scope))
            witnesses = _exact_cited_witnesses(source, item["witnesses"], "root mention witness")
            if not witnesses:
                raise _approval_invalid("stored root mention has no source witnesses")
            _revalidate_link(source, item, unit, witnesses, fact_units, words)
            mentions.append({"scope": scope, "unit": unit, "item": item, "witnesses": witnesses})
    return mentions


def _revalidate_link(source: _ChapterSource, item: dict, unit: dict, witnesses: list[dict], fact_units: set, words: list[str]) -> None:
    link, scope = item["link"], item["scope"]
    label = scope["label"]
    if not isinstance(link, dict):
        raise _approval_invalid("stored root mention link is invalid")
    if link.get("kind") == "shared_immutable_unit":
        mention_units = {(scope["chapter_sha256"], scope["unit_id"]): unit["quote"]}
        mention_units.update({_unit_key(w): w["unit_quote"] for w in witnesses})
        shared = link.get("units")
        if not isinstance(shared, list) or not shared:
            raise _approval_invalid("stored root shared-unit link names no units")
        for entry in shared:
            key = (entry.get("chapter_sha256"), entry.get("unit_id")) if isinstance(entry, dict) else None
            quote = mention_units.get(key) if key in fact_units else None
            if quote is None or not (_names_word(label, quote) or any(_names_word(word, quote) for word in words)):
                raise _approval_invalid("stored root shared-unit link is not proven by the current source")
        return
    if link.get("kind") in VARIANT_LINK_KINDS:
        proof = _exact_cited_witnesses(source, link.get("witnesses"), "root link witness")
        if not proof or not _scene_link_proven(link["kind"], proof, scope, label, words, source):
            raise _approval_invalid("stored root explicit link is not proven by the current source")
        return
    raise _approval_invalid("stored root mention link kind is unknown")


def _kin_only_label(label: str) -> bool:
    words = [word.casefold() for word in re.findall(r"\w+", label)]
    return bool(words) and any(w in KIN_LABEL_WORDS for w in words) and all(
        w in KIN_LABEL_WORDS or w in KIN_MODIFIER_WORDS or w in NON_NAME_COMPOUND_WORDS for w in words
    )


def variant_root_participant_blockers(root: dict, mentions: list[dict], registry: dict, aliases: dict, source: _ChapterSource) -> list[str]:
    """A family-only label names a relation, not a person: it may join a root only while the mention's own bounded scene shows
    exactly one named participant (the root itself). A scene that names two or more participants (e.g. two siblings)
    leaves the relation label ambiguous, so the root is blocked rather than the label guessed."""
    root_tokens = {normalized_id(word) for word in _name_words(root["name"])}
    blockers: list[str] = []
    for mention in mentions:
        scope = mention["scope"]
        if not _kin_only_label(scope["label"]):
            continue
        path = source.resolve(scope["chapter"], scope["chapter_sha256"], "root mention")
        ordered, by_id = source.units(path)
        order = [unit["id"] for unit in ordered]
        scene = scene_units_at(by_id, order, order.index(scope["unit_id"]))
        participants: set[str] = set()
        for reference in immutable_name_references(scene).values():
            label = reference["label"]
            tokens = label_components(label)
            if not tokens or not (tokens - TITLE_ROLE_TOKENS) or _kin_only_label(label):
                continue
            if tokens & root_tokens:
                participants.add("\0root")
                continue
            owners = [
                owner
                for owner in adjudication_owners(label, registry, aliases)
                if owner_relevance(label, owner, registry[owner], aliases) >= 3
            ]
            initial = not by_id[reference["unit_id"]]["quote"][: reference["start"]].strip()
            if not owners and initial and " " not in label.strip():
                # a lone sentence-initial capitalized word cannot be told from a capitalized common word by lexical evidence alone
                continue
            participants.add(owners[0] if owners else normalized_id(label))
        if len(participants) >= 2:
            blockers.append(
                f"family label {scope['label']!r}@{scope['span_start']} is ambiguous: its scene shows {len(participants)} named participants"
            )
    return blockers


def variant_root_duplicate_blockers(root: dict, registry: dict, aliases: dict, characters: dict, voices: dict) -> list[str]:
    """Independent duplicate gate: the narrator, original profiles/voices, registered names and every variant spelling."""
    actor_id, name = root["actor_id"], root["name"]
    id_norm, name_norm = normalized_id(actor_id), normalized_id(name)
    blockers: list[str] = []
    if id_norm != actor_id or not name_norm:
        blockers.append("actor id or name is not a normalized identifier")
    if "narrator" in {id_norm, name_norm} or id_norm in ANCHOR_IDS or name_norm in ANCHOR_IDS:
        blockers.append("a root can never take the narrator or an original anchor identity")
    for key, entry in registry.items():
        if key != "narrator" and isinstance(entry, dict) and normalized_id(str(entry.get("name", key))) == name_norm:
            blockers.append(f"name {name!r} is already the name of registered actor {key}")
    for store, label in ((characters, "profile"), (voices, "voice")):
        for key, info in store.items():
            names = {normalized_id(key)}
            if isinstance(info, dict) and "name" in info:
                names.add(normalized_id(str(info["name"])))
            if names & {id_norm, name_norm}:
                blockers.append(f"{actor_id} would duplicate original {label} entry {key}")
    for variant in root["variants"]:
        label = variant["label"]
        label_id = normalized_id(label)
        if label_id == "narrator":
            blockers.append(f"variant {label!r} is the narrator")
        target = aliases.get(label_id)
        if target and target != actor_id:
            blockers.append(f"variant {label!r} is already an alias of {target}")
        for owner, entry in registry.items():
            if owner != "narrator" and isinstance(entry, dict) and _shares_identifying_name(label, owner, entry, aliases):
                blockers.append(f"variant {label!r} is literally carried by registered actor {owner}")
    return list(dict.fromkeys(blockers))


def _validated_reviews(source: _ChapterSource, sha: str, reviews: object, mentions: list[dict]) -> list[dict]:
    """One exact review per included mention. A structurally invalid review is _MalformedApproval; a witness that is not the exact current source stays OperationalError."""
    if not isinstance(reviews, list) or not reviews:
        raise _MalformedApproval("approval requires one review per included mention")
    expected = {_scope_tuple(m["scope"]): m["scope"] for m in mentions}
    checked: dict[tuple, dict] = {}
    for review in reviews:
        if not isinstance(review, dict) or set(review) != ROOT_REVIEW_FIELDS or review["proposal_sha256"] != sha:
            raise _MalformedApproval("review has an invalid schema or cites a different proposal sha")
        scope = review["scope"]
        if (
            not isinstance(scope, dict)
            or set(scope) != ROOT_SCOPE_FIELDS
            or type(scope["span_start"]) is not int
            or not all(isinstance(scope[field], str) for field in ROOT_SCOPE_FIELDS - {"span_start"})
        ):
            raise _MalformedApproval("review scope is invalid")
        key = _scope_tuple(scope)
        if expected.get(key) != scope or key in checked:
            raise _MalformedApproval("review scope is not exactly one included mention of the proposal")
        if not isinstance(review["provenance"], str) or review["provenance"] not in REVIEW_PROVENANCES or review["verdict"] != DISTINCT_VERDICT:
            raise _MalformedApproval(f"review provenance must be one of {sorted(REVIEW_PROVENANCES)} with verdict {DISTINCT_VERDICT}")
        witnesses = review["factual_witnesses"]
        if review["provenance"] == NATIVE_REVIEW:
            if review["reviewer_role"] is not None or witnesses != [] or review["factual_basis"] != "":
                raise _MalformedApproval("a native review reference carries no reviewer role, witnesses or basis")
            exact: list[dict] = []
        else:
            basis, role = review["factual_basis"], review["reviewer_role"]
            if not isinstance(role, str) or role not in TRUSTED_REVIEW_ROLES or not isinstance(basis, str) or not basis.strip():
                raise _MalformedApproval("a source review needs a trusted reviewer role and a factual basis")
            if not isinstance(witnesses, list) or not witnesses or not all(
                isinstance(w, dict)
                and CONTEXT_V2_WITNESS_FIELDS <= set(w)
                and all(isinstance(w[field], str) for field in CONTEXT_V2_WITNESS_FIELDS)
                for w in witnesses
            ):
                raise _MalformedApproval("a source review needs exact factual witnesses of the documented schema")
            exact = _exact_cited_witnesses(source, witnesses, "source review witness")
        checked[key] = {**review, "factual_witnesses": exact}
    if set(checked) != set(expected):
        raise _MalformedApproval("reviews must cover every included mention exactly once")
    return [checked[key] for key in expected]


def _recorded_owner_support_holds(provenance: object, owner: str, entry: dict, registry: dict, aliases: dict) -> bool:
    """True while the support a native existing:<id> record carries is still honoured by the current registry."""
    if not isinstance(provenance, dict):
        return False
    if provenance.get("type") == "literal_witness":
        named = provenance.get("owner_name")
        return isinstance(named, str) and named.casefold() in {form.casefold() for form in owner_name_forms(owner, registry, aliases)}
    if provenance.get("type") == "content_anchor":
        anchor = provenance.get("anchor")
        return isinstance(anchor, str) and bool(anchor) and any(
            anchor.casefold() in {w.casefold() for w in re.findall(r"\b[a-zA-Z]{4,}\b", text)}
            for _field, text in extract_owner_profile_facts(entry)
        )
    return False


def _existing_owner_claim_is_current(
    native: dict, claim: str, mention: dict, registry: dict, aliases: dict, source: _ChapterSource
) -> bool:
    """A native existing:<id> claim is current only while its owner is a live registered actor AND either the support the record
    carries is still honoured by the registry or the current source scene of the exact mention re-proves it with the ordinary
    review guards. Anything else (owner gone, no recorded support and no scene proof) is stale/unusable."""
    owner = claim.split(":", 1)[1]
    entry = registry.get(owner)
    if owner == "narrator" or not isinstance(entry, dict):
        return False
    if native.get("verdict") == claim and _recorded_owner_support_holds(native.get("provenance"), owner, entry, registry, aliases):
        return True
    scope = mention["scope"]
    ordered, by_id = source.units(source.resolve(scope["chapter"], scope["chapter_sha256"], "native review mention"))
    scene = scene_units_at(by_id, [unit["id"] for unit in ordered], [unit["id"] for unit in ordered].index(scope["unit_id"]))
    return review_verdict_error(scope["label"], claim, [scope["unit_id"]], {}, {}, registry, aliases, by_id, mention["unit"], scene) is None


def _native_review_veto(native: dict, mention: dict, registry: dict, aliases: dict, source: _ChapterSource) -> str | None:
    """Why a native audit record that is not distinct_living_identity still binds the mention, or None when a source review may resolve it.

    Resolvable: `uncertain` (including an invalid/unsupported or low-confidence raw existing:/same_provisional: review) and a
    stale/unusable existing:<id>. Binding: a currently valid source-supported existing:<id> (recorded verdict, or the raw review
    behind an `uncertain` record), and every other verdict (nonidentity_fragment, same_provisional:*, unknown)."""
    verdict = native.get("verdict")
    claim = verdict
    if verdict == UNCERTAIN_VERDICT:
        raw = native.get("raw_review")
        claim = raw.get("verdict") if isinstance(raw, dict) else None
        if not (isinstance(claim, str) and claim.startswith("existing:")):
            return None
    elif not (isinstance(verdict, str) and verdict.startswith("existing:")):
        return f"native verdict {verdict!r} is a real conflict a source review cannot resolve"
    if _existing_owner_claim_is_current(native, claim, mention, registry, aliases, source):
        return f"{claim} is still supported by the current source and registry"
    return None


def _review_records_for_gate(
    reviews: list[dict], audit: list[dict], words: list[str], mentions: list[dict], registry: dict, aliases: dict, source: _ChapterSource
) -> tuple[list[dict], list[str]]:
    """Effective per-mention review records for variant_root_approval_gate, plus the reasons any review is unusable."""
    recorded = {
        (r.get("chapter_sha256"), r.get("quote_sha256"), r.get("label"), r.get("span_start")): r
        for r in audit
        if isinstance(r, dict)
    }
    mention_by_scope = {_scope_tuple(m["scope"]): m for m in mentions}
    effective: list[dict] = []
    blockers: list[str] = []
    for review in reviews:
        scope = review["scope"]
        native = recorded.get(_scope_tuple(scope))
        where = f"mention {scope['label']!r}@{scope['span_start']}"
        if review["provenance"] == NATIVE_REVIEW:
            raw = native.get("raw_review") if native else None
            if not native or native.get("verdict") != DISTINCT_VERDICT or not isinstance(raw, dict) or raw.get("verdict") != DISTINCT_VERDICT:
                blockers.append(f"{where} has no native distinct-living-identity review in the audit")
                continue
            effective.append(native)
            continue
        resolved = None
        if native and native.get("verdict") != DISTINCT_VERDICT:
            why = _native_review_veto(native, mention_by_scope[_scope_tuple(scope)], registry, aliases, source)
            if why:
                blockers.append(f"{where}: the source review contradicts the native review verdict {native.get('verdict')!r}: {why}")
                continue
            resolved = native.get("verdict")
        if not any(_names_word(scope["label"], w["unit_quote"]) or any(_names_word(x, w["unit_quote"]) for x in words) for w in review["factual_witnesses"]):
            blockers.append(f"{where}: no source-review witness literally names the mention or the identity")
            continue
        effective.append(
            {
                "chapter_sha256": scope["chapter_sha256"],
                "quote_sha256": scope["quote_sha256"],
                "label": scope["label"],
                "span_start": scope["span_start"],
                "verdict": DISTINCT_VERDICT,
                "review_provenance": SOURCE_REVIEW,
                "reviewer_role": review["reviewer_role"],
                "resolved_native_verdict": resolved,
                "own_source_witness": {
                    "unit_id": scope["unit_id"],
                    "provenance": SOURCE_REVIEW,
                    "label": scope["label"],
                    "span_start": scope["span_start"],
                },
            }
        )
    return effective, blockers


def _scoped_alias_record(actor_id: str, sha: str, mention: dict, review: dict, note: str, resolved_native: object = None) -> dict:
    scope = mention["scope"]
    return {
        "chapter_sha256": scope["chapter_sha256"],
        "quote_sha256": scope["quote_sha256"],
        "label": scope["label"],
        "span_start": scope["span_start"],
        "canonical": actor_id,
        "decision": "alias",
        "confidence": 1.0,
        "reason": f"[approved variant root {sha[:16]}; {review['provenance']}] {note or 'approved by caretaker input'}"[:300],
        "provenance": mention["item"]["provenance"] if isinstance(mention["item"]["provenance"], dict) else {},
        "witnesses": [{"chapter_sha256": w["chapter_sha256"], "unit_id": w["unit_id"]} for w in mention["witnesses"]],
        "approved_root": {
            "actor_id": actor_id,
            "proposal_sha256": sha,
            "review_provenance": review["provenance"],
            "reviewer_role": review["reviewer_role"],
            **({"resolved_native_verdict": resolved_native} if resolved_native else {}),
        },
    }


def _root_source_scope(mentions: list[dict]) -> dict:
    names = {m["scope"]["chapter"]: m["scope"]["chapter_sha256"] for m in mentions}
    return {"source": sorted(names), "source_hash": dict(sorted(names.items()))}


def _resolve_root_rows(recovery: RecoveryLedger, root: dict, sha: str, mentions: list[dict]) -> None:
    """Append resolutions for exactly this root's pending rows: its draft row and the new_actor rows of its included mentions."""
    actor_id = root["actor_id"]
    scopes = {_scope_tuple(m["scope"]) for m in mentions}
    for row in recovery.open_pending(CONTEXT_V2_STAGE):
        evidence = row["evidence"] if isinstance(row["evidence"], dict) else {}
        scope = evidence.get("scope")
        if (
            row["code"] == "context_new_actor"
            and isinstance(scope, dict)
            and evidence.get("canonical_target") == actor_id
            and (scope.get("chapter_sha256"), scope.get("quote_sha256"), scope.get("label"), scope.get("span_start")) in scopes
        ) or (
            row["code"] == VARIANT_DRAFT_ROOT
            and evidence.get("proposal_sha256") == sha
            and variant_draft_actor(row["item"]) == actor_id
        ):
            recovery.resolve(row, "approved_root_materialized", {"proposal_sha256": sha})
    for row in recovery.open_pending(ROOT_APPROVAL_STAGE):
        if row["item"].startswith(f"root_approval:{actor_id}:{sha[:16]}:"):
            recovery.resolve(row, "approved_root_materialized", {"proposal_sha256": sha})


def _close_malformed_rows(recovery: RecoveryLedger, current: set[str]) -> None:
    """Resolve every open malformed-entry row whose entry is no longer in the file (fixed, replaced or removed)."""
    for row in recovery.open_pending(ROOT_APPROVAL_STAGE):
        digest = row["evidence"].get("entry_sha256") if isinstance(row["evidence"], dict) else None
        if row["code"] == ROOT_APPROVAL_MALFORMED and digest not in current:
            recovery.resolve(row, "root_approval_entry_superseded", {"entry_sha256": digest})


def ingest_root_approvals(
    project: Path, source_sha: str, chapters: list[Path], progress: dict, recovery: RecoveryLedger
) -> dict:
    """Materialize every root the caretaker input approves and that passes every gate; return what happened.

    Fail closed (OperationalError, nothing written) on file-level problems: unreadable file, wrong contract/version/fields,
    wrong source book sha, an approval naming a proposal sha that is unknown / another actor's / not open, or any stored root
    or cited witness that is not the exact current source. An approval entry that is invalid on its own (schema, values, human
    claim, repeated actor/sha, malformed reviews) is never a crash: it becomes a typed `root_approval_malformed` pending row
    and the file is held whole (no root of it is materialized until every entry is well formed). A gate failure only blocks
    that one root (typed pending row, nothing written). Idempotent: an exact already-registered root only re-asserts its
    scoped audit and row resolutions.
    """
    outcome: dict[str, list] = {"materialized": [], "already": [], "blocked": [], "malformed": []}
    path = project / ROOT_APPROVAL_NAME
    if not path.is_file():
        _close_malformed_rows(recovery, set())
        return outcome
    payload = load_object(path, "caretaker root approvals")
    if (
        set(payload) != {"contract", "version", "source_sha256", "approvals"}
        or payload["contract"] != ROOT_APPROVAL_CONTRACT
        or payload["version"] != ROOT_APPROVAL_VERSION
        or not isinstance(payload["approvals"], list)
    ):
        raise _approval_invalid("unsupported contract or schema")
    if payload["source_sha256"] != source_sha:
        raise _approval_invalid("file belongs to a different source book")
    source = _ChapterSource(chapters)
    stored = _stored_pending_roots(project)
    open_roots = {
        row["evidence"].get("proposal_sha256")
        for row in recovery.open_pending(CONTEXT_V2_STAGE)
        if row["code"] == VARIANT_DRAFT_ROOT and isinstance(row["evidence"], dict)
    }
    registry, aliases = progress["registry"], progress["aliases"]
    plan: list[dict] = []
    malformed: list[tuple[object, str]] = []
    seen_actors: set[str] = set()
    seen_shas: set[str] = set()
    for approval in payload["approvals"]:
        try:
            actor_id, sha, _note = _checked_approval_entry(approval)
            if actor_id in seen_actors or sha in seen_shas:
                raise _MalformedApproval("approval repeats an actor or proposal")
            seen_actors.add(actor_id)
            seen_shas.add(sha)
            root = stored.get(sha)
            if root is None or root["actor_id"] != actor_id:
                raise _approval_invalid(f"no pending proposal for actor {actor_id!r} has the exact sha {sha!r}")
            already = _materialized_root_sha(registry, actor_id) == sha
            if not already and sha not in open_roots:
                raise _approval_invalid(f"proposal {sha[:16]} for {actor_id!r} is not an open pending proposal (superseded or resolved)")
            mentions = _revalidated_root(source, root)
            reviews = _validated_reviews(source, sha, approval["reviews"], mentions)
        except _MalformedApproval as error:
            malformed.append((approval, str(error)))
            continue
        plan.append({"approval": approval, "root": root, "sha": sha, "mentions": mentions, "reviews": reviews, "already": already})
    digests = {payload_hash(entry): (entry, why) for entry, why in malformed}
    _close_malformed_rows(recovery, set(digests))
    if digests:
        for digest, (entry, why) in digests.items():
            recovery.record(
                ROOT_APPROVAL_STAGE,
                f"{ROOT_APPROVAL_MALFORMED}:{digest[:16]}",
                ROOT_APPROVAL_MALFORMED,
                f"approval entry is malformed: {why}; no root of {ROOT_APPROVAL_NAME} is materialized until every entry is well formed",
                severity="pending",
                evidence={
                    "entry_sha256": digest,
                    "actor_id": entry.get("actor_id") if isinstance(entry, dict) and isinstance(entry.get("actor_id"), str) else None,
                    "problem": why,
                },
            )
            outcome["malformed"].append({"entry_sha256": digest, "problem": why})
        return outcome
    if not plan:
        return outcome
    characters = load_object(project / "characters.json", "characters profile")
    voices = load_object(project / "voices.json", "voice profiles")
    audit = load_new_identity_audit(project)
    records = load_scoped_audit(project)
    for item in plan:
        root, sha, mentions, reviews = item["root"], item["sha"], item["mentions"], item["reviews"]
        actor_id = root["actor_id"]
        known = mention_scoped_audit_index(records)
        review_by_scope = {_scope_tuple(r["scope"]): r for r in reviews}
        effective, blockers = _review_records_for_gate(reviews, audit, _name_words(root["name"]), mentions, registry, aliases, source)
        resolved_native = {
            (e["chapter_sha256"], e["quote_sha256"], e["label"], e["span_start"]): e.get("resolved_native_verdict") for e in effective
        }
        wanted = {
            _scope_tuple(m["scope"]): _scoped_alias_record(
                actor_id, sha, m, review_by_scope[_scope_tuple(m["scope"])], item["approval"]["note"], resolved_native.get(_scope_tuple(m["scope"]))
            )
            for m in mentions
        }
        if not item["already"]:
            blockers += variant_root_approval_gate(root, registry, aliases, effective)
            blockers += variant_root_duplicate_blockers(root, registry, aliases, characters, voices)
            blockers += variant_root_participant_blockers(root, mentions, registry, aliases, source)
            for scope_key, record in wanted.items():
                prior = known.get(scope_key)
                # An earlier exact-scope ambiguity is explicitly unresolved evidence, not a contrary identity decision.
                # This approval has independently re-proven that scope from current source witnesses, so supersede it
                # append-only. Every positive conflicting decision (existing alias, non-character, etc.) remains a veto.
                supersedable_ambiguity = (
                    prior
                    and prior.get("decision") == "ambiguous"
                    and prior.get("canonical") in {None, "", "none"}
                )
                if prior and not supersedable_ambiguity and (prior["decision"], prior["canonical"]) != ("alias", actor_id):
                    blockers.append(f"mention {record['label']!r}@{record['span_start']} already has scoped decision {prior['decision']}:{prior['canonical']}")
                mention = next(m for m in mentions if _scope_tuple(m["scope"]) == scope_key)
                if not any(
                    ref["label"] == record["label"] and ref["start"] == record["span_start"]
                    for ref in immutable_name_references([mention["unit"]]).values()
                ):
                    blockers.append(f"mention {record['label']!r}@{record['span_start']} is not an exact immutable name reference")
            blockers = list(dict.fromkeys(blockers))
            prefix = f"root_approval:{actor_id}:{sha[:16]}:"
            current = f"{prefix}{payload_hash(blockers)[:12]}"
            stale = [
                row
                for row in recovery.open_pending(ROOT_APPROVAL_STAGE)
                if row["item"].startswith(prefix) and (not blockers or not row["item"].startswith(current))
            ]
            for row in stale:
                recovery.resolve(row, "root_approval_superseded", {"proposal_sha256": sha})
            if blockers and not any(row["item"].startswith(current) for row in recovery.open_pending(ROOT_APPROVAL_STAGE)):
                # a generation suffix lets a blocker set that recurs after being superseded open a fresh row
                generation = sum(
                    1
                    for e in recovery.entries()
                    if e["stage"] == ROOT_APPROVAL_STAGE and e["severity"] == "pending" and e["item"].startswith(current)
                )
                recovery.record(
                    ROOT_APPROVAL_STAGE,
                    f"{current}:{generation}",
                    ROOT_APPROVAL_BLOCKED,
                    f"approval of root {actor_id} is blocked: " + "; ".join(blockers),
                    severity="pending",
                    evidence={**_root_source_scope(mentions), "actor_id": actor_id, "proposal_sha256": sha, "blockers": blockers},
                )
            if blockers:
                outcome["blocked"].append({"actor_id": actor_id, "proposal_sha256": sha, "blockers": blockers})
                continue
        # 1. scoped audit (idempotent: only absent scopes are appended)
        added = False
        for scope_key, record in wanted.items():
            prior = known.get(scope_key)
            supersedable_ambiguity = (
                prior
                and prior.get("decision") == "ambiguous"
                and prior.get("canonical") in {None, "", "none"}
            )
            if prior is None or supersedable_ambiguity:
                supersede_scoped_record(records, known, scope_key, record)
                added = True
        if added:
            mention_scoped_audit_index(records)
            atomic_json(project / SCOPED_AUDIT_NAME, {"records": records})
        # 2. registry + exact aliases, durable in the progress file
        if not item["already"]:
            look = root["look"].strip()
            registry[actor_id] = {
                "name": root["name"],
                "bio": "",
                "look": "",
                "origin": "approved_root",
                "facts": {
                    "voice": [root["bio"]],
                    "look": [] if look.rstrip(".").casefold() == NO_VISUAL_DETAILS else [root["look"]],
                },
                "approved_root": {
                    "proposal_sha256": sha,
                    "proposal_file": PROPOSALS_NAME,
                    "approval": item["approval"],
                    "variant_labels": [variant["label"] for variant in root["variants"]],
                },
            }
            for alias in (actor_id, root["name"]):
                aliases[normalized_id(alias)] = actor_id
            atomic_json(project / PROGRESS_NAME, progress)
        # 3. durable resolution of exactly this root's pending rows
        _resolve_root_rows(recovery, root, sha, mentions)
        outcome["already" if item["already"] else "materialized"].append(actor_id)
    return outcome


# ##################################################################
# source-reviewed retirement of prepared prose actors
# The ONLY way a prepared actor leaves the registry. Input is one caretaker file (docs/CAST_SOURCE_RETIREMENTS.md) that cites
# the exact book sha and the exact current registry digest (whole-file mismatch fails closed, nothing written) and, per
# actor, exact immutable source evidence plus the actor's complete set of own name references, each recorded as a
# mention-scoped non_character decision. An entry that fails any gate becomes a typed pending row for that actor alone
# (blocking freeze) and writes nothing. Original anchors/profiles are never retirable, and an actor anything else still
# references (another alias, a scoped alias, an alias audit record, a context/root/draft row) is never dropped, so no
# speaker can silently fall back to the narrator or to an inactive alias. History is append-only.
SOURCE_RETIREMENT_NAME = "cast_source_reviewed_retirements.json"
SOURCE_RETIREMENT_CONTRACT = "cast_source_reviewed_retirement"
SOURCE_RETIREMENT_VERSION = 1
SOURCE_RETIREMENT_STAGE = "cast_source_retirement"
SOURCE_RETIREMENT_BLOCKED = "source_retirement_blocked"
SOURCE_RETIRE_ACTION = "retire_actor"
SOURCE_RETIREMENT_HISTORY = "source_reviewed_retirements"
SOURCE_RETIREMENT_INPUTS = "source_retirement_inputs"
SOURCE_RETIREMENT_FIELDS = frozenset({"action", "actor_id", "reviewer_role", "factual_basis", "evidence", "own_refs"})
PREPARED_ORIGIN = "prepared"
PENDING_ACTOR_KEYS = ("canonical_target", "actor_id", "canonical", "owner", "target")


def registry_digest(registry: dict) -> str:
    """The exact digest a retirement input must cite for the registry the caretaker reviewed (stable JSON of all of it)."""
    return json_digest(registry)


def _retirement_invalid(message: str) -> OperationalError:
    return OperationalError("cast_integrity", f"caretaker source-reviewed retirement: {message}")


def _source_own_refs(source: _ChapterSource, chapters: list[Path], label_ids: set[str]) -> dict[tuple, set[tuple[str, str]]]:
    """Every exact immutable name reference of the whole book whose label is one of the actor's own name ids."""
    expected: dict[tuple, set[tuple[str, str]]] = {}
    for path in chapters:
        try:
            units, by_id = source.units(path)
        except DataIssue as error:
            raise _retirement_invalid(f"source chapter {path.name} cannot be read as exact evidence: {error}") from error
        for reference in immutable_name_references(units).values():
            if normalized_id(reference["label"]) in label_ids:
                unit = by_id[reference["unit_id"]]
                key = (unit["chapter_sha256"], text_digest(unit["quote"]), reference["label"], reference["start"])
                expected.setdefault(key, set()).add((path.name, unit["id"]))
    return expected


def _row_references_actor(row: dict, actor_id: str) -> bool:
    evidence = row["evidence"] if isinstance(row.get("evidence"), dict) else {}
    owners = evidence.get("owners")
    return (
        any(evidence.get(key) == actor_id for key in PENDING_ACTOR_KEYS)
        or (isinstance(owners, list) and actor_id in owners)
        or str(row.get("item", "")).startswith(f"{VARIANT_DRAFT_ITEM}{actor_id}:")
    )


def _retirement_own_refs(entry: dict, expected: dict, blockers: list[str]) -> list[dict]:
    raw = entry["own_refs"]
    if not expected:
        blockers.append("the actor has no exact own source reference, so nothing proves it is a non-character")
        return []
    if not isinstance(raw, list) or not raw:
        blockers.append("own_refs must list every exact own source reference of the actor")
        return []
    scopes: dict[tuple, dict] = {}
    for scope in raw:
        if (
            not isinstance(scope, dict)
            or set(scope) != ROOT_SCOPE_FIELDS
            or not all(isinstance(scope[f], str) and scope[f] for f in ROOT_SCOPE_FIELDS - {"span_start"})
            or type(scope["span_start"]) is not int
        ):
            blockers.append("an own_refs entry has an invalid scope schema")
            return []
        key = _scope_tuple(scope)
        if key in scopes or (scope["chapter"], scope["unit_id"]) not in expected.get(key, set()):
            blockers.append(f"own ref {scope['label']!r}@{scope['span_start']} is repeated or not an exact own reference of the actor in the current source")
            return []
        scopes[key] = scope
    missing = [f"{k[2]!r}@{k[3]}" for k in expected if k not in scopes]
    if missing:
        blockers.append(f"own_refs omit {len(missing)} exact own source reference(s): {', '.join(sorted(missing)[:5])}")
        return []
    return [scopes[key] for key in sorted(scopes)]


def _retirement_blockers(entry: dict, ctx: dict) -> tuple[list[str], dict]:
    """Every reason this one entry may not retire its actor, plus the validated evidence/own refs it would apply."""
    registry, aliases, actor_id = ctx["registry"], ctx["aliases"], entry["actor_id"]
    blockers: list[str] = []
    held = registry.get(actor_id)
    if actor_id in ANCHOR_IDS:
        blockers.append(f"{actor_id!r} is an original anchor identity and is never retired")
    if not isinstance(held, dict):
        return [*blockers, f"{actor_id!r} is not an active registry actor"], {}
    if held.get("origin") != PREPARED_ORIGIN:
        blockers.append(f"{actor_id!r} is not a prepared prose actor (origin {held.get('origin')!r})")
    if actor_id in ctx["characters"] or actor_id in ctx["voices"]:
        blockers.append(f"{actor_id!r} has an original character/voice profile that is never retired")
    name = str(held.get("name", actor_id))
    label_ids = {actor_id, normalized_id(name)} - {""}
    role, basis = entry["reviewer_role"], entry["factual_basis"]
    if role not in TRUSTED_REVIEW_ROLES or not isinstance(basis, str) or not basis.strip():
        blockers.append("a retirement needs a trusted reviewer role and a factual basis")
    evidence: list[dict] = []
    raw = entry["evidence"]
    if not isinstance(raw, list) or not raw or any(not isinstance(w, dict) or set(w) != CONTEXT_V2_WITNESS_FIELDS for w in raw):
        blockers.append("evidence must be a non-empty list of exact witness units")
    else:
        try:
            evidence = _exact_cited_witnesses(ctx["source"], raw, "retirement evidence")
        except (OperationalError, DataIssue) as error:
            blockers.append(str(error))
        else:
            if evidence != raw:
                blockers.append("evidence repeats a unit")
            elif not any(_names_word(word, w["unit_quote"]) for w in evidence for word in (name, actor_id.replace("_", " "))):
                blockers.append(f"no evidence unit literally names the actor {name!r}")
    own_refs = _retirement_own_refs(entry, _source_own_refs(ctx["source"], ctx["chapters"], label_ids), blockers)
    keys = sorted(key for key, target in aliases.items() if target == actor_id and key not in label_ids)
    if keys:
        blockers.append(f"alias(es) {keys} still resolve to the actor, so retiring would drop a referenced actor")
    if aliases.get(actor_id, actor_id) != actor_id:
        blockers.append(f"the actor id is itself an alias of {aliases[actor_id]!r}")
    for record in ctx["scoped"]:
        if record["decision"] == "alias" and record["canonical"] == actor_id:
            blockers.append(f"mention {record['label']!r}@{record['span_start']} is scoped to the actor as an alias")
            break
    for record in ctx["audit"]:
        if isinstance(record, dict) and record.get("decision") != "distinct" and actor_id in {
            normalized_id(str(record.get("alias", ""))),
            normalized_id(str(record.get("canonical", ""))),
        }:
            blockers.append("the alias audit references the actor")
            break
    for stage in (CONTEXT_V2_STAGE, ROOT_APPROVAL_STAGE, "cast"):
        if any(_row_references_actor(row, actor_id) for row in ctx["recovery"].open_pending(stage)):
            blockers.append(f"an open {stage} pending row still references the actor")
    known = mention_scoped_audit_index(ctx["scoped"])
    for scope in own_refs:
        prior = known.get(_scope_tuple(scope))
        if prior and prior["decision"] == "alias":
            blockers.append(f"own ref {scope['label']!r}@{scope['span_start']} already has scoped alias decision {prior['canonical']!r}")
    return list(dict.fromkeys(blockers)), {"evidence": evidence, "own_refs": own_refs, "entry": held, "label_ids": label_ids}


def _non_character_record(entry: dict, entry_sha: str, scope: dict, evidence: list[dict]) -> dict:
    return {
        "chapter_sha256": scope["chapter_sha256"],
        "quote_sha256": scope["quote_sha256"],
        "label": scope["label"],
        "span_start": scope["span_start"],
        "canonical": "none",
        "decision": "non_character",
        "confidence": 1.0,
        "reason": f"[source-reviewed retirement of {entry['actor_id']}; {entry['reviewer_role']}] {entry['factual_basis']}"[:300],
        "provenance": {"author": entry["reviewer_role"]},
        "witnesses": [{"chapter_sha256": w["chapter_sha256"], "unit_id": w["unit_id"]} for w in evidence],
        "source_reviewed_retirement": {"actor_id": entry["actor_id"], "entry_sha256": entry_sha},
    }


def _record_retirement_blocked(recovery: RecoveryLedger, actor_id: str, entry_sha: str, blockers: list[str], source_sha: str) -> None:
    current = f"{SOURCE_RETIREMENT_STAGE_ITEM}{actor_id}:{entry_sha[:16]}:{payload_hash(blockers)[:12]}"
    if any(row["item"].startswith(current) for row in recovery.open_pending(SOURCE_RETIREMENT_STAGE)):
        return
    # a generation suffix lets a blocker set that recurs after being superseded open a fresh row
    generation = sum(
        1
        for e in recovery.entries()
        if e["stage"] == SOURCE_RETIREMENT_STAGE and e["severity"] == "pending" and e["item"].startswith(current)
    )
    recovery.record(
        SOURCE_RETIREMENT_STAGE,
        f"{current}:{generation}",
        SOURCE_RETIREMENT_BLOCKED,
        f"source-reviewed retirement of {actor_id} is blocked: " + "; ".join(blockers),
        severity="pending",
        evidence={
            "source": ["book"],
            "source_hash": {"book": source_sha},
            "actor_id": actor_id,
            "entry_sha256": entry_sha,
            "blockers": blockers,
        },
    )


SOURCE_RETIREMENT_STAGE_ITEM = "source_retirement:"


def ingest_source_reviewed_retirements(
    project: Path, source_sha: str, chapters: list[Path], progress: dict, recovery: RecoveryLedger
) -> dict:
    """Retire every prepared prose actor the caretaker input names and that passes every gate; return what happened.

    The whole file (contract, book sha, exact current registry digest, entry schema) fails closed before any write. A gate
    failure only blocks that one actor (typed pending row, nothing written for it). An input file is processed once
    (its hash is recorded append-only), so a file left in place never replays against a registry it no longer matches.
    """
    outcome: dict = {"retired": [], "blocked": [], "already": False}
    path = project / SOURCE_RETIREMENT_NAME
    if not path.is_file():
        return outcome
    payload = load_object(path, "caretaker source-reviewed retirements")
    if (
        set(payload) != {"contract", "version", "source_sha256", "registry_sha256", "retirements"}
        or payload["contract"] != SOURCE_RETIREMENT_CONTRACT
        or payload["version"] != SOURCE_RETIREMENT_VERSION
        or not isinstance(payload["registry_sha256"], str)
        or not isinstance(payload["retirements"], list)
    ):
        raise _retirement_invalid("unsupported contract or schema")
    if payload["source_sha256"] != source_sha:
        raise _retirement_invalid("file belongs to a different source book")
    history, inputs = progress.get(SOURCE_RETIREMENT_HISTORY, []), progress.get(SOURCE_RETIREMENT_INPUTS, [])
    if not isinstance(history, list) or not isinstance(inputs, list):
        raise _retirement_invalid("progress retirement history is invalid")
    input_sha = payload_hash(payload)
    if input_sha in inputs:
        outcome["already"] = True
        return outcome
    registry, aliases = progress["registry"], progress["aliases"]
    before = registry_digest(registry)
    if payload["registry_sha256"] != before:
        raise _retirement_invalid("file was reviewed against a different registry than the current one")
    seen: set[str] = set()
    for entry in payload["retirements"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != SOURCE_RETIREMENT_FIELDS
            or entry["action"] != SOURCE_RETIRE_ACTION
            or not isinstance(entry["actor_id"], str)
            or not IDENTIFIER.match(entry["actor_id"])
            or entry["actor_id"] in seen
            or HUMAN_REVIEW_CLAIM.search(json.dumps(entry))
        ):
            raise _retirement_invalid(
                f"entry has an invalid schema, repeats an actor or claims a human review (only {SOURCE_RETIRE_ACTION!r} exists)"
            )
        seen.add(entry["actor_id"])
    ctx = {
        "registry": registry,
        "aliases": aliases,
        "chapters": chapters,
        "source": _ChapterSource(chapters),
        "characters": load_object(project / "characters.json", "characters profile"),
        "voices": load_object(project / "voices.json", "voice profiles"),
        "scoped": load_scoped_audit(project),
        "recovery": recovery,
        "audit": _alias_audit_records(project),
    }
    ready: list[tuple[dict, str, dict]] = []
    for entry in payload["retirements"]:
        entry_sha = payload_hash(entry)
        blockers, plan = _retirement_blockers(entry, ctx)
        actor_id = entry["actor_id"]
        current = f"{SOURCE_RETIREMENT_STAGE_ITEM}{actor_id}:{entry_sha[:16]}:{payload_hash(blockers)[:12]}"
        for row in recovery.open_pending(SOURCE_RETIREMENT_STAGE):
            if row["item"].startswith(f"{SOURCE_RETIREMENT_STAGE_ITEM}{actor_id}:") and (not blockers or not row["item"].startswith(current)):
                recovery.resolve(row, "source_retirement_superseded", {"entry_sha256": entry_sha})
        if blockers:
            _record_retirement_blocked(recovery, actor_id, entry_sha, blockers, source_sha)
            outcome["blocked"].append({"actor_id": actor_id, "entry_sha256": entry_sha, "blockers": blockers})
        else:
            ready.append((entry, entry_sha, plan))
    # 1. scoped audit: one non_character decision per own reference (idempotent; a superseded decision stays as history)
    records, wrote = ctx["scoped"], False
    for entry, entry_sha, plan in ready:
        for scope in plan["own_refs"]:
            prior = mention_scoped_audit_index(records).get(_scope_tuple(scope))
            if prior and prior["decision"] == "non_character":
                continue
            supersede_scoped_record(records, mention_scoped_audit_index(records), _scope_tuple(scope), _non_character_record(entry, entry_sha, scope, plan["evidence"]))
            wrote = True
    if wrote:
        mention_scoped_audit_index(records)
        atomic_json(project / SCOPED_AUDIT_NAME, {"records": records})
    # 2. registry/aliases and append-only history, durable in the progress file together with the processed input hash
    for entry, entry_sha, plan in ready:
        actor_id = entry["actor_id"]
        removed = sorted(key for key, target in aliases.items() if target == actor_id)
        history.append(
            {
                "actor_id": actor_id,
                "entry_sha256": entry_sha,
                "input_sha256": input_sha,
                "source_sha256": source_sha,
                "registry_sha256_before": before,
                "reviewer_role": entry["reviewer_role"],
                "factual_basis": entry["factual_basis"],
                "evidence": plan["evidence"],
                "own_refs": plan["own_refs"],
                "retired_entry": copy.deepcopy(plan["entry"]),
                "removed_aliases": removed,
            }
        )
        del registry[actor_id]
        for key in removed:
            del aliases[key]
        outcome["retired"].append(actor_id)
    progress[SOURCE_RETIREMENT_HISTORY] = history
    progress[SOURCE_RETIREMENT_INPUTS] = [*inputs, input_sha]
    atomic_json(project / PROGRESS_NAME, progress)
    # 3. durable resolution of the quality flags the retirement answers (rows are appended, never rewritten)
    for entry, entry_sha, _ in ready:
        for row in recovery.open_pending("cast_quality"):
            evidence = row["evidence"] if isinstance(row.get("evidence"), dict) else {}
            if str(row.get("code", "")).startswith("quality_") and evidence.get("registry_id") == entry["actor_id"]:
                recovery.resolve(row, "actor_retired_source_reviewed", {"actor_id": entry["actor_id"], "entry_sha256": entry_sha})
    return outcome


def _alias_audit_records(project: Path) -> list:
    path = project / AUDIT_NAME
    if not path.is_file():
        return []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise OperationalError("cast_integrity", "alias audit is unreadable") from error
    records = report if isinstance(report, list) else report.get("records") if isinstance(report, dict) else None
    if not isinstance(records, list):
        raise OperationalError("cast_integrity", "alias audit requires a records array")
    return records


# ##################################################################
# exact scope
# a recorded row is retried only while every chapter it names still exists with its exact recorded bytes hash; anything else is stale and stays open.
def exact_scope(row: dict, by_name: dict[str, Path]) -> bool:
    names, hashes = row_scope(row)
    return bool(
        isinstance(names, list)
        and isinstance(hashes, dict)
        and names
        and all(name in by_name and hashes.get(name) == file_digest(by_name[name]) for name in names)
    )


# ##################################################################
# replay pending
# re-runs each open pending row's exact scope (only when every recorded chapter still has its exact recorded hash), bounded per row; the structural cursor and semantic ledger never move. A scope that now yields no pending warning is resolved durably and coverage records it; otherwise the row stays open and freezes publication.
def replay_pending(
    project: Path,
    chapters: list[Path],
    progress: dict,
    ambiguous: set[str],
    source_text: str,
    recovery: RecoveryLedger,
    ask,
) -> dict:
    by_name = {path.name: path for path in chapters}
    coverage = semantic_coverage(progress)

    def retry(row: dict) -> tuple[str, dict] | None:
        names, hashes = row_scope(row)
        paths = [by_name[name] for name in names]
        RecoveryLedger.presented.clear()
        start = chapters.index(paths[0])
        try:
            units = immutable_evidence_units(paths)
            work = {"registry": copy.deepcopy(progress["registry"]), "aliases": dict(progress["aliases"])}
            for window in unit_windows(units, work["registry"], work["aliases"]):
                discoveries, _ = discover_batch(project, start, paths, window, source_text, work, ambiguous, None, ask)
                apply_discoveries(work["registry"], work["aliases"], discoveries)
        except DataIssue:
            return None
        key = (row["stage"], row["item_sha256"], row["code"], row.get("dedup_scope", ""), row.get("dedup_hash", ""))
        if key in RecoveryLedger.presented:
            return None
        progress["registry"], progress["aliases"] = (work["registry"], work["aliases"])
        # A clean replay may have resolved the row by a mention-scoped binding; disclose only the
        # bindings inspected from its exact source scope instead of presenting the outcome as owner-free.
        bindings = replay_model_bindings(project, units)
        coverage.setdefault("resolved_pending", []).append(
            {"item": row["item"], "code": row["code"], "source_hash": hashes, "model_bindings": bindings}
        )
        return "replayed_exact_scope_clean", {"attempt": len(recovery.attempts(row)), "model_bindings": bindings}

    return recovery.revalidate(
        "cast",
        lambda row: exact_scope(row, by_name),
        retry,
        severities=("pending",),
        attempt_limit=PENDING_REPLAY_LIMIT,
    )


# ##################################################################
# revalidate quarantines
# generic bounded exact-hash retry of every open cast quarantine row (legacy severity rows and new parse/model quarantines alike, never only pending). A chapter is retried only while its exact recorded hash still matches; a clean result (including an empty chapter, now a zero-coverage non-character attestation) replaces the quarantined coverage entry in place and keeps the superseded entry, discoveries, rejection archive and ledger rows as append-only history. The structural cursor never moves and source bytes are never changed or reinterpreted.
def revalidate_quarantines(
    project: Path,
    chapters: list[Path],
    progress: dict,
    ambiguous: set[str],
    source_text: str,
    recovery: RecoveryLedger,
    ask,
    limit: int | None = QUARANTINE_REVALIDATION_LIMIT,
    earlier: set[tuple[str, str, str]] | None = None,
) -> dict:
    by_name = {path.name: path for path in chapters}
    coverage = semantic_coverage(progress)
    progress_path = project / PROGRESS_NAME

    def quarantined_entries(names: list[str]) -> list[dict]:
        return [
            batch
            for batch in coverage["completed_batches"]
            if batch.get("quarantined") and set(batch["chapter_sha256"]) & set(names)
        ]

    def attested(names: list[str], hashes: dict) -> bool:
        return all(
            any(
                name in batch["chapter_sha256"] and batch["chapter_sha256"][name] == hashes[name]
                for batch in coverage["completed_batches"]
                if not batch.get("quarantined")
            )
            for name in names
        )

    def supersede(old: dict, units: list[dict]) -> None:
        entries = [
            write_semantic_record(
                project,
                unit["start"],
                unit["batch"],
                unit["units"],
                unit["classifications"],
                True,
                unit["quarantine"] is not None,
                unit["quarantine"] is None,
            )
            for unit in units
        ]
        batches = coverage["completed_batches"]
        position = batches.index(old)
        batches[position : position + 1] = entries
        structural = progress["completed_batches"]
        position = next(i for i, b in enumerate(structural) if (b["start"], b["end"]) == (old["start"], old["end"]))
        structural[position : position + 1] = [
            {
                "start": entry["start"],
                "end": entry["end"],
                "chapter_sha256": entry["chapter_sha256"],
                **({"quarantined": True} if entry.get("quarantined") else {}),
            }
            for entry in entries
        ]
        coverage.setdefault("superseded_quarantines", []).append({"entry": old, "replaced_by": entries})

    def rerun(old: dict) -> list[dict]:
        """Fresh recoverable attempt over exactly the old entry's chapters; clean units update the working registry."""
        units: list[dict] = []
        position = old["start"]
        while position < old["end"]:
            for unit in recoverable_batches(
                project, chapters[: old["end"]], position, progress, ambiguous, source_text, recovery, ask, None
            ):
                if unit["quarantine"] is None:
                    progress["registry"], progress["aliases"] = unit["registry"], unit["aliases"]
                units.append(unit)
                position = unit["start"] + len(unit["batch"])
        return units

    def retry(row: dict) -> tuple[str, dict] | None:
        names, hashes = row_scope(row)
        attempt = {"attempt": len(recovery.attempts(row))}
        open_entries = quarantined_entries(names)
        if not open_entries:
            # A previous pass replaced the coverage entry but stopped before this row's resolution was appended.
            return ("coverage_already_attested", attempt) if attested(names, hashes) else None
        for old in open_entries:
            exact_names = [path.name for path in chapters[old["start"] : old["end"]]]
            if exact_names != list(old["chapter_sha256"]) or not set(exact_names) <= set(names):
                return None
            units = rerun(old)
            if any(unit["quarantine"] is None for unit in units):
                supersede(old, units)
                atomic_json(progress_path, progress)
        if quarantined_entries(names):
            return None
        return "revalidated_clean", {
            **attempt,
            "empty_chapters": empty_chapter_attestations([by_name[n] for n in names]),
        }

    return recovery.revalidate(
        "cast",
        lambda row: exact_scope(row, by_name),
        retry,
        severities=("quarantine",),
        limit=limit,
        # A row recorded during this very run just failed; it waits for the next pass instead of burning an attempt.
        defer=None if earlier is None else lambda row: row_key(row) not in earlier,
    )


# ##################################################################
# plan quarantine revalidation
# read-only proof preparation: classifies every open cast quarantine row as eligible (exact hash matches, attempts remain), stale (source bytes differ or hash not recorded), or exhausted, plus quarantined coverage chapters, without any inference, write, or cursor change.
def plan_quarantine_revalidation(source: Path, project: Path | None = None) -> dict:
    source = source.resolve()
    project = project or get_output_dir(source)
    _, _, chapters = source_chapters(source, project)
    by_name = {path.name: path for path in chapters}
    recovery = RecoveryLedger(project)
    rows = recovery.entries()
    counts: dict[str, dict[str, int]] = {}
    for row in recovery.open_rows("cast", ("quarantine",)):
        if not exact_scope(row, by_name):
            state = "stale"
        elif len(recovery.attempts(row, rows)) >= REVALIDATION_ATTEMPT_LIMIT:
            state = "exhausted"
        else:
            state = "eligible"
        bucket = counts.setdefault(row["code"], {"eligible": 0, "stale": 0, "exhausted": 0})
        bucket[state] += 1
    progress_path = project / PROGRESS_NAME
    quarantined = empty = []
    if progress_path.exists():
        progress = load_object(progress_path, "cast preparation progress")
        quarantined = quarantined_chapter_names(progress)
        empty = sorted(empty_chapter_names(progress))
    return {
        "open_quarantine_rows": sum(sum(bucket.values()) for bucket in counts.values()),
        "by_code": counts,
        "quarantined_chapters": len(quarantined),
        "empty_chapters_attested": len(empty),
        "empty_chapters_in_source": len(empty_chapter_attestations(chapters)),
    }


# ##################################################################
# commit batch
# atomically advances registry, semantic ledger and (for production batches) the structural cursor for one recovered unit.
def commit_batch(project: Path, progress: dict, coverage: dict, unit: dict, revalidation: bool) -> None:
    batch, start = unit["batch"], unit["start"]
    if unit["quarantine"] is None:
        progress["registry"], progress["aliases"] = unit["registry"], unit["aliases"]
    record_semantic_batch(
        project,
        start,
        batch,
        unit["units"],
        unit["classifications"],
        coverage,
        revalidation,
        unit["quarantine"] is not None,
    )
    if revalidation:
        return
    progress["next_chapter"] = start + len(batch)
    progress["completed_batches"].append(
        {
            "start": start,
            "end": start + len(batch),
            "chapter_sha256": {path.name: file_digest(path) for path in batch},
            **({"quarantined": True} if unit["quarantine"] is not None else {}),
        }
    )


# ##################################################################
# quarantined chapter names
def quarantined_chapter_names(progress: dict) -> list[str]:
    return sorted(
        {
            name
            for batch in semantic_coverage(progress)["completed_batches"]
            if batch.get("quarantined")
            for name in batch["chapter_sha256"]
        }
    )


# ##################################################################
# empty chapter names
# chapters attested as zero-coverage non-character (exact hash recorded); they never veto a freeze.
def empty_chapter_names(progress: dict) -> dict[str, str]:
    return {
        name: digest
        for batch in semantic_coverage(progress)["completed_batches"]
        for name, digest in sorted((batch.get("empty_chapters") or {}).items())
    }


# ##################################################################
# prepare cast
# resumes each bounded native-Ollama batch from durable progress and atomically publishes a freeze only after all chapters and assets validate.
def prepare_cast(
    source: Path,
    verify_only: bool = False,
    max_batches: int | None = None,
    ask=None,
    max_revalidations: int | None = QUARANTINE_REVALIDATION_LIMIT,
) -> dict:
    source = source.resolve()
    if not source.is_file():
        raise OperationalError("source_missing", f"input source is not a file: {source}")
    if LLM_STYLE != "ollama" and not verify_only and ask is None:
        raise OperationalError(
            "cast_integrity", "prepare-cast requires schema-constrained native Ollama configured in local/config.toml"
        )
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
        raise OperationalError("cast_integrity", "no frozen cast manifest exists")
    source_sha = source_fingerprint(source)
    source_text = source.read_text(encoding="utf-8")
    progress_path = project / PROGRESS_NAME
    if progress_path.exists():
        progress = load_object(progress_path, "cast preparation progress")
        if progress.get("source_sha256") != source_sha or progress.get("chapter_count") != len(chapters):
            raise OperationalError("cast_integrity", "cast preparation progress belongs to a different source")
        if progress.get("version") not in {1, 2, 3}:
            raise OperationalError("cast_integrity", "cast preparation progress has an unsupported version")
        if progress.get("version") in {1, 2}:
            # Version 3 adds a separate semantic ledger beginning at source chapter zero; structural cursor and cached media remain unchanged.
            progress["version"] = 3
            semantic_coverage(progress)
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
            raise OperationalError("cast_integrity", "existing project is missing original Part 1 canonical anchors")
        progress = {
            "version": 3,
            "source_sha256": source_sha,
            "chapter_count": len(chapters),
            "next_chapter": 0,
            "registry": registry,
            "aliases": {actor_id: actor_id for actor_id in registry},
            "completed_batches": [],
        }
        atomic_json(progress_path, progress)
    validate_preparation_coverage(progress, chapters)
    validate_semantic_coverage(progress, chapters)
    # Always refresh: an audit can be safely appended after an interrupted batch and must apply before its next discovery even when old progress says audit_applied.
    inactive, ambiguous = refresh_alias_audit(project, source_text, progress)
    atomic_json(progress_path, progress)
    batches = 0
    coverage = semantic_coverage(progress)
    recovery = RecoveryLedger(project)
    ingest_context_quality_proposals(project, source_sha, chapters, progress["registry"], recovery, progress)
    ingest_context_resolution_v2(project, source_sha, chapters, progress["registry"], progress["aliases"], recovery)
    ingest_root_approvals(project, source_sha, chapters, progress, recovery)
    ingest_source_reviewed_retirements(project, source_sha, chapters, progress, recovery)
    earlier = {row_key(row) for row in recovery.open_quarantined("cast")}
    # Historical 0..cursor batches had structural hashes only. Reclassify that prefix under the semantic ledger before touching the next production batch.
    # A data problem quarantines only the offending chapter (semantic ledger only; the structural cursor never moves backwards).
    while int(coverage["next_chapter"]) < int(progress["next_chapter"]):
        start = int(coverage["next_chapter"])
        prefix = chapters[: int(progress["next_chapter"])]
        for unit in recoverable_batches(project, prefix, start, progress, ambiguous, source_text, recovery, ask, None):
            commit_batch(project, progress, coverage, unit, True)
            atomic_json(progress_path, progress)
    while int(progress["next_chapter"]) < len(chapters):
        # Audit records may be appended while this resumable preparation is paused; refresh is idempotent and leaves cursor and media untouched.
        inactive, ambiguous = refresh_alias_audit(project, source_text, progress)
        ingest_context_quality_proposals(project, source_sha, chapters, progress["registry"], recovery, progress)
        ingest_context_resolution_v2(project, source_sha, chapters, progress["registry"], progress["aliases"], recovery)
        ingest_root_approvals(project, source_sha, chapters, progress, recovery)
        ingest_source_reviewed_retirements(project, source_sha, chapters, progress, recovery)
        atomic_json(progress_path, progress)
        start = int(progress["next_chapter"])
        for unit in recoverable_batches(
            project, chapters, start, progress, ambiguous, source_text, recovery, ask, start
        ):
            commit_batch(project, progress, coverage, unit, False)
            atomic_json(progress_path, progress)
            batches += 1
        if max_batches is not None and batches >= max_batches:
            return {
                "status": "preparing",
                "chapters": len(chapters),
                "next_chapter": progress["next_chapter"],
                "actors": len(progress["registry"]),
                "warnings": len(recovery.entries()),
            }
    replay_pending(project, chapters, progress, ambiguous, source_text, recovery, ask)
    atomic_json(progress_path, progress)
    revalidate_quarantines(
        project, chapters, progress, ambiguous, source_text, recovery, ask, max_revalidations, earlier
    )
    atomic_json(progress_path, progress)
    blocked = quarantined_chapter_names(progress)
    pending_rows = recovery.open_pending("cast")
    quality_rows = [
        *active_quality_pending(recovery, progress["registry"]),
        *recovery.open_pending(CONTEXT_V2_STAGE),
        *recovery.open_pending(ROOT_APPROVAL_STAGE),
        *recovery.open_pending(SOURCE_RETIREMENT_STAGE),
    ]
    if blocked or pending_rows or quality_rows:
        # Publication refuses (typed status, no manifest) until cast uncertainty and every
        # active country/garble/duplicate-actor quality flag have durable resolutions.
        return {
            "status": "blocked",
            "chapters": len(chapters),
            "pending": len(blocked) + len(pending_rows) + len(quality_rows),
            "quarantined_chapters": blocked,
            "quality_pending": len(quality_rows),
            "warnings": len(recovery.entries()),
        }
    if int(semantic_coverage(progress)["next_chapter"]) != len(chapters):
        raise OperationalError("cast_integrity", "cannot freeze cast before full semantic coverage attestation")
    # Reapply the audited transitive map against retained legacy profiles before
    # publication; no inactive identifier can escape into the approved registry.
    legacy_registry = {**load_object(project / "characters.json", "characters profile"), **progress["registry"]}
    final_inactive, final_ambiguous = apply_alias_audit(project, source_text, legacy_registry, progress["aliases"])
    if final_ambiguous != ambiguous:
        raise OperationalError("cast_integrity", "alias audit ambiguity changed during cast preparation")
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
        "scoped_audit": scoped_audit_attestation(load_scoped_audit(project)),
        "quarantined_chapters": quarantined_chapter_names(progress),
        "empty_chapters": empty_chapter_names(progress),
    }
    atomic_json(project / MANIFEST_NAME, manifest)
    verify_frozen_cast(source, project)
    return {
        "status": "frozen",
        "chapters": len(chapters),
        "actors": len(actors),
        "manifest": str(project / MANIFEST_NAME),
        "warnings": len(recovery.entries()),
        "quarantined_chapters": manifest["quarantined_chapters"],
    }
