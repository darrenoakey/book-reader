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
REJECTIONS_NAME = "cast_preparation_rejections.jsonl"
BATCH_CHAPTERS = 6
MAX_BATCH_CHARACTERS = 24
EVIDENCE_REPAIR_ATTEMPTS = 2
SEMANTIC_COVERAGE_VERSION = 1
CLASSIFICATION_CHUNK_SIZE = 16
GENERIC_PRONOUN_ALIASES = frozenset({"i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs"})
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
NON_NAME_COMPOUND_PREFIXES = frozenset({"A", "An", "The", "All", "Each", "Every", "Some", "Any", "No", "Not", "Only", "As", "If", "When", "While", "After", "Before", "Because", "Although", "Though", "Since", "Unless", "And", "But", "Or", "Nor", "So", "Yet", "Then", "Also", "However", "Therefore", "In", "On", "At", "By", "From", "With", "Without", "For", "To", "Of", "Into", "Out", "Up", "Down", "Over", "Under", "Around", "Through", "Across", "During", "Beyond", "Within", "Against", "Between", "Among", "About"})
NON_ENTITY_LABELS = frozenset({"Someone", "Anyone", "Everyone", "Nobody", "Nothing", "Something", "He", "She", "Him", "Her", "His", "Hers", "They", "Them", "Their", "Theirs", "It", "Its", "We", "Us", "Our", "Ours", "I", "Me", "My", "Mine", "You", "Your", "Yours"})
TITLE_WORDS = frozenset({"Professor", "Master", "Doctor", "Captain", "Commander"})
NARRATIVE_ATTRIBUTION_VERBS = frozenset({"added", "announced", "asked", "called", "continued", "intervened", "murmured", "ordered", "replied", "said", "shouted", "spoke", "whispered"})


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
        chapter_text = chapter.read_text(encoding="utf-8")
        chapter_hash = hashlib.sha256(chapter_text.encode("utf-8")).hexdigest()
        for sentence_index, quote in enumerate(immutable_spans(chapter_text)):
            units.append(
                {
                    "id": f"c{chapter_index:02d}s{sentence_index:05d}",
                    "chapter": chapter.name,
                    "chapter_sha256": chapter_hash,
                    "quote": quote,
                }
            )
    if not units:
        raise ValueError("source batch has no immutable evidence units")
    return units


# ##################################################################
# immutable name references
# enumerates only exact contiguous capitalized lexical spans; source text is never copied or offset-calculated by a model.
def immutable_name_references(units: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    references: dict[str, dict[str, str]] = {}
    word = re.compile(r"[A-Za-z][A-Za-z'-]*")
    name_word = re.compile(r"[A-Z][a-z]*(?:-[A-Za-z]+)?$")
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
                if end == start + 1 and token.group() in TITLE_WORDS and end + 1 < len(tokens) and tokens[end + 1].group().casefold() in NARRATIVE_ATTRIBUTION_VERBS:
                    break
                current = tokens[end]
                possessive = current.group().endswith("'s")
                current_label = current.group()[:-2] if possessive else current.group()
                if (end > start and unit["quote"][tokens[end - 1].end() : current.start()].strip()) or not name_word.fullmatch(current_label):
                    break
                label_end = current.end() - 2 if possessive else current.end()
                label = unit["quote"][token.start() : label_end]
                if end > start and token.group() in NON_NAME_COMPOUND_PREFIXES:
                    break
                references[f"{unit['id']}n{index:03d}"] = {"unit_id": unit["id"], "label": label, "start": token.start(), "suffix": start > 0 and not unit["quote"][tokens[start - 1].end() : token.start()].strip() and bool(name_word.fullmatch(tokens[start - 1].group())) and tokens[start - 1].group() not in NON_NAME_COMPOUND_PREFIXES}
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
        if not isinstance(record, dict) or any(not isinstance(record.get(field), str) or not record[field] for field in SCOPED_AUDIT_FIELDS if field != "confidence") or not isinstance(record.get("confidence"), (int, float)) or type(record.get("span_start")) is not int or record["span_start"] < 0:
            raise ValueError("mention-scoped audit record is incomplete")
        if "owners" in record and (not isinstance(record["owners"], list) or not all(isinstance(owner, str) for owner in record["owners"])):
            raise ValueError("mention-scoped audit record has invalid offered owners")
        if "history" in record and (not isinstance(record["history"], list) or not all(isinstance(item, dict) and isinstance(item.get("decision"), str) and isinstance(item.get("reason"), str) for item in record["history"])):
            raise ValueError("mention-scoped audit record has invalid history")
        scope = (record["chapter_sha256"], record["quote_sha256"], record["label"], record["span_start"])
        if scope in index and index[scope] != record:
            raise ValueError("mention-scoped audit has conflicting records for one mention")
        index[scope] = record
    return index


NON_NAME_COMPOUND_WORDS = frozenset(word.casefold() for word in NON_NAME_COMPOUND_PREFIXES)
ADJUDICATION_ROSTER_MAX = 40
ADJUDICATION_SNIPPET_CHARS = 80


def label_components(text: str) -> set[str]:
    """Normalized name words of a label, without articles/prepositions that are not part of a name."""
    return {normalized_id(word) for word in re.split(r"[\s_]+", text) if len(word) > 1 and word.casefold() not in NON_NAME_COMPOUND_WORDS}


def adjudication_owners(label: str, registry: dict, aliases: dict | None = None, proposed: str | None = None) -> list[str]:
    """Candidate canonical owners for one label, led by the primary's contextual proposal, then exact name-word
    matches, any shared name component (Ren Dove -> ren), the audited alias owner, and, only when none of those exist, a bounded
    roster of the whole canonical cast (Mom -> mother) ordered so owners whose own profile/facts mention the label come first."""
    label_id = normalized_id(label)
    components = label_components(label)
    owners: list[str] = []

    def add(actor_id: str | None) -> None:
        if actor_id in registry and actor_id != "narrator" and actor_id not in owners:
            owners.append(actor_id)

    for actor_id, info in sorted(registry.items()):
        words = {normalized_id(word) for word in str(info.get("name", actor_id)).split()} | set(actor_id.split("_"))
        if label_id in words or components & words:
            add(actor_id)
    add((aliases or {}).get(label_id))
    if not owners:
        needle = label.casefold()
        mentioning = [actor_id for actor_id, info in sorted(registry.items()) if needle in json.dumps(info, ensure_ascii=False).casefold()]
        for actor_id in [*mentioning, *sorted(registry)]:
            add(actor_id)
        del owners[ADJUDICATION_ROSTER_MAX:]
    if proposed in registry and proposed != "narrator":
        # the proposal leads but never narrows: the offered set always covers the proposal-free set, so staleness is decidable without it
        if proposed in owners:
            owners.remove(proposed)
        owners.insert(0, proposed)
    return owners


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
    return bool(scoped) and scoped["decision"] == "alias" and scoped["canonical"] == canonical


# ##################################################################
# candidate coverage ledger
# groups exact lexical labels while retaining immutable witness IDs, so every possible named span receives a durable explicit decision.
def candidate_coverage_ledger(units: list[dict[str, str]], registry: dict[str, dict], aliases: dict[str, str], scoped_audit: list[dict] | None = None) -> list[dict]:
    scoped = mention_scoped_audit_index(scoped_audit or [])
    grouped: dict[str, dict] = {}
    units_by_id = {unit["id"]: unit for unit in units}
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
        candidate = grouped.setdefault(key, {"label": label, "ref_ids": [], "has_standalone": False, "scoped": record, "records": []})
        candidate["ref_ids"].append(ref_id)
        if record:
            candidate["records"].append(record)
        candidate["has_standalone"] = candidate["has_standalone"] or not reference["suffix"]
    for candidate in grouped.values():
        label_id = normalized_id(candidate["label"])
        direct = aliases.get(label_id)
        names = [actor_id for actor_id, entry in registry.items() if label_id in {normalized_id(actor_id), normalized_id(str(entry.get("name", actor_id)))}]
        candidate["known_owner"] = direct or (names[0] if len(names) == 1 else None)
        candidate["nonentity"] = False
        candidate["stale"] = False
        if candidate["scoped"]:
            ledger_owners = adjudication_owners(candidate["label"], registry, aliases)
            candidate["stale"] = any(readjudication_due(item, ledger_owners) for item in candidate["records"])
            candidate["known_owner"] = candidate["scoped"]["canonical"] if candidate["scoped"]["decision"] == "alias" else None
            if candidate["scoped"]["decision"] != "alias":
                candidate["nonentity"] = candidate["scoped"]["decision"] == "non_character"
        for ref_id in candidate["ref_ids"]:
            unit = units_by_id[references[ref_id]["unit_id"]]
            label = re.escape(candidate["label"])
            if re.search(rf"(?i)\b(?:impact of|attack of) {label}(?:\s+and\s+[A-Z][a-z]+)?\b", unit["quote"]) or re.search(rf"(?i)\b{label}\s+(?:air|team|group|clan|family|house)\b", unit["quote"]):
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
    retained = [(key, value) for key, value in grouped.items() if (" " in value["label"] or value["has_standalone"]) and (" " in value["label"] or normalized_id(value["label"]) in qualified_components or not has_lowercase_occurrence(value["label"]))]
    retained.sort(key=lambda item: (-len(item[1]["label"].split()), item[0]))
    return [
        {"id": f"p{index:04d}", "label": value["label"], "ref_ids": value["ref_ids"], "known_owner": value["known_owner"], "nonentity": value["nonentity"], **({"scoped_audit": value["scoped"]} if value["scoped"] else {}), **({"scoped_stale": True} if value["stale"] else {})}
        for index, (_, value) in enumerate(retained)
    ]


# ##################################################################
# classification schema
# forces one bounded native decision for every local candidate; all labels, IDs and witness bytes remain program-derived.
def discovery_schema(known_ids: list[str], candidates: list[dict], identity_candidates: list[dict] | None = None, allow_new: bool = True) -> dict:
    if not candidates:
        raise ValueError("classification schema requires lexical candidates")
    # candidates with an established owner are never legal identity targets: their canonical ID is already legal
    identity_ids = [candidate["id"] for candidate in (identity_candidates or candidates) if not candidate.get("known_owner")]
    unit_ids = sorted({ref_id.rsplit("n", 1)[0] for candidate in candidates for ref_id in candidate["ref_ids"]})

    def record_schema(candidate: dict) -> dict:
        evidence = {"type": "array", "minItems": 1, "maxItems": 3, "uniqueItems": True, "items": {"type": "string", "enum": unit_ids}}
        fixed_owner = candidate.get("known_owner")
        scoped = candidate.get("scoped_audit")
        audited_target = candidate.get("audited_target")
        base = {"type": "object", "properties": {"candidate_id": {"type": "string", "enum": [candidate["id"]]}, "evidence_unit_ids": evidence}, "required": ["candidate_id", "status", "identity", "evidence_unit_ids"], "additionalProperties": False}
        if scope_final(candidate) and scoped["decision"] != "alias":
            return {**base, "properties": {**base["properties"], "status": {"type": "string", "enum": [scoped["decision"]]}, "identity": {"type": "string", "enum": ["none"]}}}
        if fixed_owner:
            return {**base, "properties": {**base["properties"], "status": {"type": "string", "enum": ["known"]}, "identity": {"type": "string", "enum": [fixed_owner]}}}
        if audited_target:
            return {**base, "properties": {**base["properties"], "status": {"type": "string", "enum": ["known"]}, "identity": {"type": "string", "enum": [audited_target]}}}
        known_targets = sorted(set(known_ids + [identity for identity in identity_ids if identity != candidate["id"]]))
        branches = [
            {**base, "properties": {**base["properties"], "status": {"type": "string", "enum": ["known"]}, "identity": {"type": "string", "enum": known_targets}}},
            {**base, "properties": {**base["properties"], "status": {"type": "string", "enum": ["non_character", "ambiguous"]}, "identity": {"type": "string", "enum": ["none"]}}},
        ]
        if allow_new:
            branches.insert(0, {**base, "properties": {**base["properties"], "status": {"type": "string", "enum": ["new"]}, "identity": {"type": "string", "enum": [candidate["id"]]}}})
        return {"oneOf": branches}

    return {"type": "object", "properties": {"classifications": {"type": "array", "minItems": len(candidates), "maxItems": len(candidates), "uniqueItems": True, "items": {"oneOf": [record_schema(candidate) for candidate in candidates]}}}, "required": ["classifications"], "additionalProperties": False}


# ##################################################################
# classification prompt
# sends compact candidate-owned witnesses rather than an open-ended prose scan and requires exhaustive classifications in schema order.
def discovery_prompt(chapters: list[Path], registry: dict, aliases: dict[str, str], candidates: list[dict] | None = None, all_candidates: list[dict] | None = None, allow_new: bool = True) -> str:
    units = immutable_evidence_units(chapters)
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
        audited_target = f" FIXED_AUDITED_TARGET={candidate['audited_target']}" if candidate.get("audited_target") else ""
        rows.append(f"{candidate['id']} label={candidate['label']!r}{fixed}{audited_target} witnesses: {contexts}")
    global_ids = "; ".join(f"{candidate['id']}={candidate['label']!r}" for candidate in all_candidates)
    full_narrative = "\n".join(f"[{unit['id']}] {unit['quote']}" for unit in units)
    return f"""Classify EVERY candidate exactly once using only the response schema. Candidate labels and source witnesses are immutable local evidence; never copy a name, quote, offset, or invented ID into JSON.

status=new means this candidate is a distinct named living person/creature and identity MUST equal its own candidate_id. A named weapon, equipment item, attack, skill, species, group, or action is non_character even when capitalized; require source behavior/description proving a living entity before new. When both a source-qualified full name and a shorter component occur, make the full name the new identity and map the shorter label only when source evidence proves it is that identity. status=known means it is the same identity as an approved canonical ID or another new candidate in the global ledger; identity MUST name that target. An alias of a new full-name owner MUST be status=known targeting that owner, never status=new with a different identity. status=non_character means the lexical capitalisation is not a person/creature. status=ambiguous means source evidence cannot safely decide; identity MUST be none. For known mappings select source units proving identity; co-occurrence in one sentence alone is NOT proof. Do not merge spelling variants on similarity. A kinship/role label (Mom, Dad) or a partly matching full name may be proposed status=known to an approved canonical ID when its witness context supports that person; a separate per-mention validation then decides it, so prefer a contextual known proposal over ambiguous when the cast plausibly holds the owner. Bare Xiao is ambiguous unless a source witness identifies it. Indefinite sentence words Someone, Anyone, Everyone, Nobody, Nothing, and Something are non_character, never unresolved people. A bare surname or title fragment such as Crest is non_character unless it is an approved alias or its own selected witness explicitly identifies the same person; sharing a longer name is not identity proof. House, clan, family, place, group, team, species, and organization labels are non_character even when they mention or surround a known person; classify the exact label, never merge a house or clan into its member. Existing identities may only use their canonical name or a pre-approved alias below. A new identity may have zero aliases. Every evidence_unit_ids list must include a witness for its candidate. For an alias-to-new-identity link, include distinct witnesses for both spellings; a shared co-occurrence sentence alone is invalid.

Known canonical IDs: {roster or '(none)'}. Every ledger row carrying FIXED_KNOWN_OWNER MUST be status=known with exactly that identity; never create a new actor for it.
Approved aliases: {audited or '(none)'}
Global candidate identities (for cross-chunk links only): {global_ids}

CANDIDATE LEDGER:\n""" + "\n".join(rows) + "\n\nFULL BOUNDED SOURCE NARRATIVE (use it to resolve source-proven variant groups; never copy its text into JSON):\n" + full_narrative


# ##################################################################
# approved known label
# accepts audited aliases and a unique literal component of an established full name only when the selected source witness contains that full name.
def approved_known_label(label: str, canonical: str, evidence_unit_ids: list[str], units: list[dict[str, str]], registry: dict, aliases: dict) -> bool:
    label_id = normalized_id(label)
    entry = registry[canonical]
    full_name = str(entry.get("name", canonical))
    if aliases.get(label_id) == canonical or label_id in {normalized_id(canonical), normalized_id(full_name)}:
        return True
    if " " in label.strip() or not label_id:
        return False
    owners = [actor_id for actor_id, info in registry.items() if label_id in {normalized_id(word) for word in str(info.get("name", actor_id)).split()}]
    selected = [unit for unit in units if unit["id"] in evidence_unit_ids]
    return owners == [canonical] and any(source_label_present(full_name, [unit]) for unit in selected)


# ##################################################################
# materialize classifications
# validates exhaustive schema transport and creates discoveries only from source-derived candidate labels and links.
def materialize_classifications(value: object, units: list[dict[str, str]], candidates: list[dict], registry: dict, aliases: dict) -> tuple[list[dict], list[dict]]:
    if not isinstance(value, dict) or set(value) != {"classifications"} or not isinstance(value["classifications"], list):
        raise ValueError("classification response is not the exact object schema")
    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    unit_ids = {unit["id"] for unit in units}
    classifications = value["classifications"]
    if len(classifications) != len(candidates):
        raise ValueError("classification response omitted or duplicated lexical candidates")
    seen: set[str] = set()
    validated: dict[str, dict] = {}
    for item in classifications:
        if not isinstance(item, dict) or set(item) != {"candidate_id", "status", "identity", "evidence_unit_ids"}:
            raise ValueError("classification record has invalid fields")
        candidate_id, status, identity, evidence = item.get("candidate_id"), item.get("status"), item.get("identity"), item.get("evidence_unit_ids")
        if candidate_id not in candidate_by_id or candidate_id in seen or status not in {"known", "new", "non_character", "ambiguous"} or not isinstance(identity, str) or not isinstance(evidence, list) or not evidence or len(set(evidence)) != len(evidence) or not set(evidence) <= unit_ids:
            raise ValueError("classification record has invalid candidate or source evidence")
        candidate_units = {ref_id.rsplit("n", 1)[0] for ref_id in candidate_by_id[candidate_id]["ref_ids"]}
        if not candidate_units.intersection(evidence):
            raise ValueError(f"classification lacks a source witness for {candidate_id}")
        if status in {"non_character", "ambiguous"} and identity != "none":
            raise ValueError(f"non-identity classification has a target for {candidate_id}")
        if status == "new" and identity != candidate_id:
            raise ValueError(f"new classification must own its candidate ID: {candidate_id}")
        if status == "known" and identity not in registry and identity not in candidate_by_id:
            raise ValueError(f"known classification has unknown identity target: {identity}")
        seen.add(candidate_id)
        validated[candidate_id] = item
    if seen != set(candidate_by_id):
        raise ValueError("classification response did not cover the complete candidate ledger")
    discoveries: list[dict] = []
    for candidate in candidates:
        item = validated[candidate["id"]]
        if item["status"] != "new":
            continue
        label = candidate["label"]
        actor_id = normalized_id(label)
        if not IDENTIFIER.fullmatch(actor_id):
            raise ValueError(f"candidate cannot form a canonical ID: {label!r}")
        if actor_id in registry or aliases.get(actor_id) not in {None, actor_id}:
            raise ValueError(f"new candidate conflicts with established identity: {label!r}")
        unit_order = [unit["id"] for unit in units]
        context_ids: list[str] = []
        for evidence_id in [*item["evidence_unit_ids"], *(ref_id.rsplit("n", 1)[0] for ref_id in candidate["ref_ids"])]:
            position = unit_order.index(evidence_id)
            for nearby_id in unit_order[max(0, position - 1) : position + 3]:
                if nearby_id not in context_ids:
                    context_ids.append(nearby_id)
        source_quotes = [next(unit["quote"] for unit in units if unit["id"] == evidence_id) for evidence_id in context_ids]
        source_facts = " ".join(source_quotes)
        discoveries.append({"canonical_id": "new", "id": actor_id, "name": label, "aliases": [], "voice_facts": source_facts, "look_facts": source_facts, "evidence": source_quotes})
    for candidate in candidates:
        item = validated[candidate["id"]]
        if item["status"] == "ambiguous":
            raise RuntimeError(f"semantic classification remains unresolved for {candidate['label']!r}")
        if item["status"] != "known":
            continue
        target = item["identity"]
        label_id = normalized_id(candidate["label"])
        if target in registry:
            if not scoped_alias_approved(candidate, target) and not approved_known_label(candidate["label"], target, item["evidence_unit_ids"], units, registry, aliases):
                raise ValueError(f"known classification reassigns an unapproved alias: {candidate['label']!r}")
        else:
            owner = validated[target]
            if owner["status"] != "new":
                raise ValueError(f"candidate identity target is not a new identity: {target}")
            owner_label = candidate_by_id[target]["label"]
            owner_id = normalized_id(owner_label)
            if owner_id == label_id:
                continue
            candidate_units = {ref_id.rsplit("n", 1)[0] for ref_id in candidate["ref_ids"]}
            target_units = {ref_id.rsplit("n", 1)[0] for ref_id in candidate_by_id[target]["ref_ids"]}
            link_units = set(item["evidence_unit_ids"])
            full_name_component = normalized_id(candidate["label"]) in {normalized_id(word) for word in candidate_by_id[target]["label"].split()}
            if not candidate.get("audited_target") and not full_name_component and (not candidate_units.intersection(link_units) or not target_units.intersection(link_units) or candidate_units.intersection(target_units).intersection(link_units) == link_units):
                raise ValueError(f"candidate link lacks distinct source identity evidence: {candidate['label']!r}")
            alias_unit_ids = list(dict.fromkeys([*item["evidence_unit_ids"], *(ref_id.rsplit("n", 1)[0] for ref_id in candidate["ref_ids"])]))
            alias_quotes = [next(unit["quote"] for unit in units if unit["id"] == evidence_id) for evidence_id in alias_unit_ids]
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
# source-audited variant targets
# converts only read-only audit groups whose full owner is present in the current ledger into schema targets, preserving source-audited rather than guessed identity links.
def source_audited_variant_targets(project: Path, candidates: list[dict], source_text: str, registry: dict) -> dict[str, str]:
    labels = {candidate["label"]: candidate["id"] for candidate in candidates}
    targets: dict[str, str] = {}
    for filename, key in (("qa-klein-team-alias-proposal.json", "decisions"), ("qa-batch61-62-semantic-audit.json", "batch61_62_expected_new_named")):
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
                aliases = [part.strip() for part in re.sub(r"[()]", "", aliases).replace("also written", ",").split(",")]
            if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
                continue
            owner_label = next((alias for alias in aliases if alias in labels and " " in alias), next((alias for alias in aliases if alias in labels), None))
            if owner_label is None:
                continue
            evidence = record.get("evidence")
            if not isinstance(evidence, list) or not evidence or not all(isinstance(item, dict) and isinstance(item.get("excerpt"), str) and item["excerpt"] in source_text for item in evidence):
                raise RuntimeError("source-audited variant evidence is absent from immutable source")
            owner_candidate = next(candidate for candidate in candidates if candidate["id"] == labels[owner_label])
            audited_canonical = record.get("canonical")
            owner = audited_canonical if isinstance(audited_canonical, str) and audited_canonical in registry else owner_candidate.get("known_owner") or owner_candidate["id"]
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
            groups.append(f"SOURCE-AUDITED VARIANT GROUP: {', '.join(present)}; use full source label {full[0]!r} as the one new owner and classify other listed labels known to it only with their source witnesses. {note}")
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
                groups.append(f"SOURCE-AUDITED VARIANT GROUP: {', '.join(present)}; use full source label {full!r} as the one new owner and classify other listed labels known to it only with their source witnesses. {note}")
    return "\n".join(groups)


# ##################################################################
# classification chunks
# partitions only native response cardinality while retaining the complete batch ledger as legal identity targets for every chunk.
def classification_chunks(candidates: list[dict]) -> list[list[dict]]:
    return [candidates[index : index + CLASSIFICATION_CHUNK_SIZE] for index in range(0, len(candidates), CLASSIFICATION_CHUNK_SIZE)]


# ##################################################################
# validate classification chunk
# rejects malformed or incomplete chunk transport before its response can be composed into the batch-wide exact-once ledger.
def validate_classification_chunk(value: object, candidates: list[dict], registry: dict, aliases: dict, all_candidates: list[dict], units: list[dict], pending: list[dict] | None = None) -> list[dict]:
    if not isinstance(value, dict) or set(value) != {"classifications"} or not isinstance(value["classifications"], list):
        raise ValueError("classification chunk is not the exact object schema")
    expected = {candidate["id"] for candidate in candidates}
    records = value["classifications"]
    actual = [record.get("candidate_id") for record in records if isinstance(record, dict)]
    identities = {candidate["id"] for candidate in all_candidates} | set(registry) | {"none"}
    if len(records) != len(candidates) or set(actual) != expected or len(actual) != len(set(actual)):
        raise ValueError("classification chunk omitted or duplicated lexical candidates")
    for record in records:
        if not isinstance(record, dict) or set(record) != {"candidate_id", "status", "identity", "evidence_unit_ids"} or record["status"] not in {"known", "new", "non_character", "ambiguous"} or record["identity"] not in identities:
            raise ValueError("classification chunk has invalid record")
        candidate = next(candidate for candidate in candidates if candidate["id"] == record["candidate_id"])
        own_units = {ref_id.rsplit("n", 1)[0] for ref_id in candidate["ref_ids"]}
        evidence = record["evidence_unit_ids"]
        if not isinstance(evidence, list) or not evidence or len(evidence) != len(set(evidence)) or not set(evidence) <= {unit["id"] for unit in units} or not own_units.intersection(evidence):
            raise ValueError(f"classification lacks a unique own source witness: {record['candidate_id']}")
        if record["status"] == "new" and record["identity"] != record["candidate_id"]:
            raise ValueError(f"new classification must own its candidate ID: {record['candidate_id']}")
        if record["status"] in {"non_character", "ambiguous"} and record["identity"] != "none":
            raise ValueError(f"non-identity classification has a target: {record['candidate_id']}")
        if record["status"] == "known" and record["identity"] == "none":
            raise ValueError(f"known classification lacks an identity target: {record['candidate_id']}")
        if record["status"] == "known" and record["identity"] == record["candidate_id"]:
            raise ValueError(f"known classification must target another identity: {record['candidate_id']}")
        target_candidate = next((candidate for candidate in all_candidates if candidate["id"] == record["identity"]), None)
        if record["status"] == "known" and target_candidate is not None and target_candidate.get("known_owner") in registry:
            # chain through a known-owner candidate resolves to its established canonical before any new-alias evidence guard; the mention still needs contextual approval below
            record["identity"] = target_candidate["known_owner"]
            target_candidate = None
            if record["identity"] == record["candidate_id"]:
                raise ValueError(f"known classification must target another identity: {record['candidate_id']}")
        if record["status"] == "known" and record["identity"] in registry:
            canonical = record["identity"]
            if scoped_alias_approved(candidate, canonical):
                pass
            elif pending is not None and not scope_final(candidate) and not approved_known_label(candidate["label"], canonical, record["evidence_unit_ids"], units, registry, aliases):
                pending.append({"candidate": candidate, "proposed": canonical})
            elif not approved_known_label(candidate["label"], canonical, record["evidence_unit_ids"], units, registry, aliases):
                raise ValueError(f"known classification assigns prose or cross-owner label to {canonical}: {candidate['label']!r}")
        if record["status"] == "ambiguous" and pending is not None and not scope_final(candidate):
            pending.append({"candidate": candidate, "proposed": None})
        if candidate.get("scoped_stale") and pending is not None and not any(item["candidate"] is candidate for item in pending):
            # a cached ambiguous decision is re-judged per mention whatever the primary says; it must not be silently replaced by a label-level answer
            pending.append({"candidate": candidate, "proposed": record["identity"] if record["identity"] in registry else None})
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
                full_name_component = normalized_id(candidate["label"]) in {normalized_id(word) for word in target_candidate["label"].split()}
                evidence = set(record["evidence_unit_ids"])
                if not full_name_component and (not candidate_units.intersection(evidence) or not target_units.intersection(evidence) or candidate_units.intersection(target_units).intersection(evidence) == evidence):
                    raise ValueError(f"known candidate link lacks distinct source identity evidence: {candidate['label']!r}")
    return records


# ##################################################################
# verify proposed living entities
# performs a second native, schema-bound source review only for proposed new identities, refusing equipment, attacks, groups, and labels without a living-agent witness.
def verify_proposed_living_entities(batch_units: list[dict], discoveries: list[dict], ask) -> set[str]:
    if not discoveries:
        return set()
    unit_ids = [unit["id"] for unit in batch_units]
    identities = [item["id"] for item in discoveries]
    roster = "; ".join(f"{item['id']} name={item['name']!r} aliases={item['aliases']!r}" for item in discoveries)
    schema = {"type": "object", "properties": {"entities": {"type": "array", "minItems": len(identities), "maxItems": len(identities), "items": {"type": "object", "properties": {"id": {"type": "string", "enum": identities}, "eligibility": {"type": "string", "enum": ["living", "nonliving", "uncertain"]}, "evidence_unit_ids": {"type": "array", "minItems": 1, "maxItems": 3, "uniqueItems": True, "items": {"type": "string", "enum": unit_ids}}}, "required": ["id", "eligibility", "evidence_unit_ids"], "additionalProperties": False}}}, "required": ["entities"], "additionalProperties": False}
    narrative = "\n".join(f"[{unit['id']}] {unit['quote']}" for unit in batch_units)
    prompt = f"For each proposed identity decide only living, nonliving, or uncertain. A living result needs a cited witness naming this exact name or alias and proving a named living individual/creature. Equipment, weapons, attacks, skills, groups, places, and captions are nonliving. Use uncertain when source does not prove either result; uncertain fails closed. Output every ID exactly once.\nProposed identities: {roster}\nSOURCE:\n{narrative}"
    response = ask(prompt, max_tokens=1200, max_attempts=1, response_schema=schema)
    value = json.loads(response)
    records = value.get("entities") if isinstance(value, dict) else None
    if not isinstance(records, list) or len(records) != len(identities) or {record.get("id") for record in records if isinstance(record, dict)} != set(identities):
        raise RuntimeError("living-entity verifier omitted or duplicated a proposed identity")
    units_by_id = {unit["id"]: unit for unit in batch_units}
    approved = set()
    by_id = {item["id"]: item for item in discoveries}
    for record in records:
        if not isinstance(record, dict) or record.get("eligibility") not in {"living", "nonliving", "uncertain"} or not isinstance(record.get("evidence_unit_ids"), list) or not record["evidence_unit_ids"] or not set(record["evidence_unit_ids"]) <= set(unit_ids):
            raise RuntimeError("living-entity verifier returned invalid source evidence")
        identity = by_id[record["id"]]
        labels = [identity["name"], *identity["aliases"]]
        if not any(source_label_present(label, [units_by_id[evidence_id]]) for label in labels for evidence_id in record["evidence_unit_ids"]):
            raise RuntimeError(f"living-entity verifier lacks own identity witness: {record['id']}")
        if record["eligibility"] == "uncertain":
            raise RuntimeError(f"living-entity verifier is uncertain for {record['id']}")
        if record["eligibility"] == "living":
            approved.add(record["id"])
    return approved


# ##################################################################
# scoped native adjudication
# decides each pending mention separately (exact chapter/quote scope, never a global label alias), and persists the interpretive record before any mapping uses it.
SCOPED_AUDIT_NAME = "mention-scoped-audit.json"
ADJUDICATION_MENTIONS_PER_CALL = 12
ADJUDICATION_MAX_ROUNDS = 4
ADJUDICATION_MIN_CONFIDENCE = 0.7
ADJUDICATION_SCENE_CHARS = 2400
ADJUDICATION_OWNER_FACT_CHARS = 700
ADJUDICATION_FACTS_TOTAL_CHARS = 6000


def bounded_scene(units_by_id: dict, order: list[str], position: int) -> str:
    """Contiguous same-chapter source around one mention, grown alternately both ways within a fixed character budget; the mention's own unit is always whole."""
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
                if used + size <= ADJUDICATION_SCENE_CHARS:
                    used += size
                    low, high = (edge, high) if step < 0 else (low, edge)
                    grew = True
    return " ".join(units_by_id[order[index]]["quote"] for index in range(low, high + 1))


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
            parts.append(f"prior facts mentioning {label!r}: ...{flat[max(0, hit - ADJUDICATION_SNIPPET_CHARS) : hit + len(label) + ADJUDICATION_SNIPPET_CHARS]}...")
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
    ordered = sorted(mention_scoped_audit_index(records).values(), key=lambda r: (r["chapter_sha256"], r["quote_sha256"], r["span_start"], r["label"]))
    return {"version": 1, "count": len(ordered), "sha256": json_digest(ordered)}


def adjudicate_pending_mentions(project: Path, pending: list[dict], units: list[dict], registry: dict, ask, aliases: dict | None = None) -> int:
    """Adjudicate every pending mention lacking a binding decision; returns the number of audit records written (0 means no new context)."""
    written = 0
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
            if scope not in known or readjudication_due(known[scope], owners):
                mentions.setdefault(scope, unit)
        scopes = list(mentions)
        for offset in range(0, len(scopes), ADJUDICATION_MENTIONS_PER_CALL):
            chunk = scopes[offset : offset + ADJUDICATION_MENTIONS_PER_CALL]
            ids = [f"m{index}" for index in range(len(chunk))]
            decisions = ["alias", "non_character", "ambiguous"] if owners else ["non_character", "ambiguous"]
            item = {"type": "object", "properties": {"mention_id": {"type": "string", "enum": ids}, "refers_to_person": {"type": "string", "enum": ["yes", "no", "unclear"]}, "candidate_kind": {"type": "string", "enum": ["individual_name", "specific_role", "endearment", "prose_fragment", "nonliving", "unclear"]}, "decision": {"type": "string", "enum": decisions}, "canonical": {"type": "string", "enum": [*owners, "none"]}, "confidence": {"type": "number", "minimum": 0, "maximum": 1}, "reason": {"type": "string", "minLength": 1, "maxLength": 300}}, "required": ["mention_id", "refers_to_person", "candidate_kind", "decision", "canonical", "confidence", "reason"], "additionalProperties": False, "allOf": [{"if": {"properties": {"decision": {"const": "non_character"}}}, "then": {"properties": {"candidate_kind": {"enum": ["endearment", "prose_fragment", "nonliving"]}}}}, {"if": {"properties": {"candidate_kind": {"enum": ["individual_name", "specific_role"]}}}, "then": {"properties": {"decision": {"enum": ["alias", "ambiguous"]}}}}]}
            schema = {"type": "object", "properties": {"mentions": {"type": "array", "minItems": len(chunk), "maxItems": len(chunk), "items": item}}, "required": ["mentions"], "additionalProperties": False}
            roster = "; ".join(f"{owner}={registry[owner].get('name', owner)!r}" for owner in owners) or "(no candidate owner)"
            rows = []
            for mention_id, scope in zip(ids, chunk, strict=True):
                position = order.index(mentions[scope]["id"])
                rows.append(f"{mention_id} [{mentions[scope]['chapter']}] mention: {mentions[scope]['quote']}\n   bounded scene: {bounded_scene(units_by_id, order, position)}")
                if scope in known:
                    prior = known[scope]
                    rows.append(f"   previous decision (made when only {prior.get('owners', 'unrecorded owners')} were offered): {prior['decision']} confidence={prior['confidence']} because: {prior['reason']}")
            prompt = f"Decide separately, for each exact mention of the label {candidate['label']!r}, whether THIS mention refers to an existing character. Never decide by spelling or sound similarity, and never generalise from one mention to another. alias requires the mention or its bounded scene to prove identity with the named owner, consistent with that owner's canonical prior facts; canonical must be that owner. The possible owners are only candidates (a shared name component, or a bounded roster of the whole cast when the label matches no name): a kinship or role label such as Mom or Dad may be a vocative or reference for a cast member whose prior facts say they are that person's parent, and a full name may extend a shorter canonical name; accept such a link only when the scene and prior facts support it. A previous decision is shown only where one exists; reconsider it with the fuller owner information. First answer refers_to_person for this mention: yes when it names or addresses a person/creature in the story (a vocative such as Mom, or a full name, counts), no only when it is a place, object, group, title word or other non-living thing, unclear when you cannot tell. Also classify candidate_kind: individual_name is a stable actor identity; specific_role can map only when the scene identifies its owner; endearment and prose_fragment address/describes a person but are not actor names; nonliving is not a person. Then decide: alias only for individual_name or source-owned specific_role with a supported owner. non_character is correct for nonliving, endearment, or prose_fragment even when refers_to_person=yes. ambiguous only when a potential individual_name/specific_role owner cannot be decided. Give honest confidence and a short source-based reason.\nPossible owners: {roster}\nCanonical prior facts for the possible owners:\n{owner_prior_facts(registry, owners, candidate['label'])}\nMENTIONS:\n" + "\n".join(rows)
            if len(prompt) > PREPARATION_PROMPT_MAX_CHARS:
                raise RuntimeError(f"scoped adjudication prompt exceeds native context budget for {candidate['label']!r}")
            value = json.loads(ask(prompt, max_tokens=1500, max_attempts=1, response_schema=schema))
            returned = value.get("mentions") if isinstance(value, dict) else None
            if not isinstance(returned, list) or {r.get("mention_id") for r in returned if isinstance(r, dict)} != set(ids) or len(returned) != len(ids):
                raise RuntimeError(f"scoped adjudication omitted or duplicated mentions for {candidate['label']!r}")
            for result in returned:
                scope = chunk[ids.index(result["mention_id"])]
                decision, canonical = result["decision"], result["canonical"]
                person = result["refers_to_person"]
                reason = str(result["reason"])
                kind = result["candidate_kind"]
                if decision == "alias" and (canonical not in owners or result["confidence"] < ADJUDICATION_MIN_CONFIDENCE or person != "yes" or kind not in {"individual_name", "specific_role"}):
                    decision, canonical = "ambiguous", "none"
                elif decision in {"non_character", "ambiguous"} and person == "yes" and result["canonical"] in owners:
                    relationship_schema = {"type": "object", "properties": {"relationship": {"type": "string", "enum": ["same_owner", "distinct", "not_identity", "unclear"]}, "witness_unit_ids": {"type": "array", "minItems": 1, "maxItems": 3, "uniqueItems": True, "items": {"type": "string", "enum": [unit["id"] for unit in units]}}, "reason": {"type": "string", "minLength": 1, "maxLength": 300}}, "required": ["relationship", "witness_unit_ids", "reason"], "additionalProperties": False}
                    relationship_prompt = f"For this one exact mention only, determine its relationship to proposed canonical {result['canonical']!r}: same_owner only with source scene continuity; distinct/not_identity/unclear otherwise. Do not infer from spelling. Cite witness IDs. Mention [{mentions[scope]['id']}]: {mentions[scope]['quote']!r}. Bounded scene: {bounded_scene(units_by_id, order, order.index(mentions[scope]['id']))}. Raw rationale: {reason!r}"
                    relationship = json.loads(ask(relationship_prompt, max_tokens=500, max_attempts=1, response_schema=relationship_schema))
                    if relationship["relationship"] == "same_owner":
                        decision, canonical, reason = "alias", result["canonical"], f"[relationship review {relationship['witness_unit_ids']}] {relationship['reason']}"[:300]
                    else:
                        decision, canonical, reason = "ambiguous", "none", f"[relationship review={relationship['relationship']}] {relationship['reason']}"[:300]
                elif decision == "non_character" and person != "no" and kind not in {"endearment", "prose_fragment"}:
                    # One bounded native tiebreak distinguishes a bad coupled transport from a genuinely unresolved identity.
                    # It sees only this immutable mention/scene and the raw rationale; it cannot create a global alias.
                    review_schema = {"type": "object", "properties": {"semantic_type": {"type": "string", "enum": ["individual_identity", "actor_reference", "endearment", "prose_fragment", "nonliving", "unclear"]}, "witness_unit_ids": {"type": "array", "minItems": 1, "maxItems": 3, "uniqueItems": True, "items": {"type": "string", "enum": [unit["id"] for unit in units]}}, "reason": {"type": "string", "minLength": 1, "maxLength": 300}}, "required": ["semantic_type", "witness_unit_ids", "reason"], "additionalProperties": False}
                    review_prompt = f"Classify ONE exact source span using exactly one semantic_type and cite immutable witness IDs. individual_identity is a stable actor name; actor_reference is a known person's name/reference but not a new name; endearment/prose_fragment/nonliving are not stable actor identities. Do not output a decision, canonical ID, or person flag; the caller derives those mechanically. Raw inconsistent result: person={person!r}, kind={kind!r}, decision={decision!r}, reason={reason!r}. Mention [{mentions[scope]['id']}]: {mentions[scope]['quote']!r}. Bounded scene: {bounded_scene(units_by_id, order, order.index(mentions[scope]['id']))}"
                    reviewed = json.loads(ask(review_prompt, max_tokens=500, max_attempts=1, response_schema=review_schema))
                    review_type, review_reason = reviewed["semantic_type"], reviewed["reason"]
                    if review_type in {"endearment", "prose_fragment", "nonliving"}:
                        kind, decision, canonical, reason = review_type, "non_character", "none", f"[semantic-type review {reviewed['witness_unit_ids']}] {review_reason}"[:300]
                    else:
                        decision, canonical, reason = "ambiguous", "none", f"[inconsistent non_character with refers_to_person={person}; semantic-type review={review_type}] {review_reason}"[:300]
                if decision != "alias":
                    canonical = "none"
                record = {"chapter_sha256": scope[0], "quote_sha256": scope[1], "label": candidate["label"], "span_start": scope[3], "canonical": canonical, "decision": decision, "confidence": float(result["confidence"]), "reason": reason, "owners": list(owners), "raw_adjudication": {"decision": result["decision"], "canonical": result["canonical"], "refers_to_person": person, "candidate_kind": kind, "confidence": result["confidence"], "reason": result["reason"]}}
                prior = known.get(scope)
                if prior:
                    # keep every superseded decision and its reason; the audit is append-only history, not an overwrite
                    record["history"] = [*prior.get("history", []), {key: prior[key] for key in ("decision", "canonical", "confidence", "reason", "owners") if key in prior}]
                    records[:] = [item for item in records if item is not prior]
                records.append(record)
                known[scope] = record
                written += 1
            mention_scoped_audit_index(records)
            atomic_json(project / SCOPED_AUDIT_NAME, {"records": records})
    return written


# ##################################################################
# discover batch
# executes bounded schema calls for ledger chunks, then materializes their composed complete classification once so omissions cannot be hidden between calls.
def discover_batch(project: Path, start: int, batch: list[Path], batch_units: list[dict[str, str]], batch_text: str, progress: dict, ambiguous: set[str], prompt: str | None = None, ask=None) -> tuple[list[dict], list[dict]]:
    del ambiguous, prompt
    ask = ask or ask_sync
    for adjudication_round in range(ADJUDICATION_MAX_ROUNDS):
        candidates = candidate_coverage_ledger(batch_units, progress["registry"], progress["aliases"], load_scoped_audit(project))
        if not candidates:
            return [], []
        pending: list[dict] = []
        # Binding mention-scoped decisions are already source-validated exact references.
        # Materialize them deterministically; asking the provider to reclassify them can
        # overwrite a valid Lou→Lu scope with an unrelated lexical candidate target.
        bound = [candidate for candidate in candidates if candidate.get("scoped_audit", {}).get("decision") in {"alias", "non_character"} and not candidate.get("scoped_stale")]
        records: list[dict] = [
            {"candidate_id": candidate["id"], "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]], "status": "known" if candidate["scoped_audit"]["decision"] == "alias" else "non_character", "identity": candidate["scoped_audit"]["canonical"] if candidate["scoped_audit"]["decision"] == "alias" else "none"}
            for candidate in bound
        ]
        unresolved = [candidate for candidate in candidates if candidate not in bound]
        allow_new = True
        variant_targets = source_audited_variant_targets(project, candidates, batch_text, progress["registry"])
        for candidate in candidates:
            candidate["audited_target"] = variant_targets.get(candidate["id"])
        audited_context = source_audited_variant_context(project, candidates)
        for chunk_index, chunk in enumerate(classification_chunks(unresolved)):
            chunk_prompt = discovery_prompt(batch, progress["registry"], progress["aliases"], chunk, candidates, allow_new)
            if audited_context:
                chunk_prompt += "\n\n" + audited_context
            schema = discovery_schema(list(progress["registry"]), chunk, candidates, allow_new)
            response = ask(chunk_prompt, max_tokens=1800, max_attempts=1, response_schema=schema)
            for attempt in range(EVIDENCE_REPAIR_ATTEMPTS + 1):
                try:
                    records.extend(validate_classification_chunk(json.loads(response), chunk, progress["registry"], progress["aliases"], candidates, batch_units, pending))
                    break
                except (ValueError, json.JSONDecodeError) as error:
                    record_rejected_discovery(project, start, batch, batch_units, response, RuntimeError(f"chunk {chunk_index}: {error}"), attempt)
                    if attempt == EVIDENCE_REPAIR_ATTEMPTS:
                        raise RuntimeError(f"cast semantic classification {start}-{start + len(batch) - 1} chunk {chunk_index} rejected after bounded repairs; archived evidence: {error}") from error
                    response = ask(f"Your chunk classification was rejected: {error}. Return the complete schema object for this chunk only.\n\n{chunk_prompt}", max_tokens=1800, max_attempts=1, response_schema=schema)
        if not pending:
            break
        # monotonic: only mentions without a binding decision are adjudicated, so a round that writes nothing has no new context and cannot change the next classification
        if not adjudicate_pending_mentions(project, pending, batch_units, progress["registry"], ask, progress["aliases"]):
            break
        if adjudication_round == ADJUDICATION_MAX_ROUNDS - 1:
            raise RuntimeError(f"scoped adjudication did not converge within {ADJUDICATION_MAX_ROUNDS} classification rounds")
    discoveries, classifications = materialize_classifications({"classifications": records}, batch_units, candidates, progress["registry"], progress["aliases"])
    approved = verify_proposed_living_entities(batch_units, discoveries, ask)
    rejected = {item["id"] for item in discoveries} - approved
    if rejected:
        for record in classifications:
            candidate = next(item for item in candidates if item["id"] == record["candidate_id"])
            candidate_id = normalized_id(candidate["label"])
            if candidate_id in rejected or record["identity"] in {next(item["id"] for item in discoveries if item["id"] == rejected_id) for rejected_id in rejected}:
                record["status"], record["identity"] = "non_character", "none"
        discoveries = [item for item in discoveries if item["id"] in approved]
    return discoveries, classifications


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
        source_bytes = sum(len(path.read_text(encoding="utf-8")) for path in candidate)
        candidate_prompt = "" if source_bytes * 2 > PREPARATION_PROMPT_MAX_CHARS else discovery_prompt(candidate, registry, aliases)
        conservative_size = len(candidate_prompt) + (source_bytes * 2)
        if conservative_size > PREPARATION_PROMPT_MAX_CHARS:
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
# source label present
# requires a whole source label rather than accepting a coincidental substring in a selected immutable sentence.
def source_label_present(label: str, units: list[dict[str, str]]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(label.strip())}(?!\w)", unit["quote"], re.IGNORECASE) for unit in units)


# ##################################################################
# record rejected discovery
# retains every complete native response and immutable unit citations so each original or repair failure is auditable without a lossy exception snippet.
def record_rejected_discovery(
    project: Path, start: int, batch: list[Path], units: list[dict[str, str]], response: str, error: Exception, attempt: int
) -> None:
    payload = {
        "start_chapter": start,
        "chapters": [path.name for path in batch],
        "attempt": attempt,
        "error": str(error),
        "evidence_units": units,
        "response": response,
    }
    with (project / REJECTIONS_NAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
# refresh audited aliases
# reapplies verified audit records before each discovery without moving the cursor or generating media, retaining every earlier inactive identity and ambiguity.
def refresh_alias_audit(project: Path, source_text: str, progress: dict) -> tuple[set[str], set[str]]:
    legacy_registry = {**load_object(project / "characters.json", "characters profile"), **progress["registry"]}
    inactive, ambiguous = apply_alias_audit(project, source_text, legacy_registry, progress["aliases"])
    inactive.update(progress.get("inactive_legacy_ids", []))
    ambiguous.update(progress.get("ambiguous_new_ids", []))
    if inactive:
        progress["registry"] = {actor_id: entry for actor_id, entry in progress["registry"].items() if actor_id not in inactive}
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
    # A manifest frozen before scoped audit existed attests an empty record set; any record appearing later is unattested.
    attested = manifest.get("scoped_audit", scoped_audit_attestation([]))
    if attested != scoped_audit_attestation(load_scoped_audit(project)):
        raise RuntimeError("frozen cast manifest does not attest the current mention-scoped audit records")
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
    for record in sorted(load_scoped_audit(project), key=lambda r: (r["chapter_sha256"], r["quote_sha256"], r["span_start"], r["label"])):
        if record["decision"] != "alias":
            continue
        unit = units.get((record["chapter_sha256"], record["quote_sha256"]))
        if unit is None:
            raise RuntimeError("mention-scoped alias is bound to source text that does not exist")
        if not any(ref["label"] == record["label"] and ref["start"] == record["span_start"] for ref in immutable_name_references([unit]).values()):
            raise RuntimeError("mention-scoped alias does not name an exact source mention at its offset")
        if record["canonical"] not in manifest["actors"]:
            raise RuntimeError("mention-scoped alias targets an actor outside the frozen cast")
        if record["confidence"] < ADJUDICATION_MIN_CONFIDENCE:
            raise RuntimeError("mention-scoped alias is below the adjudication confidence floor")
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
        if not isinstance(batch, dict) or set(batch) != {"start", "end", "chapter_sha256"}:
            raise RuntimeError("cast preparation has an invalid completed batch record")
        start, end, hashes = batch["start"], batch["end"], batch["chapter_sha256"]
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start != expected_start
            or end <= start
            or end > len(chapters)
        ):
            raise RuntimeError("cast preparation completed batches do not have unique consecutive chapter coverage")
        expected_hashes = {path.name: file_digest(path) for path in chapters[start:end]}
        if not isinstance(hashes, dict) or hashes != expected_hashes or seen.intersection(hashes):
            raise RuntimeError("cast preparation completed batches lack unique exact chapter hashes")
        seen.update(hashes)
        expected_start = end
    if progress.get("next_chapter") != expected_start:
        raise RuntimeError("cast preparation cursor does not match completed exact chapter coverage")


# ##################################################################
# semantic coverage state
# migrates historical structural batches into a separate source-bound semantic ledger without moving the established production cursor.
def semantic_coverage(progress: dict) -> dict:
    coverage = progress.get("semantic_coverage")
    if coverage is None:
        coverage = {"version": SEMANTIC_COVERAGE_VERSION, "next_chapter": 0, "completed_batches": []}
        progress["semantic_coverage"] = coverage
    if not isinstance(coverage, dict) or coverage.get("version") != SEMANTIC_COVERAGE_VERSION:
        raise RuntimeError("cast preparation has an unsupported semantic coverage ledger")
    return coverage


# ##################################################################
# validate semantic coverage
# proves every semantic ledger entry is consecutive, source-hashed, and bound to the exact deterministic candidate ledger used for its native classification.
def validate_semantic_coverage(progress: dict, chapters: list[Path]) -> None:
    coverage = semantic_coverage(progress)
    expected_start = 0
    for batch in coverage["completed_batches"]:
        if not isinstance(batch, dict) or set(batch) != {"start", "end", "chapter_sha256", "ledger_sha256"}:
            raise RuntimeError("semantic coverage has an invalid completed batch record")
        start, end = batch["start"], batch["end"]
        if not isinstance(start, int) or not isinstance(end, int) or start != expected_start or end <= start or end > len(chapters):
            raise RuntimeError("semantic coverage batches are not unique consecutive source coverage")
        if batch["chapter_sha256"] != {path.name: file_digest(path) for path in chapters[start:end]} or not isinstance(batch["ledger_sha256"], str):
            raise RuntimeError("semantic coverage batch is not bound to exact source and candidate ledger")
        expected_start = end
    if coverage.get("next_chapter") != expected_start:
        raise RuntimeError("semantic coverage cursor does not match its completed batches")


# ##################################################################
# record semantic batch
# writes the complete local ledger and native classifications before any cursor advances, making omitted candidates auditable across restarts.
def record_semantic_batch(project: Path, start: int, batch: list[Path], units: list[dict], classifications: list[dict], coverage: dict, revalidation: bool) -> None:
    ledger = candidate_coverage_ledger(units, {}, {})
    ledger_sha256 = json_digest(ledger)
    payload = {"start_chapter": start, "chapters": [path.name for path in batch], "semantic_revalidation": revalidation, "candidate_ledger": ledger, "classifications": classifications}
    with (project / DISCOVERIES_NAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    coverage["completed_batches"].append({"start": start, "end": start + len(batch), "chapter_sha256": {path.name: file_digest(path) for path in batch}, "ledger_sha256": ledger_sha256})
    coverage["next_chapter"] = start + len(batch)


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
        if progress.get("version") not in {1, 2, 3}:
            raise RuntimeError("cast preparation progress has an unsupported version")
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
            raise RuntimeError("existing project is missing original Part 1 canonical anchors")
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
    # Historical 0..cursor batches had structural hashes only. Reclassify that prefix under the semantic ledger before touching the next production batch.
    while int(coverage["next_chapter"]) < int(progress["next_chapter"]):
        start = int(coverage["next_chapter"])
        prefix = chapters[: int(progress["next_chapter"])]
        batch, _ = context_safe_batch(prefix, start, progress["registry"], progress["aliases"])
        units = immutable_evidence_units(batch)
        discoveries, classifications = discover_batch(project, start, batch, units, source_text, progress, ambiguous)
        apply_discoveries(progress["registry"], progress["aliases"], discoveries)
        record_semantic_batch(project, start, batch, units, classifications, coverage, True)
        atomic_json(progress_path, progress)
    while int(progress["next_chapter"]) < len(chapters):
        # Audit records may be appended while this resumable preparation is paused; refresh is idempotent and leaves cursor and media untouched.
        inactive, ambiguous = refresh_alias_audit(project, source_text, progress)
        atomic_json(progress_path, progress)
        start = int(progress["next_chapter"])
        batch, prompt = context_safe_batch(chapters, start, progress["registry"], progress["aliases"])
        batch_units = immutable_evidence_units(batch)
        discoveries, classifications = discover_batch(project, start, batch, batch_units, source_text, progress, ambiguous, prompt)
        apply_discoveries(progress["registry"], progress["aliases"], discoveries)
        record_semantic_batch(project, start, batch, batch_units, classifications, coverage, False)
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
    if int(semantic_coverage(progress)["next_chapter"]) != len(chapters):
        raise RuntimeError("cannot freeze cast before full semantic coverage attestation")
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
        "scoped_audit": scoped_audit_attestation(load_scoped_audit(project)),
    }
    atomic_json(project / MANIFEST_NAME, manifest)
    verify_frozen_cast(source, project)
    return {
        "status": "frozen",
        "chapters": len(chapters),
        "actors": len(actors),
        "manifest": str(project / MANIFEST_NAME),
    }
