"""Deterministic whole-book cast source index: exact name occurrences, exact-label candidate groups, factual contexts.

Additive and read-only: it reuses the immutable evidence units, name references, scope hashes and proof guards of
src.cast_freeze unchanged, performs no model call, and never writes to a project.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.cast_freeze import (
    ADJUDICATION_SCENE_CHARS,
    PROGRESS_NAME,
    SCOPED_AUDIT_NAME,
    CastDataIssue,
    adjudication_owners,
    bounded_scene_range,
    file_digest,
    immutable_evidence_units,
    immutable_name_references,
    load_object,
    load_scoped_audit,
    mention_scoped_audit_index,
    normalized_id,
    readjudication_due,
    scene_units_at,
    scoped_alias_proof,
    text_digest,
)
from src.hour_runner import chapter_order

INDEX_VERSION = 1
DEFAULT_CONTEXT_LIMIT = 6
DEFAULT_CHUNK_CHARS = 600_000
PARTITION_ORDER = ("literal_owner", "alias", "non_character", "ambiguous", "stale", "open")
NEEDS_DECISION = frozenset({"stale", "open"})
NONENTITY_CUE = re.compile(r"(?i)impact of|attack of|\s(?:air|team|group|clan|family|house)")


# ##################################################################
# source index
# every capitalized name occurrence of the whole source with the exact chapter/quote hashes of its immutable scope; columns are parallel tuples ordered by source position, so equal bytes always give an equal digest.
@dataclass(frozen=True, slots=True)
class SourceIndex:
    chapters: tuple[dict, ...]
    units: tuple[dict, ...]
    unit_position: dict[str, int]
    by_id: dict[str, dict]
    order: tuple[str, ...]
    quote_sha256: tuple[str, ...]
    unit_chapter: tuple[int, ...]
    unit_offset: tuple[int, ...]
    occurrence_ref: tuple[str, ...]
    occurrence_unit: tuple[int, ...]
    occurrence_label: tuple[str, ...]
    occurrence_start: tuple[int, ...]
    occurrence_suffix: tuple[bool, ...]
    digest: str

    def scope(self, occurrence: int) -> tuple[str, str, str, int]:
        unit = self.occurrence_unit[occurrence]
        return (
            self.units[unit]["chapter_sha256"],
            self.quote_sha256[unit],
            self.occurrence_label[occurrence],
            self.occurrence_start[occurrence],
        )


# ##################################################################
# index digest
# one streamed hash over the chapter hashes, unit quote hashes and every occurrence, binding the complete index to the exact source.
def index_digest(chapters: Sequence[dict], units: Sequence[dict], quote_sha256: Sequence[str], occurrences) -> str:
    digest = hashlib.sha256(f"cast-index-v{INDEX_VERSION}\n".encode())
    for chapter in chapters:
        digest.update(
            f"C\0{chapter['name']}\0{chapter['text_sha256']}\0{chapter['file_sha256']}\0{chapter['units']}\0{chapter['chars']}\n".encode()
        )
    digest.update("\n".join(f"U\0{unit['id']}\0{sha}" for unit, sha in zip(units, quote_sha256)).encode())
    digest.update(
        "\n".join(f"O\0{unit}\0{start}\0{int(suffix)}\0{label}" for unit, start, suffix, label in occurrences).encode()
    )
    return digest.hexdigest()


# ##################################################################
# chapter records
# one exact record per source chapter, including a chapter that has no immutable span, so nothing is silently absent from the index. The immutable units must tile the chapter text exactly (only a whitespace-only tail may be unspanned); anything else fails closed because exact coverage could not be proven.
def chapter_records(chapters: list[Path], units: list[dict]) -> tuple[list[dict], list[int], list[int]]:
    members: dict[str, list[int]] = {}
    for position, unit in enumerate(units):
        members.setdefault(unit["chapter"], []).append(position)
    records, unit_chapter, unit_offset = [], [0] * len(units), [0] * len(units)
    for index, path in enumerate(chapters):
        text = path.read_text(encoding="utf-8")
        offset = 0
        for position in members.get(path.name, []):
            unit_chapter[position], unit_offset[position] = index, offset
            offset += len(units[position]["quote"])
        if (
            not text.startswith("".join(units[position]["quote"] for position in members.get(path.name, [])))
            or text[offset:].strip()
        ):
            raise CastDataIssue(
                f"source chapter {path.name} is not exactly tiled by its immutable units", code="source_not_tiled"
            )
        records.append(
            {
                "position": index,
                "name": path.name,
                "text_sha256": text_digest(text),
                "file_sha256": file_digest(path),
                "units": len(members.get(path.name, [])),
                "first_unit": members[path.name][0] if path.name in members else 0,
                "chars": len(text),
                "tail": text[offset:],
            }
        )
    return records, unit_chapter, unit_offset


# ##################################################################
# build source index
# one pass over the exact immutable units and name references of src.cast_freeze; an undecodable chapter raises its typed CastDataIssue (fail closed) for the caller to quarantine before indexing.
def build_source_index(chapters: list[Path]) -> SourceIndex:
    units = immutable_evidence_units(chapters)
    references = immutable_name_references(units)
    records, unit_chapter, unit_offset = chapter_records(chapters, units)
    unit_position = {unit["id"]: index for index, unit in enumerate(units)}
    quote_sha256 = tuple(text_digest(unit["quote"]) for unit in units)
    ref_ids = tuple(references)
    occurrence_unit = tuple(unit_position[references[ref]["unit_id"]] for ref in ref_ids)
    occurrence_label = tuple(references[ref]["label"] for ref in ref_ids)
    occurrence_start = tuple(references[ref]["start"] for ref in ref_ids)
    occurrence_suffix = tuple(bool(references[ref]["suffix"]) for ref in ref_ids)
    digest = index_digest(
        records, units, quote_sha256, zip(occurrence_unit, occurrence_start, occurrence_suffix, occurrence_label)
    )
    return SourceIndex(
        tuple(records),
        tuple(units),
        unit_position,
        {unit["id"]: unit for unit in units},
        tuple(unit["id"] for unit in units),
        quote_sha256,
        tuple(unit_chapter),
        tuple(unit_offset),
        ref_ids,
        occurrence_unit,
        occurrence_label,
        occurrence_start,
        occurrence_suffix,
        digest,
    )


# ##################################################################
# joined text
# every unit quote joined by newlines with the start of each quote, so an arbitrary literal can be located as a whole word and mapped back to the immutable unit that holds it.
class JoinedText:
    def __init__(self, units: Sequence[dict]) -> None:
        self.text = "\n".join(unit["quote"] for unit in units)
        self.starts = []
        position = 0
        for unit in units:
            self.starts.append(position)
            position += len(unit["quote"]) + 1

    def word_char(self, position: int) -> bool:
        if position < 0 or position >= len(self.text):
            return False
        char = self.text[position]
        return char.isalnum() or char == "_"

    def whole_word(self, needle: str, limit: int | None = None) -> list[int]:
        found: list[int] = []
        at = self.text.find(needle) if needle else -1
        while at >= 0 and (limit is None or len(found) < limit):
            if not self.word_char(at - 1) and not self.word_char(at + len(needle)):
                found.append(at)
            at = self.text.find(needle, at + 1)
        return found

    def unit_of(self, position: int) -> int:
        return bisect.bisect_right(self.starts, position) - 1


# ##################################################################
# lowercase oracle
# answers "does this label occur in lowercase anywhere in the source" for every label from one pass over the text; identical to a bounded regex search per label but without rescanning the book for each one.
class LowercaseOracle:
    def __init__(self, units: Sequence[dict]) -> None:
        self.joined = JoinedText(units)
        self.runs = set(re.findall(r"\w+", self.joined.text))
        self.known: dict[str, bool] = {}

    def present(self, label: str) -> bool:
        folded = label.casefold()
        if folded not in self.known:
            self.known[folded] = (
                folded in self.runs if re.fullmatch(r"\w+", folded) else bool(self.joined.whole_word(folded, 1))
            )
        return self.known[folded]


# ##################################################################
# candidate group
# all occurrences of one exact label (no spelling variant is ever merged) with a content hash of their exact immutable scopes.
@dataclass(frozen=True, slots=True)
class CandidateGroup:
    id: str
    label: str
    normalized: str
    occurrences: tuple[int, ...]
    chapters: int
    standalone: int
    nonentity: bool
    scope_sha256: str


@dataclass(frozen=True, slots=True)
class Grouping:
    groups: tuple[CandidateGroup, ...]
    dropped_labels: int
    dropped_occurrences: int


# ##################################################################
# scope hash
# binds a group to the exact chapter hash, quote hash, label and span offset of every occurrence in source order.
def scope_hash(index: SourceIndex, label: str, occurrences: Sequence[int]) -> str:
    lines = (
        f"{index.units[index.occurrence_unit[occurrence]]['chapter_sha256']}\0"
        f"{index.quote_sha256[index.occurrence_unit[occurrence]]}\0{label}\0{index.occurrence_start[occurrence]}"
        for occurrence in occurrences
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ##################################################################
# nonentity evidence
# the ledger's literal "impact of / attack of / team-house-clan" cue, checked only in units that contain any such cue.
def nonentity_label(label: str, quotes: Sequence[str]) -> bool:
    escaped = re.escape(label)
    first = re.compile(rf"(?i)\b(?:impact of|attack of) {escaped}(?:\s+and\s+[A-Z][a-z]+)?\b")
    second = re.compile(rf"(?i)\b{escaped}\s+(?:air|team|group|clan|family|house)\b")
    return any(first.search(quote) or second.search(quote) for quote in quotes)


# ##################################################################
# group candidates
# one pass: exact-label groups, then the same retention rule as the ledger (multi-word, or a standalone capitalized label with no lowercase occurrence), in a deterministic order (more words first, then exact label).
def group_candidates(index: SourceIndex) -> Grouping:
    by_label: dict[str, list[int]] = {}
    for occurrence, label in enumerate(index.occurrence_label):
        by_label.setdefault(label, []).append(occurrence)
    oracle = LowercaseOracle(index.units)
    qualified = {
        normalized_id(word)
        for label in by_label
        if " " in label and any(not oracle.present(word) for word in label.split())
        for word in label.split()
    }
    retained = [
        label
        for label, occurrences in by_label.items()
        if normalized_id(label)
        and (" " in label or any(not index.occurrence_suffix[item] for item in occurrences))
        and (" " in label or normalized_id(label) in qualified or not oracle.present(label))
    ]
    retained.sort(key=lambda label: (-len(label.split()), label))
    cue: dict[int, bool] = {}
    groups = []
    for label in retained:
        occurrences = tuple(by_label[label])
        units = sorted({index.occurrence_unit[item] for item in occurrences})
        for unit in units:
            if unit not in cue:
                cue[unit] = bool(NONENTITY_CUE.search(index.units[unit]["quote"]))
        scope = scope_hash(index, label, occurrences)
        groups.append(
            CandidateGroup(
                f"g{scope[:16]}",
                label,
                normalized_id(label),
                occurrences,
                len({index.unit_chapter[unit] for unit in units}),
                sum(1 for item in occurrences if not index.occurrence_suffix[item]),
                nonentity_label(label, [index.units[unit]["quote"] for unit in units if cue[unit]]),
                scope,
            )
        )
    kept = set(retained)
    dropped = [label for label in by_label if label not in kept]
    return Grouping(tuple(groups), len(dropped), sum(len(by_label[label]) for label in dropped))


# ##################################################################
# partition
# a subset of one group's occurrences that share one reuse status: literal_owner (exact approved name/alias), alias or non_character (a proven mention-scoped decision), ambiguous (a final recorded non-decision), stale (a recorded decision that must be re-decided) or open (nothing established).
@dataclass(frozen=True, slots=True)
class Partition:
    kind: str
    canonical: str | None
    occurrences: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ResolvedGroup:
    group: CandidateGroup
    known_owner: str | None
    partitions: tuple[Partition, ...]

    def needs_decision(self) -> tuple[int, ...]:
        return tuple(
            sorted(item for part in self.partitions if part.kind in NEEDS_DECISION for item in part.occurrences)
        )


# ##################################################################
# owner names
# registry facts only: normalized canonical id / display name -> actors carrying it, in registry order.
def owner_names(registry: dict) -> dict[str, list[str]]:
    names: dict[str, list[str]] = {}
    for actor_id, entry in registry.items():
        for key in {normalized_id(actor_id), normalized_id(str(entry.get("name", actor_id)))}:
            names.setdefault(key, []).append(actor_id)
    return names


# ##################################################################
# literal owner
# the ledger's known_owner rule (audited alias, else a unique exact name), additionally required to be a real cast member.
def literal_owner(label: str, registry: dict, aliases: dict, names: dict[str, list[str]]) -> str | None:
    label_id = normalized_id(label)
    owners = names.get(label_id, [])
    owner = aliases.get(label_id) or (owners[0] if len(owners) == 1 else None)
    return owner if owner in registry else None


# ##################################################################
# scoped status
# applies the existing guards to one exact recorded mention: an alias stays bound only while scoped_alias_proof still proves it from the source, a recorded non-decision is final only while every plausible owner was offered, anything else is re-decided.
def scoped_status(
    index: SourceIndex,
    occurrence: int,
    record: dict,
    registry: dict,
    aliases: dict,
    owners_of: Callable[[str], list[str]],
) -> str:
    decision = record["decision"]
    if decision == "non_character":
        return "non_character"
    if decision == "ambiguous":
        return "stale" if readjudication_due(record, owners_of(record["label"])) else "ambiguous"
    if decision != "alias":
        return "stale"
    unit = index.occurrence_unit[occurrence]
    scene = scene_units_at(index.by_id, index.order, unit)
    proven = scoped_alias_proof(record["label"], record["canonical"], index.units[unit], scene, registry, aliases)
    return "stale" if proven[1] else "alias"


# ##################################################################
# resolve known
# partitions each group by what is already established under existing guards: exact scoped records first, then the literal owner, otherwise open. Scope is the exact (chapter hash, quote hash, label, span start); similar spellings are never reused.
def resolve_known(
    index: SourceIndex, grouping: Grouping, registry: dict, aliases: dict, scoped_audit: list[dict] | None = None
) -> tuple[ResolvedGroup, ...]:
    scoped = mention_scoped_audit_index(scoped_audit or [])
    names = owner_names(registry)
    relevant: dict[str, list[str]] = {}

    def owners_of(label: str) -> list[str]:
        if label not in relevant:
            relevant[label] = adjudication_owners(label, registry, aliases)
        return relevant[label]

    resolved = []
    for group in grouping.groups:
        known = literal_owner(group.label, registry, aliases, names)
        buckets: dict[tuple[str, str | None], list[int]] = {}
        for occurrence in group.occurrences:
            record = scoped.get(index.scope(occurrence)) if scoped else None
            if record is None:
                key = ("literal_owner", known) if known else ("open", None)
            else:
                kind = scoped_status(index, occurrence, record, registry, aliases, owners_of)
                key = (kind, record["canonical"] if kind == "alias" else None)
            buckets.setdefault(key, []).append(occurrence)
        ordered = sorted(buckets, key=lambda key: (PARTITION_ORDER.index(key[0]), key[1] or ""))
        resolved.append(ResolvedGroup(group, known, tuple(Partition(k, c, tuple(buckets[(k, c)])) for k, c in ordered)))
    return tuple(resolved)


# ##################################################################
# diversified contexts
# a deterministic, purely positional choice of at most `limit` distinct-sentence occurrences spread over the book: the first, the last, then repeatedly the occurrence in a not-yet-used chapter farthest from every pick. No text is generated; nothing is ranked by a model.
def diversified_occurrences(index: SourceIndex, occurrences: Sequence[int], limit: int) -> list[int]:
    seen: set[str] = set()
    distinct = []
    for occurrence in occurrences:
        sha = index.quote_sha256[index.occurrence_unit[occurrence]]
        if sha not in seen:
            seen.add(sha)
            distinct.append(occurrence)
    if limit <= 0:
        return []
    if len(distinct) <= limit:
        return distinct
    picked = [distinct[0], distinct[-1]][:limit]
    while len(picked) < limit:
        positions = sorted(index.occurrence_unit[item] for item in picked)
        covered = {index.unit_chapter[unit] for unit in positions}
        best, best_score = None, None
        for occurrence in distinct:
            if occurrence in picked:
                continue
            unit = index.occurrence_unit[occurrence]
            at = bisect.bisect_left(positions, unit)
            near = min(abs(unit - positions[j]) for j in (at - 1, at) if 0 <= j < len(positions))
            score = (index.unit_chapter[unit] not in covered, near)
            if best_score is None or score > best_score:
                best, best_score = occurrence, score
        picked.append(best)
    return sorted(picked)


# ##################################################################
# context record
# source facts only for one occurrence: ids, exact hashes, the sentence itself and the bounded scene's unit range.
def context_record(index: SourceIndex, occurrence: int) -> dict:
    unit = index.occurrence_unit[occurrence]
    low, high = bounded_scene_range(index.by_id, index.order, unit, ADJUDICATION_SCENE_CHARS)
    return {
        "ref_id": index.occurrence_ref[occurrence],
        "unit_id": index.units[unit]["id"],
        "chapter": index.units[unit]["chapter"],
        "chapter_sha256": index.units[unit]["chapter_sha256"],
        "quote_sha256": index.quote_sha256[unit],
        "label": index.occurrence_label[occurrence],
        "span_start": index.occurrence_start[occurrence],
        "quote": index.units[unit]["quote"],
        "scene": [index.units[low]["id"], index.units[high]["id"]],
    }


# ##################################################################
# group payload
# the compact, JSON-safe record a later whole-book decision step consumes: counts, hashes, the status of every partition, and contexts drawn only from occurrences that still need a decision.
def group_payload(index: SourceIndex, resolved: ResolvedGroup, limit: int = DEFAULT_CONTEXT_LIMIT) -> dict:
    group = resolved.group
    picks = diversified_occurrences(index, resolved.needs_decision(), limit)
    return {
        "id": group.id,
        "label": group.label,
        "occurrences": len(group.occurrences),
        "chapters": group.chapters,
        "standalone": group.standalone,
        "nonentity": group.nonentity,
        "known_owner": resolved.known_owner,
        "scope_sha256": group.scope_sha256,
        "partitions": [
            {"kind": part.kind, "canonical": part.canonical, "count": len(part.occurrences)}
            for part in resolved.partitions
        ],
        "contexts": [context_record(index, occurrence) for occurrence in picks],
    }


# ##################################################################
# cast index
# the complete result of one whole-book pass with per-stage timings.
@dataclass(frozen=True, slots=True)
class CastIndex:
    source: SourceIndex
    grouping: Grouping
    resolved: tuple[ResolvedGroup, ...]
    timings: dict[str, float]

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for item in self.resolved:
            for part in item.partitions:
                kinds[part.kind] = kinds.get(part.kind, 0) + len(part.occurrences)
        return {
            "version": INDEX_VERSION,
            "digest": self.source.digest,
            "chapters": len(self.source.chapters),
            "units": len(self.source.units),
            "occurrences": len(self.source.occurrence_label),
            "groups": len(self.grouping.groups),
            "dropped_labels": self.grouping.dropped_labels,
            "dropped_occurrences": self.grouping.dropped_occurrences,
            "occurrences_by_status": dict(sorted(kinds.items())),
            "timings": self.timings,
        }


# ##################################################################
# chunk plan
# the whole source cut into consecutive paragraph-boundary chunks, each source character in exactly one chunk. A segment is an exact slice of one chapter's text (hash of its bytes); a chunk may hold several segments and a chapter may span chunks, but a paragraph is never split. How a chunk is rendered into any request is the caller's business; only the verbatim slices are fixed here.
PARAGRAPH = re.compile(r".+?(?:\n[^\S\n]*\n\s*|\Z)", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Segment:
    chapter: int
    start: int
    end: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    segments: tuple[Segment, ...]
    chars: int
    sha256: str
    oversize: bool


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    max_chars: int
    chunks: tuple[Chunk, ...]
    coverage: dict
    occurrence_chunk: tuple[int, ...]
    straddling: tuple[int, ...]
    digest: str


def chapter_text(index: SourceIndex, position: int) -> str:
    record = index.chapters[position]
    first = record["first_unit"]
    return "".join(unit["quote"] for unit in index.units[first : first + record["units"]]) + record["tail"]


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in PARAGRAPH.finditer(text) if match.end() > match.start()]


def chunk_of(index: SourceIndex, ordinal: int, parts: list, oversize: bool) -> Chunk:
    segments = tuple(Segment(chapter, start, end, hasher.hexdigest()) for chapter, start, end, hasher in parts)
    lines = (f"{index.chapters[seg.chapter]['text_sha256']}\0{seg.start}\0{seg.end}\0{seg.sha256}" for seg in segments)
    return Chunk(
        f"k{ordinal:04d}",
        segments,
        sum(seg.end - seg.start for seg in segments),
        hashlib.sha256("\n".join(lines).encode()).hexdigest(),
        oversize,
    )


# ##################################################################
# pack chunks
# greedy, in source order: a paragraph joins the open chunk while the total stays within max_chars, otherwise it opens the next chunk; a single paragraph larger than max_chars is its own chunk and is flagged oversize.
def pack_chunks(index: SourceIndex, max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    parts: list = []
    size = 0
    for chapter in range(len(index.chapters)):
        text = chapter_text(index, chapter)
        for start, end in paragraph_spans(text):
            if parts and size + (end - start) > max_chars:
                chunks.append(chunk_of(index, len(chunks), parts, False))
                parts, size = [], 0
            if parts and parts[-1][0] == chapter and parts[-1][2] == start:
                parts[-1][2] = end
            else:
                parts.append([chapter, start, end, hashlib.sha256()])
            parts[-1][3].update(text[start:end].encode("utf-8"))
            size += end - start
    if parts:
        chunks.append(chunk_of(index, len(chunks), parts, False))
    return [Chunk(chunk.id, chunk.segments, chunk.chars, chunk.sha256, chunk.chars > max_chars) for chunk in chunks]


# ##################################################################
# verify chunk coverage
# recounts, independently of the packer, that every chapter is covered start-to-end by consecutive segments with no gap, no overlap, and the exact slice hash; returns the facts rather than a verdict.
def verify_chunk_coverage(index: SourceIndex, chunks: Sequence[Chunk]) -> dict:
    cursor = [0] * len(index.chapters)
    gaps = overlaps = mismatches = 0
    texts: dict[int, str] = {}
    for chunk in chunks:
        for seg in chunk.segments:
            if seg.start > cursor[seg.chapter]:
                gaps += 1
            elif seg.start < cursor[seg.chapter]:
                overlaps += 1
            cursor[seg.chapter] = max(cursor[seg.chapter], seg.end)
            text = texts.setdefault(seg.chapter, chapter_text(index, seg.chapter))
            if hashlib.sha256(text[seg.start : seg.end].encode("utf-8")).hexdigest() != seg.sha256:
                mismatches += 1
        texts = {key: value for key, value in texts.items() if key == chunk.segments[-1].chapter}
    short = sum(1 for record in index.chapters if cursor[record["position"]] != record["chars"])
    chars = sum(chunk.chars for chunk in chunks)
    total = sum(record["chars"] for record in index.chapters)
    return {
        "chapters": len(index.chapters),
        "chunks": len(chunks),
        "chars": total,
        "chunk_chars": chars,
        "gaps": gaps,
        "overlaps": overlaps,
        "hash_mismatches": mismatches,
        "chapters_not_fully_covered": short,
        "exactly_once": not (gaps or overlaps or mismatches or short) and chars == total,
    }


# ##################################################################
# occurrence chunks
# the chunk holding each name occurrence, decided by the absolute character offset of its first character; an occurrence whose label runs past its segment is listed as straddling (a label split across a paragraph break).
def occurrence_chunks(index: SourceIndex, chunks: Sequence[Chunk]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    bounds: dict[int, list[tuple[int, int, int]]] = {}
    for ordinal, chunk in enumerate(chunks):
        for seg in chunk.segments:
            bounds.setdefault(seg.chapter, []).append((seg.start, seg.end, ordinal))
    starts = {chapter: [item[0] for item in items] for chapter, items in bounds.items()}
    placed, straddling = [], []
    for occurrence, unit in enumerate(index.occurrence_unit):
        chapter = index.unit_chapter[unit]
        offset = index.unit_offset[unit] + index.occurrence_start[occurrence]
        _, end, ordinal = bounds[chapter][bisect.bisect_right(starts[chapter], offset) - 1]
        placed.append(ordinal)
        if offset + len(index.occurrence_label[occurrence]) > end:
            straddling.append(occurrence)
    return tuple(placed), tuple(straddling)


# ##################################################################
# build chunk plan
# pack, verify, and place every occurrence; fails closed with a typed issue unless every source character is in exactly one chunk.
def build_chunk_plan(index: SourceIndex, max_chars: int) -> ChunkPlan:
    if max_chars <= 0:
        raise ValueError("chunk size must be positive")
    chunks = pack_chunks(index, max_chars)
    coverage = verify_chunk_coverage(index, chunks)
    if not coverage["exactly_once"]:
        raise CastDataIssue("chunk plan does not cover the source exactly once", coverage, "chunk_coverage_failed")
    placed, straddling = occurrence_chunks(index, chunks)
    coverage = {
        **coverage,
        "oversize_chunks": sum(1 for chunk in chunks if chunk.oversize),
        "straddling_labels": len(straddling),
    }
    lines = [f"{index.digest}\0{max_chars}", *(chunk.sha256 for chunk in chunks)]
    return ChunkPlan(
        max_chars, tuple(chunks), coverage, placed, straddling, hashlib.sha256("\n".join(lines).encode()).hexdigest()
    )


# ##################################################################
# claim
# one entity asserted by any upstream extractor, as plain data: an opaque id and the exact names it claims. Nothing about how it was produced is assumed.
@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    names: tuple[str, ...]


# ##################################################################
# claim names
# validates the claims and returns exact (whitespace-trimmed) name -> claim ids; an empty name or a repeated claim id is malformed data.
def claim_names(claims: Sequence[Claim]) -> dict[str, list[str]]:
    names: dict[str, list[str]] = {}
    seen: set[str] = set()
    for claim in claims:
        if claim.claim_id in seen:
            raise ValueError(f"duplicate claim id {claim.claim_id!r}")
        seen.add(claim.claim_id)
        for name in claim.names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"claim {claim.claim_id!r} has an empty or non-text name")
            names.setdefault(name.strip(), []).append(claim.claim_id)
    return {name: sorted(set(ids)) for name, ids in sorted(names.items())}


# ##################################################################
# component names
# every proper whole-word sub-phrase of each multi-word claimed name -> the claims holding it; a factual relation only, never evidence that two spellings are one identity.
def component_names(names: dict[str, list[str]]) -> dict[str, list[str]]:
    parts: dict[str, set[str]] = {}
    for name, ids in names.items():
        words = name.split()
        for low in range(len(words)):
            for high in range(low + 1, len(words) + 1):
                if high - low < len(words):
                    parts.setdefault(" ".join(words[low:high]), set()).update(ids)
    return {phrase: sorted(ids) for phrase, ids in parts.items()}


# ##################################################################
# group chunks
# per-chunk occurrence counts of one group, so an omitted or unreconciled entity names exactly the chunks that contain it.
def group_chunks(plan: ChunkPlan | None, group: CandidateGroup) -> dict[str, int]:
    if plan is None:
        return {}
    counts: dict[str, int] = {}
    for occurrence in group.occurrences:
        key = plan.chunks[plan.occurrence_chunk[occurrence]].id
        counts[key] = counts.get(key, 0) + 1
    return counts


# ##################################################################
# reconcile
# compares the exact-label candidate groups of the source with any set of claimed entities and reports, deterministically and with exact counts: covered (a claim name equals the label), component (the label is a whole-word part of a claimed name; needs a decision), settled (already established under existing guards), omitted (nothing claims it and it still needs a decision, with factual contexts), and unsupported claim names (not a whole word anywhere in the source). Every retained occurrence lands in exactly one status.
def reconcile(
    cast: CastIndex, claims: Sequence[Claim], plan: ChunkPlan | None = None, limit: int = DEFAULT_CONTEXT_LIMIT
) -> dict:
    index = cast.source
    names = claim_names(claims)
    parts = component_names(names)
    report: dict[str, list] = {"covered": [], "component": [], "settled": [], "omitted": []}
    totals = {key: 0 for key in report}
    for resolved in cast.resolved:
        group = resolved.group
        phrase = " ".join(group.label.split())
        if group.label in names:
            status, extra = "covered", {"claims": names[group.label]}
        elif phrase in parts:
            status, extra = "component", {"component_of": parts[phrase]}
        elif not resolved.needs_decision():
            status, extra = "settled", {}
        else:
            status, extra = "omitted", {}
        totals[status] += len(group.occurrences)
        entry = {"id": group.id, "label": group.label, "occurrences": len(group.occurrences), **extra}
        if status in {"component", "omitted"}:
            entry["chunks"] = group_chunks(plan, group)
        if status == "omitted":
            entry = {**group_payload(index, resolved, limit), "chunks": entry["chunks"]}
        report[status].append(entry)
    joined = JoinedText(index.units)
    located = {name: joined.whole_word(name) for name in names}
    unsupported = [{"name": name, "claims": ids} for name, ids in names.items() if not located[name]]
    artifact = {
        "version": INDEX_VERSION,
        "index_digest": index.digest,
        "plan_digest": plan.digest if plan else None,
        "claims": len(claims),
        "claim_names": len(names),
        "unsupported_claim_names": unsupported,
        "occurrences": {**totals, "retained": sum(totals.values()), "dropped": cast.grouping.dropped_occurrences},
        "groups": {key: len(value) for key, value in report.items()},
        **report,
    }
    artifact["digest"] = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return artifact


# ##################################################################
# build cast index
# index, group, and resolve the whole book in one pass each; the first two stages depend only on source bytes and can be cached by digest, the resolve stage depends on the registry, aliases and scoped records.
def build_cast_index(
    chapters: list[Path], registry: dict, aliases: dict, scoped_audit: list[dict] | None = None
) -> CastIndex:
    timings: dict[str, float] = {}
    clock = time.perf_counter()
    source = build_source_index(chapters)
    timings["index_s"], clock = round(time.perf_counter() - clock, 3), time.perf_counter()
    grouping = group_candidates(source)
    timings["group_s"], clock = round(time.perf_counter() - clock, 3), time.perf_counter()
    resolved = resolve_known(source, grouping, registry, aliases, scoped_audit)
    timings["resolve_s"] = round(time.perf_counter() - clock, 3)
    return CastIndex(source, grouping, resolved, timings)


# ##################################################################
# project inputs
# the same chapter list prepare-cast sees (no intro) plus the read-only registry, aliases and scoped audit of an existing project.
def project_inputs(project: Path) -> tuple[list[Path], dict, dict, list[dict]]:
    chapters = sorted(
        (path for path in (project / "chapters").glob("*.txt") if path.name != "00-intro.txt"), key=chapter_order
    )
    progress_path = project / PROGRESS_NAME
    progress = load_object(progress_path, "cast preparation progress") if progress_path.is_file() else {}
    scoped = load_scoped_audit(project) if (project / SCOPED_AUDIT_NAME).is_file() else []
    return chapters, progress.get("registry", {}), progress.get("aliases", {}), scoped


# ##################################################################
# registry claims
# the established cast as plain claims (display name plus every audited alias), used only to exercise reconciliation on a real project.
def registry_claims(registry: dict, aliases: dict) -> list[Claim]:
    return [
        Claim(
            actor_id,
            tuple(
                dict.fromkeys(
                    [str(entry.get("name", actor_id)).strip()]
                    + [alias.replace("_", " ") for alias, target in sorted(aliases.items()) if target == actor_id]
                )
            ),
        )
        for actor_id, entry in registry.items()
        if str(entry.get("name", actor_id)).strip()
    ]


# ##################################################################
# benchmark
# builds the cast index, a chunk plan and a reconciliation for one project and times each stage; the contexts stage covers every group that still needs a decision.
def benchmark(project: Path, limit: int = DEFAULT_CONTEXT_LIMIT, chunk_chars: int = DEFAULT_CHUNK_CHARS) -> dict:
    chapters, registry, aliases, scoped = project_inputs(project)
    built = build_cast_index(chapters, registry, aliases, scoped)
    clock = time.perf_counter()
    payloads = [group_payload(built.source, item, limit) for item in built.resolved]
    contexts_s, clock = round(time.perf_counter() - clock, 3), time.perf_counter()
    plan = build_chunk_plan(built.source, chunk_chars)
    plan_s, clock = round(time.perf_counter() - clock, 3), time.perf_counter()
    report = reconcile(built, registry_claims(registry, aliases), plan, limit)
    reconcile_s = round(time.perf_counter() - clock, 3)
    return {
        **built.summary(),
        "timings": {**built.timings, "contexts_s": contexts_s, "plan_s": plan_s, "reconcile_s": reconcile_s},
        "groups_needing_decision": sum(1 for item in payloads if item["contexts"]),
        "contexts": sum(len(item["contexts"]) for item in payloads),
        "chunk_plan": {"digest": plan.digest, "max_chars": chunk_chars, **plan.coverage},
        "reconcile": {
            "digest": report["digest"],
            "groups": report["groups"],
            "occurrences": report["occurrences"],
            "unsupported_claim_names": len(report["unsupported_claim_names"]),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only whole-book cast index benchmark for a book-reader project")
    parser.add_argument("project", type=Path)
    parser.add_argument("--contexts", type=int, default=DEFAULT_CONTEXT_LIMIT)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    args = parser.parse_args(argv)
    print(json.dumps(benchmark(args.project.resolve(), args.contexts, args.chunk_chars), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
