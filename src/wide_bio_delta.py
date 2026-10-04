"""Resumable delta runner for wide-context biography extraction, built on src.wide_bio and src.cast_index.

The authoritative original source is cut at paragraph boundaries into tokenizer-budget chunks (exact local token counts,
configurable target) and each chunk is sent exactly once per attempt with a compact DELTA prompt: traits that are
already cited by a locally accepted claim are listed once (so they are not repeated), but every source paragraph of the
chunk is always shown, a character that is not established is always expressible (`subject_ref` = `novel`), and an
unclear subject is always expressible (`subject_ref` = `ambiguous`). Nothing is dropped from the source to save tokens.

Every response is saved verbatim with its hashes before it is judged, and every fact is judged locally: exact literal
quote inside the cited paragraph, chapter witness, and subject/candidate reconciliation against the exact-label cast
index. Anything unsupported, ambiguous or incomplete becomes a typed `pending` item; nothing is merged into any registry
or cast. Progress lives in an append-only, hash-chained journal, so an interrupted run resumes without re-asking a saved
chunk, and a bounded deadline always leaves a durable `not_ready` summary (it never claims readiness it does not have).

`plan` and `validate` are offline. Only `run --execute` can reach a model, through the explicit proof config's single
primary route. This module reports exact token and call counts; it makes no throughput or completion-time claim.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.cast_freeze import (
    TITLE_ROLE_TOKENS,
    CastDataIssue,
    candidate_coverage_ledger,
    exact_scope,
    immutable_evidence_units,
    label_components,
    normalized_id,
    owner_name_forms,
    source_bridge_predicate,
)
from src.cast_index import (
    PARAGRAPH,
    CastIndex,
    Claim,
    build_cast_index,
    component_names,
    project_inputs,
    reconcile,
)
from src.llm import request_for
from src.wide_bio import (
    HARD_DEADLINE_S,
    MAX_FIXED_OVERHEAD_TOKENS,
    SOFT_DEADLINE_S,
    ContractError,
    Counter,
    Paragraph,
    ProofConfig,
    Transport,
    build_counter,
    canonical_json,
    chat_transport,
    check_output_dir,
    load_proof_config,
    pack_source,
    padded,
    project_chapters,
    read_source,
    sha256_text,
    validate_calibration,
    write_atomic,
)
from src.wide_bio_tokenizer import TokenizerRefusal

DELTA_VERSION = 5
QUOTE_MAX = 240
VALUE_MAX = 100
REF_NOVEL = "novel"
REF_AMBIGUOUS = "ambiguous"
# short category enum: appearance, role, gender, kinship, gene/beast, age (changes included), source alias
CATEGORIES = (
    "look",
    "role",
    "gender",
    "kin",
    "beast",
    "age",
    "alias",
    "voice",
    "personality",
    "power",
)
# compact-density contract (delta v2). Raw 4096/8192 runs hit the output cap because the model returned long dialogue/action
# sentences as "facts". Density is repaired at the source (prompt + schema + value length) and judged locally with purely
# syntactic rules: a value is the shortest contiguous phrase of the category, never a quotation, a question, a sentence, or a
# clause. Nothing is capped per subject or category: every distinct explicit trait stays expressible.
VALUE_WORDS_MAX = {
    "look": 10,
    "role": 7,
    "gender": 4,
    "kin": 10,
    "beast": 8,
    "age": 8,
    "alias": 6,
    "voice": 8,
    "personality": 8,
    "power": 10,
}
QUOTE_MARKS = '"\u201c\u201d\u00ab\u00bb\u201e'
# closed-class function words: a subject pronoun or a speech verb marks a clause or an utterance, never a trait phrase
CLAUSE_SUBJECT_WORDS = frozenset(
    [
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "them",
        "said",
        "says",
        "say",
        "asked",
        "asks",
        "replied",
        "cried",
        "shouted",
        "whispered",
        "muttered",
        "answered",
        "called",
        "exclaimed",
        "demanded",
        "told",
    ]
)
# copulas, auxiliaries and negations: a finite clause, so only phrases of the plain-noun categories are checked against them
CLAUSE_VERB_WORDS = frozenset(
    [
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "not",
        "no",
        "never",
    ]
)
# Voice and power are also compact attribute phrases: an explanation of
# dialogue is not a voice trait, and a full action clause is not a power name.
PHRASE_CATEGORIES = frozenset(CATEGORIES)
VOICE_REPORT_WORDS = frozenset(
    {"explain", "explained", "explaining", "say", "said", "telling", "told", "replying", "replied", "asking", "asked", "arguing", "argued"}
)
LEADING_WORDS = frozenset(
    ["the", "a", "an", "his", "her", "their", "its", "my", "your", "our"]
)
FACT_FIELDS = ("subject", "category", "value", "paragraph_id")
SUBJECT_FIELDS = ("name", "ref")
PENDING_REASONS = (
    "incomplete_fields",
    "unsupported_category",
    "unknown_paragraph",
    "unsupported_value",
    "value_too_long",
    "no_chapter_witness",
    "unknown_subject_ref",
    "subject_ref_mismatch",
    "subject_not_in_paragraph",
    "ambiguous_subject",
    "novel_collides_established",
    "component_of_other_subject",
    "subject_not_candidate",
    "subject_nonentity",
    "alias_collision",
    "role_scope_uncertain",
    "alias_unbridged",
    "actor_id_collision",
    "dialogue_value",
    "value_not_compact",
    "clause_value",
    "category_incompatible_value",
)
NO_RESPONSE = frozenset({"transport_error", "over_budget", "token_count_refused"})
SYSTEM_PROMPT = (
    "You extract NEW character biography facts from a book excerpt. Return only JSON matching the schema. "
    "Each fact has `subject` (who: `name` as written in the paragraph, and `ref`), `category`, `value` and `paragraph_id`. "
    "`value` is the SHORTEST phrase copied exactly, unmodified and contiguous, from the paragraph named by `paragraph_id` "
    "(one of the [[P id]] markers) that states the trait; it is the evidence itself, so never paraphrase it and never copy "
    "a sentence, a quotation, dialogue, an action or a general observation. "
    f"Categories: {', '.join(CATEGORIES)}. look=physical appearance (one row per distinct feature: hair, eyes, build, clothing worn habitually), "
    "role=an explicit occupation, rank or title only, kin=an explicit named relationship (one row per relative), "
    "beast=an explicit creature or species, gender=explicit gender, age=an explicit age or age change, "
    "alias=another name the source explicitly gives the same character. voice=ONLY a persistent acoustic/vocal quality (pitch, tone, rasp, cadence or volume), such as 'a voice like gravel' or 'a soft voice'; NEVER what someone said, an explanation, advice, dialogue content, an occupation or a personality. "
    "personality=an explicitly named enduring disposition such as 'gruff and patient', never a transient action. "
    "power=ONLY a specific supernatural, cultivation, beast, innate or named-system capability such as 'call lightning' or 'speak with gulls'; NEVER an ordinary action, learned advice, dialogue, explanation or occupation. "
    "role=ONLY an explicit occupation, formal rank, title or social status; NEVER speech, explaining, advice, or an action. "
    "Report every distinct explicit trait of every character; several rows for one character and category are expected when the source states several. "
    "Before emitting EACH row, compare its normalised (subject name, category, value) to every row already emitted. If it is the same trait, OMIT it even if another paragraph also contains it: emit one row using its earliest paragraph id. Do not loop or repeat a trait under different paragraph ids. "
    "Return compact MINIFIED JSON: no indentation or explanatory text. Skip what the source does not state outright: no transient action, no speech, no inference. Do not repeat a trait twice. "
    "An 'Already established' section may list characters and traits already cited: do not repeat those traits, and use the "
    "listed [id] as `subject.ref` for those characters. It is only a de-duplication aid. Read every paragraph: still report "
    "every new trait of a known character, every character that is not listed (`ref` = novel), and any fact whose subject is "
    "unclear (`ref` = ambiguous; never guess an actor). Do not infer, merge identities, or invent; if nothing new qualifies return an empty facts list."
)


# ##################################################################
# text helpers
# exact, deterministic normalisation shared by validation and replay.
def norm_space(text: str) -> str:
    return " ".join(text.split())


def norm_value(text: str) -> str:
    return norm_space(text).casefold().strip(" .,;:!")


def norm_slot(text: str) -> str:
    """Trait slot identity: the normalised value without leading articles or possessives (`his grey eyes` == `grey eyes`)."""
    words = norm_value(text).split()
    while len(words) > 1 and words[0] in LEADING_WORDS:
        words.pop(0)
    return " ".join(words)


def compactness_problem(
    category: str, value: str, subject: str, names: Sequence[str]
) -> str | None:
    """Deterministic syntactic density rules for one literal value; None when it is a compact trait phrase."""
    if any(mark in value for mark in QUOTE_MARKS) or "?" in value or "!" in value:
        return "dialogue_value"
    words = re.findall(r"\w+(?:['\u2019-]\w+)*", value)
    if (
        not words
        or len(words) > VALUE_WORDS_MAX[category]
        or re.search(r"[.;:\n]\s*\S", value)
    ):
        return "value_not_compact"
    lowered = {word.casefold() for word in words}
    if lowered & CLAUSE_SUBJECT_WORDS or (
        category in PHRASE_CATEGORIES and lowered & CLAUSE_VERB_WORDS
    ):
        return "clause_value"
    if category == "voice" and (
        lowered & VOICE_REPORT_WORDS or "that" in lowered
    ):
        return "category_incompatible_value"
    known = {norm_space(name).casefold() for name in [subject, *names]}
    if category not in ("alias", "role") and norm_space(value).casefold() in known:
        return "category_incompatible_value"
    # a phrase led by the character's own name is that character doing something, not a trait of them
    lead = norm_space(value).casefold()
    if category != "alias" and any(
        lead.startswith(name + " ") for name in {norm_space(subject).casefold()}
    ):
        return "clause_value"
    return None


def subject_id(name: str) -> str:
    return "S" + sha256_text(norm_space(name))[:8]


def whole_word(needle: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", text) is not None


# ##################################################################
# per-chunk schema
# every request constrains `paragraph_id` to that chunk's exact paragraph ids and `subject_ref` to novel, ambiguous and the established ids actually shown in that prompt; short quote/value lengths are part of the schema.
def delta_schema(paragraph_ids: Sequence[str], shown_ids: Sequence[str]) -> dict:
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": list(FACT_FIELDS),
        "properties": {
            "subject": {
                "type": "object",
                "additionalProperties": False,
                "required": list(SUBJECT_FIELDS),
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 80},
                    "ref": {
                        "type": "string",
                        "enum": [REF_NOVEL, REF_AMBIGUOUS, *shown_ids],
                    },
                },
            },
            "category": {
                "type": "string",
                "enum": list(CATEGORIES),
                "description": "voice is acoustic quality only; personality is an enduring named disposition; power is a named supernatural/cultivation/beast/innate capability; role is occupation/rank/status only, never speech or action.",
            },
            "value": {"type": "string", "minLength": 1, "maxLength": VALUE_MAX},
            "paragraph_id": {"type": "string", "enum": list(paragraph_ids)},
        },
    }
    facts: dict = {
        "type": "array",
        "description": "Each distinct normalised (subject name, category, value) appears at most once; use its earliest supporting paragraph id.",
        "items": item,
    }
    if not paragraph_ids:
        facts = {"type": "array", "maxItems": 0}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["facts"],
        "properties": {"facts": facts},
    }


# ##################################################################
# settings
# the only knobs of the runner: the source-token target per chunk, the cap on the delta section, and the per-chunk attempt bound (which bounds the model calls).
@dataclass(frozen=True, slots=True)
class DeltaSettings:
    target_tokens: int
    delta_tokens: int = 1024
    max_attempts: int = 1
    temperature: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("target_tokens", "delta_tokens", "max_attempts"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ContractError(
                    f"{name} must be a positive integer", "settings_invalid"
                )
        if (self.temperature is None) != (self.seed is None):
            raise ContractError(
                "sampling needs both a temperature and a seed (or neither)",
                "sampling_invalid",
            )
        if self.temperature is not None:
            # TensorFold treats temperature <= 0 as greedy and ignores the seed, so a sampled proof needs a positive temperature.
            if (
                isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not 0.0 < self.temperature <= 2.0
            ):
                raise ContractError(
                    "sampling temperature must be a number in (0, 2]",
                    "sampling_invalid",
                )
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise ContractError(
                    "sampling seed must be an integer", "sampling_invalid"
                )

    @property
    def sampling(self) -> dict | None:
        if self.temperature is None:
            return None
        return {"temperature": float(self.temperature), "seed": self.seed}


# ##################################################################
# token-budget chunks
# the whole original text, paragraph boundaries only. `source_tokens` is the sum of exact counts of the rendered paragraphs; a lone paragraph above the target becomes its own `oversize` chunk (never split, never truncated) and is still bound by the input budget.
@dataclass(frozen=True, slots=True)
class DeltaChunk:
    id: str
    start: int
    end: int
    paragraphs: tuple[Paragraph, ...]
    source_tokens: int
    oversize: bool


def render_paragraph(paragraph: Paragraph) -> str:
    return f"[[P {paragraph.id}]]\n{paragraph.text}"


def render_user(chunk: DeltaChunk, delta_text: str) -> str:
    body = "\n\n".join(render_paragraph(paragraph) for paragraph in chunk.paragraphs)
    return f"Excerpt {chunk.id}. Extract NEW biography facts.\n\n{delta_text}Source paragraphs:\n\n{body}\n"


def pack_by_tokens(
    source: str, chapters: Sequence[Path], count: Counter, target_tokens: int
) -> tuple[list[DeltaChunk], dict]:
    whole, coverage = pack_source(source, len(source) + 1, chapters)
    paragraphs = [paragraph for chunk in whole for paragraph in chunk.paragraphs]
    if not paragraphs:
        raise ContractError("source has no paragraph text", "source_empty")
    spans = [
        (m.start(), m.end()) for m in PARAGRAPH.finditer(source) if m.end() > m.start()
    ]
    groups: list[tuple[list[Paragraph], int]] = []
    current: list[Paragraph] = []
    size = 0
    for paragraph in paragraphs:
        cost = count(render_paragraph(paragraph))
        if current and size + cost > target_tokens:
            groups.append((current, size))
            current, size = [], 0
        current.append(paragraph)
        size += cost
    groups.append((current, size))
    starts = [0] + [spans[int(group[0].id)][0] for group, _ in groups[1:]]
    ends = starts[1:] + [len(source)]
    chunks = [
        DeltaChunk(
            f"k{number:04d}", start, end, tuple(group), size, size > target_tokens
        )
        for number, ((group, size), start, end) in enumerate(zip(groups, starts, ends))
    ]
    ids = [paragraph.id for chunk in chunks for paragraph in chunk.paragraphs]
    exact = (
        "".join(source[chunk.start : chunk.end] for chunk in chunks) == source
        and ids == [paragraph.id for paragraph in paragraphs]
        and len(set(ids)) == len(ids) == coverage["shown_paragraphs"]
    )
    if not exact:
        raise ContractError(
            "chunks do not cover every source paragraph exactly once", "source_coverage"
        )
    return chunks, coverage


# ##################################################################
# established state
# what is already cited, by subject: the registry actors of the cast being prepared (seeded by direct literal name or audited alias only, their original profiles are never read for anything but de-duplication) plus every locally accepted claim. The claim part is rebuilt by replaying the journal, so it is never trusted from disk.
class Established:
    def __init__(self) -> None:
        self.subjects: dict[str, dict] = {}

    def seed(self, registry: dict, aliases: dict) -> None:
        for actor_id, entry in registry.items():
            if actor_id == "narrator" or not str(entry.get("name", actor_id)).strip():
                continue
            forms = [
                norm_space(form)
                for form in owner_name_forms(actor_id, registry, aliases)
            ]
            traits: dict[str, dict] = {}
            facts = entry.get("facts") if isinstance(entry.get("facts"), dict) else {}
            for category, values in facts.items():
                if category in CATEGORIES and isinstance(values, list):
                    traits[category] = {
                        norm_value(str(v)): str(v) for v in values if str(v).strip()
                    }
            self.subjects[actor_id] = {
                "name": forms[0],
                "aliases": [form for form in forms[1:] if form != forms[0]],
                "traits": traits,
                "claim_ids": [],
                "registry": True,
            }

    def names(self, sid: str) -> list[str]:
        subject = self.subjects[sid]
        return [subject["name"], *subject["aliases"]]

    def resolve(self, name: str) -> str | None:
        wanted = norm_space(name)
        for sid, subject in self.subjects.items():
            if wanted == subject["name"] or wanted in subject["aliases"]:
                return sid
        return None

    def all_names(self) -> list[str]:
        return [name for sid in self.subjects for name in self.names(sid)]

    def has_trait(self, sid: str, category: str, value: str) -> bool:
        subject = self.subjects.get(sid)
        return bool(subject) and norm_slot(value) in {
            norm_slot(known) for known in subject["traits"].get(category, {}).values()
        }

    def apply(self, claims: Sequence[dict]) -> None:
        for claim in claims:
            subject = self.subjects.setdefault(
                claim["subject_id"],
                {
                    "name": norm_space(claim["subject"]),
                    "aliases": [],
                    "traits": {},
                    "claim_ids": [],
                    "registry": False,
                },
            )
            subject["traits"].setdefault(claim["category"], {})[
                norm_value(claim["value"])
            ] = claim["value"].strip()
            subject["claim_ids"].append(claim["claim_id"])
            if claim["category"] == "alias":
                alias = norm_space(claim["value"])
                if alias != subject["name"] and alias not in subject["aliases"]:
                    subject["aliases"].append(alias)


# ##################################################################
# delta section
# only established subjects whose name or alias occurs in this chunk are listed, one compact line each, bounded by the delta token cap (whole subjects are dropped from the end, never partially cut, and the number dropped is recorded). The section never touches the source paragraphs, the novel-subject option or the ambiguous option.
@dataclass(frozen=True, slots=True)
class Delta:
    text: str
    shown: tuple[str, ...]
    truncated_subjects: int
    tokens: int


DELTA_HEADER = "Already established (do not repeat these traits; use the [id] as subject.ref for these characters):\n"


def subject_line(sid: str, subject: dict) -> str:
    parts = [f"[{sid}] {subject['name']}"]
    if subject["aliases"]:
        parts[0] += f" (also: {'; '.join(subject['aliases'])})"
    parts += [
        f"{category}: {'; '.join(values.values())}"
        for category, values in sorted(subject["traits"].items())
    ]
    return " | ".join(parts)


def build_delta(
    state: Established, chunk: DeltaChunk, cap: int, count: Counter
) -> Delta:
    body = "\n".join(paragraph.text for paragraph in chunk.paragraphs)
    relevant = sorted(
        (
            sid
            for sid in state.subjects
            if any(whole_word(name, body) for name in state.names(sid))
        ),
        key=lambda sid: state.subjects[sid]["name"],
    )
    lines = [subject_line(sid, state.subjects[sid]) for sid in relevant]
    keep = len(lines)
    while keep:
        text = DELTA_HEADER + "\n".join(lines[:keep]) + "\n\n"
        if count(text) <= cap:
            return Delta(text, tuple(relevant[:keep]), len(lines) - keep, count(text))
        keep -= 1
    return Delta("", (), len(lines), 0)


# ##################################################################
# candidates
# exact-label candidate groups of the chapter files from the cast index (whitespace-normalised), each with its non-entity flag; a subject that is not one of these exact labels can never be accepted as a novel character.
def candidate_labels(cast: CastIndex) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    for group in cast.grouping.groups:
        key = norm_space(group.label)
        labels[key] = labels.get(key, False) or group.nonentity
    return labels


def role_only(name: str) -> bool:
    words = label_components(name)
    return not words or words <= TITLE_ROLE_TOKENS


# ##################################################################
# validate response
# strict local judgement of one raw response against the paragraphs shown and the state established BEFORE this chunk. The model supplies only a source-literal `value` snippet and a paragraph id; the exact quote (the containing sentence), its offset and the chapter witness are reconstructed here from the source, never taken from the model. Everything not provably supported becomes `pending` with a typed reason; an exact repeat of an established trait is a `duplicate` (counted, not a claim). Only a response that is not the expected JSON object is a whole-chunk failure.
def validate_delta_response(
    raw: str,
    chunk: DeltaChunk,
    state: Established,
    shown: Sequence[str],
    candidates: dict[str, bool],
) -> dict:
    result = {
        "chunk_id": chunk.id,
        "raw_sha256": sha256_text(raw),
        "status": "ok",
        "claims": [],
        "pending": [],
        "duplicates": [],
    }
    try:
        data = json.loads(raw)
    except ValueError:
        return {**result, "status": "invalid_json"}
    facts = (
        data.get("facts") if isinstance(data, dict) and set(data) == {"facts"} else None
    )
    if not isinstance(facts, list):
        return {**result, "status": "invalid_shape"}
    by_id = {paragraph.id: paragraph for paragraph in chunk.paragraphs}
    established = state.all_names()
    chunk_names: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for position, fact in enumerate(facts):
        outcome = judge_fact(
            fact, by_id, state, shown, candidates, established, chunk_names
        )
        if outcome["kind"] == "pending":
            result["pending"].append(
                {
                    "pending_id": sha256_text(
                        canonical_json([chunk.id, position, fact])
                    ),
                    "chunk_id": chunk.id,
                    "position": position,
                    "pending_reason": outcome["reason"],
                    "fact": fact if isinstance(fact, dict) else None,
                }
            )
            continue
        key = (outcome["subject_id"], outcome["category"], norm_slot(outcome["value"]))
        known = state.has_trait(key[0], key[1], outcome["value"])
        if known or key in seen:
            result["duplicates"].append(
                {
                    "position": position,
                    "reason": "already_established"
                    if known
                    else "duplicate_in_response",
                    "subject_id": key[0],
                    "category": key[1],
                    "value": outcome["value"],
                }
            )
            continue
        seen.add(key)
        if outcome["new_subject"]:
            chunk_names.append(outcome["subject"])
        result["claims"].append(outcome["claim"](chunk, position))
    if result["pending"] or result["duplicates"]:
        result["status"] = "partial"
    return result


def pending(reason: str) -> dict:
    return {"kind": "pending", "reason": reason}


def judge_fact(
    fact,
    by_id: dict[str, Paragraph],
    state: Established,
    shown: Sequence[str],
    candidates: dict[str, bool],
    established: list[str],
    chunk_names: list[str],
) -> dict:
    who = fact.get("subject") if isinstance(fact, dict) else None
    if (
        not isinstance(fact, dict)
        or set(fact) != set(FACT_FIELDS)
        or not isinstance(who, dict)
        or set(who) != set(SUBJECT_FIELDS)
        or not all(
            isinstance(fact[field], str) and fact[field].strip()
            for field in FACT_FIELDS
            if field != "subject"
        )
        or not all(
            isinstance(who[field], str) and who[field].strip()
            for field in SUBJECT_FIELDS
        )
    ):
        return pending("incomplete_fields")
    if fact["category"] not in CATEGORIES:
        return pending("unsupported_category")
    paragraph = by_id.get(fact["paragraph_id"])
    if paragraph is None:
        return pending("unknown_paragraph")
    value = fact["value"].strip()
    if value not in paragraph.text:
        return pending("unsupported_value")
    if len(value) > VALUE_MAX:
        return pending("value_too_long")
    problem = compactness_problem(fact["category"], value, who["name"], established)
    if problem:
        return pending(problem)
    if paragraph.witness is None:
        return pending("no_chapter_witness")
    subject, ref = norm_space(who["name"]), who["ref"]
    new_subject = False
    if ref == REF_AMBIGUOUS:
        return pending("ambiguous_subject")
    if ref == REF_NOVEL:
        problem = novel_problem(
            subject, paragraph, state, candidates, established, chunk_names
        )
        if problem:
            return pending(problem)
        sid, new_subject = subject_id(subject), subject not in chunk_names
    else:
        if ref not in shown or ref not in state.subjects:
            return pending("unknown_subject_ref")
        # direct known literal reuse only: the written name must be the actor's exact name or audited alias
        if subject not in state.names(ref):
            return pending("subject_ref_mismatch")
        if not any(whole_word(name, paragraph.text) for name in state.names(ref)):
            return pending("subject_not_in_paragraph")
        sid = ref
    if fact["category"] == "alias":
        alias = norm_space(value)
        owner = state.resolve(alias)
        if (
            alias == subject
            or (owner is not None and owner != sid)
            or (alias in chunk_names and alias != subject)
        ):
            return pending("alias_collision")
    return {
        "kind": "claim",
        "subject": subject,
        "subject_id": sid,
        "category": fact["category"],
        "value": value,
        "new_subject": new_subject,
        "claim": lambda chunk, position: claim_record(
            chunk, paragraph, value, fact["category"], subject, ref, sid, position
        ),
    }


def novel_problem(
    subject: str,
    paragraph: Paragraph,
    state: Established,
    candidates: dict[str, bool],
    established: list[str],
    chunk_names: list[str],
) -> str | None:
    if not whole_word(subject, paragraph.text):
        return "subject_not_in_paragraph"
    if state.resolve(subject) is not None:
        return "novel_collides_established"
    # a role or title is never a character by itself: scope is uncertain, so it waits as pending
    if role_only(subject):
        return "role_scope_uncertain"
    others = [name for name in [*established, *chunk_names] if name != subject]
    own_parts = component_names({subject: [subject]})
    parts = component_names({name: [name] for name in others})
    if subject in parts or any(name in own_parts for name in others):
        return "component_of_other_subject"
    if subject not in candidates:
        return "subject_not_candidate"
    if candidates[subject]:
        return "subject_nonentity"
    return None


# ##################################################################
# reconstructed evidence
# the exact source quote is the sentence of the paragraph that contains the model's literal snippet (or the snippet itself when that sentence is too long), with its paragraph id, absolute offset and chapter witness.
def reconstruct_quote(text: str, value: str) -> str:
    at = text.find(value)
    low = max((text.rfind(mark, 0, at) for mark in ".!?\n"), default=-1) + 1
    ends = [text.find(mark, at + len(value)) for mark in ".!?\n"]
    high = min((end + 1 for end in ends if end >= 0), default=len(text))
    quote = text[low:high].strip()
    return quote if value in quote and len(quote) <= QUOTE_MAX else value


def claim_record(
    chunk: DeltaChunk,
    paragraph: Paragraph,
    value: str,
    category: str,
    subject: str,
    ref: str,
    sid: str,
    position: int,
) -> dict:
    quote = reconstruct_quote(paragraph.text, value)
    claim = {
        "chunk_id": chunk.id,
        "position": position,
        "subject": subject,
        "subject_id": sid,
        "subject_ref": ref,
        "category": category,
        "value": value,
        "quote": quote,
        "quote_sha256": sha256_text(quote),
        "value_offset": paragraph.start + paragraph.text.find(value),
        "source_offset": paragraph.start + paragraph.text.find(quote),
        "paragraph_id": paragraph.id,
        "paragraph_sha256": paragraph.sha256,
        "witness": paragraph.witness,
    }
    claim["claim_id"] = sha256_text(
        canonical_json(
            {
                "subject_id": sid,
                "category": category,
                "value": norm_value(value),
                "paragraph_sha256": paragraph.sha256,
                "value_offset": claim["value_offset"],
            }
        )
    )
    return claim


# ##################################################################
# plan
# the whole original text as ordered token-budget chunks. The plan fails closed if even an empty delta section plus the delta cap would overflow the input budget for any chunk (no truncation, ever). The attempt bound is not part of the plan identity, so a resume may raise it.
@dataclass(frozen=True, slots=True)
class DeltaPlan:
    source: str
    chunks: tuple[DeltaChunk, ...]
    settings: DeltaSettings
    fixed_overhead: int
    artifact: dict


def build_delta_plan(
    source: str,
    chapters: Sequence[Path],
    config: ProofConfig,
    count: Counter,
    settings: DeltaSettings,
    fixed_overhead: int = 0,
    seed_sha256: str = "",
) -> DeltaPlan:
    if (
        not isinstance(fixed_overhead, int)
        or isinstance(fixed_overhead, bool)
        or not 0 <= fixed_overhead <= MAX_FIXED_OVERHEAD_TOKENS
    ):
        raise ContractError(
            f"fixed overhead must be an integer in 0..{MAX_FIXED_OVERHEAD_TOKENS}",
            "overhead_invalid",
        )
    if settings.sampling is not None and config.backend.style != "openai":
        raise ContractError(
            "sampling temperature/seed are sent only to an openai-style (TensorFold) backend",
            "sampling_unsupported",
        )
    chunks, coverage = pack_by_tokens(source, chapters, count, settings.target_tokens)
    system_tokens = count(SYSTEM_PROMPT)
    entries, over = [], []
    for chunk in chunks:
        base = system_tokens + count(render_user(chunk, ""))
        worst = padded(
            base + settings.delta_tokens + fixed_overhead, config.tolerance_percent
        )
        if worst > config.input_budget:
            over.append(chunk.id)
        entries.append(
            {
                "id": chunk.id,
                "start": chunk.start,
                "end": chunk.end,
                "paragraphs": len(chunk.paragraphs),
                "first_paragraph_id": chunk.paragraphs[0].id,
                "last_paragraph_id": chunk.paragraphs[-1].id,
                "source_tokens": chunk.source_tokens,
                "oversize": chunk.oversize,
                "base_prompt_tokens": base,
                "max_padded_prompt_tokens": worst,
                "base_user_sha256": sha256_text(render_user(chunk, "")),
            }
        )
    if over:
        raise ContractError(
            f"chunks {over} cannot fit the input budget of {config.input_budget} tokens with a {settings.delta_tokens}-token delta",
            "input_budget_exceeded",
        )
    artifact = {
        "delta_version": DELTA_VERSION,
        "model": config.backend.model,
        "num_ctx": config.backend.num_ctx,
        "output_tokens": config.output_tokens,
        "reserve_tokens": config.reserve_tokens,
        "tolerance_percent": config.tolerance_percent,
        "fixed_overhead_tokens": fixed_overhead,
        "input_budget": config.input_budget,
        "tokenizer_capture_sha256": config.tokenizer_sha256,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "compact_contract_sha256": sha256_text(
            canonical_json(
                {
                    "value_max": VALUE_MAX,
                    "words": VALUE_WORDS_MAX,
                    "quote_marks": QUOTE_MARKS,
                    "subject_words": sorted(CLAUSE_SUBJECT_WORDS),
                    "verb_words": sorted(CLAUSE_VERB_WORDS),
                    "phrase_categories": sorted(PHRASE_CATEGORIES),
                    "leading": sorted(LEADING_WORDS),
                }
            )
        ),
        "schema_template_sha256": sha256_text(
            canonical_json(delta_schema(["p"], ["s"]))
        ),
        "target_tokens": settings.target_tokens,
        "delta_tokens": settings.delta_tokens,
        "seed_sha256": seed_sha256,
        "coverage": coverage,
        "chunks": entries,
    }
    if settings.sampling is not None:
        # present only when configured, so every unsampled plan keeps its existing fingerprint
        artifact["sampling"] = settings.sampling
    artifact["plan_sha256"] = sha256_text(canonical_json(artifact))
    return DeltaPlan(source, tuple(chunks), settings, fixed_overhead, artifact)


# ##################################################################
# journal
# append-only, hash-chained, fsynced JSONL. A torn final line (a crash mid-append, never a committed record) is quarantined beside the journal and cut off; any other inconsistency refuses the run.
class Journal:
    def __init__(self, out_dir: Path) -> None:
        self.path = out_dir / "journal.jsonl"
        self.records: list[dict] = []

    def load(self) -> list[dict]:
        data = self.path.read_bytes() if self.path.is_file() else b""
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            torn = data[cut:]
            self.path.with_name(
                f"journal.torn-{hashlib.sha256(torn).hexdigest()[:12]}"
            ).write_bytes(torn)
            self.path.write_bytes(data[:cut])
            data = data[:cut]
        previous = "0" * 64
        self.records = []
        for number, line in enumerate(data.decode("utf-8").splitlines()):
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ContractError(
                    f"journal line {number} is not JSON", "journal_corrupt"
                ) from error
            body = {
                key: value for key, value in record.items() if key != "record_sha256"
            }
            if (
                record.get("seq") != number
                or record.get("prev") != previous
                or record.get("record_sha256") != sha256_text(canonical_json(body))
            ):
                raise ContractError(
                    f"journal record {number} breaks the hash chain", "journal_corrupt"
                )
            previous = record["record_sha256"]
            self.records.append(record)
        return self.records

    def append(self, core: dict, elapsed_s: float) -> dict:
        body = {
            **core,
            "elapsed_s": round(elapsed_s, 3),
            "seq": len(self.records),
            "prev": self.records[-1]["record_sha256"] if self.records else "0" * 64,
        }
        record = {**body, "record_sha256": sha256_text(canonical_json(body))}
        with self.path.open("ab") as stream:
            stream.write((canonical_json(record) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(record)
        return record


# ##################################################################
# prepared request
# one chunk's exact request given the current established state: delta section, user prompt, per-chunk schema, payload and exact local token counts.
@dataclass(frozen=True, slots=True)
class Prepared:
    url: str
    payload: bytes
    user: str
    delta: Delta
    schema_sha256: str
    tokens: int
    padded_tokens: int

    @property
    def request_sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


def prepare(
    config: ProofConfig,
    plan: DeltaPlan,
    chunk: DeltaChunk,
    state: Established,
    count: Counter,
    output_tokens: int | None = None,
    extra_overhead: int = 0,
) -> Prepared:
    delta = build_delta(state, chunk, plan.settings.delta_tokens, count)
    user = render_user(chunk, delta.text)
    tokens = count(SYSTEM_PROMPT) + count(user)
    schema = delta_schema([paragraph.id for paragraph in chunk.paragraphs], delta.shown)
    url, payload = request_for(
        config.backend,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        plan.settings.temperature or 0.0,
        output_tokens or config.output_tokens,
        schema,
        plan.settings.seed,
    )
    return Prepared(
        url,
        payload,
        user,
        delta,
        sha256_text(canonical_json(schema)),
        tokens,
        padded(tokens + plan.fixed_overhead + extra_overhead, config.tolerance_percent),
    )


def classify(
    config: ProofConfig, meta: dict, validation: dict, padded_tokens: int
) -> str:
    if meta.get("model") != config.backend.model:
        return "fallback_route"
    if meta.get("done_reason") == "length":
        return "truncated"
    reported = meta.get("prompt_eval_count")
    if isinstance(reported, int) and reported > padded_tokens:
        return "token_drift"
    if validation["status"] in {"invalid_json", "invalid_shape"}:
        return validation["status"]
    return "ready"


META_KEYS = ("model", "done_reason", "prompt_eval_count", "eval_count")


def bounded_call(
    transport: Transport, url: str, payload: bytes, timeout: float
) -> dict:
    """One call with a true wall-clock bound: a response that has not arrived by `timeout` is abandoned (daemon thread)."""
    box: dict = {}

    def target() -> None:
        try:
            box["reply"] = transport(url, payload, timeout)
        except BaseException as error:  # noqa: BLE001 - relayed to the caller below
            box["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(max(timeout, 0.0))
    if thread.is_alive():
        raise TimeoutError(
            f"no response within the {timeout:.1f}s remaining before the hard deadline"
        )
    if "error" in box:
        raise box["error"]
    return box["reply"]


# ##################################################################
# run
# one object owns a run directory: it replays the journal (re-judging every saved raw response against rebuilt state, so nothing on disk is trusted beyond its hash), sends only what is missing, and always finishes by writing durable artifacts.
class DeltaRun:
    def __init__(
        self,
        chapters: Sequence[Path],
        config: ProofConfig,
        plan: DeltaPlan,
        out_dir: Path,
        count: Counter,
        cast: CastIndex,
        registry: dict | None = None,
        aliases: dict | None = None,
    ) -> None:
        self.chapters, self.config, self.plan, self.out_dir, self.count = (
            chapters,
            config,
            plan,
            out_dir,
            count,
        )
        self.candidates = candidate_labels(cast)
        self.cast = cast
        self.by_id = {chunk.id: chunk for chunk in plan.chunks}
        self.state = Established()
        self.state.seed(registry or {}, aliases or {})
        self.journal = Journal(out_dir)
        self.attempts: dict[str, list[dict]] = {chunk.id: [] for chunk in plan.chunks}
        self.ready: dict[str, dict] = {}
        self.calls_this_invocation = 0
        self.only_chunk: str | None = None

    # adaptive output policy ------------------------------------------
    @property
    def adaptive(self) -> dict | None:
        return self.plan.artifact.get("adaptive")

    def output_tokens_for(self, attempt: int) -> int:
        policy = self.adaptive
        if policy and attempt >= 2:
            return policy["retry_output_tokens"]
        return self.config.output_tokens

    def budget_for(self, attempt: int) -> int:
        return (
            self.config.backend.num_ctx
            - self.output_tokens_for(attempt)
            - self.config.reserve_tokens
        )

    def prep(self, chunk: DeltaChunk, attempt: int) -> Prepared:
        policy = self.adaptive
        return prepare(
            self.config,
            self.plan,
            chunk,
            self.state,
            self.count,
            self.output_tokens_for(attempt),
            policy["retry_margin_tokens"] if policy and attempt >= 2 else 0,
        )

    def length_proven(self, chunk: DeltaChunk) -> bool:
        """True only for a single saved attempt at the base cap whose hashed raw meta says done_reason length with eval_count exactly the cap."""
        records = self.attempts[chunk.id]
        if len(records) != 1:
            return False
        record = records[0]
        server = record["server"]
        return (
            record["status"] == "truncated"
            and record["called"]
            and server.get("model") == self.config.backend.model
            and server.get("done_reason") == "length"
            and server.get("eval_count") == self.config.output_tokens
        )

    def length_exhausted(self) -> list[str]:
        """Chunks whose retry attempt also hit the length cap: typed pending, never retried, never salvaged."""
        if not self.adaptive:
            return []
        return [
            chunk.id
            for chunk in self.plan.chunks
            if len(self.attempts[chunk.id]) >= 2
            and self.attempts[chunk.id][-1]["status"] == "truncated"
        ]

    def wants(self, chunk: DeltaChunk) -> bool:
        if chunk.id in self.ready:
            return False
        if not self.adaptive:
            return len(self.attempts[chunk.id]) < self.plan.settings.max_attempts
        # A bounded parent may contain only a hash-chained prefix.  Untouched
        # chunks receive their one base-cap first attempt; only a saved,
        # raw-proven base-cap length response earns the one larger retry.
        return not self.attempts[chunk.id] or self.length_proven(chunk)

    # replay ---------------------------------------------------------
    def replay(self) -> None:
        for record in self.journal.load():
            chunk = self.by_id.get(record["chunk_id"])
            if chunk is None or record["attempt"] != len(self.attempts[chunk.id]) + 1:
                raise ContractError(
                    "journal does not belong to this plan", "journal_inconsistent"
                )
            if record["status"] in NO_RESPONSE:
                self.verify_without_response(chunk, record)
                self.attempts[chunk.id].append(record)
                continue
            prepared = self.prepare_or_refuse(chunk, record["attempt"])
            raw_path, meta_path = self.raw_paths(chunk.id, record["attempt"])
            try:
                raw, meta_text = (
                    raw_path.read_text(encoding="utf-8"),
                    meta_path.read_text(encoding="utf-8"),
                )
            except OSError as error:
                raise ContractError(
                    f"saved response of {chunk.id} is unreadable",
                    "journal_inconsistent",
                ) from error
            core, validation = self.conclude(
                chunk,
                record["attempt"],
                prepared,
                raw,
                meta_text,
                record["called"],
                record["salvaged"],
            )
            if {key: record.get(key) for key in core} != core:
                raise ContractError(
                    f"saved response of {chunk.id} attempt {record['attempt']} no longer reproduces its journal record",
                    "journal_inconsistent",
                )
            self.absorb(chunk, core, validation)

    def verify_without_response(self, chunk: DeltaChunk, record: dict) -> None:
        if record["request_sha256"] is None:
            return
        try:
            prepared = self.prep(chunk, record["attempt"])
        except TokenizerRefusal as error:
            raise ContractError(str(error), "journal_inconsistent") from error
        if prepared.request_sha256 != record["request_sha256"]:
            raise ContractError(
                f"request of {chunk.id} no longer reproduces its journal record",
                "journal_inconsistent",
            )

    def prepare_or_refuse(self, chunk: DeltaChunk, attempt: int) -> Prepared:
        try:
            return self.prep(chunk, attempt)
        except TokenizerRefusal as error:
            raise ContractError(str(error), "journal_inconsistent") from error

    def raw_paths(self, chunk_id: str, attempt: int) -> tuple[Path, Path]:
        base = self.out_dir / "raw" / f"{chunk_id}.a{attempt}"
        return base.with_name(base.name + ".response.txt"), base.with_name(
            base.name + ".meta.json"
        )

    def absorb(self, chunk: DeltaChunk, core: dict, validation: dict) -> None:
        self.attempts[chunk.id].append(core)
        if core["status"] == "ready":
            self.state.apply(validation["claims"])
            self.ready[chunk.id] = validation

    # conclude -------------------------------------------------------
    def conclude(
        self,
        chunk: DeltaChunk,
        attempt: int,
        prepared: Prepared,
        raw: str,
        meta_text: str,
        called: bool,
        salvaged: bool,
    ) -> tuple[dict, dict]:
        meta = json.loads(meta_text)
        if meta.get("request_sha256") != prepared.request_sha256 or meta.get(
            "response_sha256"
        ) != sha256_text(raw):
            raise ContractError(
                f"saved response of {chunk.id} does not match its request/hash",
                "journal_inconsistent",
            )
        validation = validate_delta_response(
            raw, chunk, self.state, prepared.delta.shown, self.candidates
        )
        core = self.core(
            chunk,
            attempt,
            classify(self.config, meta, validation, prepared.padded_tokens),
            prepared,
        )
        core.update(
            called=called,
            salvaged=salvaged,
            response_sha256=sha256_text(raw),
            meta_sha256=sha256_text(meta_text),
            server={key: meta.get(key) for key in META_KEYS},
            counts={
                key: len(validation[key]) for key in ("claims", "pending", "duplicates")
            },
            result_sha256=sha256_text(canonical_json(validation)),
        )
        return core, validation

    def core(
        self,
        chunk: DeltaChunk,
        attempt: int,
        status: str,
        prepared: Prepared | None,
        error: str | None = None,
    ) -> dict:
        return {
            "chunk_id": chunk.id,
            "attempt": attempt,
            "status": status,
            "called": False,
            "salvaged": False,
            "error": error,
            "request_sha256": prepared.request_sha256 if prepared else None,
            "user_sha256": sha256_text(prepared.user) if prepared else None,
            "delta_sha256": sha256_text(prepared.delta.text) if prepared else None,
            "schema_sha256": prepared.schema_sha256 if prepared else None,
            "local_tokens": prepared.tokens if prepared else None,
            "padded_tokens": prepared.padded_tokens if prepared else None,
            "delta_shown": list(prepared.delta.shown) if prepared else [],
            "delta_truncated_subjects": prepared.delta.truncated_subjects
            if prepared
            else 0,
            "response_sha256": None,
            "meta_sha256": None,
            "server": {},
            "counts": {},
            "result_sha256": None,
        }

    # live -----------------------------------------------------------
    def settle_without_response(
        self,
        chunk: DeltaChunk,
        status: str,
        prepared: Prepared | None,
        error: str,
        elapsed_s: float,
    ) -> None:
        core = self.core(
            chunk, len(self.attempts[chunk.id]) + 1, status, prepared, error
        )
        self.journal.append(core, elapsed_s)
        self.attempts[chunk.id].append(core)

    def process(
        self,
        chunk: DeltaChunk,
        transport: Transport | None,
        clock: Callable[[], float],
        started: float,
        soft_s: float,
        hard_s: float,
    ) -> str | None:
        """Handle one missing chunk. Returns a stop reason when the run must stop, else None."""
        attempt = len(self.attempts[chunk.id]) + 1
        raw_path, meta_path = self.raw_paths(chunk.id, attempt)
        tick = clock()
        try:
            prepared = self.prep(chunk, attempt)
        except TokenizerRefusal as error:
            self.settle_without_response(
                chunk,
                "token_count_refused",
                None,
                f"{error.code}: {error}",
                clock() - tick,
            )
            return None
        if raw_path.is_file() and meta_path.is_file():
            core, validation = self.conclude(
                chunk,
                attempt,
                prepared,
                raw_path.read_text(encoding="utf-8"),
                meta_path.read_text(encoding="utf-8"),
                True,
                True,
            )
            self.journal.append(core, clock() - tick)
            self.absorb(chunk, core, validation)
            return core["status"] == "fallback_route" and "fallback_route" or None
        if prepared.padded_tokens > self.budget_for(attempt):
            self.settle_without_response(
                chunk,
                "over_budget",
                prepared,
                f"{prepared.padded_tokens} > {self.budget_for(attempt)}",
                clock() - tick,
            )
            return None
        remaining = hard_s - (clock() - started)
        if transport is None:
            return "offline"
        if clock() - started >= soft_s or remaining <= 0:
            return "deadline"
        self.calls_this_invocation += 1
        try:
            reply = bounded_call(transport, prepared.url, prepared.payload, remaining)
        except (OSError, TimeoutError, ValueError) as error:
            self.settle_without_response(
                chunk,
                "transport_error",
                prepared,
                f"{type(error).__name__}: {error}"[:300],
                clock() - tick,
            )
            self.mark_called(chunk)
            return "deadline" if isinstance(error, TimeoutError) else "transport_error"
        raw = reply.get("content") or ""
        meta = {key: reply.get(key) for key in META_KEYS} | {
            "request_sha256": prepared.request_sha256,
            "response_sha256": sha256_text(raw),
        }
        meta_text = json.dumps(meta, sort_keys=True)
        write_atomic(raw_path, raw)
        write_atomic(meta_path, meta_text)
        core, validation = self.conclude(
            chunk, attempt, prepared, raw, meta_text, True, False
        )
        self.journal.append(core, clock() - tick)
        self.absorb(chunk, core, validation)
        return "fallback_route" if core["status"] == "fallback_route" else None

    def mark_called(self, chunk: DeltaChunk) -> None:
        """A transport_error record is a call attempt even though it saved no response."""
        record = self.attempts[chunk.id][-1]
        record["called"] = True

    # artifacts ------------------------------------------------------
    def chunk_state(self, chunk: DeltaChunk) -> str:
        attempts = self.attempts[chunk.id]
        return attempts[-1]["status"] if attempts else "missing"

    def calls_total(self) -> int:
        return sum(
            1
            for records in self.attempts.values()
            for record in records
            if record["called"]
        )

    def summary(
        self,
        stop: str | None,
        offline: bool,
        elapsed_s: float,
        error: str | None = None,
    ) -> dict:
        statuses = {
            chunk.id: "ready" if chunk.id in self.ready else self.chunk_state(chunk)
            for chunk in self.plan.chunks
        }
        ready_chars = sum(
            chunk.end - chunk.start
            for chunk in self.plan.chunks
            if chunk.id in self.ready
        )
        total = self.plan.artifact["coverage"]["source_chars"]
        validations = [
            self.ready[chunk.id] for chunk in self.plan.chunks if chunk.id in self.ready
        ]
        reasons = {name: 0 for name in PENDING_REASONS}
        for validation in validations:
            for item in validation["pending"]:
                reasons[item["pending_reason"]] += 1
        complete = len(self.ready) == len(self.plan.chunks)
        reason = None
        if not complete:
            reason = (
                error
                or stop
                or ("only_chunk_proof" if self.only_chunk else None)
                or (
                    "output_length_exhausted"
                    if self.length_exhausted()
                    else ("offline_validation" if offline else "chunks_not_ready")
                )
            )
        return {
            "plan_sha256": self.plan.artifact["plan_sha256"],
            "state": "ready" if complete else "not_ready",
            "reason": reason,
            "source_chars": total,
            "source_chars_ready": ready_chars,
            "source_fraction": round(ready_chars / total, 6) if total else 0.0,
            "chunks": len(self.plan.chunks),
            "chunk_status": {
                status: list(statuses.values()).count(status)
                for status in sorted(set(statuses.values()))
            },
            "claims": sum(len(item["claims"]) for item in validations),
            "pending_claims": sum(len(item["pending"]) for item in validations),
            "pending_reasons": {
                name: number for name, number in reasons.items() if number
            },
            "duplicates": sum(len(item["duplicates"]) for item in validations),
            "length_exhausted": self.length_exhausted(),
            "calls_total": self.calls_total(),
            "calls_this_invocation": self.calls_this_invocation,
            "elapsed_s_this_invocation": round(elapsed_s, 3),
            "cast_status": "pending: nothing is merged into any registry or cast",
        } | ({"only_chunk": self.only_chunk} if self.only_chunk else {})

    def call_budget(self, soft_s: float, hard_s: float) -> dict:
        settings = self.plan.settings
        per_chunk = []
        for entry in self.plan.artifact["chunks"]:
            records = self.attempts[entry["id"]]
            done = entry["id"] in self.ready
            per_chunk.append(
                {
                    "id": entry["id"],
                    "paragraphs": entry["paragraphs"],
                    "source_tokens": entry["source_tokens"],
                    "base_prompt_tokens": entry["base_prompt_tokens"],
                    "max_delta_tokens": settings.delta_tokens,
                    "max_padded_prompt_tokens": entry["max_padded_prompt_tokens"],
                    "max_output_tokens": self.output_tokens_for(len(records) + 1)
                    if self.adaptive
                    else self.config.output_tokens,
                    "attempts_used": len(records),
                    "calls_used": sum(1 for record in records if record["called"]),
                    "ready": done,
                    "max_calls_remaining": (
                        int(self.length_proven(self.by_id[entry["id"]]))
                        if self.adaptive
                        else max(0, settings.max_attempts - len(records))
                    )
                    if not done
                    else 0,
                }
            )
        metas = [
            record["server"]
            for records in self.attempts.values()
            for record in records
            if record["called"]
        ]
        return {
            "plan_sha256": self.plan.artifact["plan_sha256"],
            "chunks": len(per_chunk),
            "max_attempts_per_chunk": settings.max_attempts,
            "max_model_calls": (
                self.calls_total() + sum(i["max_calls_remaining"] for i in per_chunk)
                if self.adaptive
                else len(per_chunk) * settings.max_attempts
            ),
            "calls_used": self.calls_total(),
            "max_calls_remaining": sum(
                item["max_calls_remaining"] for item in per_chunk
            ),
            "worst_case_input_tokens_per_call": max(
                item["max_padded_prompt_tokens"] for item in per_chunk
            ),
            "worst_case_output_tokens_per_call": self.config.output_tokens,
            "input_budget_tokens": self.config.input_budget,
            "target_tokens": settings.target_tokens,
            "delta_tokens": settings.delta_tokens,
            "measured": {
                "prompt_eval_tokens": sum(
                    meta.get("prompt_eval_count") or 0 for meta in metas
                ),
                "eval_tokens": sum(meta.get("eval_count") or 0 for meta in metas),
            },
            "deadline_per_invocation_s": {"soft": soft_s, "hard": hard_s},
            "speed_claim": "none: no latency or throughput is estimated; elapsed_s in summary.json is the wall time of the last invocation only",
            "per_chunk": per_chunk,
        }

    def write_artifacts(self, summary: dict, soft_s: float, hard_s: float) -> None:
        ordered = [
            self.ready[chunk.id] for chunk in self.plan.chunks if chunk.id in self.ready
        ]
        claims = [claim for item in ordered for claim in item["claims"]]
        pending_items = [
            item for validation in ordered for item in validation["pending"]
        ]
        write_atomic(
            self.out_dir / "claims.jsonl",
            "".join(canonical_json(claim) + "\n" for claim in claims),
        )
        write_atomic(
            self.out_dir / "pending.jsonl",
            "".join(canonical_json(item) + "\n" for item in pending_items),
        )
        write_atomic(
            self.out_dir / "chunk_results.json",
            json.dumps(
                [
                    {
                        "id": chunk.id,
                        "status": "ready"
                        if chunk.id in self.ready
                        else self.chunk_state(chunk),
                        "attempts": [
                            {
                                key: record[key]
                                for key in (
                                    "attempt",
                                    "status",
                                    "called",
                                    "salvaged",
                                    "error",
                                    "counts",
                                    "server",
                                )
                            }
                            for record in self.attempts[chunk.id]
                        ],
                    }
                    for chunk in self.plan.chunks
                ],
                indent=2,
                sort_keys=True,
            ),
        )
        write_atomic(
            self.out_dir / "pending_reconciliation.json",
            json.dumps(self.reconciliation(), indent=2, sort_keys=True),
        )
        write_atomic(
            self.out_dir / "call_budget.json",
            json.dumps(self.call_budget(soft_s, hard_s), indent=2, sort_keys=True),
        )
        write_atomic(
            self.out_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True)
        )

    def reconciliation(self) -> dict:
        subjects = {
            sid: {
                "name": item["name"],
                "aliases": item["aliases"],
                "claim_ids": item["claim_ids"],
                "status": "pending",
            }
            for sid, item in sorted(self.state.subjects.items())
        }
        report = None
        if subjects:
            claims = [Claim(sid, tuple(self.state.names(sid))) for sid in subjects]
            result = reconcile(self.cast, claims)
            report = {
                "digest": result["digest"],
                "groups": result["groups"],
                "unsupported_claim_names": result["unsupported_claim_names"],
                "omitted_candidates": [item["label"] for item in result["omitted"]],
            }
        return {
            "status": "pending",
            "plan_sha256": self.plan.artifact["plan_sha256"],
            "subjects": subjects,
            "cast_index": report,
        }


# ##################################################################
# adaptive output retry
# A new output directory that REFERENCES a finished base-cap (4096) run without ever writing to it. The parent plan must reproduce byte-for-byte from the current source/config/calibration/seed (source hash, chunk ranges, system prompt, schema template, base user hashes, overhead), the parent journal must be an untorn hash chain with exactly one attempt per chunk, and its journal + raw files are copied verbatim into the new directory, where the normal replay re-judges every saved response against its request/response hashes under identical rebuilt state (any dynamic-state or hash mismatch refuses). Only a chunk whose single attempt is raw-proven `done_reason: length` with eval_count exactly the base cap gets exactly ONE retry at RETRY_OUTPUT_TOKENS; a length at the retry cap stays typed pending (`output_length_exhausted`). No partial JSON is ever salvaged and nothing else is ever re-asked.
RETRY_OUTPUT_TOKENS = 8192
RETRY_MARGIN_TOKENS = 16


def same_or_absent(path: Path, data: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != data:
            raise ContractError(
                f"{path.name} in the adaptive output differs from the parent copy",
                "adaptive_parent_mismatch",
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def adopt_parent(
    plan: DeltaPlan, config: ProofConfig, parent: Path, out_dir: Path
) -> DeltaPlan:
    """Validate the parent run directory read-only, copy its journal/raw into `out_dir`, and return the adaptive plan."""
    parent, out_resolved = parent.resolve(), out_dir.resolve()
    if (
        parent == out_resolved
        or parent in out_resolved.parents
        or out_resolved in parent.parents
    ):
        raise ContractError(
            "adaptive output must be a new directory outside the parent",
            "adaptive_output_invalid",
        )
    if not (
        isinstance(config.output_tokens, int)
        and config.output_tokens < RETRY_OUTPUT_TOKENS
        and config.output_tokens + config.reserve_tokens < config.backend.num_ctx // 2
        and RETRY_OUTPUT_TOKENS + config.reserve_tokens < config.backend.num_ctx // 2
    ):
        raise ContractError(
            "retry output cap must exceed the base cap and fit the context",
            "adaptive_output_invalid",
        )
    try:
        parent_plan = json.loads((parent / "plan.json").read_text(encoding="utf-8"))
        journal = (parent / "journal.jsonl").read_bytes()
    except (OSError, ValueError) as error:
        raise ContractError(
            f"parent run directory is unreadable: {error}", "adaptive_parent_invalid"
        ) from error
    if parent_plan != plan.artifact:
        raise ContractError(
            "parent plan differs from the plan rebuilt from the current source, config, calibration and seed",
            "adaptive_parent_mismatch",
        )
    if not journal.endswith(b"\n"):
        raise ContractError(
            "parent journal has a torn tail; it is never repaired in place",
            "adaptive_parent_invalid",
        )
    try:
        records = [json.loads(line) for line in journal.decode("utf-8").splitlines()]
    except ValueError as error:
        raise ContractError(
            "parent journal is not JSON", "adaptive_parent_invalid"
        ) from error
    known_chunk_ids = {chunk.id for chunk in plan.chunks}
    if not records or any(not isinstance(item, dict) for item in records):
        raise ContractError(
            "parent must hold unique first attempts for known chunks only",
            "adaptive_parent_invalid",
        )
    parent_chunk_ids = [item.get("chunk_id") for item in records]
    if (
        any(chunk_id not in known_chunk_ids for chunk_id in parent_chunk_ids)
        or len(set(parent_chunk_ids)) != len(parent_chunk_ids)
        or any(item.get("attempt") != 1 for item in records)
    ):
        raise ContractError(
            "parent must hold unique first attempts for known chunks only",
            "adaptive_parent_invalid",
        )
    copies: list[tuple[Path, bytes]] = []
    for item in records:
        if item["status"] in NO_RESPONSE:
            continue
        base = f"{item['chunk_id']}.a1"
        for suffix in (".response.txt", ".meta.json"):
            try:
                data = (parent / "raw" / (base + suffix)).read_bytes()
            except OSError as error:
                raise ContractError(
                    f"parent raw {base}{suffix} is unreadable",
                    "adaptive_parent_invalid",
                ) from error
            copies.append((out_dir / "raw" / (base + suffix), data))
    mine = out_dir / "journal.jsonl"
    if mine.is_file() and not mine.read_bytes().startswith(journal):
        raise ContractError(
            "adaptive journal does not extend the parent journal",
            "adaptive_parent_mismatch",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if not mine.is_file():
        mine.write_bytes(journal)
    for path, data in copies:
        same_or_absent(path, data)
    policy = {
        "base_output_tokens": config.output_tokens,
        "retry_output_tokens": RETRY_OUTPUT_TOKENS,
        "retry_margin_tokens": RETRY_MARGIN_TOKENS,
        "max_retries_per_chunk": 1,
        "retry_only_if": "single attempt, done_reason length, eval_count == base_output_tokens",
        "parent_plan_sha256": plan.artifact["plan_sha256"],
        "parent_journal_sha256": sha256_text(journal.decode("utf-8")),
        "parent_journal_head": records[-1]["record_sha256"],
        "parent_chunks": {
            item["chunk_id"]: {
                key: item[key]
                for key in (
                    "status",
                    "request_sha256",
                    "response_sha256",
                    "meta_sha256",
                    "server",
                )
            }
            for item in records
        },
    }
    artifact = {
        key: value for key, value in plan.artifact.items() if key != "plan_sha256"
    } | {"adaptive": policy}
    artifact["plan_sha256"] = sha256_text(canonical_json(artifact))
    settings = DeltaSettings(plan.settings.target_tokens, plan.settings.delta_tokens, 2)
    return DeltaPlan(plan.source, plan.chunks, settings, plan.fixed_overhead, artifact)


@contextlib.contextmanager
def run_lock(out_dir: Path) -> Iterator[None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ContractError(
                "another run holds this run directory", "run_locked"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def ensure_plan_file(plan: DeltaPlan, out_dir: Path) -> None:
    path = out_dir / "plan.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8")).get("plan_sha256")
        if existing != plan.artifact["plan_sha256"]:
            raise ContractError(
                "run directory holds a different plan; use a new --out", "plan_mismatch"
            )
        return
    write_atomic(path, json.dumps(plan.artifact, indent=2, sort_keys=True))


# ##################################################################
# only chunk
# a caretaker sampling proof sends exactly one chunk of the FULL source plan. Every other chunk stays `missing`, so the run can never be ready and nothing here can count as coverage. Refusals happen before the output directory is touched.
FRESH_FORBIDDEN = ("journal.jsonl", "raw", "summary.json", "claims.jsonl")


def check_only_chunk(
    chunk_id: str,
    config: ProofConfig,
    plan: DeltaPlan,
    out_dir: Path,
    count: Counter,
    registry: dict | None,
    aliases: dict | None,
) -> None:
    by_id = {chunk.id: chunk for chunk in plan.chunks}
    if not plan.chunks or chunk_id not in by_id:
        raise ContractError(
            f"--only-chunk {chunk_id!r} is not a chunk of this plan ({len(plan.chunks)} chunks)",
            "only_chunk_invalid",
        )
    used = [name for name in FRESH_FORBIDDEN if (out_dir / name).exists()]
    if used:
        raise ContractError(
            f"--only-chunk needs a new output directory; {out_dir} already holds {used}",
            "only_chunk_output_not_new",
        )
    state = Established()
    state.seed(registry or {}, aliases or {})
    try:
        prepared = prepare(config, plan, by_id[chunk_id], state, count)
    except TokenizerRefusal as error:
        raise ContractError(
            f"chunk {chunk_id} cannot be counted: {error}", "only_chunk_incompatible"
        ) from error
    if prepared.padded_tokens > config.input_budget:
        raise ContractError(
            f"chunk {chunk_id} needs {prepared.padded_tokens} tokens, over the input budget {config.input_budget}",
            "only_chunk_incompatible",
        )


# ##################################################################
# run delta
# replays the journal, then asks for each chunk still missing, in order, never re-asking a ready or saved chunk, never more than `max_attempts` attempts per chunk, and never starting a request after the soft deadline. The hard deadline is a real wall-clock cut-off per request. Whatever happens (deadline, transport error, refusal, any exception) `summary.json` is rewritten as `not_ready` with its reason before control leaves. With transport=None nothing is sent (offline validation).
def run_delta(
    chapters: Sequence[Path],
    config: ProofConfig,
    plan: DeltaPlan,
    out_dir: Path,
    transport: Transport | None,
    count: Counter,
    calibration: dict | None = None,
    soft_s: float = SOFT_DEADLINE_S,
    hard_s: float = HARD_DEADLINE_S,
    clock: Callable[[], float] = time.monotonic,
    cast: CastIndex | None = None,
    registry: dict | None = None,
    aliases: dict | None = None,
    only_chunk: str | None = None,
) -> dict:
    check_output_dir(out_dir, chapters)
    if only_chunk is not None:
        check_only_chunk(
            only_chunk, config, plan, out_dir, count, registry, aliases
        )
    if transport is not None:
        measured = validate_calibration(calibration or {}, config)
        if measured["fixed_overhead_tokens"] != plan.fixed_overhead:
            raise ContractError(
                f"plan budgeted {plan.fixed_overhead} overhead tokens but calibration measured {measured['fixed_overhead_tokens']}",
                "calibration_overhead_mismatch",
            )
    if cast is None:
        try:
            cast = build_cast_index(list(chapters), registry or {}, aliases or {}, None)
        except CastDataIssue as error:
            raise ContractError(
                f"cast index unavailable: {error}", "cast_index_unavailable"
            ) from error
    with run_lock(out_dir):
        ensure_plan_file(plan, out_dir)
        run = DeltaRun(chapters, config, plan, out_dir, count, cast, registry, aliases)
        run.only_chunk = only_chunk
        started = clock()
        run.replay()
        stop: str | None = None
        error: str | None = None
        run.write_artifacts(
            run.summary("in_progress", transport is None, 0.0), soft_s, hard_s
        )
        try:
            for chunk in plan.chunks:
                if not run.wants(chunk) or (only_chunk and chunk.id != only_chunk):
                    continue
                stop = run.process(chunk, transport, clock, started, soft_s, hard_s)
                write_atomic(
                    out_dir / "summary.json",
                    json.dumps(
                        run.summary(
                            "in_progress", transport is None, clock() - started
                        ),
                        indent=2,
                        sort_keys=True,
                    ),
                )
                if stop:
                    break
        except BaseException as caught:
            error = f"error:{type(caught).__name__}"
            raise
        finally:
            summary = run.summary(stop, transport is None, clock() - started, error)
            run.write_artifacts(summary, soft_s, hard_s)
        if stop == "fallback_route":
            raise ContractError(
                "server answered with a model other than the configured primary; fallback route rejected",
                "fallback_route",
            )
        return summary


# ##################################################################
# apply to cast
# turns locally accepted claims into cast registry facts, nothing more. Existing actors (the original profiles, voices and anchors included) only gain appended `facts`; their name, bio and look, assets and ids are never rewritten. A novel subject becomes a prepared actor only through its exact label; an alias is recorded only when the source itself bridges it (src.cast_freeze.source_bridge_predicate on the cited sentence) and never takes a name another actor already owns. Anything else is returned as a typed pending item and the caller keeps it blocking the freeze. Idempotent: replaying the same claims changes nothing.
def append_unique(target: list, text: str) -> None:
    if text and text not in target:
        target.append(text)


def apply_to_cast(registry: dict, aliases: dict, claims: Sequence[dict]) -> dict:
    actor_of: dict[str, str] = {}
    applied, held, applied_claims = 0, [], []

    def hold(claim: dict, reason: str) -> None:
        held.append(
            {
                "claim_id": claim["claim_id"],
                "subject": claim["subject"],
                "category": claim["category"],
                "value": claim["value"],
                "pending_reason": reason,
                "chunk_id": claim["chunk_id"],
                "paragraph_id": claim["paragraph_id"],
            }
        )

    for claim in claims:
        sid = claim["subject_id"]
        actor = sid if sid in registry else actor_of.get(sid)
        if actor is None:
            actor = normalized_id(claim["subject"])
            existing = registry.get(actor)
            if not actor or (
                existing is not None
                and norm_space(str(existing.get("name", actor))) != claim["subject"]
            ):
                hold(claim, "actor_id_collision")
                continue
            taken = aliases.get(actor)
            if taken is not None and taken != actor:
                hold(claim, "actor_id_collision")
                continue
            if existing is None:
                registry[actor] = {
                    "name": claim["subject"],
                    "bio": "",
                    "look": "",
                    "origin": "prepared",
                    "facts": {"voice": [], "look": []},
                }
            aliases.setdefault(actor, actor)
            aliases.setdefault(normalized_id(claim["subject"]), actor)
            actor_of[sid] = actor
        facts = registry[actor].setdefault("facts", {"voice": [], "look": []})
        if not isinstance(facts, dict):
            hold(claim, "actor_id_collision")
            continue
        if claim["category"] == "alias":
            alias_key = normalized_id(claim["value"])
            proof = source_bridge_predicate(
                claim["value"],
                actor,
                [{"id": claim["claim_id"], "quote": claim["quote"]}],
                registry,
                aliases,
            )
            if proof is None or not alias_key:
                hold(claim, "alias_unbridged")
                continue
            if aliases.get(alias_key, actor) != actor:
                hold(claim, "alias_collision")
                continue
            aliases[alias_key] = actor
        append_unique(facts.setdefault(claim["category"], []), claim["value"])
        applied += 1
        applied_claims.append(claim)
    return {"applied": applied, "pending": held, "applied_claims": applied_claims}


# ##################################################################
# candidate accounting
# the native contract freezes one disposition for every deterministic candidate of every chapter (src.cast_freeze.candidate_coverage_ledger). A whole-source wide-bio scan only speaks about subjects it extracted facts for, so it may stand in for the legacy scanner only when EVERY candidate of EVERY chapter is accounted for locally, after the claims were applied: a registry/alias owner (an original actor, or one a validated claim created or bridged), a deterministic non-entity, or an exact-label typed pending item the scan retained (which keeps blocking publication). Any other candidate is unaccounted and the result is never complete.
def pending_labels(held: Sequence[dict], rejected: Sequence[dict]) -> set[str]:
    labels = set()
    for item in held:
        fact = item.get("fact")
        who = fact.get("subject") if isinstance(fact, dict) else None
        if isinstance(who, dict) and isinstance(who.get("name"), str):
            labels.add(norm_space(who["name"]))
    labels.update(norm_space(item["subject"]) for item in rejected)
    return labels


def candidate_accounting(
    chapters: Sequence[Path], registry: dict, aliases: dict, held_labels: set[str]
) -> dict:
    counts = {"owner": 0, "nonentity": 0, "pending": 0}
    unaccounted: list[dict] = []
    rows: dict[str, list[dict]] = {}
    for path in chapters:
        rows[path.name] = []
        try:
            ledger = candidate_coverage_ledger(
                immutable_evidence_units([path]), registry, aliases
            )
        except CastDataIssue as error:
            unaccounted.append({"chapter": path.name, "reason": error.code})
            continue
        for candidate in ledger:
            if candidate["known_owner"]:
                disposition, owner = "owner", candidate["known_owner"]
            elif candidate["nonentity"]:
                disposition, owner = "nonentity", "none"
            elif norm_space(candidate["label"]) in held_labels:
                disposition, owner = "pending", "none"
            else:
                unaccounted.append(
                    {
                        "chapter": path.name,
                        "candidate_id": candidate["id"],
                        "label": candidate["label"],
                    }
                )
                continue
            counts[disposition] += 1
            rows[path.name].append(
                {
                    "candidate_id": candidate["id"],
                    "label": candidate["label"],
                    "ref_ids": candidate["ref_ids"],
                    "status": f"wide_bio_{disposition}",
                    "identity": owner,
                }
            )
    return {
        "complete": not unaccounted,
        "candidates": sum(counts.values()) + len(unaccounted),
        **counts,
        "unaccounted_count": len(unaccounted),
        "unaccounted": unaccounted[:25],
        "rows": rows,
    }


# ##################################################################
# reconcile old pending
# closes an existing legacy typed-identity pending row only by exact proof: its recorded chapters still have their recorded hashes (exact_scope), and EVERY literal mention it holds is the very sentence (quote hash) of a claim this run validated and applied for exactly that label inside the very chapter text (hash). One uncovered mention, a stale scope, another row or any other code leaves the row open; nothing is ever closed in bulk.
def reconcile_old_pending(
    recovery, chapters: Sequence[Path], applied_claims: Sequence[dict]
) -> list[dict]:
    by_name = {path.name: path for path in chapters}
    proven = {
        (
            norm_space(claim["subject"]),
            claim["quote_sha256"],
            claim["witness"]["chapter_text_sha256"],
        ): claim["claim_id"]
        for claim in applied_claims
        if isinstance(claim.get("witness"), dict)
    }
    closed = []
    for row in recovery.open_pending("cast"):
        evidence = row["evidence"] if isinstance(row.get("evidence"), dict) else {}
        mentions = evidence.get("mentions")
        if (
            not str(row["code"]).startswith("pending_")
            or str(row["item"]).startswith("wide_bio:")
            or not isinstance(mentions, list)
            or not mentions
            or not exact_scope(row, by_name)
        ):
            continue
        proofs = [
            proven.get(
                (
                    norm_space(str(mention.get("label", ""))),
                    mention.get("quote_sha256"),
                    mention.get("chapter_sha256"),
                )
            )
            for mention in mentions
            if isinstance(mention, dict)
        ]
        if len(proofs) != len(mentions) or not all(proofs):
            continue
        recovery.resolve(
            row, "wide_bio_exact_claim_scope", {"claim_ids": sorted(set(proofs))}
        )
        closed.append({"item": row["item"], "code": row["code"]})
    return closed


# ##################################################################
# prepare-cast hook
# the callable handed to src.cast_freeze.prepare_cast(wide_bio=...). It runs once the structural cast batches are done and before the freeze: it pins a private seed copy of the registry (so a resumed run is judged against the same state), runs or resumes the delta runner, and only when EVERY chunk is ready applies the accepted claims to the preparation registry and records each pending item as a blocking `pending` recovery row. A `not_ready` result (offline, deadline, error, partial coverage) is returned to prepare_cast, which then neither applies anything nor freezes: a bounded or interrupted run is never approval. The model route is the explicit proof config only; with live=False nothing is sent.
def pinned_seed(out_dir: Path, registry: dict, aliases: dict) -> dict:
    """The registry/aliases this run directory is judged against: written once, then always read back, so a resume is never judged against a registry that the run itself changed."""
    path = out_dir / "seed.json"
    if not path.is_file():
        write_atomic(path, canonical_json({"registry": registry, "aliases": aliases}))
    return json.loads(path.read_text(encoding="utf-8"))


def apply_ready(
    out: Path,
    plan: DeltaPlan,
    summary: dict,
    chapters: Sequence[Path],
    progress: dict,
    recovery,
    require_accounting: bool,
) -> dict:
    """Apply a ready run's claims to `progress`, record every typed pending item, close only exactly proven old pending rows and (when required) run candidate accounting."""
    claims = [
        json.loads(line)
        for line in (out / "claims.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    held = [
        json.loads(line)
        for line in (out / "pending.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    outcome = apply_to_cast(progress["registry"], progress["aliases"], claims)
    rows = [
        {
            "item": f"wide_bio:{item['chunk_id']}:{item['pending_id'][:16]}",
            "reason": item["pending_reason"],
            "detail": item,
        }
        for item in held
    ] + [
        {
            "item": f"wide_bio:{item['chunk_id']}:{item['claim_id'][:16]}",
            "reason": item["pending_reason"],
            "detail": item,
        }
        for item in outcome["pending"]
    ]
    for row in rows:
        recovery.record(
            "cast",
            row["item"],
            f"wide_bio_pending_{row['reason']}",
            f"wide-bio fact is pending: {row['reason']}",
            severity="pending",
            evidence={
                "plan_sha256": plan.artifact["plan_sha256"],
                "source_hash": plan.artifact["coverage"]["source_sha256"],
                "fact_sha256": sha256_text(canonical_json(row["detail"])),
                "fact": row["detail"],
            },
        )
    closed = reconcile_old_pending(recovery, chapters, outcome["applied_claims"])
    result = {
        "state": "applied",
        "plan_sha256": plan.artifact["plan_sha256"],
        "claims_applied": outcome["applied"],
        "pending": len(rows),
        "closed_pending": closed,
        "calls_total": summary["calls_total"],
    }
    if not require_accounting:
        progress["wide_bio"] = result
        return result
    accounting = candidate_accounting(
        chapters,
        progress["registry"],
        progress["aliases"],
        pending_labels(held, outcome["pending"]),
    )
    result["accounting"] = {
        key: value for key, value in accounting.items() if key != "rows"
    }
    progress["wide_bio"] = result
    return {**result, "accounting_rows": accounting["rows"]}


@dataclass(frozen=True, slots=True)
class WideBioHook:
    config_path: Path
    out_dir: Path
    settings: DeltaSettings
    calibration_path: Path | None = None
    live: bool = False
    transport: Transport = chat_transport
    soft_s: float = SOFT_DEADLINE_S
    hard_s: float = HARD_DEADLINE_S

    def __call__(
        self,
        project: Path,
        chapters: Sequence[Path],
        progress: dict,
        source_text: str,
        recovery,
        require_accounting: bool = False,
    ) -> dict:
        applied = progress.get("wide_bio")
        # a required run never trusts the cache: apply, reconcile and accounting are idempotent over the same journal
        if (
            not require_accounting
            and isinstance(applied, dict)
            and applied.get("state") == "applied"
        ):
            return applied
        config = load_proof_config(self.config_path)
        out = self.out_dir
        seed = pinned_seed(out, progress["registry"], progress["aliases"])
        calibration, overhead = None, 0
        if self.live:
            try:
                calibration = json.loads(
                    (self.calibration_path or Path("/nonexistent")).read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, ValueError) as error:
                raise ContractError(
                    "calibration record is not readable JSON", "calibration_invalid"
                ) from error
            overhead = validate_calibration(calibration, config)[
                "fixed_overhead_tokens"
            ]
        count = build_counter(config)
        plan = build_delta_plan(
            source_text,
            chapters,
            config,
            count,
            self.settings,
            overhead,
            sha256_text(canonical_json(seed)),
        )
        summary = run_delta(
            chapters,
            config,
            plan,
            out,
            self.transport if self.live else None,
            count,
            calibration,
            self.soft_s,
            self.hard_s,
            registry=seed["registry"],
            aliases=seed["aliases"],
        )
        if summary["state"] != "ready":
            return {
                "state": "not_ready",
                "reason": summary["reason"],
                "source_fraction": summary["source_fraction"],
                "out_dir": str(out),
            }
        return apply_ready(
            out, plan, summary, chapters, progress, recovery, require_accounting
        )


# ##################################################################
# finalize adaptive
# no-provider finalization of a COMPLETED adaptive run. Nothing here can send a request: the verifier replays the whole adaptive output offline (source, seed, plan, journal hash chain and every saved raw response re-judged locally against rebuilt state) and accepts it only when every chunk is ready; a missing, truncated or length-exhausted chunk, an unfinished run, a different source/seed/plan or any tampered file is refused before anything is applied. The hook then applies the verified claims to the CURRENT prepare_cast progress through the same apply_ready path as the live hook (existing apply_to_cast: originals preserved, aliases only source-bridged and never taking another actor's name), records exact typed pending rows, closes only exactly proven old pending rows and runs candidate accounting; prepare_cast never freezes while a gap or pending row remains.
def verify_adaptive_output(
    config: ProofConfig, chapters: Sequence[Path], source_text: str, out: Path
) -> tuple[DeltaPlan, dict, dict]:
    try:
        stored = json.loads((out / "plan.json").read_text(encoding="utf-8"))
        seed_bytes = (out / "seed.json").read_bytes()
        seed = json.loads(seed_bytes)
        exit_code = (out / "run.exit").read_text(encoding="utf-8").strip()
        journal = (out / "journal.jsonl").read_bytes()
    except (OSError, ValueError) as error:
        raise ContractError(
            f"adaptive output is unreadable: {error}", "adaptive_output_incomplete"
        ) from error
    policy = stored.get("adaptive") if isinstance(stored, dict) else None
    if (
        not isinstance(policy, dict)
        or not isinstance(seed, dict)
        or not isinstance(seed.get("registry"), dict)
        or not isinstance(seed.get("aliases"), dict)
    ):
        raise ContractError(
            "output is not an adaptive run directory", "adaptive_output_invalid"
        )
    if exit_code != "0" or not journal.endswith(b"\n"):
        raise ContractError(
            "adaptive run did not finish cleanly (run.exit is not 0 or the journal has a torn tail)",
            "adaptive_output_incomplete",
        )
    try:
        settings = DeltaSettings(stored["target_tokens"], stored["delta_tokens"], 2)
        overhead = stored["fixed_overhead_tokens"]
    except KeyError as error:
        raise ContractError(
            "adaptive plan is missing its settings", "adaptive_output_invalid"
        ) from error
    base = build_delta_plan(
        source_text,
        chapters,
        config,
        build_counter(config),
        DeltaSettings(settings.target_tokens, settings.delta_tokens, 1),
        overhead,
        sha256_text(canonical_json(seed)),
    )
    rest = {key: value for key, value in stored.items() if key != "plan_sha256"}
    if (
        rest.pop("adaptive") != policy
        or {**base.artifact, "plan_sha256": None} != {**rest, "plan_sha256": None}
        or policy.get("parent_plan_sha256") != base.artifact["plan_sha256"]
        or stored.get("plan_sha256")
        != sha256_text(canonical_json({**rest, "adaptive": policy}))
        or policy.get("base_output_tokens") != config.output_tokens
        or policy.get("retry_output_tokens") != RETRY_OUTPUT_TOKENS
        or policy.get("retry_margin_tokens") != RETRY_MARGIN_TOKENS
    ):
        raise ContractError(
            "adaptive plan does not match the current source, config and seed",
            "adaptive_output_mismatch",
        )
    plan = DeltaPlan(base.source, base.chunks, settings, overhead, stored)
    summary = run_delta(
        chapters,
        config,
        plan,
        out,
        None,
        build_counter(config),
        registry=seed["registry"],
        aliases=seed["aliases"],
    )
    statuses = summary["chunk_status"]
    if (
        summary["state"] != "ready"
        or set(statuses) != {"ready"}
        or statuses["ready"] != len(plan.chunks)
        or summary["length_exhausted"]
        or summary["calls_this_invocation"]
        or summary["source_fraction"] != 1.0
    ):
        raise ContractError(
            f"adaptive output is not complete: {summary['reason']} {statuses}",
            "adaptive_output_incomplete",
        )
    return plan, summary, seed


@dataclass(frozen=True, slots=True)
class AdaptiveFinalizeHook:
    config_path: Path
    out_dir: Path

    def __call__(
        self,
        project: Path,
        chapters: Sequence[Path],
        progress: dict,
        source_text: str,
        recovery,
        require_accounting: bool = False,
    ) -> dict:
        plan, summary, _seed = verify_adaptive_output(
            load_proof_config(self.config_path), chapters, source_text, self.out_dir
        )
        return apply_ready(
            self.out_dir,
            plan,
            summary,
            chapters,
            progress,
            recovery,
            require_accounting,
        )


def no_ask(*_args, **_kwargs):
    raise ContractError("finalize-adaptive never calls a provider", "no_provider")


def finalize_adaptive_command(args) -> int:
    from src.cast_freeze import prepare_cast
    from src.epub_extract import get_output_dir

    config = load_proof_config(args.config)
    project = args.project.resolve()
    if get_output_dir(args.source.resolve()).resolve() != project:
        raise ContractError(
            "project is not the output directory of this source", "project_mismatch"
        )
    chapters = project_chapters(project)
    # refuse an unverifiable output before prepare_cast touches any progress
    verify_adaptive_output(config, chapters, read_source(args.source), args.out)
    result = prepare_cast(
        args.source,
        ask=no_ask,
        wide_bio=AdaptiveFinalizeHook(args.config, args.out),
        wide_bio_required=True,
    )
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result.get("status") == "frozen" else 3


def adaptive_command(args, settings: DeltaSettings, transport: Transport) -> int:
    config = load_proof_config(args.config)
    project = args.project.resolve()
    chapters = project_chapters(project)
    check_output_dir(args.out, chapters)
    try:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ContractError(
            "calibration record is not readable JSON", "calibration_invalid"
        ) from error
    overhead = validate_calibration(calibration, config)["fixed_overhead_tokens"]
    try:
        seed_bytes = (args.from_out / "seed.json").read_bytes()
    except OSError as error:
        raise ContractError(
            "parent seed.json is unreadable", "adaptive_parent_invalid"
        ) from error
    args.out.mkdir(parents=True, exist_ok=True)
    same_or_absent(args.out / "seed.json", seed_bytes)
    seed = json.loads(seed_bytes)
    count = build_counter(config)
    base = build_delta_plan(
        read_source(args.source),
        chapters,
        config,
        count,
        settings,
        overhead,
        sha256_text(canonical_json(seed)),
    )
    plan = adopt_parent(base, config, args.from_out, args.out)
    exit_file = args.out / "run.exit"
    if args.execute:
        write_atomic(args.out / "run.pid", str(os.getpid()))
    try:
        summary = run_delta(
            chapters,
            config,
            plan,
            args.out,
            transport if args.execute else None,
            count,
            calibration,
            args.soft_deadline_s,
            args.hard_deadline_s,
            registry=seed["registry"],
            aliases=seed["aliases"],
        )
    except BaseException:
        write_atomic(exit_file, "2")
        raise
    code = 0 if summary["state"] == "ready" else 3
    write_atomic(exit_file, str(code))
    print(json.dumps(summary, sort_keys=True))
    return code


# ##################################################################
# main
# plan/validate are offline; run needs --execute and a calibration record. Exit 0 = ready, 3 = not_ready (durable summary written), 2 = refused before any request. `transport` is injectable only for tests; the default is the single-route chat transport.
def main(argv: list[str] | None = None, transport: Transport = chat_transport) -> int:
    parser = argparse.ArgumentParser(
        description="Resumable delta runner for wide-context biography extraction (offline unless `--execute`)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "validate", "prepare-cast"):
        item = sub.add_parser(name)
        item.add_argument(
            "project" if name != "prepare-cast" else "source",
            type=Path,
            help="book project directory"
            if name != "prepare-cast"
            else "the original book source given to ./run prepare-cast",
        )
        if name != "prepare-cast":
            item.add_argument(
                "--source",
                type=Path,
                required=True,
                help="authoritative original UTF-8 text",
            )
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--out", type=Path, required=True)
        item.add_argument(
            "--target-tokens",
            type=int,
            required=True,
            help="source-paragraph tokens per chunk, e.g. 8000 or 32000",
        )
        item.add_argument(
            "--delta-tokens",
            type=int,
            default=1024,
            help="cap on the already-established section",
        )
        item.add_argument(
            "--max-attempts",
            type=int,
            default=1,
            help="attempts per chunk; bounds model calls",
        )
        item.add_argument("--soft-deadline-s", type=float, default=SOFT_DEADLINE_S)
        item.add_argument("--hard-deadline-s", type=float, default=HARD_DEADLINE_S)
        item.add_argument(
            "--calibration", type=Path, default=None, required=name == "run"
        )
        item.add_argument(
            "--sampling-temperature",
            type=float,
            default=None,
            help="explicit request temperature (> 0), part of the plan fingerprint; needs --sampling-seed and an openai-style primary",
        )
        item.add_argument(
            "--sampling-seed",
            type=int,
            default=None,
            help="explicit request seed, part of the plan fingerprint; needs --sampling-temperature",
        )
    sub.choices["run"].add_argument(
        "--execute", action="store_true", help="required: sends requests to the primary"
    )
    sub.choices["run"].add_argument(
        "--only-chunk",
        default=None,
        help="caretaker sampling proof: send only this chunk id (e.g. k0003) of the full plan into a NEW --out; every other chunk stays missing, so the run is never ready",
    )
    ad = sub.add_parser(
        "adaptive",
        help="new output dir that reuses a finished base-cap run read-only and retries only raw-proven length chunks once at the larger cap",
    )
    ad.add_argument("project", type=Path)
    ad.add_argument("--source", type=Path, required=True)
    ad.add_argument("--config", type=Path, required=True)
    ad.add_argument(
        "--from-out",
        type=Path,
        required=True,
        help="finished base-cap run directory (never written)",
    )
    ad.add_argument(
        "--out", type=Path, required=True, help="NEW adaptive output directory"
    )
    ad.add_argument("--target-tokens", type=int, required=True)
    ad.add_argument("--delta-tokens", type=int, default=1024)
    ad.add_argument("--calibration", type=Path, required=True)
    ad.add_argument("--soft-deadline-s", type=float, default=SOFT_DEADLINE_S)
    ad.add_argument("--hard-deadline-s", type=float, default=HARD_DEADLINE_S)
    ad.add_argument(
        "--execute",
        action="store_true",
        help="send the retry calls; without it nothing is sent",
    )
    fin = sub.add_parser(
        "finalize-adaptive",
        help="no-provider: verify a COMPLETED adaptive output and apply it to the current prepare-cast progress (never freezes while a gap or pending row remains)",
    )
    fin.add_argument("project", type=Path)
    fin.add_argument("--source", type=Path, required=True)
    fin.add_argument("--config", type=Path, required=True)
    fin.add_argument(
        "--out", type=Path, required=True, help="completed adaptive output"
    )
    pc = sub.choices["prepare-cast"]
    pc.add_argument(
        "--execute",
        action="store_true",
        help="send the wide-bio requests (the structural cast batches follow their own configuration)",
    )
    pc.add_argument("--max-batches", type=int, default=None)
    pc.add_argument(
        "--wide-bio-required",
        action="store_true",
        help="run the whole-source scan BEFORE the legacy scanner and never call the scanner: complete only when every native candidate is accounted, otherwise blocked",
    )
    args = parser.parse_args(argv)
    exit_file = None
    try:
        if args.command == "finalize-adaptive":
            return finalize_adaptive_command(args)
        settings = DeltaSettings(
            args.target_tokens,
            args.delta_tokens,
            getattr(args, "max_attempts", 1),
            getattr(args, "sampling_temperature", None),
            getattr(args, "sampling_seed", None),
        )
        if args.command == "adaptive":
            return adaptive_command(args, settings, transport)
        if args.command == "prepare-cast":
            from src.cast_freeze import prepare_cast

            if args.execute and args.calibration is None:
                raise ContractError(
                    "--execute needs --calibration", "calibration_invalid"
                )
            hook = WideBioHook(
                args.config,
                args.out,
                settings,
                args.calibration,
                args.execute,
                transport,
                args.soft_deadline_s,
                args.hard_deadline_s,
            )
            result = prepare_cast(
                args.source,
                max_batches=args.max_batches,
                wide_bio=hook,
                wide_bio_required=args.wide_bio_required,
            )
            print(json.dumps(result, sort_keys=True, default=str))
            return 0 if result.get("status") == "frozen" else 3
        config = load_proof_config(args.config)
        chapters = project_chapters(args.project.resolve())
        check_output_dir(args.out, chapters)
        if args.command == "run" and not args.execute:
            raise ContractError("run requires --execute", "execute_required")
        calibration, overhead = None, 0
        if args.calibration is not None:
            try:
                calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ContractError(
                    "calibration record is not readable JSON", "calibration_invalid"
                ) from error
            overhead = validate_calibration(calibration, config)[
                "fixed_overhead_tokens"
            ]
        _, registry, aliases, _ = project_inputs(args.project.resolve())
        seed = pinned_seed(args.out, registry, aliases)
        count = build_counter(config)
        plan = build_delta_plan(
            read_source(args.source),
            chapters,
            config,
            count,
            settings,
            overhead,
            sha256_text(canonical_json(seed)),
        )
        if args.command == "plan":
            with run_lock(args.out):
                ensure_plan_file(plan, args.out)
                cast = build_cast_index(
                    list(chapters), seed["registry"], seed["aliases"], None
                )
                run = DeltaRun(
                    chapters,
                    config,
                    plan,
                    args.out,
                    count,
                    cast,
                    seed["registry"],
                    seed["aliases"],
                )
                run.replay()
                write_atomic(
                    args.out / "call_budget.json",
                    json.dumps(
                        run.call_budget(args.soft_deadline_s, args.hard_deadline_s),
                        indent=2,
                        sort_keys=True,
                    ),
                )
            print(
                json.dumps(
                    {
                        key: plan.artifact[key]
                        for key in (
                            "plan_sha256",
                            "input_budget",
                            "fixed_overhead_tokens",
                            "coverage",
                        )
                    }
                    | {
                        "chunks": len(plan.chunks),
                        "max_model_calls": len(plan.chunks) * settings.max_attempts,
                    },
                    sort_keys=True,
                )
            )
            return 0
        live = args.command == "run"
        if live:
            exit_file = args.out / "run.exit"
            write_atomic(args.out / "run.pid", str(os.getpid()))
        summary = run_delta(
            chapters,
            config,
            plan,
            args.out,
            transport if live else None,
            count,
            calibration,
            args.soft_deadline_s,
            args.hard_deadline_s,
            registry=seed["registry"],
            aliases=seed["aliases"],
            only_chunk=getattr(args, "only_chunk", None),
        )
        code = 0 if summary["state"] == "ready" else 3
        if exit_file is not None:
            write_atomic(exit_file, str(code))
        print(json.dumps(summary, sort_keys=True))
        return code
    except (ContractError, TokenizerRefusal, CastDataIssue) as error:
        print(
            json.dumps(
                {
                    "refused": getattr(error, "code", type(error).__name__),
                    "message": str(error),
                }
            ),
            file=sys.stderr,
        )
        if exit_file is not None:
            write_atomic(exit_file, "2")
        return 2


if __name__ == "__main__":
    sys.exit(main())
