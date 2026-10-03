"""Wide-context full-book biography extraction contract: isolated proof config, exact token preflight, paragraph chunks, strict fact schema, local citation validation, and an append-only evidence store.

Additive and read-only toward the book project: it reads chapter files, never writes inside a project, never imports the
production router state, and never starts inference by itself. `plan` and `validate` are offline. `run` is the only
command that can contact a model, and only with --execute, a pinned exact tokenizer, and a valid prompt_eval_count
calibration record, and only against the single primary backend of the explicit proof config.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import tomllib

from src.cast_freeze import CastDataIssue
from src.cast_index import PARAGRAPH, Claim, build_cast_index, reconcile
from src.hour_runner import chapter_order
from src.llm import Backend, request_for
from src.wide_bio_tokenizer import (
    TokenizerRefusal,
    build_capture,
    capture_digest,
    load_exact_tokenizer,
)

SOFT_DEADLINE_S = 175
HARD_DEADLINE_S = 240
FALLBACK_MODEL = re.compile(r"(?i)(^|[:\-_/])8b\b")
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})

CONTRACT_VERSION = 2
PROOF_NUM_CTX = 262144
CANONICAL_CONFIG = (
    Path(__file__).resolve().parent.parent / "local" / "config.toml"
).resolve()
SECRET_KEY = re.compile(
    r"(?i)(api_?key|secret|password|passwd|credential|auth|bearer|_key$|^key$|_token$|^token$)"
)
CATEGORIES = (
    "appearance",
    "age",
    "gender",
    "species",
    "powers",
    "profession",
    "kinship",
    "alias",
    "personality",
    "voice",
)
FACT_FIELDS = ("subject", "category", "value", "quote", "paragraph_id")
CALIBRATION_MIN_SAMPLES = 3
CALIBRATION_MIN_LARGE_TOKENS = 10_000
# the provider's chat template / BOS adds a constant number of tokens to every prompt (measured +16: 236→252, 3030→3046, 10122→10138); anything above this cap is drift, not template overhead.
MAX_FIXED_OVERHEAD_TOKENS = 64
SYSTEM_PROMPT = (
    "You extract character biography facts from a book excerpt. Return only JSON matching the schema. "
    "Every fact must be literally supported: `quote` is an exact, unmodified, contiguous substring of the paragraph "
    "named by `paragraph_id`, and `paragraph_id` is one of the [[P id]] markers. Facts are limited to the categories "
    f"{', '.join(CATEGORIES)}. `subject` is the character's name as written. Do not infer, merge identities, or invent; "
    "if nothing qualifies return an empty facts list."
)
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts"],
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(FACT_FIELDS),
                "properties": {
                    "subject": {"type": "string", "minLength": 1},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "value": {"type": "string", "minLength": 1},
                    "quote": {"type": "string", "minLength": 1},
                    "paragraph_id": {"type": "string", "minLength": 1},
                },
            },
        }
    },
}


# ##################################################################
# per-chunk response schema
# SCHEMA is only the template: each request is sent with `paragraph_id` constrained to the exact IDs of that chunk's shown paragraphs, so the server cannot emit an unshown or marker-shaped ID such as `P 000001`. A chunk with no shown paragraph can only return no facts. Local quote/witness validation still runs on every response.
def chunk_schema(chunk: Chunk) -> dict:
    schema = json.loads(json.dumps(SCHEMA))
    facts = schema["properties"]["facts"]
    ids = [paragraph.id for paragraph in chunk.paragraphs]
    if ids:
        facts["items"]["properties"]["paragraph_id"] = {"type": "string", "enum": ids}
    else:
        facts.pop("items")
        facts["maxItems"] = 0
    return schema


class ContractError(Exception):
    """A proof precondition failed; nothing was sent anywhere."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ##################################################################
# proof config
# everything a proof run may use, read from one explicitly named non-secret file; the canonical pipeline config (32k) is never consulted or modified.
@dataclass(frozen=True, slots=True)
class ProofConfig:
    path: Path
    backend: Backend
    tokenizer_capture: Path
    tokenizer_sha256: str
    output_tokens: int
    reserve_tokens: int
    tolerance_percent: float
    chunk_chars: int

    @property
    def input_budget(self) -> int:
        return self.backend.num_ctx - self.output_tokens - self.reserve_tokens


# ##################################################################
# load proof config
# explicit path only; refuses the canonical config, any secret-looking key, any primary context other than 262144, and incomplete wide_bio settings. Backup keys are never read: a proof has exactly one route.
def load_proof_config(path: Path) -> ProofConfig:
    if not isinstance(path, Path) or not path.is_file():
        raise ContractError(
            "proof config must be an explicit existing file", "config_missing"
        )
    if path.resolve() == CANONICAL_CONFIG:
        raise ContractError(
            "the canonical pipeline config cannot be used as a proof config",
            "config_is_canonical",
        )
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    llm, wide = data.get("llm", {}), data.get("wide_bio", {})
    for table in (llm, wide):
        for key in table:
            if SECRET_KEY.search(key) and key not in {
                "tokenizer_sha256",
                "tokenizer_capture",
            }:
                raise ContractError(
                    f"proof config key {key!r} looks like a secret", "config_has_secret"
                )
    url = str(llm.get("primary_url", "")).rstrip("/")
    model = str(llm.get("primary_model", ""))
    style = str(llm.get("primary_style", "ollama")).lower()
    if not url.startswith(("http://", "https://")) or not model or style != "ollama":
        raise ContractError(
            "proof config needs an ollama primary_url and primary_model",
            "config_invalid",
        )
    if (
        (urlparse(url).hostname or "") in LOOPBACK
        or FALLBACK_MODEL.search(model)
        or llm.get("primary_ping_host") in LOOPBACK
    ):
        raise ContractError(
            "proof config names the local fallback route (loopback host or 8B model)",
            "config_is_fallback",
        )
    if llm.get("primary_num_ctx") != PROOF_NUM_CTX:
        raise ContractError(
            f"primary_num_ctx must be exactly {PROOF_NUM_CTX}", "config_wrong_context"
        )
    needed = (
        "tokenizer_capture",
        "tokenizer_sha256",
        "output_tokens",
        "reserve_tokens",
        "tolerance_percent",
        "chunk_chars",
    )
    missing = [key for key in needed if key not in wide]
    if missing:
        raise ContractError(f"[wide_bio] is missing {missing}", "config_invalid")
    output_tokens, reserve, tolerance, chunk_chars = (
        wide["output_tokens"],
        wide["reserve_tokens"],
        wide["tolerance_percent"],
        wide["chunk_chars"],
    )
    if (
        not (
            isinstance(output_tokens, int)
            and isinstance(reserve, int)
            and isinstance(chunk_chars, int)
        )
        or min(output_tokens, reserve, chunk_chars) < 1
    ):
        raise ContractError(
            "output_tokens, reserve_tokens and chunk_chars must be positive integers",
            "config_invalid",
        )
    if not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 10:
        raise ContractError("tolerance_percent must be within 0..10", "config_invalid")
    if output_tokens + reserve >= PROOF_NUM_CTX // 2:
        raise ContractError(
            "output plus reserve may not take half of the context", "config_invalid"
        )
    capture = Path(str(wide["tokenizer_capture"]))
    capture = capture if capture.is_absolute() else (path.parent / capture)
    backend = Backend(
        url,
        str(llm.get("primary_ping_host", "")),
        model,
        "ollama",
        PROOF_NUM_CTX,
        bool(llm.get("primary_think", False)),
    )
    return ProofConfig(
        path.resolve(),
        backend,
        capture,
        str(wide["tokenizer_sha256"]),
        output_tokens,
        reserve,
        float(tolerance),
        chunk_chars,
    )


Counter = Callable[[str], int]


# ##################################################################
# build counter
# the pluggable exact tokenizer verifier: returns a text -> token count callable, or raises TokenizerRefusal (fail closed). Any alternative verifier only has to honour that same contract.
def build_counter(config: ProofConfig) -> Counter:
    return load_exact_tokenizer(
        config.tokenizer_capture, config.tokenizer_sha256, config.backend.model
    ).count


# ##################################################################
# source paragraphs
# the authoritative original UTF-8 text is cut into paragraphs at their exact offsets; each shown paragraph keeps its absolute [start, end) in that text, its hash, and a witness: the chapter file (hash and offset) that contains the identical bytes, or None when no chapter reproduces it (such a paragraph is still sent, but its facts stay pending).
@dataclass(frozen=True, slots=True)
class Paragraph:
    id: str
    start: int
    end: int
    text: str
    sha256: str
    witness: dict | None


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    start: int
    end: int
    paragraphs: tuple[Paragraph, ...]


def read_source(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("source is not valid UTF-8", "source_not_utf8") from error


# ##################################################################
# chapter witnesses
# forward-only search of each paragraph in the chapter files (original order, a short lookahead), so every mapped paragraph names the chapter, that chapter's text hash and the offset inside it.
LOOKAHEAD = 3


def witness_map(
    paragraphs: list[tuple[int, int, str]], chapters: Sequence[Path]
) -> list[dict | None]:
    texts = [path.read_text(encoding="utf-8") for path in chapters]
    hashes = [sha256_text(text) for text in texts]
    found: list[dict | None] = []
    chapter, cursor = 0, 0
    for _, _, text in paragraphs:
        witness = None
        for candidate in range(chapter, min(chapter + 1 + LOOKAHEAD, len(texts))):
            at = texts[candidate].find(text, cursor if candidate == chapter else 0)
            if at >= 0:
                witness = {
                    "chapter": chapters[candidate].name,
                    "chapter_text_sha256": hashes[candidate],
                    "chapter_offset": at,
                }
                chapter, cursor = candidate, at + len(text)
                break
        found.append(witness)
    return found


# ##################################################################
# pack source
# greedy paragraph-boundary chunks over the whole original text; the chunk slices [start, end) tile it from 0 to len(source) with no gap and no overlap, which is re-verified by hashing their concatenation. A paragraph larger than chunk_chars refuses the plan.
def pack_source(
    source: str, chunk_chars: int, chapters: Sequence[Path]
) -> tuple[list[Chunk], dict]:
    spans = [
        (m.start(), m.end()) for m in PARAGRAPH.finditer(source) if m.end() > m.start()
    ]
    shown = [(start, end, source[start:end].strip()) for start, end in spans]
    mapped = witness_map([item for item in shown if item[2]], chapters)
    witnesses = iter(mapped)
    cuts: list[list] = []
    size = 0
    for ordinal, (start, end) in enumerate(spans):
        if end - start > chunk_chars:
            raise ContractError(
                f"paragraph at offset {start} exceeds chunk_chars", "paragraph_oversize"
            )
        if not cuts or size + (end - start) > chunk_chars:
            cuts.append([start, end, []])
            size = 0
        cuts[-1][1] = end
        size += end - start
        text = shown[ordinal][2]
        if text:
            lead = len(source[start:end]) - len(source[start:end].lstrip())
            cuts[-1][2].append(
                Paragraph(
                    f"{ordinal:06d}",
                    start + lead,
                    start + lead + len(text),
                    text,
                    sha256_text(text),
                    next(witnesses),
                )
            )
    chunks = [
        Chunk(f"k{number:04d}", start, end, tuple(paragraphs))
        for number, (start, end, paragraphs) in enumerate(cuts)
    ]
    covered = "".join(source[chunk.start : chunk.end] for chunk in chunks)
    exact = (
        covered == source
        and all(a.end == b.start for a, b in itertools.pairwise(chunks))
        and (not chunks or chunks[0].start == 0)
    )
    if not exact:
        raise ContractError(
            "chunks do not cover the source exactly once", "source_coverage"
        )
    paragraphs = [paragraph for chunk in chunks for paragraph in chunk.paragraphs]
    if paragraphs and not any(paragraph.witness for paragraph in paragraphs):
        raise ContractError(
            "no source paragraph maps to any chapter file", "source_unmapped"
        )
    return chunks, {
        "source_chars": len(source),
        "source_bytes": len(source.encode("utf-8")),
        "source_sha256": sha256_text(source),
        "chunks": len(chunks),
        "shown_paragraphs": len(paragraphs),
        "mapped_paragraphs": sum(1 for paragraph in paragraphs if paragraph.witness),
        "unmapped_paragraphs": sum(
            1 for paragraph in paragraphs if not paragraph.witness
        ),
        "exactly_once": True,
    }


def render_user(chunk: Chunk) -> str:
    body = "\n\n".join(
        f"[[P {paragraph.id}]]\n{paragraph.text}" for paragraph in chunk.paragraphs
    )
    return f"Excerpt {chunk.id}. Extract biography facts.\n\n{body}\n"


# ##################################################################
# extraction plan
# the whole original text as ordered requests with exact token counts; fails closed if any prompt exceeds the input budget (no truncation, ever).
@dataclass(frozen=True, slots=True)
class PlannedChunk:
    chunk: Chunk
    user: str
    input_tokens: int
    padded_tokens: int


@dataclass(frozen=True, slots=True)
class ExtractionPlan:
    source: str
    chunks: tuple[PlannedChunk, ...]
    artifact: dict


def padded(tokens: int, tolerance_percent: float) -> int:
    return math.ceil(tokens * (1 + tolerance_percent / 100))


def build_plan(
    source: str,
    chapters: Sequence[Path],
    config: ProofConfig,
    count: Counter,
    fixed_overhead: int = 0,
) -> ExtractionPlan:
    if (
        not isinstance(fixed_overhead, int)
        or isinstance(fixed_overhead, bool)
        or not 0 <= fixed_overhead <= MAX_FIXED_OVERHEAD_TOKENS
    ):
        raise ContractError(
            f"fixed overhead must be an integer in 0..{MAX_FIXED_OVERHEAD_TOKENS}",
            "overhead_invalid",
        )
    chunks, coverage = pack_source(source, config.chunk_chars, chapters)
    system_tokens = count(SYSTEM_PROMPT)
    planned = []
    for chunk in chunks:
        user = render_user(chunk)
        tokens = system_tokens + count(user)
        planned.append(
            PlannedChunk(
                chunk,
                user,
                tokens,
                padded(tokens + fixed_overhead, config.tolerance_percent),
            )
        )
    over = [
        item.chunk.id for item in planned if item.padded_tokens > config.input_budget
    ]
    if over:
        raise ContractError(
            f"chunks {over} exceed the input budget of {config.input_budget} tokens",
            "input_budget_exceeded",
        )
    artifact = {
        "contract_version": CONTRACT_VERSION,
        "model": config.backend.model,
        "num_ctx": config.backend.num_ctx,
        "output_tokens": config.output_tokens,
        "reserve_tokens": config.reserve_tokens,
        "tolerance_percent": config.tolerance_percent,
        "fixed_overhead_tokens": fixed_overhead,
        "input_budget": config.input_budget,
        "tokenizer_capture_sha256": config.tokenizer_sha256,
        "schema_template_sha256": sha256_text(canonical_json(SCHEMA)),
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "calibration_required": True,
        "coverage": coverage,
        "chunks": [
            {
                "id": item.chunk.id,
                "start": item.chunk.start,
                "end": item.chunk.end,
                "paragraphs": len(item.chunk.paragraphs),
                "input_tokens": item.input_tokens,
                "padded_tokens": item.padded_tokens,
                "user_sha256": sha256_text(item.user),
                "paragraph_ids": [paragraph.id for paragraph in item.chunk.paragraphs],
                "schema_sha256": sha256_text(canonical_json(chunk_schema(item.chunk))),
            }
            for item in planned
        ],
    }
    artifact["plan_sha256"] = sha256_text(canonical_json(artifact))
    return ExtractionPlan(source, tuple(planned), artifact)


# ##################################################################
# calibration
# a measured record pairing local counts with the server's prompt_eval_count for real prompts. The server count must equal the local count plus ONE constant, non-negative, bounded chat-template/BOS overhead (returned as fixed_overhead_tokens and budgeted explicitly by the plan); a varying difference is drift and is refused. Non-zero overhead must be shown across at least two prompt sizes, the samples include a large prompt, and the record names this model and tokenizer.
def validate_calibration(record: dict, config: ProofConfig) -> dict:
    if (
        not isinstance(record, dict)
        or record.get("model") != config.backend.model
        or record.get("tokenizer_capture_sha256") != config.tokenizer_sha256
    ):
        raise ContractError(
            "calibration is for another model or tokenizer", "calibration_mismatch"
        )
    samples = record.get("samples")
    if not isinstance(samples, list) or len(samples) < CALIBRATION_MIN_SAMPLES:
        raise ContractError(
            f"calibration needs at least {CALIBRATION_MIN_SAMPLES} samples",
            "calibration_insufficient",
        )
    deltas = set()
    for sample in samples:
        local, server = (
            sample.get("local_tokens") if isinstance(sample, dict) else None,
            sample.get("prompt_eval_count") if isinstance(sample, dict) else None,
        )
        if (
            not (isinstance(local, int) and isinstance(server, int))
            or isinstance(local, bool)
            or isinstance(server, bool)
            or local < 1
            or server < 1
        ):
            raise ContractError(
                "calibration samples need positive integer local_tokens and prompt_eval_count",
                "calibration_invalid",
            )
        deltas.add(server - local)
    if len(deltas) != 1:
        raise ContractError(
            f"server-minus-local differs between samples ({sorted(deltas)}); not a constant template overhead",
            "calibration_drift",
        )
    overhead = deltas.pop()
    if overhead < 0:
        raise ContractError(
            f"server counts fewer tokens than local ({overhead}); overhead cannot be negative",
            "calibration_overhead_negative",
        )
    if overhead > MAX_FIXED_OVERHEAD_TOKENS:
        raise ContractError(
            f"fixed overhead {overhead} exceeds the {MAX_FIXED_OVERHEAD_TOKENS}-token cap; this is drift, not template overhead",
            "calibration_drift",
        )
    if overhead and len({sample["local_tokens"] for sample in samples}) < 2:
        raise ContractError(
            "a non-zero overhead needs samples of at least two different prompt sizes to prove it constant",
            "calibration_insufficient",
        )
    if max(sample["local_tokens"] for sample in samples) < CALIBRATION_MIN_LARGE_TOKENS:
        raise ContractError(
            "calibration lacks a large-prompt sample", "calibration_insufficient"
        )
    return {"samples": len(samples), "fixed_overhead_tokens": overhead}


# ##################################################################
# validate response
# strict local verification of one raw response against the paragraphs shown. Accepted claims carry the absolute source offset and chapter witness; facts on a paragraph with no chapter witness are `pending` (typed), never accepted; every rejection keeps its reason.
def validate_response(raw: str, chunk: Chunk) -> dict:
    by_id = {paragraph.id: paragraph for paragraph in chunk.paragraphs}
    result = {
        "chunk_id": chunk.id,
        "raw_sha256": sha256_text(raw),
        "status": "ok",
        "claims": [],
        "pending": [],
        "rejected": [],
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
    seen: set[str] = set()
    for position, fact in enumerate(facts):
        reason = fact_problem(fact, by_id)
        if reason:
            result["rejected"].append(
                {
                    "position": position,
                    "reason": reason,
                    "fact": fact if isinstance(fact, dict) else None,
                }
            )
            continue
        paragraph = by_id[fact["paragraph_id"]]
        offset = paragraph.text.find(fact["quote"])
        claim = {
            "chunk_id": chunk.id,
            "subject": fact["subject"].strip(),
            "category": fact["category"],
            "value": fact["value"].strip(),
            "quote": fact["quote"],
            "quote_sha256": sha256_text(fact["quote"]),
            "quote_occurrences": paragraph.text.count(fact["quote"]),
            "paragraph_id": paragraph.id,
            "paragraph_sha256": paragraph.sha256,
            "source_offset": paragraph.start + offset,
            "witness": paragraph.witness,
        }
        claim["claim_id"] = sha256_text(
            canonical_json(
                {
                    key: claim[key]
                    for key in (
                        "subject",
                        "category",
                        "value",
                        "quote_sha256",
                        "paragraph_sha256",
                        "source_offset",
                    )
                }
            )
        )
        if claim["claim_id"] in seen:
            result["rejected"].append(
                {"position": position, "reason": "duplicate_fact", "fact": fact}
            )
            continue
        seen.add(claim["claim_id"])
        if paragraph.witness is None:
            result["pending"].append({**claim, "pending_reason": "no_chapter_witness"})
        else:
            result["claims"].append(claim)
    if result["rejected"] or result["pending"]:
        result["status"] = "partial"
    return result


def fact_problem(fact, by_id: dict[str, Paragraph]) -> str | None:
    if not isinstance(fact, dict) or set(fact) != set(FACT_FIELDS):
        return "wrong_fields"
    if not all(
        isinstance(fact[field], str) and fact[field].strip() for field in FACT_FIELDS
    ):
        return "empty_or_non_text_field"
    if fact["category"] not in CATEGORIES:
        return "unknown_category"
    paragraph = by_id.get(fact["paragraph_id"])
    if paragraph is None:
        return "unknown_paragraph_id"
    if fact["quote"] not in paragraph.text:
        return "quote_not_in_paragraph"
    return None


# ##################################################################
# pending reconciliation
# accepted claims grouped by exact subject and checked against the exact-label cast index of the chapter files with an empty registry; every entry stays `pending`, nothing is merged into any cast or registry. An index that cannot be built is reported typed, not hidden.
def pending_reconciliation(
    chapters: Sequence[Path],
    claims: Sequence[dict],
    pending: Sequence[dict],
    plan_sha256: str,
) -> dict:
    subjects: dict[str, dict] = {}
    for claim in claims:
        entry = subjects.setdefault(
            claim["subject"], {"claim_ids": [], "categories": {}, "aliases": []}
        )
        entry["claim_ids"].append(claim["claim_id"])
        entry["categories"][claim["category"]] = (
            entry["categories"].get(claim["category"], 0) + 1
        )
        if claim["category"] == "alias":
            entry["aliases"].append(claim["value"])
    cast_claims = [
        Claim(f"subject:{name}", tuple(dict.fromkeys([name, *entry["aliases"]])))
        for name, entry in sorted(subjects.items())
    ]
    cast_index = None
    if cast_claims:
        try:
            report = reconcile(
                build_cast_index(list(chapters), {}, {}, None), cast_claims
            )
            cast_index = {
                "digest": report["digest"],
                "groups": report["groups"],
                "unsupported_claim_names": report["unsupported_claim_names"],
            }
        except CastDataIssue as error:
            cast_index = {
                "not_ready": error.code
                if hasattr(error, "code")
                else "cast_data_issue",
                "message": str(error),
            }
    return {
        "status": "pending",
        "plan_sha256": plan_sha256,
        "subjects": {
            name: {**entry, "status": "pending"}
            for name, entry in sorted(subjects.items())
        },
        "pending_claims": list(pending),
        "cast_index": cast_index,
    }


# ##################################################################
# store
# atomic evidence for one run directory that must live outside the book project; raw responses are kept verbatim and never overwritten.
def write_atomic(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(data, encoding="utf-8")
    temp.replace(path)


def check_output_dir(out_dir: Path, chapters: Sequence[Path]) -> None:
    resolved = out_dir.resolve()
    for chapter in chapters:
        project = chapter.resolve().parent.parent
        if resolved == project or project in resolved.parents:
            raise ContractError(
                "output directory may not be inside the book project",
                "output_inside_project",
            )


def request_record(config: ProofConfig, item: PlannedChunk) -> tuple[str, bytes]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item.user},
    ]
    return request_for(
        config.backend,
        messages,
        0.0,
        config.output_tokens,
        chunk_schema(item.chunk),
    )


Transport = Callable[[str, bytes, float], dict]


# ##################################################################
# ollama transport
# the only network call: one POST of the prepared payload to the explicit primary URL with the remaining time as its timeout. No retries, no backup route, no router. Returns content plus the server's own accounting.
def ollama_transport(url: str, payload: bytes, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return {
        key: data.get(key)
        for key in ("model", "done_reason", "prompt_eval_count", "eval_count")
    } | {"content": (data.get("message") or {}).get("content", "")}


# ##################################################################
# run extraction
# one request per chunk in order, bounded by a soft deadline (no new request after it) and a hard deadline (the request timeout). Each raw response and its server metadata are saved before validation; saved chunks are never re-asked. A chunk is `ready` only if it has a saved, untruncated, correctly-routed, valid response. The summary reports the true source fraction and is `not_ready` for anything else. With transport=None nothing is sent (offline validation).
def run_extraction(
    chapters: Sequence[Path],
    config: ProofConfig,
    plan: ExtractionPlan,
    out_dir: Path,
    transport: Transport | None,
    calibration: dict | None = None,
    soft_s: float = SOFT_DEADLINE_S,
    hard_s: float = HARD_DEADLINE_S,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    check_output_dir(out_dir, chapters)
    if transport is not None:
        measured = validate_calibration(calibration or {}, config)
        if measured["fixed_overhead_tokens"] != plan.artifact["fixed_overhead_tokens"]:
            raise ContractError(
                f"plan budgeted {plan.artifact['fixed_overhead_tokens']} overhead tokens but calibration measured {measured['fixed_overhead_tokens']}",
                "calibration_overhead_mismatch",
            )
    started = clock()
    write_atomic(
        out_dir / "plan.json", json.dumps(plan.artifact, indent=2, sort_keys=True)
    )
    results, statuses = [], {}
    for item in plan.chunks:
        raw_path = out_dir / "raw" / f"{item.chunk.id}.response.txt"
        meta_path = out_dir / "raw" / f"{item.chunk.id}.meta.json"
        url, payload = request_record(config, item)
        if not raw_path.is_file():
            remaining = min(hard_s - (clock() - started), hard_s)
            if transport is None or clock() - started >= soft_s or remaining <= 0:
                statuses[item.chunk.id] = "missing"
                continue
            reply = transport(url, payload, remaining)
            if reply.get("model") != config.backend.model:
                raise ContractError(
                    f"server answered with {reply.get('model')!r}, not {config.backend.model!r}; fallback route rejected",
                    "fallback_route",
                )
            write_atomic(
                out_dir / "raw" / f"{item.chunk.id}.request.sha256",
                hashlib.sha256(payload).hexdigest(),
            )
            write_atomic(
                meta_path,
                json.dumps(
                    {
                        key: reply.get(key)
                        for key in (
                            "model",
                            "done_reason",
                            "prompt_eval_count",
                            "eval_count",
                        )
                    },
                    sort_keys=True,
                ),
            )
            write_atomic(raw_path, reply.get("content") or "")
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.is_file()
            else {}
        )
        raw = raw_path.read_text(encoding="utf-8")
        result = validate_response(raw, item.chunk)
        result["request_sha256"] = hashlib.sha256(payload).hexdigest()
        result["server"] = meta
        if meta.get("model") not in (None, config.backend.model):
            raise ContractError(
                "a saved response came from another model", "fallback_route"
            )
        if meta.get("done_reason") == "length" or (
            isinstance(meta.get("prompt_eval_count"), int)
            and meta["prompt_eval_count"] > config.input_budget + config.reserve_tokens
        ):
            result["status"] = "truncated"
        statuses[item.chunk.id] = (
            "ready" if result["status"] in {"ok", "partial"} else result["status"]
        )
        results.append(result)
    claims = [claim for result in results for claim in result["claims"]]
    pending = [claim for result in results for claim in result["pending"]]
    write_atomic(
        out_dir / "claims.jsonl",
        "".join(canonical_json(claim) + "\n" for claim in claims),
    )
    write_atomic(
        out_dir / "chunk_results.json",
        json.dumps(
            [
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"claims", "pending"}
                }
                for result in results
            ],
            indent=2,
            sort_keys=True,
        ),
    )
    pending_report = pending_reconciliation(
        chapters, claims, pending, plan.artifact["plan_sha256"]
    )
    write_atomic(
        out_dir / "pending_reconciliation.json",
        json.dumps(pending_report, indent=2, sort_keys=True),
    )
    ready_chars = sum(
        item.chunk.end - item.chunk.start
        for item in plan.chunks
        if statuses.get(item.chunk.id) == "ready"
    )
    total = plan.artifact["coverage"]["source_chars"]
    summary = {
        "plan_sha256": plan.artifact["plan_sha256"],
        "state": "ready" if ready_chars == total else "not_ready",
        "source_chars": total,
        "source_chars_ready": ready_chars,
        "source_fraction": round(ready_chars / total, 6) if total else 0.0,
        "chunks": len(plan.chunks),
        "chunk_status": {
            status: sum(1 for value in statuses.values() if value == status)
            for status in sorted(set(statuses.values()))
        },
        "claims": len(claims),
        "pending_claims": len(pending),
        "rejected": sum(len(result["rejected"]) for result in results),
        "elapsed_s": round(clock() - started, 3),
    }
    write_atomic(
        out_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


# ##################################################################
# project chapters
# every chapter file of the project including 00-intro, in pipeline order, read-only; these are witnesses for the authoritative source, never the source itself.
def project_chapters(project: Path) -> list[Path]:
    chapters = sorted((project / "chapters").glob("*.txt"), key=chapter_order)
    if not chapters:
        raise ContractError("project has no chapters", "no_chapters")
    return chapters


# ##################################################################
# capture tokenizer
# the one read-only metadata call: POST /api/show (verbose) to the configured primary, saved as a pinned capture. It loads no model and generates nothing.
def capture_tokenizer(config_path: Path, out_path: Path) -> str:
    values = tomllib.loads(config_path.read_text(encoding="utf-8"))["llm"]
    url, model = str(values["primary_url"]).rstrip("/"), str(values["primary_model"])
    request = urllib.request.Request(
        f"{url}/api/show",
        data=json.dumps({"name": model, "verbose": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        show = json.loads(response.read().decode("utf-8"))
    write_atomic(out_path, json.dumps(build_capture(model, url, show), sort_keys=True))
    return capture_digest(out_path)


# ##################################################################
# main
# plan/validate are offline; run needs --execute. The run writes pid, log and exit files into the run directory and is bounded by the 175 s soft and 240 s hard deadlines.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wide-context full-book biography extraction contract (offline unless `run --execute`)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser(
        "capture-tokenizer",
        help="read-only /api/show capture of the tokenizer metadata; prints its sha256 pin",
    )
    cap.add_argument("--config", type=Path, required=True)
    cap.add_argument("--out", type=Path, required=True)
    for name in ("plan", "run", "validate"):
        item = sub.add_parser(name)
        item.add_argument("project", type=Path)
        item.add_argument(
            "--source",
            type=Path,
            required=True,
            help="authoritative original UTF-8 text",
        )
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--out", type=Path, required=True)
    run = sub.choices["run"]
    run.add_argument("--calibration", type=Path, required=True)
    for name in ("plan", "validate"):
        sub.choices[name].add_argument(
            "--calibration",
            type=Path,
            default=None,
            help="optional calibration record; its fixed template overhead is budgeted into the plan (run always requires it)",
        )
    run.add_argument(
        "--execute",
        action="store_true",
        help="required: sends the prepared requests to the primary",
    )
    args = parser.parse_args(argv)
    pid_file = exit_file = None
    try:
        if args.command == "capture-tokenizer":
            print(capture_tokenizer(args.config, args.out))
            return 0
        config = load_proof_config(args.config)
        chapters = project_chapters(args.project.resolve())
        check_output_dir(args.out, chapters)
        if args.command == "run" and not args.execute:
            raise ContractError("run requires --execute", "execute_required")
        calibration = None
        overhead = 0
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
        plan = build_plan(
            read_source(args.source),
            chapters,
            config,
            build_counter(config),
            overhead,
        )
        if args.command == "plan":
            write_atomic(
                args.out / "plan.json",
                json.dumps(plan.artifact, indent=2, sort_keys=True),
            )
            print(
                json.dumps(
                    {
                        key: plan.artifact[key]
                        for key in (
                            "plan_sha256",
                            "input_budget",
                            "fixed_overhead_tokens",
                            "calibration_required",
                            "coverage",
                        )
                    }
                    | {"chunks": len(plan.chunks)},
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "validate":
            print(
                json.dumps(
                    run_extraction(chapters, config, plan, args.out, None),
                    sort_keys=True,
                )
            )
            return 0
        pid_file, exit_file = args.out / "run.pid", args.out / "run.exit"
        write_atomic(pid_file, str(os.getpid()))
        summary = run_extraction(
            chapters, config, plan, args.out, ollama_transport, calibration
        )
        write_atomic(exit_file, "0")
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (ContractError, TokenizerRefusal, CastDataIssue) as error:
        message = json.dumps(
            {
                "refused": getattr(error, "code", type(error).__name__),
                "message": str(error),
            }
        )
        print(message, file=sys.stderr)
        if exit_file is not None:
            write_atomic(exit_file, "2")
        return 2


if __name__ == "__main__":
    sys.exit(main())
