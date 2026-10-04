"""Offline, deterministic, proposal-only grouping/reconciliation of new-identity review records.

Reads native `new-identity-review-audit.json` records plus the immutable chapter units of the source and reports which
records are source-proven and how exactly-normalized labels group. It never writes or mutates actors, cast, audits or
any project file, never approves anything, and makes no model call.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

from src.cast_freeze import (
    DISTINCT_VERDICT,
    FRAGMENT_VERDICT,
    NEW_IDENTITY_AUDIT_NAME,
    SAME_PROVISIONAL,
    file_digest,
    immutable_evidence_units,
    immutable_name_references,
    load_new_identity_audit,
    text_digest,
)
from src.data_recovery import OperationalError
from src.hour_runner import chapter_order

CONTRACT = "new-identity-group-report-v1"
MAX_RECORDS = 50_000
OWN_PROVENANCE = "immutable_candidate_reference"
NARRATOR_LABELS = frozenset({"narrator", "the narrator"})
SCOPE_FIELDS = ("chapter_sha256", "quote_sha256", "label", "span_start")
SHA_LENGTH = 64
# Unit ids are batch-local: cNNsMMMMM = chapter index within the review batch (not recoverable from the audit) and the
# chapter-local sentence index. The sentence index is source-provable; the batch-local chapter index is retained literally.
UNIT_ID = re.compile(r"c(\d{2,})s(\d{5,})")

CLASS_DISTINCT = "distinct_living_identity"
CLASS_NONIDENTITY = "nonidentity"
CLASS_EXISTING = "existing"
CLASS_PROVISIONAL = "same_provisional"
CLASS_UNCERTAIN = "uncertain"


# ##################################################################
# normalize label
# exact normalization only: Unicode NFKC, case fold, whitespace collapse. No title stripping, no short-form matching.
def normalize_label(label: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", label).casefold().split())


def is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA_LENGTH
        and all(c in "0123456789abcdef" for c in value)
    )


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ##################################################################
# verdict class
# maps a native verdict string to (class, target); None when the verdict is not a known native verdict.
def verdict_class(verdict: object) -> tuple[str, str] | None:
    if not isinstance(verdict, str):
        return None
    if verdict == DISTINCT_VERDICT:
        return CLASS_DISTINCT, ""
    if verdict == FRAGMENT_VERDICT:
        return CLASS_NONIDENTITY, ""
    if verdict == "uncertain":
        return CLASS_UNCERTAIN, ""
    if verdict.startswith("existing:") and verdict[len("existing:") :]:
        return CLASS_EXISTING, verdict[len("existing:") :]
    if verdict.startswith(SAME_PROVISIONAL) and verdict[len(SAME_PROVISIONAL) :]:
        return CLASS_PROVISIONAL, verdict[len(SAME_PROVISIONAL) :]
    return None


def scope_of(record: object) -> dict:
    """Literal scope fields as present (never defaulted) so a malformed row stays identifiable."""
    if not isinstance(record, dict):
        return {}
    return {field: record[field] for field in SCOPE_FIELDS if field in record}


# ##################################################################
# source proof
# returns (unit, None) when the record's scope and own witness are exactly proven by an immutable unit, else (None, reason, detail).
def prove_record(
    record: object, units_by_scope: dict, references
) -> tuple[dict | None, str, str]:
    if not isinstance(record, dict):
        return None, "malformed_record", "record is not an object"
    if not (
        is_sha(record.get("chapter_sha256"))
        and is_sha(record.get("quote_sha256"))
        and isinstance(record.get("label"), str)
        and record["label"]
        and is_int(record.get("span_start"))
        and record["span_start"] >= 0
    ):
        return (
            None,
            "malformed_record",
            "scope fields chapter_sha256/quote_sha256/label/span_start are missing or mistyped",
        )
    if not isinstance(record.get("owners"), list) or not all(
        isinstance(o, str) for o in record["owners"]
    ):
        return None, "malformed_record", "owners is not a list of strings"
    if not isinstance(record.get("witness_unit_ids"), list) or not all(
        isinstance(w, str) for w in record["witness_unit_ids"]
    ):
        return None, "malformed_record", "witness_unit_ids is not a list of strings"
    if verdict_class(record.get("verdict")) is None:
        return (
            None,
            "unsupported_verdict",
            f"verdict {record.get('verdict')!r} is not a native review verdict",
        )
    if normalize_label(record["label"]) in NARRATOR_LABELS:
        return None, "narrator_label", "a narrator label is never an identity candidate"
    witness = record.get("own_source_witness")
    if not isinstance(witness, dict):
        return (
            None,
            "missing_own_source_witness",
            "record has no own_source_witness object",
        )
    if witness.get("provenance") != OWN_PROVENANCE:
        return (
            None,
            "non_own_witness_provenance",
            f"witness provenance {witness.get('provenance')!r} is not {OWN_PROVENANCE}",
        )
    candidates = units_by_scope.get((record["chapter_sha256"], record["quote_sha256"]))
    if not candidates:
        if not any(key[0] == record["chapter_sha256"] for key in units_by_scope):
            return (
                None,
                "unknown_chapter_hash",
                "chapter_sha256 matches no immutable chapter of the source",
            )
        return (
            None,
            "quote_hash_mismatch",
            "no unit of that chapter has this quote_sha256",
        )
    label, start = record["label"], record["span_start"]
    if not any(
        c["quote"][start : start + len(label)] == label
        and any(
            ref["label"] == label and ref["start"] == start
            for ref in references(c).values()
        )
        for c in candidates
    ):
        return (
            None,
            "span_not_exact_source_reference",
            "label at span_start is not an exact immutable name reference of the unit",
        )
    witness_id = witness.get("unit_id")
    match = UNIT_ID.fullmatch(witness_id) if isinstance(witness_id, str) else None
    if (
        match is None
        or witness.get("label") != label
        or witness.get("span_start") != start
    ):
        return (
            None,
            "witness_not_own_span",
            "own_source_witness is malformed or names a different label/span",
        )
    unit = next(
        (c for c in candidates if c["sentence_index"] == int(match.group(2))), None
    )
    if unit is None:
        return (
            None,
            "witness_not_own_span",
            f"witness {witness_id} is not the unit carrying this quote and span",
        )
    if not record["witness_unit_ids"] or record["witness_unit_ids"][0] != witness_id:
        return (
            None,
            "witness_unit_ids_mismatch",
            "witness_unit_ids does not lead with the own source witness unit",
        )
    kind, target = verdict_class(record["verdict"])
    if kind == CLASS_EXISTING and target not in record["owners"]:
        return (
            None,
            "existing_target_not_offered_owner",
            f"existing target {target!r} was not an offered owner (roster coercion)",
        )
    return unit, "", ""


def literal_row(record: dict, unit: dict | None) -> dict:
    """All scope, hash, witness and ownership fields kept literally; derived fields are added, nothing is rewritten."""
    row = {field: record[field] for field in SCOPE_FIELDS}
    row.update(
        verdict=record["verdict"],
        owners=list(record["owners"]),
        witness_unit_ids=list(record["witness_unit_ids"]),
        own_source_witness=dict(record["own_source_witness"]),
        confidence=record.get("confidence"),
    )
    if unit is not None:
        row["chapter"] = unit["chapter"]
        row["sentence_index"] = unit["sentence_index"]
    return row


def scope_key(row: dict) -> tuple:
    return tuple(row[field] for field in SCOPE_FIELDS)


# ##################################################################
# build report
def build_report(records: list, units: list[dict], source_sha256: str) -> dict:
    if not isinstance(records, list):
        raise OperationalError(
            "store_invalid", "new-identity review records must be a list"
        )
    if len(records) > MAX_RECORDS:
        raise OperationalError(
            "input_too_large",
            f"{len(records)} records exceed the bounded limit {MAX_RECORDS}",
        )
    if not is_sha(source_sha256):
        raise OperationalError(
            "source_sha_invalid", "source_sha256 must be a lowercase sha256 hex digest"
        )
    units_by_scope: dict[tuple[str, str], list[dict]] = {}
    for u in units:
        units_by_scope.setdefault(
            (u["chapter_sha256"], text_digest(u["quote"])), []
        ).append(u | {"sentence_index": int(UNIT_ID.fullmatch(u["id"]).group(2))})
    reference_cache: dict[str, dict] = {}

    def references(unit: dict) -> dict:
        if unit["id"] not in reference_cache:
            reference_cache[unit["id"]] = immutable_name_references([unit])
        return reference_cache[unit["id"]]

    scope_counts: dict[tuple, int] = {}
    for record in records:
        if isinstance(record, dict) and all(field in record for field in SCOPE_FIELDS):
            try:
                key = tuple(record[field] for field in SCOPE_FIELDS)
                scope_counts[key] = scope_counts.get(key, 0) + 1
            except TypeError:
                pass
    pending: list[dict] = []
    eligible: list[tuple[dict, str, str]] = []
    for index, record in enumerate(records):
        unit, reason, detail = prove_record(record, units_by_scope, references)
        if unit is not None:
            key = tuple(record[field] for field in SCOPE_FIELDS)
            if scope_counts.get(key, 0) > 1:
                unit, reason, detail = (
                    None,
                    "duplicate_scope",
                    f"{scope_counts[key]} records share this exact scope",
                )
        if unit is None:
            pending.append(
                {
                    "reason": reason,
                    "detail": detail,
                    "input_index": index,
                    "scope": scope_of(record),
                    "record_sha256": text_digest(
                        json.dumps(record, sort_keys=True, default=repr)
                    ),
                }
            )
            continue
        kind, target = verdict_class(record["verdict"])
        eligible.append(
            (literal_row(record, unit) | {"input_index": index}, kind, target)
        )
    by_label: dict[str, list[tuple[dict, str, str]]] = {}
    for item in eligible:
        by_label.setdefault(normalize_label(item[0]["label"]), []).append(item)
    groups, nonidentity, existing, provisional, uncertain, conflicts = (
        [],
        [],
        {},
        {},
        [],
        [],
    )
    for label_key in sorted(by_label):
        items = by_label[label_key]
        kinds = {kind for _, kind, _ in items}
        existing_targets = {
            target for _, kind, target in items if kind == CLASS_EXISTING
        }
        if len(kinds) > 1 or len(existing_targets) > 1:
            conflicts.append(
                {
                    "label_key": label_key,
                    "verdicts": sorted(
                        {
                            row["verdict"]
                            if kind != CLASS_PROVISIONAL
                            else SAME_PROVISIONAL + "*"
                            for row, kind, _ in items
                        }
                    ),
                    "records": sorted((row for row, _, _ in items), key=scope_key),
                }
            )
            continue
        (kind,) = kinds
        rows = sorted((row for row, _, _ in items), key=scope_key)
        if kind == CLASS_DISTINCT:
            digest = text_digest(
                json.dumps([label_key, [scope_key(r) for r in rows]], sort_keys=True)
            )
            groups.append(
                {
                    "group_id": f"g-{digest[:16]}",
                    "label_key": label_key,
                    "literal_labels": sorted({r["label"] for r in rows}),
                    "chapter_sha256": sorted({r["chapter_sha256"] for r in rows}),
                    "quote_sha256": sorted({r["quote_sha256"] for r in rows}),
                    "record_count": len(rows),
                    "records": rows,
                }
            )
        elif kind == CLASS_NONIDENTITY:
            nonidentity.extend(rows)
        elif kind == CLASS_UNCERTAIN:
            uncertain.extend(rows)
        elif kind == CLASS_EXISTING:
            existing.setdefault(next(iter(existing_targets)), []).extend(rows)
        else:
            for row, _, target in items:
                provisional.setdefault((row["chapter_sha256"], target), []).append(row)
    existing_out = [
        {"actor_id": actor, "records": sorted(rows, key=scope_key)}
        for actor, rows in sorted(existing.items())
    ]
    provisional_out = [
        {
            "chapter_sha256": chapter,
            "provisional_id": target,
            "records": sorted(rows, key=scope_key),
        }
        for (chapter, target), rows in sorted(provisional.items())
    ]
    grouped = sum(g["record_count"] for g in groups)
    counts = {
        "input_records": len(records),
        "eligible_records": len(eligible),
        "ineligible_records": len(pending),
        "group_count": len(groups),
        "grouped_records": grouped,
        "nonidentity_records": len(nonidentity),
        "existing_records": sum(len(e["records"]) for e in existing_out),
        "same_provisional_records": sum(len(p["records"]) for p in provisional_out),
        "uncertain_records": len(uncertain),
        "conflict_records": sum(len(c["records"]) for c in conflicts),
        "conflict_labels": len(conflicts),
    }
    accounted = grouped + sum(
        counts[k]
        for k in (
            "nonidentity_records",
            "existing_records",
            "same_provisional_records",
            "uncertain_records",
            "conflict_records",
        )
    )
    pending_reasons: dict[str, int] = {}
    for row in pending:
        pending_reasons[row["reason"]] = pending_reasons.get(row["reason"], 0) + 1
    counts["pending_by_reason"] = dict(sorted(pending_reasons.items()))
    counts["balanced"] = accounted == len(eligible) and len(eligible) + len(
        pending
    ) == len(records)
    if not counts["balanced"]:
        raise OperationalError(
            "report_unbalanced",
            "group/bucket totals do not reconcile with eligible records",
        )
    return {
        "contract": CONTRACT,
        "proposal_only": True,
        "source_sha256": source_sha256,
        "groups": groups,
        "nonidentity": nonidentity,
        "existing": existing_out,
        "same_provisional": provisional_out,
        "uncertain": uncertain,
        "conflicts": conflicts,
        "pending": pending,
        "reconciliation": counts,
    }


# ##################################################################
# report for project
# strictly read-only: loads the audit and chapter files, hashes the source, returns a report.
def report_for_project(
    source: Path, project: Path, expected_source_sha256: str | None = None
) -> dict:
    source_sha = file_digest(source)
    if expected_source_sha256 is not None and expected_source_sha256 != source_sha:
        raise OperationalError(
            "source_sha_mismatch",
            "source file does not match the expected source sha256",
        )
    chapters_dir = project / "chapters"
    chapters = sorted(
        (p for p in chapters_dir.glob("*.txt") if p.name != "00-intro.txt"),
        key=chapter_order,
    )
    if not chapters:
        raise OperationalError(
            "source_no_chapters", f"no extracted chapters under {chapters_dir}"
        )
    return build_report(
        load_new_identity_audit(project), immutable_evidence_units(chapters), source_sha
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            f"usage: python -m src.identity_group_report SOURCE PROJECT  (reads {NEW_IDENTITY_AUDIT_NAME}; prints JSON)",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            report_for_project(Path(argv[0]), Path(argv[1])), indent=1, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
