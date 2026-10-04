"""Tests for the offline new-identity grouping/reconciliation report using real temporary source fixtures and the real Weakest artifacts."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.cast_freeze import (
    NEW_IDENTITY_AUDIT_NAME,
    immutable_evidence_units,
    immutable_name_references,
    text_digest,
)
from src.data_recovery import OperationalError
from src.identity_group_report import build_report, normalize_label, report_for_project

CHAPTER_ONE = "Professor Jean smiled. Mara waved at Professor Jean. The Narrator sighed. Mara laughed."
CHAPTER_TWO = "Professor Jean frowned. Mara left. Kess nodded."


def make_project(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "book.txt"
    source.write_text(CHAPTER_ONE + "\n\n" + CHAPTER_TWO, encoding="utf-8")
    project = tmp_path / "project"
    (project / "chapters").mkdir(parents=True)
    (project / "chapters" / "00-intro.txt").write_text(
        "Book by Someone", encoding="utf-8"
    )
    (project / "chapters" / "01-one.txt").write_text(CHAPTER_ONE, encoding="utf-8")
    (project / "chapters" / "02-two.txt").write_text(CHAPTER_TWO, encoding="utf-8")
    chapters = [
        project / "chapters" / "01-one.txt",
        project / "chapters" / "02-two.txt",
    ]
    return source, project, immutable_evidence_units(chapters)


def record_for(
    units: list[dict],
    unit_index: int,
    label: str,
    verdict: str = "distinct_living_identity",
    owners=(),
) -> dict:
    unit = units[unit_index]
    start = next(
        r["start"]
        for r in immutable_name_references([unit]).values()
        if r["label"] == label
    )
    return {
        "chapter_sha256": unit["chapter_sha256"],
        "quote_sha256": text_digest(unit["quote"]),
        "label": label,
        "span_start": start,
        "verdict": verdict,
        "witness_unit_ids": [unit["id"]],
        "own_source_witness": {
            "unit_id": unit["id"],
            "provenance": "immutable_candidate_reference",
            "label": label,
            "span_start": start,
        },
        "confidence": 0.95,
        "reason": "r",
        "owners": list(owners),
        "raw_review": {
            "verdict": verdict,
            "witness_unit_ids": [],
            "confidence": 0.95,
            "reason": "r",
        },
    }


def run(tmp_path: Path, records: list[dict]) -> dict:
    source, project, _units = make_project(tmp_path)
    (project / NEW_IDENTITY_AUDIT_NAME).write_text(
        json.dumps({"records": records}), encoding="utf-8"
    )
    return report_for_project(source, project)


def reasons(report: dict) -> list[str]:
    return [row["reason"] for row in report["pending"]]


def test_groups_same_label_across_chapters_keeping_every_scope_and_hash(
    tmp_path: Path,
) -> None:
    _, _, units = make_project(tmp_path)
    ids = [i for i, u in enumerate(units) if "Professor Jean" in u["quote"]]
    records = [record_for(units, i, "Professor Jean") for i in ids]
    report = run(tmp_path / "x", records)
    assert report["proposal_only"] is True
    assert (
        report["source_sha256"]
        == hashlib.sha256((tmp_path / "x" / "book.txt").read_bytes()).hexdigest()
    )
    (group,) = report["groups"]
    assert (
        group["record_count"]
        == len(records)
        == report["reconciliation"]["grouped_records"]
    )
    assert len(group["chapter_sha256"]) == 2
    got = {
        (r["chapter_sha256"], r["quote_sha256"], r["label"], r["span_start"])
        for r in group["records"]
    }
    assert got == {
        (r["chapter_sha256"], r["quote_sha256"], r["label"], r["span_start"])
        for r in records
    }
    assert report["reconciliation"]["balanced"] and not report["pending"]


def test_name_only_merge_across_different_labels_is_prevented(tmp_path: Path) -> None:
    _, _, units = make_project(tmp_path)
    records = [record_for(units, 0, "Professor Jean"), record_for(units, 1, "Mara")]
    units_jean = next(
        i
        for i, u in enumerate(units)
        if u["quote"].startswith("Professor Jean frowned")
    )
    records.append(record_for(units, units_jean, "Professor Jean"))
    report = run(tmp_path / "x", records)
    assert sorted(g["label_key"] for g in report["groups"]) == [
        "mara",
        "professor jean",
    ]
    assert normalize_label(
        "  PROFESSOR\u00a0Jean "
    ) == "professor jean" and normalize_label("Jean") != normalize_label(
        "Professor Jean"
    )


def test_conflicting_verdicts_for_one_label_are_not_grouped(tmp_path: Path) -> None:
    _, _, units = make_project(tmp_path)
    records = [
        record_for(units, 5, "Mara"),
        record_for(units, 1, "Mara", "nonidentity_fragment"),
        record_for(units, 3, "Mara", "existing:mara_prime", owners=["mara_prime"]),
    ]
    report = run(tmp_path / "x", records)
    assert report["groups"] == []
    (conflict,) = report["conflicts"]
    assert conflict["label_key"] == "mara" and len(conflict["records"]) == 3
    assert (
        report["reconciliation"]["conflict_records"] == 3
        and report["reconciliation"]["balanced"]
    )


def test_buckets_distinguish_nonidentity_existing_provisional_and_uncertain(
    tmp_path: Path,
) -> None:
    _, _, units = make_project(tmp_path)
    records = [
        record_for(units, 0, "Professor Jean", "nonidentity_fragment"),
        record_for(units, 1, "Mara", "existing:mara_prime", owners=["mara_prime"]),
        record_for(units, 6, "Kess", "same_provisional:p0001"),
    ]
    report = run(tmp_path / "x", records)
    assert len(report["nonidentity"]) == 1
    assert report["existing"][0]["actor_id"] == "mara_prime"
    assert report["same_provisional"][0]["provisional_id"] == "p0001"
    assert report["groups"] == [] and report["reconciliation"]["balanced"]


def test_uncertain_record_is_reported_not_grouped(tmp_path: Path) -> None:
    _, _, units = make_project(tmp_path)
    report = run(tmp_path / "x", [record_for(units, 1, "Mara", "uncertain")])
    assert len(report["uncertain"]) == 1 and report["groups"] == []


def mutate(units: list[dict], change) -> dict:
    record = record_for(units, 0, "Professor Jean")
    change(record)
    return record


@pytest.mark.parametrize(
    ("reason", "change"),
    [
        ("quote_hash_mismatch", lambda r: r.update(quote_sha256="0" * 64)),
        ("unknown_chapter_hash", lambda r: r.update(chapter_sha256="1" * 64)),
        (
            "span_not_exact_source_reference",
            lambda r: r.update(span_start=r["span_start"] + 1),
        ),
        ("span_not_exact_source_reference", lambda r: r.update(label="Professor Jea")),
        (
            "witness_not_own_span",
            lambda r: r["own_source_witness"].update(unit_id="c00s00001"),
        ),
        (
            "witness_not_own_span",
            lambda r: r["own_source_witness"].update(span_start=99),
        ),
        (
            "witness_unit_ids_mismatch",
            lambda r: r.update(witness_unit_ids=["c00s00001"]),
        ),
        (
            "non_own_witness_provenance",
            lambda r: r["own_source_witness"].update(provenance="invalid_provenance"),
        ),
        ("missing_own_source_witness", lambda r: r.pop("own_source_witness")),
        ("unsupported_verdict", lambda r: r.update(verdict="roster_member")),
        ("unsupported_verdict", lambda r: r.update(verdict="existing:")),
        (
            "existing_target_not_offered_owner",
            lambda r: r.update(verdict="existing:someone", owners=["other"]),
        ),
        ("malformed_record", lambda r: r.update(span_start="1")),
        ("malformed_record", lambda r: r.update(span_start=True)),
        ("malformed_record", lambda r: r.pop("owners")),
    ],
)
def test_each_failure_mode_becomes_typed_pending_never_a_group(
    tmp_path: Path, reason: str, change
) -> None:
    _, _, units = make_project(tmp_path)
    report = run(tmp_path / "x", [mutate(units, change)])
    assert reasons(report) == [reason]
    assert (
        report["groups"] == []
        and report["reconciliation"]["ineligible_records"] == 1
        and report["reconciliation"]["balanced"]
    )


def test_non_object_and_narrator_rows_are_pending_and_duplicates_are_both_withheld(
    tmp_path: Path,
) -> None:
    _, _, units = make_project(tmp_path)
    narrator = next(i for i, u in enumerate(units) if "Narrator" in u["quote"])
    good = record_for(units, 0, "Professor Jean")
    report = run(
        tmp_path / "x",
        [
            "junk",
            None,
            record_for(units, narrator, "Narrator"),
            good,
            copy.deepcopy(good),
            record_for(units, 1, "Mara"),
        ],
    )
    assert sorted(reasons(report)) == [
        "duplicate_scope",
        "duplicate_scope",
        "malformed_record",
        "malformed_record",
        "narrator_label",
    ]
    assert [g["label_key"] for g in report["groups"]] == ["mara"]
    r = report["reconciliation"]
    assert (
        r["input_records"] == 6
        and r["eligible_records"] == 1
        and r["ineligible_records"] == 5
        and r["balanced"]
    )


def test_report_is_deterministic_and_input_order_independent(tmp_path: Path) -> None:
    _, _, units = make_project(tmp_path)
    records = [
        record_for(units, 0, "Professor Jean"),
        record_for(units, 1, "Mara"),
        record_for(units, 3, "Mara"),
    ]
    first = run(tmp_path / "a", records)
    second = run(tmp_path / "b", list(reversed(records)))
    for report in (first, second):
        for group in report["groups"]:
            for row in group["records"]:
                row.pop("input_index")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_source_sha_expectation_and_bounds_fail_closed(tmp_path: Path) -> None:
    source, project, units = make_project(tmp_path)
    (project / NEW_IDENTITY_AUDIT_NAME).write_text(
        json.dumps({"records": []}), encoding="utf-8"
    )
    with pytest.raises(OperationalError):
        report_for_project(source, project, expected_source_sha256="2" * 64)
    with pytest.raises(OperationalError):
        build_report("not a list", units, "3" * 64)
    with pytest.raises(OperationalError):
        build_report([], units, "bad")


def test_project_files_are_never_modified(tmp_path: Path) -> None:
    source, project, units = make_project(tmp_path)
    (project / NEW_IDENTITY_AUDIT_NAME).write_text(
        json.dumps({"records": [record_for(units, 0, "Professor Jean")]}),
        encoding="utf-8",
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report_for_project(source, project)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


WEAKEST_SOURCE = Path(
    "/Users/darrenoakey/src/book-reader/incoming/weakest_beast_tamer.txt"
)
WEAKEST_PROJECT = Path("/Users/darrenoakey/src/book-reader/output/weakest_beast_tamer")


def test_real_weakest_artifacts_reconcile_without_touching_production() -> None:
    assert WEAKEST_SOURCE.is_file() and WEAKEST_PROJECT.is_dir(), (
        "mandatory real Weakest artifacts are unavailable"
    )
    audit = WEAKEST_PROJECT / NEW_IDENTITY_AUDIT_NAME
    before = hashlib.sha256(audit.read_bytes()).hexdigest()
    report = report_for_project(WEAKEST_SOURCE, WEAKEST_PROJECT)
    assert hashlib.sha256(audit.read_bytes()).hexdigest() == before
    r = report["reconciliation"]
    assert (
        r["balanced"]
        and r["input_records"] == r["eligible_records"] + r["ineligible_records"] > 100
    )
    assert r["group_count"] == len(report["groups"]) > 0
    assert sum(g["record_count"] for g in report["groups"]) == r["grouped_records"]
    for group in report["groups"]:
        assert len({normalize_label(label) for label in group["literal_labels"]}) == 1
        assert all(
            row["verdict"] == "distinct_living_identity" for row in group["records"]
        )
    assert len(report["pending"]) == r["ineligible_records"]
    assert (
        report["source_sha256"]
        == hashlib.sha256(WEAKEST_SOURCE.read_bytes()).hexdigest()
    )
