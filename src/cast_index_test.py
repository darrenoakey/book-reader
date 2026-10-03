"""Real-source tests for the deterministic whole-book cast index (temp chapter files, no model, no mocks)."""

import hashlib
import json
import re
import time
from pathlib import Path

import pytest

from src.cast_freeze import (
    PROGRESS_NAME,
    SCOPED_AUDIT_NAME,
    CastDataIssue,
    candidate_coverage_ledger,
    immutable_evidence_units,
    immutable_name_references,
    mention_scope,
)
from src.cast_index import (
    Claim,
    JoinedText,
    LowercaseOracle,
    build_cast_index,
    build_chunk_plan,
    build_source_index,
    diversified_occurrences,
    group_candidates,
    group_payload,
    main,
    paragraph_spans,
    reconcile,
    verify_chunk_coverage,
)

BOOK = {
    "01-one.txt": (
        "Young Ren smiled at Luna. Young bowed to the king. The young fox ran.\n\n"
        "Mary-Ann and Mary Ann met Luna Starwaver. Luna Starwaver laughed.\n\n"
        "The impact of Zed was felt. Zed Air flew by.  \n\n"
    ),
    "02-two.txt": "Luna spoke to Ren. Young sold the fruit. Ice-Wolf howled.\n\nIce-wolf tracks vanished.\n",
    "03-three.txt": "Ren walked home. Luna Starwaver waved. Quill stood alone.\n\nNo one saw Quill.\n",
}


# ##################################################################
# write book
# writes real chapter files for one test into its private directory.
def write_book(root: Path, book: dict[str, str] | None = None) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in (book or BOOK).items():
        path = root / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


# ##################################################################
# test occurrences are the exact immutable references
# the index adds no new reading of the source: ids, labels, offsets and scope hashes equal immutable_name_references / mention_scope.
def test_occurrences_equal_existing_references_and_scopes(tmp_path: Path) -> None:
    chapters = write_book(tmp_path)
    index = build_source_index(chapters)
    units = immutable_evidence_units(chapters)
    references = immutable_name_references(units)
    by_id = {unit["id"]: unit for unit in units}
    assert list(index.occurrence_ref) == list(references)
    for occurrence, (ref_id, reference) in enumerate(references.items()):
        assert index.occurrence_label[occurrence] == reference["label"]
        assert index.occurrence_start[occurrence] == reference["start"]
        assert index.occurrence_suffix[occurrence] == reference["suffix"]
        assert index.scope(occurrence) == mention_scope(by_id[reference["unit_id"]], reference)


# ##################################################################
# test digest is deterministic and exact
# equal bytes give an equal digest; one changed character, or a renamed chapter, changes it.
def test_digest_deterministic_and_sensitive(tmp_path: Path) -> None:
    first = build_source_index(write_book(tmp_path / "a")).digest
    assert build_source_index(write_book(tmp_path / "b")).digest == first
    changed = {**BOOK, "03-three.txt": BOOK["03-three.txt"].replace("Quill", "Quilt")}
    assert build_source_index(write_book(tmp_path / "c", changed)).digest != first
    renamed = {("04-three.txt" if name == "03-three.txt" else name): text for name, text in BOOK.items()}
    assert build_source_index(write_book(tmp_path / "d", renamed)).digest != first


# ##################################################################
# test undecodable chapter fails closed
# a chapter that is not UTF-8 raises the typed issue instead of being skipped.
def test_undecodable_chapter_fails_closed(tmp_path: Path) -> None:
    chapters = write_book(tmp_path)
    chapters[1].write_bytes(b"Luna spoke \xff\xfe to Ren.")
    with pytest.raises(CastDataIssue):
        build_source_index(chapters)


# ##################################################################
# test groups are exact labels and never merged by spelling
# the ledger merges Mary-Ann and Mary Ann (same normalized id); the index keeps them separate, with identical retention otherwise.
def test_exact_label_groups_match_ledger_without_spelling_merge(tmp_path: Path) -> None:
    chapters = write_book(tmp_path)
    index = build_source_index(chapters)
    grouping = group_candidates(index)
    groups = {group.label: group for group in grouping.groups}
    assert "Mary-Ann" in groups and "Mary Ann" in groups
    assert groups["Mary-Ann"].normalized == groups["Mary Ann"].normalized
    assert not set(groups["Mary-Ann"].occurrences) & set(groups["Mary Ann"].occurrences)
    ledger = candidate_coverage_ledger(immutable_evidence_units(chapters), {}, {})
    merged = next(item for item in ledger if item["label"] in {"Mary-Ann", "Mary Ann"})
    assert len(merged["ref_ids"]) == len(groups["Mary-Ann"].occurrences) + len(groups["Mary Ann"].occurrences)
    spelled = {"Mary-Ann", "Mary Ann", "Ice-Wolf", "Ice-wolf"}
    assert "Ice-Wolf" in groups and "Ice-wolf" in groups
    ledger_refs = {item["label"]: set(item["ref_ids"]) for item in ledger if item["label"] not in spelled}
    index_refs = {
        label: {index.occurrence_ref[item] for item in group.occurrences}
        for label, group in groups.items()
        if label not in spelled
    }
    assert index_refs == ledger_refs
    ledger_nonentity = {item["label"] for item in ledger if item["nonentity"]}
    assert {label for label, group in groups.items() if group.nonentity} == ledger_nonentity
    assert "Zed" in ledger_nonentity


# ##################################################################
# test every occurrence is accounted for
# retained group occurrences plus dropped ones equal all occurrences; dropped labels never include a retained one.
def test_retained_plus_dropped_equals_all_occurrences(tmp_path: Path) -> None:
    index = build_source_index(write_book(tmp_path))
    grouping = group_candidates(index)
    retained = sum(len(group.occurrences) for group in grouping.groups)
    assert retained + grouping.dropped_occurrences == len(index.occurrence_label)
    assert len({group.id for group in grouping.groups}) == len(grouping.groups)
    assert [group.scope_sha256 for group in grouping.groups] == [
        group.scope_sha256 for group in group_candidates(index).groups
    ]


# ##################################################################
# test lowercase oracle equals regex search
# the one-pass oracle answers exactly like the per-label regex the ledger uses, including hyphen, apostrophe and non-ASCII labels.
def test_lowercase_oracle_matches_regex(tmp_path: Path) -> None:
    units = [
        {"quote": "the ice-wolf and o'brien met zoë. A wolf-like café."},
        {"quote": "ice-wolfs bark; xo'brien; the_zoë"},
    ]
    oracle = LowercaseOracle(units)
    source = "\n".join(unit["quote"] for unit in units)
    for label in ["Ice-Wolf", "O'Brien", "Zoë", "Wolf", "Café", "Ice", "Brien", "Xo'Brien", "Wolf-Like", "Nothing"]:
        expected = bool(re.search(rf"(?<!\w){re.escape(label.casefold())}(?!\w)", source))
        assert oracle.present(label) == expected, label


# ##################################################################
# test scoped reuse follows existing guards
# a proven alias, a non_character record and a final ambiguous record are reused; an unproven alias is stale; a record for changed bytes or another offset never applies; a literal owner is reused only when it is in the cast.
def test_known_reuse_only_under_existing_guards(tmp_path: Path) -> None:
    chapters = write_book(tmp_path)
    index = build_source_index(chapters)
    registry = {"young_ren": {"name": "Young Ren"}, "luna_starwaver": {"name": "Luna Starwaver"}}
    young = [i for i, label in enumerate(index.occurrence_label) if label == "Young"]
    assert len(young) == 3

    def record(occurrence: int, decision: str, canonical: str, **extra: object) -> dict:
        chapter_sha, quote_sha, label, start = index.scope(occurrence)
        return {
            "chapter_sha256": chapter_sha,
            "quote_sha256": quote_sha,
            "label": label,
            "span_start": start,
            "canonical": canonical,
            "decision": decision,
            "confidence": 0.9,
            "reason": "test",
            **extra,
        }

    records = [
        record(young[0], "alias", "young_ren"),
        record(young[1], "non_character", "none"),
        record(young[2], "ambiguous", "none", owners=["young_ren", "luna_starwaver"]),
    ]
    cast = build_cast_index(chapters, registry, {}, records)
    resolved = {item.group.label: item for item in cast.resolved}
    summary = [(part.kind, part.canonical, len(part.occurrences)) for part in resolved["Young"].partitions]
    assert summary == [("alias", "young_ren", 1), ("non_character", None, 1), ("ambiguous", None, 1)]
    ledger = candidate_coverage_ledger(immutable_evidence_units(chapters), registry, {}, records)
    scoped = {
        item["scoped_audit"]["decision"]: item
        for item in ledger
        if item["label"] == "Young" and item.get("scoped_audit")
    }
    assert scoped["alias"]["known_owner"] == "young_ren" and not scoped["alias"].get("scoped_stale")
    assert not scoped["ambiguous"].get("scoped_stale") and scoped["non_character"]["nonentity"]
    stale = [records[0], {**records[2], "owners": ["luna_starwaver"]}]
    redo = next(i for i in build_cast_index(chapters, registry, {}, stale).resolved if i.group.label == "Young")
    assert [(part.kind, len(part.occurrences)) for part in redo.partitions] == [("alias", 1), ("stale", 1), ("open", 1)]
    ledger = candidate_coverage_ledger(immutable_evidence_units(chapters), registry, {}, stale)
    assert next(i for i in ledger if i.get("scoped_audit", {}).get("decision") == "ambiguous").get("scoped_stale")
    assert resolved["Luna Starwaver"].known_owner == "luna_starwaver"
    assert {part.kind for part in resolved["Luna Starwaver"].partitions} == {"literal_owner"}
    outside = build_cast_index(chapters, {"young_ren": {"name": "Young Ren"}}, {}, [])
    assert next(i for i in outside.resolved if i.group.label == "Luna Starwaver").known_owner is None
    moved = [{**records[0], "span_start": records[0]["span_start"] + 1}, {**records[1], "chapter_sha256": "0" * 64}]
    unmatched = build_cast_index(chapters, registry, {}, moved)
    assert {part.kind for item in unmatched.resolved if item.group.label == "Young" for part in item.partitions} == {
        "open"
    }
    with pytest.raises(ValueError):
        build_cast_index(chapters, registry, {}, [{k: v for k, v in records[0].items() if k != "span_start"}])


# ##################################################################
# test an unprovable alias is re-decided
# an alias record whose owner the scene never supports is stale, never trusted.
def test_unproven_alias_is_stale(tmp_path: Path) -> None:
    chapters = write_book(tmp_path, {"01-a.txt": "Young bowed to the king. The sun set.\n"})
    index = build_source_index(chapters)
    chapter_sha, quote_sha, label, start = index.scope(0)
    record = {
        "chapter_sha256": chapter_sha,
        "quote_sha256": quote_sha,
        "label": label,
        "span_start": start,
        "canonical": "young_ren",
        "decision": "alias",
        "confidence": 0.9,
        "reason": "test",
    }
    cast = build_cast_index(chapters, {"young_ren": {"name": "Young Ren"}}, {}, [record])
    young = next(item for item in cast.resolved if item.group.label == "Young")
    assert [part.kind for part in young.partitions] == ["stale"]
    assert young.needs_decision() == young.group.occurrences


# ##################################################################
# test diversified contexts are factual, distinct and deterministic
# contexts span the book (first, last, new chapters), never repeat a sentence, and copy exact source bytes and hashes.
def test_diversified_contexts_are_factual_and_spread(tmp_path: Path) -> None:
    book = {f"{n:02d}-c.txt": f"Ren met Luna in part {n}. Ren left.\n\nRen waved.\n" for n in range(1, 21)}
    book["99-dup.txt"] = "Ren left. Ren left.\n"
    chapters = write_book(tmp_path, book)
    index = build_source_index(chapters)
    ren = [i for i, label in enumerate(index.occurrence_label) if label == "Ren"]
    picks = diversified_occurrences(index, ren, 5)
    assert picks == diversified_occurrences(index, ren, 5) and len(picks) == 5
    assert picks[0] == ren[0] and index.unit_chapter[index.occurrence_unit[picks[-1]]] >= 19
    assert len({index.quote_sha256[index.occurrence_unit[i]] for i in picks}) == 5
    assert len({index.unit_chapter[index.occurrence_unit[i]] for i in picks}) >= 4
    assert diversified_occurrences(index, ren, 0) == []
    cast = build_cast_index(chapters, {}, {}, [])
    payload = group_payload(cast.source, next(i for i in cast.resolved if i.group.label == "Ren"), 5)
    for context in payload["contexts"]:
        unit = cast.source.by_id[context["unit_id"]]
        assert context["quote"] == unit["quote"]
        assert context["quote_sha256"] == hashlib.sha256(unit["quote"].encode()).hexdigest()
        assert context["chapter_sha256"] == unit["chapter_sha256"]
        assert context["scene"][0] <= context["unit_id"] <= context["scene"][1]
    json.dumps(payload)


# ##################################################################
# test chunks cover every source character exactly once
# paragraph-boundary chunks tile every chapter with no gap or overlap, never split a paragraph, and verification detects tampering.
def test_chunk_plan_covers_source_exactly_once(tmp_path: Path) -> None:
    chapters = write_book(tmp_path)
    index = build_source_index(chapters)
    for max_chars in (10, 80, 150, 10_000):
        plan = build_chunk_plan(index, max_chars)
        assert plan.coverage["exactly_once"] and plan.coverage["chars"] == sum(len(p.read_text()) for p in chapters)
        text = "".join(chapter_slice(chapters, seg) for chunk in plan.chunks for seg in chunk.segments)
        assert text == "".join(p.read_text() for p in chapters)
        for chunk in plan.chunks:
            for seg in chunk.segments:
                body = (tmp_path / index.chapters[seg.chapter]["name"]).read_text()
                assert seg.start == 0 or body[seg.start - 1] in " \n\t"
                assert seg.start == 0 or any(start == seg.start for start, _ in paragraph_spans(body))
        assert plan.digest == build_chunk_plan(index, max_chars).digest
        assert all(chunk.oversize == (chunk.chars > max_chars) for chunk in plan.chunks)
        assert len(plan.occurrence_chunk) == len(index.occurrence_label)
    assert len(build_chunk_plan(index, 10_000).chunks) == 1
    assert len(build_chunk_plan(index, 10).chunks) > len(build_chunk_plan(index, 150).chunks) > 1
    plan = build_chunk_plan(index, 150)
    broken = list(plan.chunks)
    first = broken[1]
    broken[1] = type(first)(first.id, first.segments[1:] or first.segments, first.chars, first.sha256, first.oversize)
    assert not verify_chunk_coverage(index, broken)["exactly_once"] or len(first.segments) == 1
    assert not verify_chunk_coverage(index, plan.chunks[:-1])["exactly_once"]
    with pytest.raises(ValueError):
        build_chunk_plan(index, 0)


def chapter_slice(chapters: list[Path], seg: object) -> str:
    return chapters[seg.chapter].read_text()[seg.start : seg.end]


# ##################################################################
# test occurrences land in the chunk holding their first character
# every occurrence maps to one chunk whose segment contains its absolute offset; none straddle in this source.
def test_occurrence_chunk_placement(tmp_path: Path) -> None:
    chapters = write_book(tmp_path)
    index = build_source_index(chapters)
    plan = build_chunk_plan(index, 120)
    for occurrence, ordinal in enumerate(plan.occurrence_chunk):
        unit = index.occurrence_unit[occurrence]
        chapter = index.unit_chapter[unit]
        offset = index.unit_offset[unit] + index.occurrence_start[occurrence]
        assert (
            chapters[chapter].read_text()[offset : offset + len(index.occurrence_label[occurrence])]
            == index.occurrence_label[occurrence]
        )
        assert any(seg.chapter == chapter and seg.start <= offset < seg.end for seg in plan.chunks[ordinal].segments)
    assert plan.straddling == ()


# ##################################################################
# test reconciliation classifies every retained occurrence once
# exact claim names are covered, whole-word parts of claimed names are component (never merged), unclaimed undecided labels are omitted with contexts and chunks, unsupported claim names are reported, and the artifact is deterministic.
def test_reconcile_statuses_and_totals(tmp_path: Path) -> None:
    chapters = write_book(tmp_path)
    registry = {"luna_starwaver": {"name": "Luna Starwaver"}}
    cast = build_cast_index(chapters, registry, {}, [])
    plan = build_chunk_plan(cast.source, 120)
    claims = [
        Claim("c1", ("Luna Starwaver", "Starwaver")),
        Claim("c2", ("Young Ren",)),
        Claim("c3", ("Nobody Real",)),
    ]
    report = reconcile(cast, claims, plan)
    assert report == reconcile(cast, claims, plan)
    status = {entry["label"]: key for key in ("covered", "component", "settled", "omitted") for entry in report[key]}
    assert status["Luna Starwaver"] == "covered"
    assert status["Young Ren"] == "covered"
    assert status["Ren"] == "component" and status["Young"] == "component"
    assert status["Quill"] == "omitted" and status["Mary-Ann"] == "omitted" and status["Mary Ann"] == "omitted"
    omitted = next(entry for entry in report["omitted"] if entry["label"] == "Quill")
    assert omitted["contexts"] and sum(omitted["chunks"].values()) == omitted["occurrences"]
    assert [item["name"] for item in report["unsupported_claim_names"]] == ["Nobody Real"]
    occ = report["occurrences"]
    assert occ["retained"] + occ["dropped"] == len(cast.source.occurrence_label)
    assert sum(report["groups"].values()) == len(cast.grouping.groups)
    settled = reconcile(build_cast_index(chapters, {"quill": {"name": "Quill"}}, {}, []), [], plan)
    assert "Quill" in {entry["label"] for entry in settled["settled"]}
    with pytest.raises(ValueError):
        reconcile(cast, [Claim("x", ("A",)), Claim("x", ("B",))])
    with pytest.raises(ValueError):
        reconcile(cast, [Claim("x", ("  ",))])


# ##################################################################
# test joined text locates whole words
# exact, case-sensitive whole-word positions map back to the holding unit.
def test_joined_text_whole_word(tmp_path: Path) -> None:
    index = build_source_index(write_book(tmp_path))
    joined = JoinedText(index.units)
    found = joined.whole_word("Luna")
    assert found and all(index.units[joined.unit_of(at)]["quote"].count("Luna") for at in found)
    assert joined.whole_word("luna") == [] and joined.whole_word("Lun") == []


# ##################################################################
# test benchmark cli on a real temp project
# the read-only CLI reports every stage for a project directory with progress and scoped audit files.
def test_cli_benchmark_on_project(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    write_book(tmp_path / "chapters", {"00-intro.txt": "Title by Someone.", **BOOK})
    (tmp_path / PROGRESS_NAME).write_text(
        json.dumps({"registry": {"luna_starwaver": {"name": "Luna Starwaver"}}, "aliases": {}}), encoding="utf-8"
    )
    (tmp_path / SCOPED_AUDIT_NAME).write_text(json.dumps({"records": []}), encoding="utf-8")
    assert main([str(tmp_path), "--chunk-chars", "200"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["chapters"] == 3 and result["chunk_plan"]["exactly_once"] is True
    assert {"index_s", "group_s", "resolve_s", "contexts_s", "plan_s", "reconcile_s"} <= set(result["timings"])


# ##################################################################
# test index scales near linearly
# a 40x larger synthetic book indexes, groups, plans and reconciles well inside a generous bound.
def test_benchmark_synthetic_book_scales(tmp_path: Path) -> None:
    names = ["Ren", "Luna", "Julius", "Taro", "Leora", "Min Chen", "Aster Blackwood", "Quill"]
    book = {
        f"{n:04d}-part.txt": "\n\n".join(
            " ".join(
                f"{names[(n + p + s) % len(names)]} spoke to {names[(n + s) % len(names)]} about part {n}."
                for s in range(6)
            )
            for p in range(5)
        )
        + "\n"
        for n in range(1, 161)
    }
    chapters = write_book(tmp_path, book)
    clock = time.perf_counter()
    cast = build_cast_index(chapters, {}, {}, [])
    plan = build_chunk_plan(cast.source, 20_000)
    report = reconcile(cast, [Claim("c", ("Ren", "Min Chen"))], plan)
    assert time.perf_counter() - clock < 60
    assert len(cast.source.units) == 160 * 30 and plan.coverage["exactly_once"]
    assert report["occurrences"]["retained"] + report["occurrences"]["dropped"] == len(cast.source.occurrence_label)
