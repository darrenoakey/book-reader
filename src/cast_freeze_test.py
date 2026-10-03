"""Filesystem and source-evidence tests for frozen whole-book cast preparation."""

import hashlib
import json
import re
import shutil
import tempfile
import wave
from pathlib import Path

import pytest
from PIL import Image

from src.cast_freeze import (
    ANCHOR_IDS,
    CONTEXT_PROPOSALS_NAME,
    CONTEXT_RESOLUTION_V2_CONTRACT,
    CONTEXT_RESOLUTION_V2_NAME,
    CONTEXT_V2_STAGE,
    DISTINCT_VERDICT,
    MANIFEST_NAME,
    NATIVE_REVIEW,
    NEW_IDENTITY_AUDIT_NAME,
    PROGRESS_NAME,
    PROPOSALS_NAME,
    REJECTIONS_NAME,
    ROOT_APPROVAL_CONTRACT,
    ROOT_APPROVAL_NAME,
    ROOT_APPROVAL_STAGE,
    SCOPED_AUDIT_NAME,
    SOURCE_RETIREMENT_CONTRACT,
    SOURCE_RETIREMENT_NAME,
    SOURCE_RETIREMENT_STAGE,
    SOURCE_REVIEW,
    active_quality_pending,
    adapt_context_resolution_v2,
    apply_alias_audit,
    asset_hashes,
    cache_model_records,
    candidate_coverage_ledger,
    classification_offer_fingerprint,
    collect_classifications,
    context_safe_batch,
    discovery_schema,
    immutable_evidence_units,
    immutable_name_references,
    ingest_context_quality_proposals,
    ingest_context_resolution_v2,
    ingest_root_approvals,
    ingest_source_reviewed_retirements,
    materialize_classifications,
    memoized_model_ask,
    mention_scoped_audit_index,
    partition_classification_chunk,
    prepared_profile,
    proposal_sha,
    record_rejected_discovery,
    refresh_alias_audit,
    registry_digest,
    restore_cached_model_records,
    source_label_present,
    validate_classification_chunk,
    validate_preparation_coverage,
    variant_draft_actor,
    variant_root_approval_gate,
    variant_root_duplicate_blockers,
    verify_frozen_cast,
)
from src.data_recovery import OperationalError, RecoveryLedger
from src.epub_extract import get_output_dir
from src.hour_runner import source_fingerprint


# ##################################################################
# write real voice WAV
# writes an actual PCM file so freeze validation binds a usable local media artifact rather than a placeholder byte string.
def write_voice(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b"\0\0" * 240)


# ##################################################################
# write frozen project
# creates an isolated complete source, profiles, WAVs, and valid PNG references using the real project filesystem convention.
def write_frozen_project(root: Path) -> tuple[Path, Path]:
    source = root / f"cast_freeze_{root.name}.txt"
    source.write_text("Ren said, Hello. Ron Blackfire answered, Welcome.", encoding="utf-8")
    project = get_output_dir(source)
    chapters = project / "chapters"
    chapters.mkdir(parents=True)
    (chapters / "00-intro.txt").write_text("Frozen Book by Tester, narrated by Narrator", encoding="utf-8")
    (chapters / "01-part.txt").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    characters = {
        actor_id: {"name": actor_id.replace("_", " ").title(), "bio": "source profile", "look": "ordinary human"}
        for actor_id in ANCHOR_IDS
    }
    characters["narrator"]["look"] = ""
    voices = {actor_id: {"description": "clear source-grounded voice"} for actor_id in ANCHOR_IDS}
    appearances = {
        actor_id: "ordinary human with source-described neutral appearance"
        for actor_id in ANCHOR_IDS
        if actor_id != "narrator"
    }
    breeze = {}
    for actor_id in ANCHOR_IDS:
        wav = project / "voices" / f"{actor_id}.wav"
        write_voice(wav)
        breeze[actor_id] = {
            "ref_wav": str(wav.relative_to(project)),
            "ref_text": "Listen.",
            "description": voices[actor_id]["description"],
        }
        if actor_id != "narrator":
            portrait = project / "refs" / f"{actor_id}.png"
            portrait.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(31, 72, 123)).save(portrait)
    for name, value in (
        ("characters.json", characters),
        ("voices.json", voices),
        ("appearances.json", appearances),
        ("breeze_voices.json", breeze),
    ):
        (project / name).write_text(json.dumps(value), encoding="utf-8")
    manifest = {
        "version": 1,
        "source_sha256": source_fingerprint(source),
        "chapter_sha256": {"01-part.txt": hashlib.sha256((chapters / "01-part.txt").read_bytes()).hexdigest()},
        "actors": characters,
        "approved_aliases": {actor_id: actor_id for actor_id in ANCHOR_IDS},
        "inactive_legacy_ids": [],
        "asset_hashes": asset_hashes(project, ANCHOR_IDS),
    }
    (project / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return source, project


# ##################################################################
# test exhaustive candidate schema
# constrains native output to one classification for every locally-derived lexical candidate.
def test_discovery_schema_requires_exact_complete_candidate_classification() -> None:
    candidates = [
        {"id": "p0000", "label": "Han", "ref_ids": ["c00s00000n000"]},
        {"id": "p0001", "label": "The", "ref_ids": ["c00s00001n000"]},
    ]
    schema = discovery_schema(["ren"], candidates)["properties"]["classifications"]
    assert schema["minItems"] == schema["maxItems"] == 2
    branches = schema["items"]["oneOf"]
    assert len(branches) == 2
    assert {branch["oneOf"][0]["properties"]["candidate_id"]["enum"][0] for branch in branches} == {"p0000", "p0001"}
    assert branches[0]["oneOf"][0]["properties"]["status"]["enum"] == ["new"]
    assert branches[0]["oneOf"][2]["properties"]["identity"]["enum"] == ["none"]


# ##################################################################
# test classifications preserve exact labels
# materializes participants locally, while omissions cannot pass an exhaustive candidate ledger.
def test_classifications_materialize_exact_labels_and_reject_omissions() -> None:
    with tempfile.TemporaryDirectory() as directory:
        chapter = Path(directory) / "01-part.txt"
        chapter.write_text("Foam Xiao waved. Aster Blackwood spoke. The beasts ran.", encoding="utf-8")
        units = immutable_evidence_units([chapter])
        candidates = candidate_coverage_ledger(units, {}, {})
        response = {
            "classifications": [
                {
                    "candidate_id": candidate["id"],
                    "status": "new",
                    "identity": candidate["id"],
                    "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]],
                }
                if candidate["label"] in {"Foam Xiao", "Aster Blackwood"}
                else {
                    "candidate_id": candidate["id"],
                    "status": "non_character",
                    "identity": "none",
                    "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]],
                }
                for candidate in candidates
            ]
        }
        discoveries, classifications = materialize_classifications(response, units, candidates, {}, {})
        assert {entry["id"] for entry in discoveries} == {"foam_xiao", "aster_blackwood"}
        assert len(classifications) == len(candidates)
        response["classifications"].pop()
        with pytest.raises(ValueError, match="omitted or duplicated"):
            materialize_classifications(response, units, candidates, {}, {})


# ##################################################################
# test rejected discovery audit
# keeps the entire native response and immutable citations rather than truncating the data needed to diagnose a rejected batch.
def test_rejected_discovery_audit_keeps_full_response_and_units() -> None:
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        chapter = project / "01-part.txt"
        chapter.write_text("Ren said, Hello.", encoding="utf-8")
        units = immutable_evidence_units([chapter])
        response = "x" * 1200
        record_rejected_discovery(project, 0, [chapter], units, response, ValueError("invalid alias"), 1)
        saved = json.loads((project / REJECTIONS_NAME).read_text(encoding="utf-8"))
        assert saved["response"] == response
        assert saved["attempt"] == 1
        assert saved["evidence_units"] == units
        assert saved["chapters"] == ["01-part.txt"]


# ##################################################################
# test resumable coverage
# rejects duplicate or skipped batch coverage without discarding any durable registry or discovery buffers.
def test_preparation_coverage_requires_unique_consecutive_hashes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        chapters = []
        for index in range(2):
            chapter = root / f"{index + 1:02d}-part.txt"
            chapter.write_text(f"Chapter {index}.", encoding="utf-8")
            chapters.append(chapter)
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in chapters}
        progress = {"next_chapter": 2, "completed_batches": [{"start": 0, "end": 2, "chapter_sha256": hashes}]}
        validate_preparation_coverage(progress, chapters)
        progress["completed_batches"].append({"start": 2, "end": 2, "chapter_sha256": {}})
        with pytest.raises(RuntimeError, match="unique consecutive"):
            validate_preparation_coverage(progress, chapters)


# ##################################################################
# test context-safe source batch
# reduces only the number of complete consecutive chapters when the compact registry and source would exceed native context, never truncating source text.
def test_context_safe_batch_reduces_without_source_truncation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        chapters = []
        for index in range(3):
            chapter = root / f"{index + 1:02d}-part.txt"
            chapter.write_text("X. " * 1_000, encoding="utf-8")
            chapters.append(chapter)
        batch, prompt = context_safe_batch(chapters, 0, {"narrator": {"name": "Narrator"}}, {})
        assert [path.name for path in batch] == ["01-part.txt", "02-part.txt"]
        assert len(prompt) <= 53_536
        chapters[0].write_text("X " * 30_000, encoding="utf-8")
        # an oversized lone chapter is returned whole (never truncated) for immutable-unit windowing
        oversized, no_prompt = context_safe_batch(chapters, 0, {"narrator": {"name": "Narrator"}}, {})
        assert oversized == [chapters[0]] and no_prompt == ""


# ##################################################################
# test audit aliases flatten transitively
# preserves legacy actor files while resolving a source-proven alias chain to one approved canonical identity.
def test_alias_audit_flattens_transitive_merges_without_guessing() -> None:
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        source = "Gene taunted Jean, the tiger boy."
        (project / "cast_alias_audit.json").write_text(
            json.dumps(
                [
                    {"alias": "gene", "canonical": "jean", "evidence": ["Gene taunted Jean"], "decision": "merge"},
                    {
                        "alias": "jean",
                        "canonical": "tiger_boy",
                        "evidence": ["Jean, the tiger boy"],
                        "decision": "merge",
                    },
                    {"alias": "jin", "canonical": "jean", "evidence": ["Gene taunted Jean"], "decision": "ambiguous"},
                ]
            ),
            encoding="utf-8",
        )
        registry = {"gene": {}, "jean": {}, "tiger_boy": {}}
        aliases = {actor_id: actor_id for actor_id in registry}
        inactive, ambiguous = apply_alias_audit(project, source, registry, aliases)
        assert aliases["gene"] == "tiger_boy"
        assert aliases["jean"] == "tiger_boy"
        assert "jin" not in aliases
        assert inactive == {"gene", "jean"}
        assert ambiguous == {"jin"}


# ##################################################################
# test frozen media detects mutation
# validates a real complete project then proves changing an established portrait byte prevents all later-hour production.
def test_verify_frozen_cast_detects_changed_anchor_portrait_bytes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source, project = write_frozen_project(Path(directory))
        try:
            assert verify_frozen_cast(source, project)["source_sha256"] == source_fingerprint(source)
            (project / "refs" / "ren.png").write_bytes(b"changed-original-anchor")
            with pytest.raises(RuntimeError, match="portrait bytes changed"):
                verify_frozen_cast(source, project)
        finally:
            shutil.rmtree(project, ignore_errors=True)


# ##################################################################
# test source label boundary
# proves aliases match whole source labels and never a substring of a longer personal name.
def test_source_label_requires_whole_word_boundary() -> None:
    units = [{"id": "c00s00000", "chapter": "a", "quote": "Foam Xiao spoke; the foamy sea rose."}]
    assert source_label_present("Foam", units)
    assert source_label_present("foam xiao", units)
    assert not source_label_present("Fo", units)
    assert not source_label_present("oam", units)
    assert not source_label_present("Foa", units)
    assert not source_label_present("Xia", units)
    assert source_label_present("Xiao", [{"id": "x", "chapter": "a", "quote": "Hi, Xiao."}])


# ##################################################################
# test additive audit refresh
# applies newly appended verified aliases at cursor 60 without moving progress or rewriting media, and remains identical on a second refresh.
def test_audit_refresh_is_additive_idempotent_and_preserves_cursor(tmp_path: Path) -> None:
    source_text = "Klein Goldrest greeted Klene Goldest. Kai is the lizard boy."
    (tmp_path / "cast_alias_audit.json").write_text(
        json.dumps(
            [
                {"alias": "Klein", "canonical": "k_goldest", "evidence": ["Klein Goldrest"], "decision": "merge"},
                {"alias": "Klene", "canonical": "k_goldest", "evidence": ["Klene Goldest"], "decision": "merge"},
                {"alias": "kai", "canonical": "lizard_boy", "evidence": ["Kai is the lizard boy"], "decision": "merge"},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "characters.json").write_text(
        json.dumps({"k_goldest": {"name": "K Goldest"}, "kai": {"name": "Kai"}, "lizard_boy": {"name": "Lizard Boy"}}),
        encoding="utf-8",
    )
    progress = {
        "next_chapter": 60,
        "completed_batches": [{"start": 0, "end": 60, "chapter_sha256": {"01.txt": "unchanged"}}],
        "registry": {"k_goldest": {"name": "K Goldest"}},
        "aliases": {"k_goldest": "k_goldest", "kai": "kai", "lizard_boy": "lizard_boy"},
        "inactive_legacy_ids": ["old_legacy"],
        "ambiguous_new_ids": [],
        "audit_applied": True,
    }
    inactive, _ = refresh_alias_audit(tmp_path, source_text, progress)
    assert progress["next_chapter"] == 60
    assert progress["completed_batches"] == [{"start": 0, "end": 60, "chapter_sha256": {"01.txt": "unchanged"}}]
    assert progress["aliases"]["klein"] == "k_goldest"
    assert progress["aliases"]["klene"] == "k_goldest"
    assert progress["aliases"]["kai"] == "lizard_boy"
    assert "kloene" not in progress["aliases"]
    assert {"old_legacy", "klein", "klene", "kai"} <= inactive
    before = json.loads(json.dumps(progress, sort_keys=True))
    refresh_alias_audit(tmp_path, source_text, progress)
    assert progress == before


# ##################################################################
# test semantic prefix migration
# installs a separate semantic ledger at zero while retaining an existing structural cursor until each historical chapter is reclassified.
def test_semantic_coverage_migration_requires_prefix_revalidation() -> None:
    from src.cast_freeze import semantic_coverage, validate_semantic_coverage

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        chapters = []
        for index in range(2):
            chapter = root / f"{index + 1:02d}.txt"
            chapter.write_text(f"Han {index}.", encoding="utf-8")
            chapters.append(chapter)
        progress = {
            "next_chapter": 2,
            "completed_batches": [
                {
                    "start": 0,
                    "end": 2,
                    "chapter_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in chapters},
                }
            ],
        }
        coverage = semantic_coverage(progress)
        assert coverage["next_chapter"] == 0
        validate_semantic_coverage(progress, chapters)
        coverage["next_chapter"] = 1
        with pytest.raises(RuntimeError, match="cursor does not match"):
            validate_semantic_coverage(progress, chapters)


# ##################################################################
# test bounded candidate chunks
# retains every candidate exactly once while limiting a native schema response to a bounded cardinality.
def test_classification_chunks_cover_ledger_without_overlap() -> None:
    from src.cast_freeze import CLASSIFICATION_CHUNK_SIZE, classification_chunks

    candidates = [
        {"id": f"p{index:04d}", "label": f"Name{index}", "ref_ids": [f"c00s{index:05d}n000"]}
        for index in range(CLASSIFICATION_CHUNK_SIZE * 2 + 1)
    ]
    chunks = classification_chunks(candidates)
    assert [len(chunk) for chunk in chunks] == [CLASSIFICATION_CHUNK_SIZE, CLASSIFICATION_CHUNK_SIZE, 1]
    assert [candidate["id"] for chunk in chunks for candidate in chunk] == [candidate["id"] for candidate in candidates]


# ##################################################################
# test unresolved and alias evidence fail closed
# prevents ambiguous people from checkpointing and requires a new spelling link to carry distinct witnesses for both source labels.
def test_classification_blocks_ambiguity_and_requires_distinct_alias_witnesses() -> None:
    with tempfile.TemporaryDirectory() as directory:
        chapter = Path(directory) / "01-part.txt"
        chapter.write_text("June approached. Jun appeared with a monkey.", encoding="utf-8")
        units = immutable_evidence_units([chapter])
        candidates = candidate_coverage_ledger(units, {}, {})
        by_label = {candidate["label"]: candidate for candidate in candidates}
        june, jun = by_label["June"], by_label["Jun"]
        records = []
        for candidate in candidates:
            status, identity = (
                ("new", june["id"])
                if candidate == june
                else (("known", june["id"]) if candidate == jun else ("non_character", "none"))
            )
            evidence = [candidate["ref_ids"][0].rsplit("n", 1)[0]]
            if candidate == jun:
                evidence.append(june["ref_ids"][0].rsplit("n", 1)[0])
            records.append(
                {"candidate_id": candidate["id"], "status": status, "identity": identity, "evidence_unit_ids": evidence}
            )
        discoveries, _ = materialize_classifications({"classifications": records}, units, candidates, {}, {})
        june_discovery = next(item for item in discoveries if item["id"] == "june")
        assert "Jun" in june_discovery["aliases"]
        assert "June approached." in june_discovery["look_facts"]
        records[0]["status"], records[0]["identity"] = "ambiguous", "none"
        with pytest.raises(RuntimeError, match="remains unresolved"):
            materialize_classifications({"classifications": records}, units, candidates, {}, {})


# ##################################################################
# test discourse prefixes never form aliases
# removes only grammar-led compounds such as As Ren while retaining the legitimate lexical Ren witness.
def test_discourse_prefix_does_not_form_prose_name_compound() -> None:
    from src.cast_freeze import immutable_name_references

    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "As Ren watched, Aster Blackwood arrived."}]
    labels = {reference["label"] for reference in immutable_name_references(units).values()}
    assert "As Ren" not in labels
    assert {"Ren", "Aster Blackwood"} <= labels


# ##################################################################
# test grammar labels excluded before semantic ledger
# treats capitalized pronouns and indefinite grammar words as non-entity lexical material while retaining real names for explicit classification.
def test_pronoun_only_words_are_not_name_candidates() -> None:
    from src.cast_freeze import immutable_name_references

    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "Someone warned Ren. He followed Sora."}]
    labels = {reference["label"] for reference in immutable_name_references(units).values()}
    assert not labels.intersection({"Someone", "He"})
    assert {"Ren", "Sora"} <= labels


# ##################################################################
# test suffix-only fragments excluded
# removes a surname/title fragment occurring only inside a longer capitalized label, while preserving a first-name candidate with source evidence.
def test_suffix_only_capitalized_fragment_is_not_a_candidate() -> None:
    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "Ron Blackfire met the Gold Crest airship."}]
    labels = {candidate["label"] for candidate in candidate_coverage_ledger(units, {}, {})}
    assert "Crest" not in labels
    assert "Ron" in labels


# ##################################################################
# test house labels cannot take actor identity
# rejects a clan or house phrase mapped to a person even when the actor's name shares one lexical component.
def test_house_label_cannot_be_known_actor_alias() -> None:
    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "The Gold Crest airship carried Klein Goldest."}]
    candidates = candidate_coverage_ledger(units, {"k_goldest": {"name": "Klein Goldest"}}, {})
    gold_crest = next(candidate for candidate in candidates if candidate["label"] == "Gold Crest")
    response = {
        "classifications": [
            {
                "candidate_id": candidate["id"],
                "status": "known",
                "identity": "k_goldest",
                "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]],
            }
            if candidate == gold_crest
            else {
                "candidate_id": candidate["id"],
                "status": "non_character",
                "identity": "none",
                "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]],
            }
            for candidate in candidates
        ]
    }
    with pytest.raises(ValueError, match="unapproved alias"):
        materialize_classifications(response, units, candidates, {"k_goldest": {"name": "Klein Goldest"}}, {})


# ##################################################################
# test full names lead candidate ownership
# orders source-qualified full names before their components so new actor IDs remain Foam Xiao and Aster Blackwood rather than shortened fragments.
def test_full_name_candidates_precede_short_components() -> None:
    units = [
        {
            "id": "c00s00000",
            "chapter": "01.txt",
            "quote": "Foam Xiao and Aster Blackwood arrived. Foam and Aster followed.",
        }
    ]
    labels = [candidate["label"] for candidate in candidate_coverage_ledger(units, {}, {})]
    assert labels.index("Foam Xiao") < labels.index("Foam")
    assert labels.index("Aster Blackwood") < labels.index("Aster")


# ##################################################################
# test approved full names carry fixed owner
# labels already audited to an established actor are visibly fixed in the native ledger rather than left eligible for a spurious new identity.
def test_candidate_ledger_marks_approved_full_name_owner() -> None:
    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "Klein Goldest confronted Ren."}]
    candidates = candidate_coverage_ledger(
        units, {"k_goldest": {"name": "Klein Goldest"}}, {"klein_goldest": "k_goldest"}
    )
    assert (
        next(candidate for candidate in candidates if candidate["label"] == "Klein Goldest")["known_owner"]
        == "k_goldest"
    )


# ##################################################################
# test conjunction never forms a person compound
# excludes grammar-plus-name spans such as But Cass while retaining the real Cass label for source-grounded classification.
def test_conjunction_does_not_form_person_compound() -> None:
    from src.cast_freeze import immutable_name_references

    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "But Cass answered Ren."}]
    labels = {reference["label"] for reference in immutable_name_references(units).values()}
    assert "But Cass" not in labels
    assert {"Cass", "Ren"} <= labels


# ##################################################################
# test function-word compound eligibility
# prevents every grammatical determiner, conjunction, or preposition from becoming a multiword person label while retaining its following role or name token.
def test_function_word_prefixes_never_form_multiword_candidates() -> None:
    from src.cast_freeze import immutable_name_references

    units = [
        {
            "id": "c00s00000",
            "chapter": "01.txt",
            "quote": "The Ceremony Master greeted Ren. In Luna Starwaver's hall, Cass waited.",
        }
    ]
    labels = {reference["label"] for reference in immutable_name_references(units).values()}
    assert not {"The Ceremony Master", "In Luna", "In Luna Starwaver"}.intersection(labels)
    assert {"Ceremony Master", "Luna Starwaver", "Cass"} <= labels


# ##################################################################
# test lowercase lexical usage is not a standalone name
# excludes a sentence-initial ordinary word when the exact lower-case lexical form appears in source, while preserving a name component of a full label.
def test_lowercase_lexical_usage_excludes_single_word_candidate() -> None:
    units = [
        {
            "id": "c00s00000",
            "chapter": "01.txt",
            "quote": "Keep walking. Please keep walking. Foam Xiao arrived. Foam stayed.",
        }
    ]
    labels = {candidate["label"] for candidate in candidate_coverage_ledger(units, {}, {})}
    assert "Keep" not in labels
    assert {"Foam Xiao", "Foam"} <= labels


# ##################################################################
# test historical prefix semantic migration accepts chapter sixty introductions
# reads the real frozen source and durable cursor to prove source-backed Han, Sora, and Jun remain eligible for new identities while replaying a pre-cursor prefix.
def test_historical_semantic_migration_keeps_chapter_sixty_new_candidates_eligible() -> None:
    source = Path("/Users/darrenoakey/src/book-reader/incoming/weakest_beast_tamer.txt")
    project = Path("/Users/darrenoakey/src/book-reader/output/weakest_beast_tamer")
    assert source.is_file(), "mandatory real weakest-beast-tamer source fixture is unavailable"
    assert project.is_dir(), "mandatory real weakest-beast-tamer project fixture is unavailable"
    from src.hour_runner import source_chapters

    # Immutable pre-batch-61/62 snapshot: the live cursor keeps advancing, so the historical phase is read from the frozen QA backup.
    snapshot = json.loads((project / "qa-new-identity-before-correction.json").read_text(encoding="utf-8"))
    progress = snapshot["progress"]
    assert progress["version"] == 3 and progress["next_chapter"] == 60
    assert progress["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(snapshot["alias_audit"]) == 25, "historical alias audit snapshot changed"
    _, _, chapters = source_chapters(source, project)
    batch, _ = context_safe_batch(chapters, 59, progress["registry"], progress["aliases"])
    candidates = candidate_coverage_ledger(immutable_evidence_units(batch), progress["registry"], progress["aliases"])
    labels = {candidate["label"] for candidate in candidates}
    assert {"Han", "Sora", "Jun", "June"} <= labels
    assert not {"han", "sora", "june"}.intersection(progress["registry"])
    schema = discovery_schema(list(progress["registry"]), candidates, allow_new=True)
    han_id = next(candidate["id"] for candidate in candidates if candidate["label"] == "Han")
    han = next(
        branch
        for branch in schema["properties"]["classifications"]["items"]["oneOf"]
        if "oneOf" in branch and branch["oneOf"][0]["properties"]["candidate_id"]["enum"] == [han_id]
    )
    assert han["oneOf"][0]["properties"]["status"]["enum"] == ["new"]


# ##################################################################
# test full-name component link is local source proof
# permits Aster to target the source-qualified Aster Blackwood candidate without requiring the model to duplicate a second witness already encoded by the full lexical span.
def test_full_name_component_alias_uses_local_full_span_proof() -> None:
    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "Aster Blackwood arrived. Aster charged."}]
    candidates = candidate_coverage_ledger(units, {}, {})
    full = next(candidate for candidate in candidates if candidate["label"] == "Aster Blackwood")
    short = next(candidate for candidate in candidates if candidate["label"] == "Aster")
    response = {
        "classifications": [
            {
                "candidate_id": candidate["id"],
                "status": "new",
                "identity": candidate["id"],
                "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]],
            }
            if candidate == full
            else (
                {
                    "candidate_id": candidate["id"],
                    "status": "known",
                    "identity": full["id"],
                    "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]],
                }
                if candidate == short
                else {
                    "candidate_id": candidate["id"],
                    "status": "non_character",
                    "identity": "none",
                    "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]],
                }
            )
            for candidate in candidates
        ]
    }
    discoveries, _ = materialize_classifications(response, units, candidates, {}, {})
    assert next(item for item in discoveries if item["id"] == "aster_blackwood")["aliases"] == ["Aster"]


# ##################################################################
# test bounded prompt includes complete source narrative
# gives every chunk the complete context-safe batch narrative so name variants can be linked from source facts rather than isolated mention snippets.
def test_discovery_prompt_carries_full_bounded_source_narrative() -> None:
    with tempfile.TemporaryDirectory() as directory:
        chapter = Path(directory) / "01.txt"
        chapter.write_text("Foam Xiao led the cobra team. Fong carried cobra daggers.", encoding="utf-8")
        from src.cast_freeze import discovery_prompt

        prompt = discovery_prompt([chapter], {}, {})
        assert "FULL BOUNDED SOURCE NARRATIVE" in prompt
        assert "Fong carried cobra daggers." in prompt


# ##################################################################
# test source-audited variant context is advisory and full-name preserving
# derives group guidance from the read-only audit file without writing it or turning its short aliases into canonical IDs.
def test_source_audited_variant_context_prefers_full_candidate_owner(tmp_path: Path) -> None:
    from src.cast_freeze import source_audited_variant_context

    (tmp_path / "qa-klein-team-alias-proposal.json").write_text(
        json.dumps({"decisions": [{"alias": ["Foam", "Fong", "Fo", "Fang", "Foam Xiao"], "note": "red cobra"}]}),
        encoding="utf-8",
    )
    candidates = [
        {"id": "p0000", "label": label, "ref_ids": ["c00s00000n000"]}
        for label in ["Foam", "Fong", "Fo", "Fang", "Foam Xiao"]
    ]
    context = source_audited_variant_context(tmp_path, candidates)
    assert "Foam Xiao" in context and "one new owner" in context


# ##################################################################
# test grammatical predecessor cannot suppress a real name
# preserves Han and a role token when their preceding capital word is a grammar prefix, while true proper-name suffix filtering remains available.
def test_grammar_prefix_does_not_mark_following_name_as_suffix_only() -> None:
    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "Only Han maintained composure. The Director spoke."}]
    labels = {candidate["label"] for candidate in candidate_coverage_ledger(units, {}, {})}
    assert {"Han", "Director"} <= labels


# ##################################################################
# test alias witnesses enrich only new owner preparation facts
# carries exact source quotes from an approved new-owner alias into the prepared actor facts without touching established profiles.
def test_new_owner_alias_witnesses_enrich_source_facts() -> None:
    units = [
        {"id": "c00s00000", "chapter": "01.txt", "quote": "Foam Xiao arrived."},
        {"id": "c00s00001", "chapter": "01.txt", "quote": "Foam's scarlet cobra marked his neck."},
    ]
    candidates = [
        {"id": "p0000", "label": "Foam Xiao", "ref_ids": ["c00s00000n000"]},
        {"id": "p0001", "label": "Foam", "ref_ids": ["c00s00001n000"], "audited_target": "p0000"},
    ]
    response = {
        "classifications": [
            {"candidate_id": "p0000", "status": "new", "identity": "p0000", "evidence_unit_ids": ["c00s00000"]},
            {"candidate_id": "p0001", "status": "known", "identity": "p0000", "evidence_unit_ids": ["c00s00001"]},
        ]
    }
    discoveries, _ = materialize_classifications(response, units, candidates, {}, {})
    foam = discoveries[0]
    assert foam["aliases"] == ["Foam"]
    assert "scarlet cobra" in foam["look_facts"]


# ##################################################################
# test action and group clues do not force nonentity
# a living actor may be named in an impact or team phrase, so those source clues remain native eligibility questions rather than a blanket negative schema force.
def test_action_and_group_clues_do_not_force_nonentity() -> None:
    units = [
        {
            "id": "c00s00000",
            "chapter": "01.txt",
            "quote": "The impact of Han and Hammer stopped the beast. Mira team advanced.",
        }
    ]
    candidates = candidate_coverage_ledger(units, {}, {})
    schema = discovery_schema([], candidates)
    branches = schema["properties"]["classifications"]["items"]["oneOf"]
    for label in ("Han", "Mira"):
        candidate = next(item for item in candidates if item["label"] == label)
        branch = next(
            item
            for item in branches
            if "oneOf" in item and item["oneOf"][0]["properties"]["candidate_id"]["enum"] == [candidate["id"]]
        )
        assert branch["oneOf"][0]["properties"]["status"]["enum"] == ["new"]


# ##################################################################
# test historical chapter sixty carries casting facts
# proves the real migration boundary includes the animal and physical context needed for source-grounded new voice and portrait preparation.
def test_historical_chapter_sixty_preserves_casting_context() -> None:
    source = Path("/Users/darrenoakey/src/book-reader/incoming/weakest_beast_tamer.txt")
    project = Path("/Users/darrenoakey/src/book-reader/output/weakest_beast_tamer")
    assert source.is_file(), "mandatory real weakest-beast-tamer source fixture is unavailable"
    assert project.is_dir(), "mandatory real weakest-beast-tamer project fixture is unavailable"
    from src.hour_runner import source_chapters

    _, _, chapters = source_chapters(source, project)
    text = "\n".join(chapters[index].read_text(encoding="utf-8") for index in (59, 60, 61))
    assert "Han" in text and "spider" in text.casefold()
    assert "Sora" in text and "deer" in text.casefold()
    assert "Jun" in text and "monkey" in text.casefold()
    assert "Aster" in text and "rhinoceros" in text.casefold()


# ##################################################################
# scoped mention audit splits candidates
# one audited Young occurrence links to its owner while unscoped Young stays separate and never inherits the alias.
def test_scoped_audit_splits_young_without_global_alias() -> None:
    CH1 = hashlib.sha256(b"chapter one body").hexdigest()
    units = [
        {"id": "c00s00000", "chapter": "ch1.txt", "chapter_sha256": CH1, "quote": "Young Ren smiled."},
        {"id": "c00s00001", "chapter": "ch1.txt", "chapter_sha256": CH1, "quote": "Young stood up. Young left."},
        {
            "id": "c00s00002",
            "chapter": "ch2.txt",
            "chapter_sha256": hashlib.sha256(b"chapter two body").hexdigest(),
            "quote": "Young stood up. Young left.",
        },
    ]
    record = {
        "chapter_sha256": CH1,
        "quote_sha256": hashlib.sha256(units[1]["quote"].encode()).hexdigest(),
        "label": "Young",
        "span_start": 0,
        "canonical": "young_ren",
        "decision": "alias",
        "confidence": 0.9,
        "reason": "context names Ren",
    }
    registry = {"young_ren": {"name": "Young Ren"}}
    plain = candidate_coverage_ledger(units, registry, {})
    assert next(c for c in plain if c["label"] == "Young").get("scoped_audit") is None
    ledger = candidate_coverage_ledger(units, registry, {}, [record])
    young = [c for c in ledger if c["label"] == "Young"]
    scoped = [c for c in young if c.get("scoped_audit")]
    other = [c for c in young if not c.get("scoped_audit")]
    assert len(scoped) == 1 and len(other) == 1
    assert scoped[0]["known_owner"] == "young_ren" and other[0]["known_owner"] is None
    assert all(ref.startswith("c00s00001") for ref in scoped[0]["ref_ids"])
    assert len(scoped[0]["ref_ids"]) == 1  # only the offset-0 Young in the quote, not the second Young
    assert next(c for c in ledger if c["label"] == "Young" and not c.get("scoped_audit"))["known_owner"] is None
    with pytest.raises(ValueError):
        candidate_coverage_ledger(units, registry, {}, [{k: v for k, v in record.items() if k != "span_start"}])
    renamed = [{**u, "chapter": "other.txt"} for u in units]
    assert any(c.get("scoped_audit") for c in candidate_coverage_ledger(renamed, registry, {}, [record]))
    changed = [{**u, "chapter_sha256": "0" * 64} if u["id"] == "c00s00001" else u for u in units]
    assert not any(c.get("scoped_audit") for c in candidate_coverage_ledger(changed, registry, {}, [record]))
    assert not set(scoped[0]["ref_ids"]) & set(other[0]["ref_ids"])
    assert any(c["label"] == "Young Ren" and c["known_owner"] == "young_ren" for c in ledger)
    with pytest.raises(ValueError):
        candidate_coverage_ledger(units, registry, {}, [{**record, "reason": ""}])


# ##################################################################
# scoped adjudication end to end
# an unapproved existing-owner link is decided per mention, persisted with hashes first, and never becomes a global Young alias.
def test_discover_batch_adjudicates_each_mention_and_persists_before_mapping(tmp_path: Path) -> None:
    from src.cast_freeze import SCOPED_AUDIT_NAME, discover_batch

    chapter = tmp_path / "ch1.txt"
    chapter.write_text(
        "Young Ren smiled. Young bowed to the king. The young fox ran. Young sold the fruit.", encoding="utf-8"
    )
    units = immutable_evidence_units([chapter])
    progress = {"registry": {"young_ren": {"name": "Young Ren"}}, "aliases": {}}
    calls: list[str] = []

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        calls.append(prompt)
        schema = response_schema or {}
        if "mentions" in schema["properties"]:
            rows = []
            for line in prompt.splitlines():
                if line.startswith("m") and "mention:" in line:
                    mention_id = line.split()[0]
                    bows = "bowed" in line
                    rows.append(
                        {
                            "mention_id": mention_id,
                            "refers_to_person": "yes" if bows else "no",
                            "candidate_kind": "individual_name" if bows else "nonliving",
                            "decision": "alias" if bows else "non_character",
                            "canonical": "young_ren" if bows else "none",
                            "confidence": 0.9,
                            "reason": "context",
                        }
                    )
            return json.dumps({"mentions": rows})
        out = []
        for option in schema["properties"]["classifications"]["items"]["oneOf"]:
            branches = option.get("oneOf", [option])
            cid = branches[0]["properties"]["candidate_id"]["enum"][0]
            row = next(line for line in prompt.splitlines() if line.startswith(cid + " label="))
            witness = row.split("witnesses: [")[1].split("]")[0]
            label = row.split("label='")[1].split("'")[0]
            statuses = {st: b["properties"] for b in branches for st in b["properties"]["status"]["enum"]}
            if len(branches) == 1:
                status = next(iter(statuses))
            elif label == "Young":
                status = "known" if "known" in statuses else "non_character"
            else:
                status = "new"
            identity = statuses[status]["identity"]["enum"]
            out.append(
                {
                    "candidate_id": cid,
                    "status": status,
                    "identity": "young_ren" if "young_ren" in identity else identity[0],
                    "evidence_unit_ids": ["c00s00001" if label == "Young" and len(branches) > 1 else witness],
                }
            )
        return json.dumps({"classifications": out})

    def cid_label(prompt: str, cid: str) -> str:
        for line in prompt.splitlines():
            if line.startswith(cid + " label="):
                return line.split("label=")[1].split()[0].strip("'")
        return ""

    discoveries, classifications = discover_batch(
        tmp_path, 0, [chapter], units, chapter.read_text(), progress, set(), ask=ask
    )
    assert discoveries == []
    records = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert {r["label"] for r in records} == {"Young"}
    assert {r["decision"] for r in records} == {"alias", "non_character"}
    for r in records:
        assert (
            len(r["chapter_sha256"]) == 64 and len(r["quote_sha256"]) == 64 and r["reason"] and r["confidence"] == 0.9
        )
    aliased = [r for r in records if r["decision"] == "alias"]
    assert len(aliased) == 1 and aliased[0]["canonical"] == "young_ren"
    assert aliased[0]["quote_sha256"] == hashlib.sha256(units[1]["quote"].encode()).hexdigest()
    assert aliased[0]["chapter_sha256"] == hashlib.sha256(chapter.read_bytes()).hexdigest()
    assert aliased[0]["span_start"] == units[1]["quote"].index("Young") and isinstance(aliased[0]["span_start"], int)
    assert any(c["status"] == "known" and c["identity"] == "young_ren" for c in classifications)
    assert any(c["status"] == "non_character" for c in classifications)


# ##################################################################
# monotonic scoped adjudication loop on the production audit snapshot
# a label first classified as a non-character and only proposed as an unapproved known alias in a later classification round is adjudicated rather than failing, while the 53 real audit records stay intact.
def test_discover_batch_adjudicates_second_round_darling_with_production_snapshot(tmp_path: Path) -> None:
    from src.cast_freeze import SCOPED_AUDIT_NAME, discover_batch

    snapshot = Path(__file__).parent / "testdata" / "weakest_scoped_audit_snapshot.json"
    records_before = json.loads(snapshot.read_text())["records"]
    assert len(records_before) == 53
    (tmp_path / SCOPED_AUDIT_NAME).write_text(snapshot.read_text())
    chapter = tmp_path / "ch1.txt"
    chapter.write_text("Young bowed to the king. Mother smiled. Darling, said Mother softly.", encoding="utf-8")
    units = immutable_evidence_units([chapter])
    progress = {
        "registry": {
            "young_ren": {"name": "Young Ren", "bio": "Young Ren is the child whom Mother calls Darling"},
            "mother": {"name": "Mother"},
        },
        "aliases": {},
    }
    darling_rounds = {"count": 0}

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        schema = response_schema or {}
        if "mentions" in schema["properties"]:
            rows = []
            for line in prompt.splitlines():
                if line.startswith("m") and "mention:" in line:
                    rows.append(
                        {
                            "mention_id": line.split()[0],
                            "refers_to_person": "yes",
                            "candidate_kind": "individual_name",
                            "decision": "alias",
                            "canonical": "young_ren",
                            "confidence": 0.9,
                            "reason": "context",
                        }
                    )
            return json.dumps({"mentions": rows})
        out = []
        for option in schema["properties"]["classifications"]["items"]["oneOf"]:
            branches = option.get("oneOf", [option])
            cid = branches[0]["properties"]["candidate_id"]["enum"][0]
            row = next(line for line in prompt.splitlines() if line.startswith(cid + " label="))
            witness = row.split("witnesses: [")[1].split("]")[0]
            label = row.split("label='")[1].split("'")[0]
            statuses = {st: b["properties"] for b in branches for st in b["properties"]["status"]["enum"]}
            if len(branches) == 1:
                status = next(iter(statuses))
            elif label == "Darling":
                darling_rounds["count"] += 1
                status = (
                    "known"
                    if darling_rounds["count"] > 1 and "known" in statuses
                    else "non_character"
                    if "non_character" in statuses
                    else next(iter(statuses))
                )
            elif label == "Young":
                status = "known" if "known" in statuses else "non_character"
            else:
                status = "new"
            identity = statuses[status]["identity"]["enum"]
            target = "young_ren" if "young_ren" in identity else identity[0]
            out.append({"candidate_id": cid, "status": status, "identity": target, "evidence_unit_ids": [witness]})
        return json.dumps({"classifications": out})

    discover_batch(tmp_path, 0, [chapter], units, chapter.read_text(), progress, set(), ask=ask)
    after = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert all(record in after for record in records_before)
    assert any(
        record["label"] == "Darling" and record["decision"] == "alias" and record["canonical"] == "young_ren"
        for record in after
    )


# ##################################################################
# scoped audit freeze attestation
# the freeze manifest attests the exact scoped record set: appended, altered, or removed records fail closed, and only validated alias records become exact references.
def test_freeze_attests_scoped_audit_and_exports_exact_references() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source, project = write_frozen_project(Path(directory))
        try:
            run_scoped_freeze_checks(source, project)
        finally:
            shutil.rmtree(project, ignore_errors=True)


def run_scoped_freeze_checks(source: Path, project: Path) -> None:
    from src.cast_freeze import (
        SCOPED_AUDIT_NAME,
        scoped_audit_attestation,
        validated_scoped_references,
    )

    chapters = sorted((project / "chapters").glob("*.txt"))
    units = immutable_evidence_units(chapters)
    unit = next(u for u in units if "Ron Blackfire" in u["quote"])
    start = unit["quote"].index("Ron")

    def record(**changes: object) -> dict:
        return {
            "chapter_sha256": unit["chapter_sha256"],
            "quote_sha256": hashlib.sha256(unit["quote"].encode()).hexdigest(),
            "label": "Ron",
            "span_start": start,
            "canonical": "ron_blackfire",
            "decision": "alias",
            "confidence": 0.9,
            "reason": "context names Ron Blackfire",
            **changes,
        }

    manifest_path = project / MANIFEST_NAME
    legacy = json.loads(manifest_path.read_text())
    assert "scoped_audit" not in legacy and verify_frozen_cast(source, project)  # legacy manifest attests no records
    assert validated_scoped_references(source, project) == []
    records = [record(), record(label="Ren", span_start=0, decision="non_character", canonical="none", reason="x")]
    (project / SCOPED_AUDIT_NAME).write_text(json.dumps({"records": records}))
    with pytest.raises(RuntimeError, match="attest"):
        verify_frozen_cast(source, project)  # records appeared after an empty attestation
    manifest_path.write_text(json.dumps({**legacy, "scoped_audit": scoped_audit_attestation(records)}))
    assert verify_frozen_cast(source, project)
    references = validated_scoped_references(source, project)
    assert references == [
        {k: records[0][k] for k in ("chapter_sha256", "quote_sha256", "label", "span_start", "canonical")}
    ]
    assert (
        json.loads(manifest_path.read_text())["approved_aliases"] == legacy["approved_aliases"]
        and "ron" not in legacy["approved_aliases"]
    )
    reordered = list(reversed(records))
    assert scoped_audit_attestation(reordered) == scoped_audit_attestation(records)
    (project / SCOPED_AUDIT_NAME).write_text(json.dumps({"records": [records[0]]}))
    with pytest.raises(RuntimeError, match="attest"):
        verify_frozen_cast(source, project)
    for bad, message in (
        (record(span_start=start + 1), "exact source mention"),
        (record(canonical="not_frozen"), "outside the frozen cast"),
        (record(quote_sha256="0" * 64), "does not exist"),
        (record(confidence=0.5), "confidence"),
    ):
        (project / SCOPED_AUDIT_NAME).write_text(json.dumps({"records": [bad]}))
        manifest_path.write_text(json.dumps({**legacy, "scoped_audit": scoped_audit_attestation([bad])}))
        with pytest.raises(RuntimeError, match=message):
            validated_scoped_references(source, project)


# ##################################################################
# adjudication prompt carries canonical prior facts and the full bounded scene
# the owner's profile and source-derived facts plus a same-chapter scene beyond one neighbouring sentence reach every generic adjudication call.
def test_adjudication_prompt_includes_prior_facts_and_bounded_scene(tmp_path: Path) -> None:
    from src.cast_freeze import (
        ADJUDICATION_SCENE_CHARS,
        adjudicate_pending_mentions,
        immutable_name_references,
    )

    chapter = tmp_path / "ch1.txt"
    filler = " ".join(f"Filler sentence number {i} about nothing." for i in range(200))
    chapter.write_text(
        f"Opening remark about the harbour. Elena of the north was proud. Marta wept. Young bowed low. Later Marta said farewell. {filler}",
        encoding="utf-8",
    )
    units = immutable_evidence_units([chapter])
    registry = {
        "young_ren": {
            "name": "Young Ren",
            "bio": "a boy raised by Elena of the north",
            "look": "freckled",
            "facts": {"voice": ["soft tenor"], "look": ["scar on chin"]},
        }
    }
    candidate = {
        "id": "p0000",
        "label": "Young",
        "ref_ids": [r for r, ref in immutable_name_references(units).items() if ref["label"] == "Young"],
    }
    prompts: list[str] = []

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        prompts.append(prompt)
        return json.dumps(
            {
                "mentions": [
                    {
                        "mention_id": "m0",
                        "refers_to_person": "unclear",
                        "candidate_kind": "unclear",
                        "decision": "ambiguous",
                        "canonical": "none",
                        "confidence": 0.5,
                        "reason": "x",
                    }
                ]
            }
        )

    adjudicate_pending_mentions(tmp_path, [{"candidate": candidate, "proposed": "young_ren"}], units, registry, ask)
    prompt = prompts[0]
    assert (
        "a boy raised by Elena of the north" in prompt
        and "soft tenor" in prompt
        and "scar on chin" in prompt
        and "freckled" in prompt
    )
    scene = prompt.split("bounded scene: ")[1].splitlines()[0]
    assert (
        "Opening remark about the harbour." in scene
        and "Marta wept." in scene
        and "Later Marta said farewell." in scene
    )
    assert len(scene) <= ADJUDICATION_SCENE_CHARS + len(units[3]["quote"])
    assert "Filler sentence number 199" not in scene


# ##################################################################
# start-6 regression: component-overlap and bounded-roster owners, history-preserving re-adjudication
# replays the production start-6 shape: cached ambiguous Ren Dove and Mom records (decided before any owner was offered) are re-judged per mention with Ren / the bounded cast, keep their prior reason as history, and never leak to other mentions or other source bytes.
def test_start6_ren_dove_mom_and_third_context_readjudicate_with_history(tmp_path: Path) -> None:
    from src.cast_freeze import (
        SCOPED_AUDIT_NAME,
        adjudicate_pending_mentions,
        adjudication_owners,
        discover_batch,
    )

    registry = {
        "ren": {"name": "Ren", "bio": "a ten year old boy"},
        "mother": {"name": "Mother", "bio": "Ren's parent, referred to as 'Mom' by her son"},
        "narrator": {"name": "Narrator"},
    }
    assert adjudication_owners("Ren Dove", registry) == ["ren"]  # component overlap, not the whole cast
    assert adjudication_owners("Mom", registry) == ["mother"]  # exact profile evidence, never roster filler
    assert adjudication_owners("Mom", registry, proposed="ren") == ["mother"]  # model proposal is not evidence
    chapter_one, chapter_two = tmp_path / "ch1.txt", tmp_path / "ch2.txt"
    chapter_one.write_text("Ren Dove hid under the covers. Ren whispered, I love you, Mom.", encoding="utf-8")
    chapter_two.write_text("Someone shouted at Mom about the broken cart.", encoding="utf-8")
    units = immutable_evidence_units([chapter_one, chapter_two])
    stale_reason = "only the full name is shown; no owner was offered"

    def cached(label: str, quote: str) -> dict:
        unit = next(u for u in units if u["quote"] == quote)
        return {
            "chapter_sha256": unit["chapter_sha256"],
            "quote_sha256": hashlib.sha256(quote.encode()).hexdigest(),
            "label": label,
            "span_start": quote.index(label),
            "canonical": "none",
            "decision": "ambiguous",
            "confidence": 0.5,
            "reason": stale_reason,
        }

    legacy = [cached("Ren Dove", units[0]["quote"]), cached("Mom", units[1]["quote"]), cached("Mom", units[2]["quote"])]
    (tmp_path / SCOPED_AUDIT_NAME).write_text(json.dumps({"records": legacy}))
    ledger = candidate_coverage_ledger(units, registry, {}, legacy)
    assert {c["label"] for c in ledger if c.get("scoped_stale")} == {
        "Ren Dove",
        "Mom",
    }  # cached ambiguity is stale, not final
    # a record bound to other bytes or another offset is not applied at all: the source guards still decide scope
    moved = [{**legacy[1], "span_start": legacy[1]["span_start"] + 1}, {**legacy[2], "chapter_sha256": "0" * 64}]
    assert not any(
        c.get("scoped_audit") for c in candidate_coverage_ledger(units, registry, {}, moved) if c["label"] == "Mom"
    )

    adjudications: list[str] = []

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        schema = response_schema or {}
        if "mentions" in schema["properties"]:
            adjudications.append(prompt)
            assert "previous decision" in prompt and stale_reason in prompt
            rows = []
            for line in prompt.splitlines():
                if line.startswith("m") and "mention:" in line:
                    allowed = schema["properties"]["mentions"]["items"]["properties"]["canonical"]["enum"]
                    if "shouted" in line:
                        rows.append(
                            {
                                "mention_id": line.split()[0],
                                "refers_to_person": "no",
                                "candidate_kind": "nonliving",
                                "decision": "non_character",
                                "canonical": "none",
                                "confidence": 0.9,
                                "reason": "a stranger's mom, not the cast",
                            }
                        )
                    else:
                        target = "ren" if "Dove" in line else "mother"
                        assert target in allowed
                        rows.append(
                            {
                                "mention_id": line.split()[0],
                                "refers_to_person": "yes",
                                "candidate_kind": "individual_name",
                                "decision": "alias",
                                "canonical": target,
                                "confidence": 0.9,
                                "reason": "scene and prior facts support it",
                            }
                        )
            return json.dumps({"mentions": rows})
        out = []
        for option in schema["properties"]["classifications"]["items"]["oneOf"]:
            branches = option.get("oneOf", [option])
            cid = branches[0]["properties"]["candidate_id"]["enum"][0]
            row = next(line for line in prompt.splitlines() if line.startswith(cid + " label="))
            label = row.split("label='")[1].split("'")[0]
            witness = row.split("witnesses: [")[1].split("]")[0]
            statuses = {st: b["properties"] for b in branches for st in b["properties"]["status"]["enum"]}
            if len(branches) == 1:
                status = next(iter(statuses))
            elif label in {"Ren Dove", "Mom"}:
                status = "known" if "known" in statuses else "ambiguous"
            else:
                status = "non_character"
            identity = statuses[status]["identity"]["enum"]
            if label == "Ren Dove" and status == "known":
                identity = ["ren"]
            elif label == "Mom" and status == "known":
                identity = ["mother"]
            out.append({"candidate_id": cid, "status": status, "identity": identity[0], "evidence_unit_ids": [witness]})
        return json.dumps({"classifications": out})

    progress = {"registry": registry, "aliases": {}}
    discoveries, classifications = discover_batch(
        tmp_path, 0, [chapter_one, chapter_two], units, "", progress, set(), ask=ask
    )
    assert discoveries == [] and adjudications
    records = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert len(records) == 3  # replaced in place of the cached records, never duplicated
    by_quote = {(r["label"], r["quote_sha256"]): r for r in records}
    dove = by_quote[("Ren Dove", hashlib.sha256(units[0]["quote"].encode()).hexdigest())]
    mom_vocative = by_quote[("Mom", hashlib.sha256(units[1]["quote"].encode()).hexdigest())]
    mom_third = by_quote[("Mom", hashlib.sha256(units[2]["quote"].encode()).hexdigest())]
    assert (dove["decision"], dove["canonical"], dove["owners"]) == ("alias", "ren", ["ren"])
    assert (mom_vocative["decision"], mom_vocative["canonical"], mom_vocative["owners"]) == (
        "alias",
        "mother",
        ["mother"],
    )
    assert (mom_third["decision"], mom_third["canonical"]) == (
        "non_character",
        "none",
    )  # the same label elsewhere is decided on its own scene
    for record in (dove, mom_vocative, mom_third):
        assert record["history"] == [
            {"decision": "ambiguous", "canonical": "none", "confidence": 0.5, "reason": stale_reason}
        ]
    assert any(c["status"] == "known" and c["identity"] == "ren" for c in classifications) and any(
        c["identity"] == "mother" for c in classifications
    )
    # idempotent: decisions that already saw every plausible owner are never re-asked
    calls = len(adjudications)
    candidates = candidate_coverage_ledger(units, registry, {}, records)
    assert not any(c.get("scoped_stale") for c in candidates)
    adjudicate_pending_mentions(
        tmp_path,
        [
            {
                "candidate": next(
                    c for c in candidates if c["label"] == "Mom" and c["scoped_audit"]["decision"] == "alias"
                ),
                "proposed": "mother",
            }
        ],
        units,
        registry,
        ask,
    )
    assert len(adjudications) == calls
    # an ambiguous answer that already saw every owner stays final with its reason; it is not re-asked
    stuck = {**mom_vocative, "decision": "ambiguous", "canonical": "none", "owners": ["mother"]}
    assert not any(c.get("scoped_stale") for c in candidate_coverage_ledger(units, registry, {}, [stuck]))


# ##################################################################
# adjudication verdicts must agree with the model's own person answer
# A capitalized continuation may be malformed prose, but component overlap alone is not identity proof; preserve raw output and leave it scoped-ambiguous for an independently evidenced decision.
def test_adjudication_rejects_verdict_contradicting_person_answer(tmp_path: Path) -> None:
    from src.cast_freeze import adjudicate_pending_mentions

    chapter = tmp_path / "ch1.txt"
    chapter.write_text("Ren Dove hid under the covers. Mom wept softly. Mom left.", encoding="utf-8")
    units = immutable_evidence_units([chapter])
    registry = {"ren": {"name": "Ren"}, "mother": {"name": "Mother", "bio": "called Mom by her son"}}
    answers = iter([("yes", "non_character", "none"), ("no", "alias", "mother")])
    ledger = candidate_coverage_ledger(units, registry, {})
    prompts: list[str] = []

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        prompts.append(prompt)
        if response_schema and "mentions" not in response_schema["properties"]:
            return json.dumps(
                {
                    "semantic_type": "prose_fragment",
                    "witness_unit_ids": ["c00s00000"],
                    "reason": "the capitalized continuation is prose",
                }
            )
        person, decision, canonical = next(answers)
        return json.dumps(
            {
                "mentions": [
                    {
                        "mention_id": "m0",
                        "refers_to_person": person,
                        "candidate_kind": "individual_name",
                        "decision": decision,
                        "canonical": canonical,
                        "confidence": 0.95,
                        "reason": "he is Ren's parent",
                    }
                ]
            }
        )

    for label in ("Ren Dove", "Mom"):
        candidate = next(c for c in ledger if c["label"] == label)
        candidate = {**candidate, "ref_ids": candidate["ref_ids"][:1]}
        progressed = adjudicate_pending_mentions(
            tmp_path, [{"candidate": candidate, "proposed": None}], units, registry, ask, binding_progress_only=True
        )
        assert progressed is (label == "Ren Dove")
    records = json.loads((tmp_path / "mention-scoped-audit.json").read_text())["records"]
    assert [(r["label"], r["decision"], r["canonical"]) for r in records] == [
        ("Ren Dove", "non_character", "none"),
        ("Mom", "ambiguous", "none"),
    ]
    assert (
        records[0]["raw_adjudication"]["decision"] == "non_character"
        and records[0]["raw_adjudication"]["refers_to_person"] == "yes"
    )
    assert any(
        "refers_to_person" in prompt and "candidate_kind" in prompt and "endearment" in prompt for prompt in prompts
    )
    assert any("semantic_type" in prompt and "Do not output a decision" in prompt for prompt in prompts)


# ##################################################################
# caretaker quality proposal intake
# Caretaker input is source-bound, quote-relative Unicode scoped, append-only in recovery,
# and cannot approve or alter an actor mapping.
def test_context_quality_proposal_requires_exact_scope_and_preserves_resolution_history(tmp_path: Path) -> None:
    chapter = tmp_path / "01-part_01.txt"
    chapter.write_text("Ren greeted Zed.", encoding="utf-8")
    unit = immutable_evidence_units([chapter])[0]
    scope = {
        "chapter": chapter.name,
        "chapter_sha256": unit["chapter_sha256"],
        "unit_id": unit["id"],
        "quote": unit["quote"],
        "quote_sha256": hashlib.sha256(unit["quote"].encode()).hexdigest(),
        "label": "Ren",
        "span_start": unit["quote"].index("Ren"),
    }
    witness = {key: scope[key] for key in ("chapter", "chapter_sha256", "unit_id", "quote", "quote_sha256")}
    pending = {
        "proposal_id": "quality-ren-1",
        "registry_id": "ren",
        "kind": "garble",
        "status": "pending",
        "scope": scope,
        "witnesses": [witness],
        "note": "review source spelling",
        "resolution": None,
    }
    payload = {"version": 1, "source_sha256": "source-sha", "proposals": [pending]}
    (tmp_path / CONTEXT_PROPOSALS_NAME).write_text(json.dumps(payload), encoding="utf-8")
    recovery = RecoveryLedger(tmp_path)
    assert len(ingest_context_quality_proposals(tmp_path, "source-sha", [chapter], {"ren": {"name": "Ren"}}, recovery)) == 1
    row = recovery.open_pending("cast_quality")[0]
    assert row["evidence"]["proposal_id"] == "quality-ren-1" and row["evidence"]["proposal_sha256"]
    assert active_quality_pending(recovery, {"ren": {"name": "Ren"}}) == [row]
    assert active_quality_pending(recovery, {}) == []
    resolved = {**pending, "status": "resolved", "resolution": {"reason": "caretaker verified source"}}
    (tmp_path / CONTEXT_PROPOSALS_NAME).write_text(
        json.dumps({"version": 1, "source_sha256": "source-sha", "proposals": [resolved]}), encoding="utf-8"
    )
    assert ingest_context_quality_proposals(tmp_path, "source-sha", [chapter], {"ren": {"name": "Ren"}}, recovery) == []
    assert recovery.open_pending("cast_quality") == []
    assert any(entry["severity"] == "resolved" for entry in recovery.entries())
    invalid = {**pending, "scope": {**scope, "span_start": 1}}
    (tmp_path / CONTEXT_PROPOSALS_NAME).write_text(
        json.dumps({"version": 1, "source_sha256": "source-sha", "proposals": [invalid]}), encoding="utf-8"
    )
    with pytest.raises(OperationalError):
        ingest_context_quality_proposals(tmp_path, "source-sha", [chapter], {"ren": {"name": "Ren"}}, recovery)


# ##################################################################
# caretaker context resolution v2 adapter
# Real temp chapters: source identity fails closed; a decision applies only when the existing guards prove it,
# and everything else (new_actor, hold, unproven alias/non_character) is typed pending, never a whole-file failure.
V2_SOURCE_SHA = "b" * 64


def v2_fixture(tmp_path: Path) -> tuple[list[Path], dict]:
    first = tmp_path / "01-part_01.txt"
    second = tmp_path / "02-part_02.txt"
    first.write_text(
        "Ana ran across the yard. Ana Reed shouted at the sky. The Zephyr howled over the wall. "
        "Zed appeared at the gate. Professor, he called in a tense voice. Mom, said Ana.",
        encoding="utf-8",
    )
    second.write_text("Zed carried a lantern. Ana waited at the door.", encoding="utf-8")
    return [first, second], {"ana": {"name": "Ana", "bio": "a scout"}}


def v2_row(chapters: list[Path], chapter: int, label: str, decision: str, target: str, witness_chapter: int) -> dict:
    path = chapters[chapter]
    units = immutable_evidence_units([path])
    unit = next(item for item in units if label in item["quote"])
    other_units = immutable_evidence_units([chapters[witness_chapter]])
    witness = other_units[0] if witness_chapter != chapter else units[max(units.index(unit) - 1, 0)]
    # unit ids are computed on the single chapter file exactly as the caretaker did (c00 prefix)
    assert unit["id"].startswith("c00") and witness["id"].startswith("c00")
    return {
        "chapter_file": path.name,
        "chapter_sha256": unit["chapter_sha256"],
        "main_unit_id": unit["id"],
        "main_unit_quote": unit["quote"],
        "main_unit_quote_sha256": hashlib.sha256(unit["quote"].encode()).hexdigest(),
        "label": label,
        "span_start": unit["quote"].index(label),
        "label_matches_span": True,
        "witnesses": [
            {
                "chapter_file": chapters[witness_chapter].name,
                "chapter_sha256": witness["chapter_sha256"],
                "unit_id": witness["id"],
                "unit_quote": witness["quote"],
                "unit_quote_sha256": hashlib.sha256(witness["quote"].encode()).hexdigest(),
                "role": "identity_fact_witness",
            }
        ],
        "decision": decision,
        "canonical_target": target,
        "target_registered": target in {"ana"},
        "reason": f"caretaker {decision}",
        "provenance": {"author": "caretaker"},
    }


def v2_witness(chapters: list[Path], chapter: int, index: int) -> dict:
    unit = immutable_evidence_units([chapters[chapter]])[index]
    return {
        "chapter_file": chapters[chapter].name,
        "chapter_sha256": unit["chapter_sha256"],
        "unit_id": unit["id"],
        "unit_quote": unit["quote"],
        "unit_quote_sha256": hashlib.sha256(unit["quote"].encode()).hexdigest(),
    }


def v2_draft(chapters: list[Path], actor_id: str = "zed", name: str = "Zed", cited: bool = True) -> dict:
    """A draft shaped like the caretaker's: bio/look text plus (when cited) per-field source citations."""
    draft = {
        "actor_id": actor_id,
        "name": name,
        "kind": "person",
        "bio": "Appears at the gate and later carries a lantern.",
        "look": "Carries a lantern.",
        "status": "DRAFT - not approved",
        "witnesses": [v2_witness(chapters, 0, 3), v2_witness(chapters, 1, 0)],
    }
    if cited:
        draft["citations"] = {"bio": [v2_witness(chapters, 0, 3), v2_witness(chapters, 1, 0)], "look": [v2_witness(chapters, 1, 0)]}
    return draft


def v2_payload(rows: list[dict], sha: str = V2_SOURCE_SHA, drafts: list[dict] | None = None) -> dict:
    return {
        "contract": CONTEXT_RESOLUTION_V2_CONTRACT,
        "source_book_sha256": sha,
        "source_sha256": sha,
        "decisions": ["alias", "new_actor", "non_character", "hold"],
        "rows": rows,
        "unhandled_labels": [],
        "actor_quality_flags": [{"registry_id": "ana", "proposal": "ignored"}],
        "new_actor_drafts": drafts or [],
        "summary": {},
    }


def test_context_resolution_v2_applies_only_proven_and_types_the_rest_pending(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    first_refs = {ref["label"] for ref in immutable_name_references(immutable_evidence_units([chapters[0]])).values()}
    assert {"Ana Reed", "Zephyr", "Zed", "Professor", "Mom"} <= first_refs
    rows = [
        v2_row(chapters, 0, "Ana Reed", "alias", "ana", 1),  # cross-chapter witness, proven by scoped_alias_proof
        v2_row(chapters, 0, "Zephyr", "non_character", "none", 1),  # no plausible owner exists
        v2_row(chapters, 0, "Zed", "new_actor", "zed", 1),
        v2_row(chapters, 0, "Professor", "hold", "none", 0),
        v2_row(chapters, 0, "Mom", "alias", "ana", 0),  # no literal tie to Ana: unproven
    ]
    rows[-1]["witnesses"].append(dict(rows[0]["witnesses"][0]))
    rows[-1]["witnesses"].append(dict(rows[0]["witnesses"][0]))  # a repeated exact unit is one witness
    (tmp_path / CONTEXT_RESOLUTION_V2_NAME).write_text(json.dumps(v2_payload(rows)), encoding="utf-8")
    before = (tmp_path / CONTEXT_RESOLUTION_V2_NAME).read_bytes()
    recovery = RecoveryLedger(tmp_path)

    open_rows = ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, recovery)

    assert sorted(row["code"] for row in open_rows) == [
        "context_alias_unproven",
        "context_hold",
        "context_new_actor",
    ]
    assert all(row["stage"] == CONTEXT_V2_STAGE and row["severity"] == "pending" for row in open_rows)
    assert all(row["evidence"]["witnesses"] for row in open_rows)
    assert all(len(row["evidence"]["witnesses"]) <= 2 for row in open_rows)
    records = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text(encoding="utf-8"))["records"]
    assert sorted((r["label"], r["decision"], r["canonical"]) for r in records) == [
        ("Ana Reed", "alias", "ana"),
        ("Zephyr", "non_character", "none"),
    ]
    assert next(r for r in records if r["label"] == "Ana Reed")["proof"]["literal"] == "shared_name_word"
    # input is never modified, registry/aliases are never touched, and re-ingest is idempotent
    assert (tmp_path / CONTEXT_RESOLUTION_V2_NAME).read_bytes() == before and registry == {
        "ana": {"name": "Ana", "bio": "a scout"}
    }
    ledger_size = len(recovery.entries())
    assert len(ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, recovery)) == 3
    assert len(recovery.entries()) == ledger_size


def test_context_resolution_v2_guards_block_unsafe_application_and_resolve_when_proven(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    rows = [v2_row(chapters, 0, "Ana Reed", "alias", "ana", 1)]
    (tmp_path / CONTEXT_RESOLUTION_V2_NAME).write_text(json.dumps(v2_payload(rows)), encoding="utf-8")
    recovery = RecoveryLedger(tmp_path)
    # owner not registered yet: the alias stays pending and nothing is written to the audit
    pending = ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, {}, {}, recovery)
    assert [row["code"] for row in pending] == ["context_alias_unproven"]
    assert not (tmp_path / SCOPED_AUDIT_NAME).exists()
    # once the registry proves the owner, the alias applies and its pending row gets an appended resolution
    assert ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, recovery) == []
    entries = recovery.entries()
    assert [e["severity"] for e in entries] == ["pending", "resolved"]
    records = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text(encoding="utf-8"))["records"]
    assert [(r["label"], r["decision"]) for r in records] == [("Ana Reed", "alias")]
    # a non_character claim about a label an existing actor plausibly owns is never applied
    guarded = [v2_row(chapters, 0, "Ana", "non_character", "none", 1)]
    (tmp_path / CONTEXT_RESOLUTION_V2_NAME).write_text(json.dumps(v2_payload(guarded)), encoding="utf-8")
    blocked = ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, recovery)
    assert [row["code"] for row in blocked] == ["context_non_character_unproven"]
    assert [r["label"] for r in json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]] == ["Ana Reed"]
    # an unproven mention that is absent from the current file is never resolved by omission
    (tmp_path / CONTEXT_RESOLUTION_V2_NAME).write_text(json.dumps(v2_payload([])), encoding="utf-8")
    assert len(ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, recovery)) == 1


def test_context_resolution_v2_source_violations_fail_closed(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    good = [v2_row(chapters, 0, "Zed", "hold", "none", 1)]

    def adapt(payload: dict, sha: str = V2_SOURCE_SHA) -> dict:
        return adapt_context_resolution_v2(payload, sha, chapters, registry, {"ana": "ana"}, [])

    assert len(adapt(v2_payload(good))["pending"]) == 1
    mutate = [
        ("main span", lambda r: r.update(span_start=r["span_start"] + 1)),
        ("main quote", lambda r: r.update(main_unit_quote=r["main_unit_quote"] + "!")),
        ("main sha", lambda r: r.update(main_unit_quote_sha256="0" * 64)),
        ("chapter sha", lambda r: r.update(chapter_sha256="0" * 64)),
        ("main unit", lambda r: r.update(main_unit_id="c00s99999")),
        ("witness sha", lambda r: r["witnesses"][0].update(chapter_sha256="0" * 64)),
        ("witness name", lambda r: r["witnesses"][0].update(chapter_file="99-missing.txt")),
        ("witness quote", lambda r: r["witnesses"][0].update(unit_quote="invented")),
        ("no witnesses", lambda r: r.update(witnesses=[])),
        ("decision", lambda r: r.update(decision="promote")),
    ]
    for _name, change in mutate:
        row = json.loads(json.dumps(good[0]))
        change(row)
        with pytest.raises(OperationalError):
            adapt(v2_payload([row]))
    with pytest.raises(OperationalError):
        adapt(v2_payload(good), "c" * 64)
    with pytest.raises(OperationalError):
        adapt({**v2_payload(good), "source_book_sha256": "c" * 64})
    with pytest.raises(OperationalError):
        adapt({**v2_payload(good), "contract": "other"})
    with pytest.raises(OperationalError):
        adapt(v2_payload([good[0], good[0]]))  # duplicate exact mention scope
    # a changed chapter byte invalidates the whole file, and nothing is written
    chapters[1].write_text("Zed carried a lantern. Ana waited at the door!", encoding="utf-8")
    (tmp_path / CONTEXT_RESOLUTION_V2_NAME).write_text(json.dumps(v2_payload(good)), encoding="utf-8")
    with pytest.raises(OperationalError):
        ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, registry, {}, RecoveryLedger(tmp_path))
    assert not (tmp_path / SCOPED_AUDIT_NAME).exists()


def _write_v2(tmp_path: Path, rows: list[dict], drafts: list[dict]) -> bytes:
    (tmp_path / CONTEXT_RESOLUTION_V2_NAME).write_text(json.dumps(v2_payload(rows, drafts=drafts)), encoding="utf-8")
    return (tmp_path / CONTEXT_RESOLUTION_V2_NAME).read_bytes()


def _draft_rows(chapters: list[Path]) -> list[dict]:
    rows = [
        v2_row(chapters, 0, "Zed", "new_actor", "zed", 1),
        v2_row(chapters, 1, "Zed", "new_actor", "zed", 0),
        v2_row(chapters, 0, "Professor", "new_actor", "zed", 0),  # a title variant of the same draft
    ]
    rows[2]["provenance"] = {"author": "caretaker", "method": "variant"}
    return rows


def _draft_pending(open_rows: list[dict]) -> list[dict]:
    return [row for row in open_rows if row["item"].startswith("variant_draft:")]


def test_variant_draft_becomes_one_pending_root_with_per_scope_provenance(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    before_input = _write_v2(tmp_path, _draft_rows(chapters), [v2_draft(chapters)])
    aliases = {"ana": "ana"}
    recovery = RecoveryLedger(tmp_path)

    open_rows = ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, registry, aliases, recovery)

    roots = _draft_pending(open_rows)
    assert [row["code"] for row in roots] == ["context_variant_draft_root"]
    assert sorted(row["code"] for row in open_rows if row not in roots) == ["context_new_actor"] * 3  # still block freeze
    stored = [json.loads(line) for line in (tmp_path / "cast_pending_proposals.jsonl").read_text().splitlines()]
    assert len(stored) == 1
    root = stored[0]
    assert root["status"] == "pending_proposal_not_approved" and root["actor_id"] == "zed"
    assert roots[0]["evidence"]["proposal_sha256"] == proposal_sha(root)
    assert roots[0]["evidence"]["proposal_sha256"] == hashlib.sha256(
        (tmp_path / "cast_pending_proposals.jsonl").read_text().splitlines()[0].encode()
    ).hexdigest()
    assert sorted(variant["label"] for variant in root["variants"]) == ["Professor", "Zed"]
    zed = next(variant for variant in root["variants"] if variant["label"] == "Zed")
    assert len(zed["scopes"]) == 2
    professor = next(variant for variant in root["variants"] if variant["label"] == "Professor")
    assert professor["scopes"][0]["provenance"] == {"author": "caretaker", "method": "variant"}
    for variant in root["variants"]:
        for scope in variant["scopes"]:
            assert scope["scope"]["chapter_sha256"] and scope["scope"]["quote_sha256"] and scope["witnesses"]
    # complete citations travel with the draft, each with its exact source quote
    assert root["citations"]["bio"] and root["citations"]["look"]
    assert all(c["unit_quote"] and c["unit_quote_sha256"] for c in root["citations"]["bio"] + root["citations"]["look"])
    # every included mention carries its validated linkage: here each shares an immutable unit with the draft's facts
    for variant in root["variants"]:
        for scope in variant["scopes"]:
            assert scope["link"]["kind"] == "shared_immutable_unit" and scope["link"]["units"]
    assert roots[0]["evidence"]["unlinked_excluded"] == [] and roots[0]["evidence"]["scope_count"] == 3
    assert "source_clusters" not in root
    # nothing was approved: input, registry, aliases and scoped audit are untouched; replay is idempotent
    assert (tmp_path / CONTEXT_RESOLUTION_V2_NAME).read_bytes() == before_input
    assert registry == {"ana": {"name": "Ana", "bio": "a scout"}} and aliases == {"ana": "ana"}
    assert not (tmp_path / SCOPED_AUDIT_NAME).exists()
    size = len(recovery.entries())
    again = ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, registry, aliases, recovery)
    assert len(again) == len(open_rows) and len(recovery.entries()) == size
    assert len((tmp_path / "cast_pending_proposals.jsonl").read_text().splitlines()) == 1


def test_variant_draft_without_full_bio_look_citations_never_becomes_a_root(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    rows = _draft_rows(chapters)

    def stage(draft: dict) -> list[dict]:
        result = adapt_context_resolution_v2(
            v2_payload(rows, drafts=[draft]), V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, []
        )
        draft_rows = [p for p in result["pending"] if p["item"].startswith("variant_draft:")]
        assert bool(result["roots"]) == (draft_rows[0]["code"] == "context_variant_draft_root")
        return draft_rows

    uncited = stage(v2_draft(chapters, cited=False))  # the legacy shape: witnesses only, no per-field citations
    assert [p["code"] for p in uncited] == ["context_variant_draft_incomplete"]
    assert uncited[0]["missing"] == ["bio has no source citations", "look has no source citations"]
    only_bio = v2_draft(chapters)
    only_bio["citations"]["look"] = []
    assert stage(only_bio)[0]["missing"] == ["look has no source citations"]
    only_look = v2_draft(chapters)
    only_look["citations"]["bio"] = []
    assert stage(only_look)[0]["missing"] == ["bio has no source citations"]
    # the explicit no-visual-details sentinel is the only look that needs no citation
    sentinel = v2_draft(chapters)
    sentinel["look"] = "no visual details given"
    sentinel["citations"]["look"] = []
    assert [p["code"] for p in stage(sentinel)] == ["context_variant_draft_root"]
    empty = v2_draft(chapters)
    empty["bio"] = " "
    assert stage(empty)[0]["missing"] == ["bio text is empty"]
    # a draft with no new_actor mention has no per-scope provenance, so it is not a root either
    lone = adapt_context_resolution_v2(
        v2_payload([], drafts=[v2_draft(chapters)]), V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, []
    )
    assert lone["roots"] == [] and [p["code"] for p in lone["pending"]] == ["context_variant_draft_no_mentions"]


def test_variant_draft_never_mints_a_duplicate(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    rows = _draft_rows(chapters)

    def codes(drafts: list[dict], reg: dict, aliases: dict, with_rows: list[dict] = rows) -> dict[str, str]:
        result = adapt_context_resolution_v2(v2_payload(with_rows, drafts=drafts), V2_SOURCE_SHA, chapters, reg, aliases, [])
        return {
            f"variant_draft:{variant_draft_actor(p['item'])}": p["code"]
            for p in result["pending"]
            if p["item"].startswith("variant_draft:")
        } | {"roots": str(len(result["roots"]))}

    assert codes([v2_draft(chapters)], registry, {"ana": "ana"}) == {"variant_draft:zed": "context_variant_draft_root", "roots": "1"}
    # already registered, already an alias, or sharing a name word with a registered actor
    assert codes([v2_draft(chapters)], {**registry, "zed": {"name": "Zed"}}, {"zed": "zed"})["roots"] == "0"
    assert codes([v2_draft(chapters)], registry, {"ana": "ana", "zed": "ana"}) == {
        "variant_draft:zed": "context_variant_draft_duplicate",
        "roots": "0",
    }
    named = v2_draft(chapters, actor_id="lord_ana", name="Lord Ana")
    assert codes([named], registry, {"ana": "ana"}, [v2_row(chapters, 0, "Zed", "new_actor", "lord_ana", 1)]) == {
        "variant_draft:lord_ana": "context_variant_draft_duplicate",
        "roots": "0",
    }
    # two drafts with the same name, or both claiming one label variant, block each other
    twin = v2_draft(chapters, actor_id="zed_two", name="Zed")
    assert codes([v2_draft(chapters), twin], registry, {"ana": "ana"}) == {
        "variant_draft:zed": "context_variant_draft_duplicate",
        "variant_draft:zed_two": "context_variant_draft_duplicate",
        "roots": "0",
    }
    contested = [rows[0], v2_row(chapters, 1, "Zed", "new_actor", "zephyr", 0)]  # the spelling "Zed" is offered to both
    assert codes([v2_draft(chapters), v2_draft(chapters, "zephyr", "Zephyr")], registry, {"ana": "ana"}, contested) == {
        "variant_draft:zed": "context_variant_draft_duplicate",
        "variant_draft:zephyr": "context_variant_draft_duplicate",
        "roots": "0",
    }
    # duplicate draft ids are an invalid file, not a silent merge
    with pytest.raises(OperationalError):
        codes([v2_draft(chapters), v2_draft(chapters)], registry, {"ana": "ana"})


def test_variant_draft_source_and_approval_violations_fail_closed(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    rows = _draft_rows(chapters)

    def adapt(draft: object) -> dict:
        return adapt_context_resolution_v2(
            {**v2_payload(rows), "new_actor_drafts": [draft]}, V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, []
        )

    assert len(adapt(v2_draft(chapters))["roots"]) == 1
    mutate = [
        ("approved", lambda d: d.update(status="APPROVED")),
        ("bad id", lambda d: d.update(actor_id="Bad Id")),
        ("no name", lambda d: d.update(name=" ")),
        ("missing key", lambda d: d.pop("bio")),
        ("bio citation quote", lambda d: d["citations"]["bio"][0].update(unit_quote="invented")),
        ("look citation sha", lambda d: d["citations"]["look"][0].update(chapter_sha256="0" * 64)),
        ("witness unit", lambda d: d["witnesses"][0].update(unit_id="c00s99999")),
        ("citation field", lambda d: d["citations"].update(voice=[])),
        ("citations type", lambda d: d.update(citations=[])),
    ]
    for _name, change in mutate:
        draft = json.loads(json.dumps(v2_draft(chapters)))
        change(draft)
        with pytest.raises(OperationalError):
            adapt(draft)
    with pytest.raises(OperationalError):
        adapt("not a draft")
    with pytest.raises(OperationalError):
        adapt_context_resolution_v2(
            {**v2_payload(rows), "new_actor_drafts": "x"}, V2_SOURCE_SHA, chapters, registry, {}, []
        )


def test_variant_draft_pending_rows_close_only_on_real_state_change(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    rows = _draft_rows(chapters)
    recovery = RecoveryLedger(tmp_path)

    def ingest(drafts: list[dict], reg: dict, use: list[dict] = rows) -> list[dict]:
        _write_v2(tmp_path, use, drafts)
        return _draft_pending(ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, reg, {"ana": "ana"}, recovery))

    assert [r["code"] for r in ingest([v2_draft(chapters, cited=False)], registry)] == ["context_variant_draft_incomplete"]
    # citations arrive: the incomplete row is superseded by an appended resolution and the root is pending
    assert [r["code"] for r in ingest([v2_draft(chapters)], registry)] == ["context_variant_draft_root"]
    assert [e["severity"] for e in recovery.entries() if variant_draft_actor(e["item"]) == "zed"] == ["pending", "pending", "resolved"]
    # omitting the draft while its new_actor mentions remain never resolves it
    assert [r["code"] for r in ingest([], registry)] == ["context_variant_draft_root"]
    # once nothing in the file names the draft, the draft-level row closes (each mention row has its own pending/resolution)
    assert ingest([], registry, []) == []
    # a root whose actor was registered through the real registry path is closed, never minted here
    assert [r["code"] for r in ingest([v2_draft(chapters, "zed2", "Zed Two")], registry, [v2_row(chapters, 0, "Zed", "new_actor", "zed2", 1)])] == [
        "context_variant_draft_root"
    ]
    registered = {**registry, "zed2": {"name": "Zed Two"}}
    assert [r["code"] for r in ingest([], registered, [v2_row(chapters, 0, "Zed", "new_actor", "zed2", 1)])] == []
    assert registered["zed2"] == {"name": "Zed Two"}


def kin_fixture(tmp_path: Path) -> list[Path]:
    path = tmp_path / "01-part_01.txt"
    path.write_text(
        "Zed appeared at the gate. The hall was quiet. Auntie waved from the porch. Auntie is the sister of Zed. "
        "Rain fell on the roof. Dust blew.",
        encoding="utf-8",
    )
    other = tmp_path / "02-part_02.txt"
    other.write_text("Zed left. Auntie is the sister of Zed.", encoding="utf-8")
    return [path, other]


def kin_row(chapters: list[Path], label: str, chapter: int, index: int, witness_index: int) -> dict:
    unit = immutable_evidence_units([chapters[chapter]])[index]
    row = v2_row(chapters, chapter, label, "new_actor", "zed", chapter)
    witness = v2_witness(chapters, chapter, witness_index)
    row.update(
        main_unit_id=unit["id"],
        main_unit_quote=unit["quote"],
        main_unit_quote_sha256=hashlib.sha256(unit["quote"].encode()).hexdigest(),
        span_start=unit["quote"].index(label),
        witnesses=[{**witness, "role": "preceding_unit"}],
    )
    return row


def kin_link(chapters: list[Path], kind: str, witnesses: list[tuple[int, int]], chapter: int = 0, index: int = 2) -> dict:
    unit = immutable_evidence_units([chapters[chapter]])[index]
    return {
        "kind": kind,
        "chapter_file": chapters[chapter].name,
        "chapter_sha256": unit["chapter_sha256"],
        "main_unit_id": unit["id"],
        "label": "Auntie",
        "span_start": unit["quote"].index("Auntie"),
        "witnesses": [v2_witness(chapters, c, i) for c, i in witnesses],
    }


def stage_kin(chapters: list[Path], registry: dict, links: list[dict]) -> dict:
    draft = {**v2_draft(chapters), "variant_links": links}
    draft["witnesses"] = draft["citations"]["bio"] = [v2_witness(chapters, 0, 0)]
    draft["citations"]["look"] = [v2_witness(chapters, 0, 0)]
    rows = [kin_row(chapters, "Zed", 0, 0, 0), kin_row(chapters, "Auntie", 0, 2, 1)]
    return adapt_context_resolution_v2(v2_payload(rows, drafts=[draft]), V2_SOURCE_SHA, chapters, registry, {}, [])


def test_variant_root_includes_only_mentions_with_validated_linkage(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    linked = _draft_rows(chapters)
    # "Mom" shares no immutable unit with the draft's facts (its neighbour names Professor, not the draft) and has no link
    stray = v2_row(chapters, 0, "Mom", "new_actor", "zed", 0)
    result = adapt_context_resolution_v2(
        v2_payload([*linked, stray], drafts=[v2_draft(chapters)]), V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, []
    )
    (root,) = result["roots"]
    assert sorted(variant["label"] for variant in root["variants"]) == ["Professor", "Zed"]
    row = next(item for item in result["pending"] if item["code"] == "context_variant_draft_root")
    assert row["scope_count"] == 3
    assert [(item["scope"]["label"], item["why"]) for item in row["unlinked_excluded"]] == [
        ("Mom", "no shared immutable unit with the draft facts and no explicit link")
    ]
    # the stray mention itself stays its own pending context_new_actor row and is never silently merged
    assert [item["code"] for item in result["pending"] if item["item"].startswith("context:") and "Mom" in item["item"]] == [
        "context_new_actor"
    ]
    # with no linked mention at all there is no root, only a typed unlinked gap
    only = adapt_context_resolution_v2(
        v2_payload([stray], drafts=[v2_draft(chapters)]), V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, []
    )
    assert only["roots"] == [] and [p["code"] for p in only["pending"] if p["item"].startswith("variant_draft:")] == [
        "context_variant_draft_unlinked"
    ]
    # a shared unit counts only when it literally carries the mention's label or the draft's name
    off_topic = v2_draft(chapters)
    off_topic["witnesses"] = off_topic["citations"]["bio"] = off_topic["citations"]["look"] = [v2_witness(chapters, 0, 0)]
    quiet = adapt_context_resolution_v2(
        v2_payload([stray], drafts=[off_topic]), V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, []
    )
    assert quiet["roots"] == []


def test_variant_root_accepts_explicit_source_validated_kinship_and_participant_links(tmp_path: Path) -> None:
    chapters = kin_fixture(tmp_path)
    registry = {"ana": {"name": "Ana", "bio": "a scout"}}
    # without a link the Auntie mention is excluded; Zed alone still forms the root
    bare = stage_kin(chapters, registry, [])
    assert [v["label"] for v in bare["roots"][0]["variants"]] == ["Zed"]
    kin = stage_kin(chapters, registry, [kin_link(chapters, "kinship", [(0, 3)])])
    auntie = next(v for v in kin["roots"][0]["variants"] if v["label"] == "Auntie")
    assert auntie["scopes"][0]["link"]["kind"] == "kinship" and auntie["scopes"][0]["link"]["witnesses"]
    scene = stage_kin(chapters, registry, [kin_link(chapters, "continuous_participant", [(0, 2), (0, 0)])])
    auntie = next(v for v in scene["roots"][0]["variants"] if v["label"] == "Auntie")
    assert auntie["scopes"][0]["link"]["kind"] == "continuous_participant"
    # a link whose witnesses do not prove it leaves the mention excluded, never merged
    for links in (
        [kin_link(chapters, "kinship", [(0, 1)])],  # real unit, but names neither Auntie nor Zed
        [kin_link(chapters, "kinship", [(0, 0)])],  # names Zed but not the mention's label
        [kin_link(chapters, "kinship", [(1, 1)])],  # names both, but in another chapter: outside this mention's scene
        [kin_link(chapters, "continuous_participant", [(0, 2)])],  # label but no draft name in the scene
        [kin_link(chapters, "continuous_participant", [(1, 1), (0, 0)])],  # the label unit is outside the mention's scene
    ):
        staged = stage_kin(chapters, registry, links)
        assert [v["label"] for v in staged["roots"][0]["variants"]] == ["Zed"]
    # source-invalid or foreign links fail the whole file closed
    forged = kin_link(chapters, "kinship", [(0, 3)])
    forged["witnesses"][0]["unit_quote"] = "invented"
    with pytest.raises(OperationalError):
        stage_kin(chapters, registry, [forged])
    for bad in (
        {**kin_link(chapters, "kinship", [(0, 3)]), "kind": "friendship"},
        {**kin_link(chapters, "kinship", [(0, 3)]), "witnesses": []},
        {**kin_link(chapters, "kinship", [(0, 3)]), "span_start": 3},
        {**kin_link(chapters, "kinship", [(0, 3)]), "extra": 1},
        kin_link(chapters, "kinship", [(0, 3)], index=3),  # an exact span, but not one of this draft's mentions
    ):
        with pytest.raises(OperationalError):
            stage_kin(chapters, registry, [bad])


def test_variant_root_serializes_to_the_pending_proposal_file_without_overwrite_or_replay_duplicates(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    existing = {"type": "new_identity", "id": "someone", "mentions": []}  # a line the demoted-identity lane already stored
    (tmp_path / "cast_pending_proposals.jsonl").write_text(json.dumps(existing, sort_keys=True) + "\n", encoding="utf-8")
    recovery = RecoveryLedger(tmp_path)
    rows = _draft_rows(chapters)
    _write_v2(tmp_path, rows, [v2_draft(chapters)])
    first = _draft_pending(ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, recovery))
    lines = (tmp_path / "cast_pending_proposals.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0]) == existing  # appended, the prior proposal is untouched
    for _ in range(3):  # replays neither duplicate the proposal nor the ledger row
        ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, recovery)
    assert (tmp_path / "cast_pending_proposals.jsonl").read_text().splitlines() == lines
    assert len([e for e in recovery.entries() if e["item"].startswith("variant_draft:")]) == 1
    # a changed root (a new mention is added) is a NEW proposal line and a NEW ledger row; the old row is superseded
    more = [*rows, v2_row(chapters, 0, "Mom", "new_actor", "zed", 0)]
    more[-1]["witnesses"] = [{**v2_witness(chapters, 0, 3), "role": "keyword_witness:Zed"}]
    _write_v2(tmp_path, more, [v2_draft(chapters)])
    second = _draft_pending(ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, registry, {"ana": "ana"}, recovery))
    after = (tmp_path / "cast_pending_proposals.jsonl").read_text().splitlines()
    assert len(after) == 3 and after[:2] == lines
    assert [r["evidence"]["proposal_sha256"] for r in first] != [r["evidence"]["proposal_sha256"] for r in second]
    assert second[0]["evidence"]["proposal_sha256"] == hashlib.sha256(after[2].encode()).hexdigest()
    assert len(second) == 1  # the superseded row is closed, only the current root stays pending


def _review_records(root: dict) -> list[dict]:
    return [
        {
            "chapter_sha256": item["scope"]["chapter_sha256"],
            "quote_sha256": item["scope"]["quote_sha256"],
            "label": item["scope"]["label"],
            "span_start": item["scope"]["span_start"],
            "verdict": DISTINCT_VERDICT,
            "own_source_witness": {
                "unit_id": item["scope"]["unit_id"],
                "provenance": "immutable_candidate_reference",
                "label": item["scope"]["label"],
                "span_start": item["scope"]["span_start"],
            },
            "confidence": 0.0,  # confidence is never evidence in either direction
        }
        for variant in root["variants"]
        for item in variant["scopes"]
    ]


def test_variant_root_approval_gate_needs_strong_facts_and_never_changes_the_registry(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    aliases = {"ana": "ana"}
    result = adapt_context_resolution_v2(
        v2_payload(_draft_rows(chapters), drafts=[v2_draft(chapters)]), V2_SOURCE_SHA, chapters, registry, aliases, []
    )
    (root,) = result["roots"]
    reviews = _review_records(root)
    snapshot = json.loads(json.dumps([registry, aliases, root, reviews]))

    assert variant_root_approval_gate(root, registry, aliases, reviews) == []  # positive: spelling/confidence play no part

    def blockers(changed: dict | None = None, reg: dict | None = None, ali: dict | None = None, revs: list | None = None) -> list[str]:
        return variant_root_approval_gate(
            {**root, **(changed or {})}, registry if reg is None else reg, aliases if ali is None else ali, reviews if revs is None else revs
        )

    assert any("kind" in b for b in blockers({"kind": "creature"}))
    assert any("unapproved" in b for b in blockers({"status": "approved"}))
    assert blockers({"type": "other"}) == ["not a variant draft root"]
    assert any("already registered" in b for b in blockers(reg={**registry, "zed": {"name": "Zed"}}))
    assert any("alias" in b for b in blockers(ali={**aliases, "zed": "ana"}))
    assert any("shares a name" in b for b in blockers({"actor_id": "zed_two", "name": "Ana Zed"}))
    assert any("bio citation" in b for b in blockers({"citations": {**root["citations"], "bio": [root["citations"]["look"][0] | {"unit_quote": "Unrelated."}]}}))
    assert any("look is not cited" in b for b in blockers({"citations": {**root["citations"], "look": []}}))
    # every included mention needs its own exact-scope distinct-living verdict; a high-confidence wrong verdict is no help
    assert any("lacks an exact-scope" in b for b in blockers(revs=reviews[1:]))
    uncertain = [{**reviews[0], "verdict": "uncertain", "confidence": 1.0}, *reviews[1:]]
    assert any("lacks an exact-scope" in b for b in blockers(revs=uncertain))
    existing = [{**reviews[0], "verdict": "existing:ana"}, *reviews[1:]]
    assert any("lacks an exact-scope" in b for b in blockers(revs=existing))
    moved = [{**reviews[0], "own_source_witness": {**reviews[0]["own_source_witness"], "unit_id": "c00s99999"}}, *reviews[1:]]
    assert any("lacks an exact-scope" in b for b in blockers(revs=moved))
    assert "root has no mentions" in blockers({"variants": []})
    # the gate is pure: nothing it saw was modified
    assert [registry, aliases, root, reviews] == snapshot


def test_variant_root_gate_title_only_overlap_never_blocks_a_different_full_name(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    aliases = {"ana": "ana"}
    result = adapt_context_resolution_v2(
        v2_payload(_draft_rows(chapters), drafts=[v2_draft(chapters)]), V2_SOURCE_SHA, chapters, registry, aliases, []
    )
    (root,) = result["roots"]
    reviews = _review_records(root)
    cast = {
        **registry,
        "lord_bloodwin": {"name": "Lord Bloodwin"},
        "professor_venmont": {"name": "Professor Venmont"},
        "bram": {"name": "Bram"},
    }
    cast_aliases = {**aliases, "lord_bloodwin": "lord_bloodwin", "professor_venmont": "professor_venmont", "bram": "bram", "the_professor": "professor_venmont", "old_wick": "bram"}

    def duplicates(actor_id: str, name: str, reg: dict | None = None, ali: dict | None = None) -> list[str]:
        found = variant_root_approval_gate(
            {**root, "actor_id": actor_id, "name": name}, cast if reg is None else reg, cast_aliases if ali is None else ali, reviews
        )
        return [b for b in found if "shares a name" in b or "already" in b]

    # a shared title/role word alone is not an identity overlap
    assert duplicates("lord_ravenspire", "Lord Ravenspire") == []
    assert duplicates("lord_venmont_two", "Lord Selia") == []
    assert duplicates("professor_orrin", "Professor Orrin") == []
    assert duplicates("captain_orrin", "Captain Orrin", reg={**cast, "captain_hale": {"name": "Captain Hale"}}) == []
    # a real overlap still blocks: exact full name, shared given name/surname, alias, id
    assert any("already registered" in b for b in duplicates("lord_bloodwin", "Lord Bloodwin"))
    assert any("shares a name" in b for b in duplicates("lord_two", "Lord Bloodwin"))
    assert any("shares a name" in b for b in duplicates("bloodwin_two", "Bloodwin"))
    assert any("shares a name" in b for b in duplicates("professor_two", "Professor Venmont"))
    assert any("shares a name" in b for b in duplicates("venmont_two", "Lady Venmont"))
    assert any("shares a name" in b for b in duplicates("bram_two", "Lord Bram"))
    assert any("alias" in b for b in duplicates("the_professor", "Someone Else"))
    assert any("shares a name" in b for b in duplicates("wick_two", "Wick Smith"))


# ##################################################################
# variant root approval and materialization
# Real temp chapters, real ledger, real files: a root becomes a registry actor only through the documented approval input,
# after every gate; everything else leaves registry, aliases, scoped audit and original profile files byte-identical.
def staged_root_project(
    tmp_path: Path, chapters: list[Path], registry: dict, rows: list[dict], drafts: list[dict]
) -> tuple[dict, RecoveryLedger, dict[str, dict]]:
    aliases = {key: key for key in registry}
    _write_v2(tmp_path, rows, drafts)
    progress = {"registry": registry, "aliases": aliases}
    (tmp_path / PROGRESS_NAME).write_text(json.dumps(progress), encoding="utf-8")
    (tmp_path / "characters.json").write_text(
        json.dumps({"ana": {"name": "Ana", "bio": "a scout", "look": "tall"}}), encoding="utf-8"
    )
    (tmp_path / "voices.json").write_text(json.dumps({"ana": {"description": "calm"}}), encoding="utf-8")
    recovery = RecoveryLedger(tmp_path)
    ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, registry, aliases, recovery)
    roots = {}
    for line in (tmp_path / PROPOSALS_NAME).read_text().splitlines():
        root = json.loads(line)
        roots[root["actor_id"]] = root
    return progress, recovery, roots


def native_audit(tmp_path: Path, root: dict, verdict: str = DISTINCT_VERDICT, skip: int | None = None) -> None:
    records = [
        {**record, "verdict": verdict, "raw_review": {"verdict": verdict, "witness_unit_ids": [], "confidence": 0.9}}
        for index, record in enumerate(_review_records(root))
        if index != skip
    ]
    (tmp_path / NEW_IDENTITY_AUDIT_NAME).write_text(json.dumps({"records": records}), encoding="utf-8")


def root_reviews(root: dict, chapters: list[Path], provenance: str = NATIVE_REVIEW, witness: tuple[int, int] = (0, 3)) -> list[dict]:
    sha = proposal_sha(root)
    reviews = []
    for variant in root["variants"]:
        for item in variant["scopes"]:
            native = provenance == NATIVE_REVIEW
            reviews.append(
                {
                    "proposal_sha256": sha,
                    "scope": item["scope"],
                    "provenance": provenance,
                    "verdict": DISTINCT_VERDICT,
                    "reviewer_role": None if native else "caretaker",
                    "factual_witnesses": [] if native else [v2_witness(chapters, *witness)],
                    "factual_basis": "" if native else "the cited unit names this identity as a living person",
                }
            )
    return reviews


def write_root_approval(tmp_path: Path, root: dict, reviews: list[dict], sha: str | None = None, **over) -> None:
    approval = {
        "action": "approve_root",
        "actor_id": root["actor_id"],
        "proposal_sha256": sha or proposal_sha(root),
        "reviews": reviews,
        "note": "caretaker checked the cited source",
        **over,
    }
    payload = {"contract": ROOT_APPROVAL_CONTRACT, "version": 1, "source_sha256": V2_SOURCE_SHA, "approvals": [approval]}
    (tmp_path / ROOT_APPROVAL_NAME).write_text(json.dumps(payload), encoding="utf-8")


def untouched_state(tmp_path: Path) -> dict:
    names = (PROGRESS_NAME, "characters.json", "voices.json", SCOPED_AUDIT_NAME, PROPOSALS_NAME)
    return {name: (tmp_path / name).read_bytes() if (tmp_path / name).exists() else None for name in names}


def approve(tmp_path: Path, chapters: list[Path], progress: dict, recovery: RecoveryLedger) -> dict:
    return ingest_root_approvals(tmp_path, V2_SOURCE_SHA, chapters, progress, recovery)


def test_root_approval_materializes_native_reviewed_root_once_and_only_exactly(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    root = roots["zed"]
    native_audit(tmp_path, root)
    write_root_approval(tmp_path, root, root_reviews(root, chapters))
    originals = {name: (tmp_path / name).read_bytes() for name in ("characters.json", "voices.json")}
    ana_before = json.loads(json.dumps(progress["registry"]["ana"]))
    assert any(row["code"] == "context_variant_draft_root" for row in recovery.open_pending(CONTEXT_V2_STAGE))

    outcome = approve(tmp_path, chapters, progress, recovery)

    assert outcome == {"materialized": ["zed"], "already": [], "blocked": [], "malformed": []}
    entry = progress["registry"]["zed"]
    assert entry["origin"] == "approved_root" and entry["approved_root"]["proposal_sha256"] == proposal_sha(root)
    assert entry["approved_root"]["approval"]["reviews"], "raw review refs are retained"
    assert prepared_profile(entry) == {"name": "Zed", "bio": root["bio"], "look": root["look"]}
    assert progress["registry"]["ana"] == ana_before
    # only the exact actor id/name are global aliases; Professor stays mention-scoped
    assert progress["aliases"] == {"ana": "ana", "zed": "zed"}
    saved = json.loads((tmp_path / PROGRESS_NAME).read_text())
    assert saved["registry"]["zed"]["name"] == "Zed" and saved["aliases"]["zed"] == "zed"
    records = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert sorted((r["label"], r["decision"], r["canonical"]) for r in records) == [
        ("Professor", "alias", "zed"),
        ("Zed", "alias", "zed"),
        ("Zed", "alias", "zed"),
    ]
    assert all(r["approved_root"]["review_provenance"] == NATIVE_REVIEW for r in records)
    mention_scoped_audit_index(records)
    # original profiles and voices are never rewritten; the pending rows for this root are closed
    assert {name: (tmp_path / name).read_bytes() for name in originals} == originals
    assert recovery.open_pending(CONTEXT_V2_STAGE) == [] and recovery.open_pending(ROOT_APPROVAL_STAGE) == []
    # idempotent: replaying the approval and re-ingesting the unchanged v2 file change nothing and leave nothing pending
    snapshot, size = untouched_state(tmp_path), len(recovery.entries())
    assert approve(tmp_path, chapters, progress, recovery) == {"materialized": [], "already": ["zed"], "blocked": [], "malformed": []}
    assert ingest_context_resolution_v2(tmp_path, V2_SOURCE_SHA, chapters, progress["registry"], progress["aliases"], recovery) == []
    assert untouched_state(tmp_path) == snapshot and len(recovery.entries()) == size
    fresh = RecoveryLedger(tmp_path)
    assert fresh.open_pending(CONTEXT_V2_STAGE) == []


def test_root_approval_title_only_overlap_with_registered_actors_never_blocks(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    # registered court actors share only the title/role word "Professor"/"Lord" with the Zed root and its "Professor" variant
    registry = {**registry, "professor_venmont": {"name": "Professor Venmont"}, "lord_bloodwin": {"name": "Lord Bloodwin"}}
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    root = roots["zed"]
    native_audit(tmp_path, root)
    write_root_approval(tmp_path, root, root_reviews(root, chapters))

    assert approve(tmp_path, chapters, progress, recovery) == {"materialized": ["zed"], "already": [], "blocked": [], "malformed": []}


def test_root_approval_real_overlap_with_a_title_bearing_registered_actor_still_blocks(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    root = roots["zed"]
    native_audit(tmp_path, root)
    write_root_approval(tmp_path, root, root_reviews(root, chapters))
    # the registry changes after staging: a titled actor now carries the root's own name word "Zed"
    progress["registry"]["professor_zed_venmont"] = {"name": "Professor Zed Venmont"}

    outcome = approve(tmp_path, chapters, progress, recovery)

    assert outcome["materialized"] == [] and "zed" not in progress["registry"]
    blockers = [b for entry in outcome["blocked"] for b in entry["blockers"]]
    assert any("shares a name with registered actor professor_zed_venmont" in b for b in blockers)
    assert any("variant 'Zed' is literally carried by registered actor professor_zed_venmont" in b for b in blockers)
    assert not any("variant 'Professor'" in b for b in blockers)  # the title-only variant is never what blocks


def test_variant_root_duplicate_blockers_ignore_title_only_labels_but_keep_real_identification(tmp_path: Path) -> None:
    registry = {
        "ana": {"name": "Ana"},
        "professor_venmont": {"name": "Professor Venmont"},
        "lord_bloodwin": {"name": "Lord Bloodwin"},
    }
    aliases = {key: key for key in registry} | {"old_wick": "ana"}

    def blockers(*labels: str) -> list[str]:
        root = {"actor_id": "zed", "name": "Zed", "variants": [{"label": label} for label in labels]}
        return variant_root_duplicate_blockers(root, registry, aliases, {}, {})

    # title/role-only overlap with a different full name is no identification
    assert blockers("Professor", "Lord", "Lord Ravenspire", "Professor Orrin", "Zed") == []
    # exact full name, a shared given-name/surname word, and a known alias still identify a registered actor
    assert any("carried by registered actor lord_bloodwin" in b for b in blockers("Lord Bloodwin"))
    assert any("carried by registered actor professor_venmont" in b for b in blockers("Venmont"))
    assert any("carried by registered actor professor_venmont" in b for b in blockers("Lady Venmont"))
    assert any("carried by registered actor ana" in b for b in blockers("Old Wick"))
    assert any("carried by registered actor ana" in b for b in blockers("Ana"))


def test_root_approval_accepts_source_reviewed_trusted_role_proof_with_truthful_provenance(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    root = roots["zed"]
    reviews = root_reviews(root, chapters, SOURCE_REVIEW)
    reviews[0] = root_reviews(root, chapters)[0]  # one mention native...
    native_audit(tmp_path, root)  # ...backed by the audit, the others source-reviewed
    write_root_approval(tmp_path, root, reviews)
    assert approve(tmp_path, chapters, progress, recovery)["materialized"] == ["zed"]
    records = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert {r["approved_root"]["review_provenance"] for r in records} == {NATIVE_REVIEW, SOURCE_REVIEW}
    assert all("human" not in json.dumps(r).casefold() for r in records)
    assert {r["approved_root"]["reviewer_role"] for r in records} == {None, "caretaker"}


def test_root_approval_input_violations_fail_closed_before_any_write(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    root = roots["zed"]
    native_audit(tmp_path, root)
    good = root_reviews(root, chapters)
    source_good = root_reviews(root, chapters, SOURCE_REVIEW)
    before = untouched_state(tmp_path)

    def attempt(reviews: list[dict], **over) -> None:
        write_root_approval(tmp_path, root, reviews, **over)
        with pytest.raises(OperationalError):
            approve(tmp_path, chapters, progress, recovery)
        assert untouched_state(tmp_path) == before and "zed" not in progress["registry"]

    def mutated(reviews: list[dict], **change) -> list[dict]:
        return [{**reviews[0], **change}, *reviews[1:]]

    attempt(good, sha="0" * 64)  # unknown proposal sha (hash mismatch with the stored proposals)
    attempt(good, actor_id="someone_else")  # a sha that belongs to another actor
    forged = {**v2_witness(chapters, 0, 3), "unit_quote": "invented"}
    attempt(mutated(source_good, factual_witnesses=[forged]))  # a witness that is not the exact current source
    # wrong book, wrong contract, and a changed chapter byte
    write_root_approval(tmp_path, root, good)
    approval_path = tmp_path / ROOT_APPROVAL_NAME
    payload = json.loads(approval_path.read_text())
    for change in ({"source_sha256": "c" * 64}, {"contract": "other"}, {"version": 2}, {"extra": 1}, {"approvals": {}}):
        approval_path.write_text(json.dumps({**payload, **change}), encoding="utf-8")
        with pytest.raises(OperationalError):
            approve(tmp_path, chapters, progress, recovery)
    approval_path.write_text(json.dumps(payload), encoding="utf-8")
    chapters[1].write_text("Zed carried a lantern. Ana waited at the door!", encoding="utf-8")
    with pytest.raises(OperationalError):
        approve(tmp_path, chapters, progress, recovery)
    assert untouched_state(tmp_path) == before and "zed" not in progress["registry"]


def test_root_approval_blocks_every_failed_gate_without_writing_anything(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    root = roots["zed"]
    before = untouched_state(tmp_path)

    def blocked(reviews: list[dict], contains: str) -> None:
        write_root_approval(tmp_path, root, reviews)
        result = approve(tmp_path, chapters, progress, recovery)
        assert result["materialized"] == [] and len(result["blocked"]) == 1
        assert any(contains in blocker for blocker in result["blocked"][0]["blockers"]), result["blocked"]
        assert untouched_state(tmp_path) == before and "zed" not in progress["registry"]
        assert [r["code"] for r in recovery.open_pending(ROOT_APPROVAL_STAGE)] == ["root_approval_blocked"]

    # no native review at all, an uncertain/existing verdict, or a raw-less record is no native proof
    blocked(root_reviews(root, chapters), "no native distinct-living-identity review")
    native_audit(tmp_path, root, skip=0)
    blocked(root_reviews(root, chapters), "no native distinct-living-identity review")
    native_audit(tmp_path, root, verdict="uncertain")
    blocked(root_reviews(root, chapters), "no native distinct-living-identity review")
    # a source review can never override a real native conflict (a fragment verdict here; see the resolution tests for existing:<id>)
    native_audit(tmp_path, root, verdict="nonidentity_fragment")
    blocked(root_reviews(root, chapters, SOURCE_REVIEW), "contradicts the native review verdict")
    # a source-review witness that names neither the mention nor the identity proves nothing
    (tmp_path / NEW_IDENTITY_AUDIT_NAME).unlink()
    blocked(root_reviews(root, chapters, SOURCE_REVIEW, witness=(0, 0)), "no source-review witness literally names")
    blocked(root_reviews(root, chapters, SOURCE_REVIEW, witness=(0, 1)), "no source-review witness literally names")
    # the same proposal after the blockers change supersedes the older blocked row instead of stacking
    assert len(recovery.open_pending(ROOT_APPROVAL_STAGE)) == 1


def malformed_audit(tmp_path: Path, root: dict, per_mention: list[dict]) -> None:
    """Native audit with one hand-shaped record per mention (verdict/raw_review/provenance/... merged over a distinct record)."""
    records = [{**record, **extra} for record, extra in zip(_review_records(root), per_mention, strict=True)]
    (tmp_path / NEW_IDENTITY_AUDIT_NAME).write_text(json.dumps({"records": records}), encoding="utf-8")


def uncertain_record(raw_verdict: str = "uncertain", **extra) -> dict:
    return {"verdict": "uncertain", "raw_review": {"verdict": raw_verdict, "witness_unit_ids": [], "confidence": 0.3}, **extra}


def test_source_review_resolves_native_uncertain_and_stale_unsupported_reviews(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    root = roots["zed"]
    # mention 0 is genuinely uncertain, mention 1 is an unsupported (rejected) existing:ana review recorded as uncertain,
    # mention 2 is an existing:<id> whose owner is no longer a registered actor (stale)
    malformed_audit(
        tmp_path,
        root,
        [
            uncertain_record(),
            uncertain_record("existing:ana", pending_type="unapproved_alias", invalid_reason="lacks source/registry support"),
            {"verdict": "existing:departed", "raw_review": {"verdict": "existing:departed", "witness_unit_ids": [], "confidence": 0.9}},
        ],
    )
    write_root_approval(tmp_path, root, root_reviews(root, chapters, SOURCE_REVIEW))
    assert approve(tmp_path, chapters, progress, recovery)["materialized"] == ["zed"]
    records = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert sorted(r["approved_root"]["resolved_native_verdict"] for r in records) == ["existing:departed", "uncertain", "uncertain"]
    assert all(r["approved_root"]["review_provenance"] == SOURCE_REVIEW and "human" not in json.dumps(r).casefold() for r in records)
    assert recovery.open_pending(ROOT_APPROVAL_STAGE) == []


def test_native_unsupported_existing_without_owner_proof_does_not_veto(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    root = roots["zed"]
    # ana is registered but neither recorded support nor the mention's scene ties "Zed"/"Professor" to her
    stale = {"verdict": "existing:ana", "raw_review": {"verdict": "existing:ana", "witness_unit_ids": [], "confidence": 0.9}}
    gone = {**stale, "provenance": {"type": "literal_witness", "owner_name": "Anna", "owner_profile_field": "name", "witness_unit_id": "c00s00000"}}
    malformed_audit(tmp_path, root, [stale, gone, stale])
    write_root_approval(tmp_path, root, root_reviews(root, chapters, SOURCE_REVIEW))
    assert approve(tmp_path, chapters, progress, recovery)["materialized"] == ["zed"]


def test_currently_valid_source_supported_existing_still_vetoes_a_source_review(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    root = roots["zed"]
    before = untouched_state(tmp_path)
    ana_support = {"type": "literal_witness", "owner_name": "Ana", "owner_profile_field": "name", "witness_unit_id": "c00s00000"}
    valid = {"verdict": "existing:ana", "raw_review": {"verdict": "existing:ana", "witness_unit_ids": [], "confidence": 0.9}, "provenance": ana_support}
    uncertain_ok = uncertain_record()

    def vetoed(per_mention: list[dict], contains: str = "contradicts the native review verdict") -> None:
        malformed_audit(tmp_path, root, per_mention)
        write_root_approval(tmp_path, root, root_reviews(root, chapters, SOURCE_REVIEW))
        result = approve(tmp_path, chapters, progress, recovery)
        assert result["materialized"] == [] and any(contains in b for b in result["blocked"][0]["blockers"]), result
        assert untouched_state(tmp_path) == before and "zed" not in progress["registry"]

    vetoed([uncertain_ok, valid, uncertain_ok])  # recorded support still honoured by the registry (Ana is still named Ana)
    # the source scene of the exact mention re-proves the owner (no recorded support, "Professor" is written in Ana's profile)
    progress["registry"]["ana"]["bio"] = "a scout everyone calls Professor"
    vetoed([{"verdict": "existing:ana", "raw_review": {"verdict": "existing:ana"}}, uncertain_ok, uncertain_ok])
    # a low-confidence record (uncertain) whose raw review was this valid existing:ana is the same real conflict
    vetoed([uncertain_record("existing:ana"), uncertain_ok, uncertain_ok], "still supported by the current source")
    progress["registry"]["ana"]["bio"] = "a scout"
    # every other real native verdict binds too
    for verdict in ("nonidentity_fragment", "same_provisional:c7", "something_else"):
        vetoed([{"verdict": verdict, "raw_review": {"verdict": verdict}}, uncertain_ok, uncertain_ok], "real conflict")
    # a native-provenance claim never rides on an uncertain record
    malformed_audit(tmp_path, root, [uncertain_ok] * 3)
    write_root_approval(tmp_path, root, root_reviews(root, chapters))
    result = approve(tmp_path, chapters, progress, recovery)
    assert result["materialized"] == [] and any("no native distinct-living-identity review" in b for b in result["blocked"][0]["blockers"])


def test_source_review_resolving_uncertain_never_overrides_participant_duplicate_or_scoped_conflicts(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    root = roots["zed"]
    malformed_audit(tmp_path, root, [uncertain_record()] * 3)
    write_root_approval(tmp_path, root, root_reviews(root, chapters, SOURCE_REVIEW))
    before = untouched_state(tmp_path)
    # original-anchor / duplicate conflict
    progress["registry"]["old_zed"] = {"name": "Zed"}
    result = approve(tmp_path, chapters, progress, recovery)
    assert result["materialized"] == [] and any("already the name of registered actor" in b for b in result["blocked"][0]["blockers"])
    del progress["registry"]["old_zed"]
    # an exact-scope scoped decision already binds one mention
    scope = root["variants"][1]["scopes"][0]["scope"]
    (tmp_path / SCOPED_AUDIT_NAME).write_text(
        json.dumps(
            {
                "records": [
                    {
                        "chapter_sha256": scope["chapter_sha256"], "quote_sha256": scope["quote_sha256"], "label": scope["label"],
                        "span_start": scope["span_start"], "canonical": "ana", "decision": "alias", "confidence": 1.0, "reason": "earlier",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = approve(tmp_path, chapters, progress, recovery)
    assert result["materialized"] == [] and any("already has scoped decision" in b for b in result["blocked"][0]["blockers"])
    assert "zed" not in progress["registry"]
    (tmp_path / SCOPED_AUDIT_NAME).unlink()
    assert untouched_state(tmp_path) == before
    # a witness that names neither the mention nor the identity proves nothing even over an uncertain native
    write_root_approval(tmp_path, root, root_reviews(root, chapters, SOURCE_REVIEW, witness=(0, 0)))
    result = approve(tmp_path, chapters, progress, recovery)
    assert any("no source-review witness literally names" in b for b in result["blocked"][0]["blockers"])
    # the distinct-participant gate still blocks a family label whose scene names two participants
    pair = tmp_path / "pair"
    pair.mkdir()
    siblings = (
        "Zed appeared at the gate beside Mara. The hall was quiet. Auntie waved from the porch. Auntie is the sister of Zed. "
        "Rain fell on the roof. Dust blew."
    )
    chapters, progress, recovery, root = kin_staged(pair, siblings)
    malformed_audit(pair, root, [uncertain_record()] * sum(len(v["scopes"]) for v in root["variants"]))
    write_root_approval(pair, root, root_reviews(root, chapters, SOURCE_REVIEW))
    result = approve(pair, chapters, progress, recovery)
    assert result["materialized"] == [] and any("family label 'Auntie'" in b for b in result["blocked"][0]["blockers"])
    assert "zed" not in progress["registry"]


def test_root_approval_supersedes_exact_ambiguous_scopes_with_append_only_history(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    root = roots["zed"]
    # A prior ambiguity is unresolved evidence, not an identity/non-character decision. Strong exact-source reviews may
    # replace it, retaining why the scope was previously held. This deliberately does not loosen non-ambiguous conflicts.
    prior = []
    for record in _review_records(root):
        prior.append(
            {
                "chapter_sha256": record["chapter_sha256"],
                "quote_sha256": record["quote_sha256"],
                "label": record["label"],
                "span_start": record["span_start"],
                "canonical": "none",
                "decision": "ambiguous",
                "confidence": 0.0,
                "reason": "earlier exact-source uncertainty",
            }
        )
    (tmp_path / SCOPED_AUDIT_NAME).write_text(json.dumps({"records": prior}), encoding="utf-8")
    write_root_approval(tmp_path, root, root_reviews(root, chapters, SOURCE_REVIEW))

    assert approve(tmp_path, chapters, progress, recovery)["materialized"] == ["zed"]
    records = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert all(record["decision"] == "alias" and record["canonical"] == "zed" for record in records)
    assert all(record["history"] == [{"decision": "ambiguous", "canonical": "none", "confidence": 0.0, "reason": "earlier exact-source uncertainty"}] for record in records)


def test_malformed_individual_approval_entries_become_typed_rows_and_hold_the_whole_file(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    root = roots["zed"]
    native_audit(tmp_path, root)
    good = root_reviews(root, chapters)
    source_good = root_reviews(root, chapters, SOURCE_REVIEW)
    before = untouched_state(tmp_path)

    def mutated(reviews: list[dict], **change) -> list[dict]:
        return [{**reviews[0], **change}, *reviews[1:]]

    def held(reviews: list[dict], **over) -> dict:
        write_root_approval(tmp_path, root, reviews, **over)
        result = approve(tmp_path, chapters, progress, recovery)  # never raises
        assert result["materialized"] == [] and result["blocked"] == [] and len(result["malformed"]) == 1, result
        assert untouched_state(tmp_path) == before and "zed" not in progress["registry"] and "zed" not in progress["aliases"]
        (row,) = recovery.open_pending(ROOT_APPROVAL_STAGE)
        assert row["code"] == "root_approval_malformed" and row["evidence"]["entry_sha256"] == result["malformed"][0]["entry_sha256"]
        assert row["evidence"]["problem"] == result["malformed"][0]["problem"]
        return result

    held(good, action="approve_all")
    held(good, extra="field")
    held(good, note="approved after human review of the book")  # an untruthful provenance claim
    held(good, note=7)
    held(good, sha="not-a-sha")
    held(good[1:])  # a mention without a review
    held([*good, good[0]])  # a repeated review
    held([])
    held(mutated(good, proposal_sha256="1" * 64))
    held(mutated(good, provenance="human_reviewed"))
    held(mutated(good, provenance=["unhashable"]))
    held(mutated(good, verdict="uncertain"))
    held(mutated(good, scope={**good[0]["scope"], "span_start": good[0]["scope"]["span_start"] + 1}))
    held(mutated(good, scope={**good[0]["scope"], "label": ["unhashable"]}))
    held(mutated(good, reviewer_role="caretaker"))
    held(mutated(source_good, reviewer_role="reviewer"))
    held(mutated(source_good, reviewer_role=["unhashable"]))
    held(mutated(source_good, factual_basis=" "))
    held(mutated(source_good, factual_witnesses=[]))
    held(mutated(source_good, factual_witnesses=["not an object"]))
    held(mutated(source_good, factual_witnesses=[{"chapter_file": "x"}]))
    # a non-object entry and a repeated actor in one file are malformed entries too (one row each, deduplicated across batches)
    approval = json.loads((tmp_path / ROOT_APPROVAL_NAME).read_text())
    entry = {"action": "approve_root", "actor_id": "zed", "proposal_sha256": proposal_sha(root), "reviews": good, "note": "ok"}
    (tmp_path / ROOT_APPROVAL_NAME).write_text(json.dumps({**approval, "approvals": ["junk", entry, entry]}), encoding="utf-8")
    result = approve(tmp_path, chapters, progress, recovery)
    assert result["materialized"] == [] and len(result["malformed"]) == 2 and "zed" not in progress["registry"]
    size = len(recovery.entries())
    assert approve(tmp_path, chapters, progress, recovery)["malformed"] and len(recovery.entries()) == size
    assert {r["code"] for r in recovery.open_pending(ROOT_APPROVAL_STAGE)} == {"root_approval_malformed"}
    assert untouched_state(tmp_path) == before
    # fixing the file closes the typed rows and the root materializes; nothing stays pending
    write_root_approval(tmp_path, root, good)
    assert approve(tmp_path, chapters, progress, recovery)["materialized"] == ["zed"]
    assert recovery.open_pending(ROOT_APPROVAL_STAGE) == []
    # a removed file also closes a malformed row
    other = tmp_path / "gone"
    other.mkdir()
    chapters, registry = v2_fixture(other)
    progress, recovery, roots = staged_root_project(other, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    write_root_approval(other, roots["zed"], [])
    assert len(approve(other, chapters, progress, recovery)["malformed"]) == 1 and len(recovery.open_pending(ROOT_APPROVAL_STAGE)) == 1
    (other / ROOT_APPROVAL_NAME).unlink()
    assert approve(other, chapters, progress, recovery) == {"materialized": [], "already": [], "blocked": [], "malformed": []}
    assert recovery.open_pending(ROOT_APPROVAL_STAGE) == []


def test_root_approval_independent_duplicate_and_kind_gates(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, _draft_rows(chapters), [v2_draft(chapters)])
    root = roots["zed"]
    reviews = root_reviews(root, chapters, SOURCE_REVIEW)
    write_root_approval(tmp_path, root, reviews)

    def blockers_with(**state) -> list[str]:
        local = json.loads(json.dumps(progress))
        local["registry"].update(state.get("registry", {}))
        local["aliases"].update(state.get("aliases", {}))
        for name, value in state.get("files", {}).items():
            (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
        result = approve(tmp_path, chapters, local, recovery)
        assert result["materialized"] == [] and "zed" not in local["registry"]
        return [b for entry in result["blocked"] for b in entry["blockers"]]

    assert any("name 'Zed' is already the name of registered actor" in b for b in blockers_with(registry={"old_zed": {"name": "Zed"}}))
    assert any("variant 'Professor' is already an alias of ana" in b for b in blockers_with(aliases={"professor": "ana"}))
    assert any("literally carried by registered actor" in b for b in blockers_with(registry={"zed_lin": {"name": "Zed Lin"}}))
    assert any("original profile entry" in b for b in blockers_with(files={"characters.json": {"ana": {"name": "Ana"}, "zed": {"name": "Z"}}}))
    assert any("original voice entry" in b for b in blockers_with(files={"characters.json": {"ana": {"name": "Ana"}}, "voices.json": {"zed": {}}}))
    (tmp_path / "voices.json").write_text(json.dumps({"ana": {}}), encoding="utf-8")


def test_root_approval_never_collapses_into_the_narrator_or_a_non_living_kind(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    creature = {**v2_draft(chapters, "zephyr", "Zephyr"), "kind": "creature"}
    creature["witnesses"] = creature["citations"]["bio"] = creature["citations"]["look"] = [v2_witness(chapters, 0, 2)]
    progress, recovery, roots = staged_root_project(
        tmp_path, chapters, registry, [v2_row(chapters, 0, "Zephyr", "new_actor", "zephyr", 1)], [creature]
    )
    before = untouched_state(tmp_path)
    root = roots["zephyr"]
    write_root_approval(tmp_path, root, root_reviews(root, chapters, SOURCE_REVIEW, witness=(0, 2)))
    result = approve(tmp_path, chapters, progress, recovery)
    assert result["materialized"] == [] and any("living-capable kind" in b for b in result["blocked"][0]["blockers"])
    assert untouched_state(tmp_path) == before and "zephyr" not in progress["registry"]
    # staging does not refuse a draft that names the narrator, so the approval step must: no review can collapse a root into it
    (tmp_path / PROPOSALS_NAME).unlink()
    other = tmp_path / "narr"
    other.mkdir()
    chapters, registry = v2_fixture(other)
    named = v2_draft(chapters, "narrator", "Narrator")
    progress, recovery, roots = staged_root_project(
        other, chapters, registry, [v2_row(chapters, 0, "Zed", "new_actor", "narrator", 1)], [named]
    )
    root = roots["narrator"]
    write_root_approval(other, root, root_reviews(root, chapters, SOURCE_REVIEW))
    result = approve(other, chapters, progress, recovery)
    assert result["materialized"] == [] and any("narrator" in b for b in result["blocked"][0]["blockers"])
    assert "narrator" not in progress["registry"] and not (other / SCOPED_AUDIT_NAME).exists()


def test_root_approval_materializes_only_the_root_the_input_names(tmp_path: Path) -> None:
    chapters, registry = v2_fixture(tmp_path)
    other = v2_draft(chapters, "zephyr", "Zephyr")
    other["witnesses"] = other["citations"]["bio"] = other["citations"]["look"] = [v2_witness(chapters, 0, 2)]
    rows = [*_draft_rows(chapters), v2_row(chapters, 0, "Zephyr", "new_actor", "zephyr", 1)]
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, rows, [v2_draft(chapters), other])
    assert set(roots) == {"zed", "zephyr"}
    root = roots["zed"]
    native_audit(tmp_path, root)
    write_root_approval(tmp_path, root, root_reviews(root, chapters))
    assert approve(tmp_path, chapters, progress, recovery)["materialized"] == ["zed"]
    assert "zephyr" not in progress["registry"] and "zephyr" not in progress["aliases"]
    records = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert {r["canonical"] for r in records} == {"zed"}
    still = {row["code"] for row in recovery.open_pending(CONTEXT_V2_STAGE)}
    assert still == {"context_variant_draft_root", "context_new_actor"}  # the unapproved root and its mention stay pending
    # a stale approval (the proposal changed) names a sha that is no longer pending: refused, not re-targeted
    stale = {**root, "bio": root["bio"] + " Changed."}
    write_root_approval(tmp_path, root, root_reviews(root, chapters), sha=proposal_sha(stale))
    with pytest.raises(OperationalError):
        approve(tmp_path, chapters, progress, recovery)


def kin_staged(tmp_path: Path, sibling_text: str) -> tuple[list[Path], dict, RecoveryLedger, dict]:
    chapters = kin_fixture(tmp_path)
    chapters[0].write_text(sibling_text, encoding="utf-8")
    registry = {"ana": {"name": "Ana", "bio": "a scout"}}
    draft = {**v2_draft(chapters), "variant_links": [kin_link(chapters, "kinship", [(0, 3)])]}
    draft["witnesses"] = draft["citations"]["bio"] = draft["citations"]["look"] = [v2_witness(chapters, 0, 0)]
    rows = [kin_row(chapters, "Zed", 0, 0, 0), kin_row(chapters, "Auntie", 0, 2, 1)]
    progress, recovery, roots = staged_root_project(tmp_path, chapters, registry, rows, [draft])
    return chapters, progress, recovery, roots["zed"]


def test_family_only_label_cannot_join_a_root_when_the_scene_shows_two_participants(tmp_path: Path) -> None:
    alone = (
        "Zed appeared at the gate. The hall was quiet. Auntie waved from the porch. Auntie is the sister of Zed. "
        "Rain fell on the roof. Dust blew."
    )
    chapters, progress, recovery, root = kin_staged(tmp_path / "alone", alone) if (tmp_path / "alone").mkdir() is None else None
    write_root_approval(tmp_path / "alone", root, root_reviews(root, chapters, SOURCE_REVIEW))
    assert approve(tmp_path / "alone", chapters, progress, recovery)["materialized"] == ["zed"]  # one participant: the kinship link stands
    siblings = alone.replace("Zed appeared at the gate.", "Zed appeared at the gate beside Mara.")
    (tmp_path / "pair").mkdir()
    chapters, progress, recovery, root = kin_staged(tmp_path / "pair", siblings)
    assert {v["label"] for v in root["variants"]} == {"Zed", "Auntie"}
    before = untouched_state(tmp_path / "pair")
    write_root_approval(tmp_path / "pair", root, root_reviews(root, chapters, SOURCE_REVIEW))
    result = approve(tmp_path / "pair", chapters, progress, recovery)
    assert result["materialized"] == [] and any("family label 'Auntie'" in b and "ambiguous" in b for b in result["blocked"][0]["blockers"])
    assert untouched_state(tmp_path / "pair") == before and "zed" not in progress["registry"]


# ##################################################################
# exact decision memo
# identical source/options/schema calls reuse a syntactically valid result, while malformed
# JSON never becomes a reusable model decision.
def test_memoized_model_ask_reuses_only_valid_exact_request() -> None:
    calls = 0

    def ask(prompt: str, **kwargs) -> str:
        nonlocal calls
        calls += 1
        return '{"ok":true}' if prompt == "valid" else "not-json"

    cached = memoized_model_ask(ask)
    schema = {"type": "object"}
    assert cached("valid", max_tokens=10, response_schema=schema) == '{"ok":true}'
    assert cached("valid", max_tokens=10, response_schema=schema) == '{"ok":true}'
    assert cached("valid", max_tokens=11, response_schema=schema) == '{"ok":true}'
    assert cached("bad", max_tokens=10, response_schema=schema) == "not-json"
    assert cached("bad", max_tokens=10, response_schema=schema) == "not-json"
    assert calls == 4


# ##################################################################
# per-record classification recovery
# a citation failure is held once as typed pending/raw evidence while neighbouring valid rows
# remain checkpointable; it must not repair/replay the whole chunk.
def test_collect_classifications_holds_bad_citation_per_record_without_chunk_replay(tmp_path: Path) -> None:
    chapter = tmp_path / "01.txt"
    chapter.write_text("Ren greeted Zed. Zed thanked Ren.", encoding="utf-8")
    units = immutable_evidence_units([chapter])
    calls = 0

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        nonlocal calls
        calls += 1
        rows = []
        for index, option in enumerate(response_schema["properties"]["classifications"]["items"]["oneOf"]):
            branch = option.get("oneOf", [option])[0]
            candidate_id = branch["properties"]["candidate_id"]["enum"][0]
            witness = (
                next(line for line in prompt.splitlines() if line.startswith(candidate_id + " label="))
                .split("[")[1]
                .split("]")[0]
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": "new",
                    "identity": candidate_id,
                    "evidence_unit_ids": [] if index == 0 else [witness],
                }
            )
        return json.dumps({"classifications": rows})

    records, candidates, rejected = collect_classifications(
        tmp_path,
        0,
        [chapter],
        units,
        chapter.read_text(encoding="utf-8"),
        {"registry": {}, "aliases": {}},
        ask,
    )
    assert calls == 1
    assert len(records) == len(candidates) >= 2
    assert len(rejected) == 1
    failed = rejected[0]["candidate"]["id"]
    assert next(row for row in records if row["candidate_id"] == failed)["status"] == "ambiguous"
    archived = [json.loads(line) for line in (tmp_path / REJECTIONS_NAME).read_text(encoding="utf-8").splitlines()]
    assert len(archived) == 1 and archived[0]["code"] == "cast_record_rejected"
    assert rejected[0]["archive"]["line"] == 0


# ##################################################################
# cross-chunk cardinality recovery
# a genuine provider row for an adjacent known chunk is raw provenance, not a reason to
# discard this chunk's valid rows; the omitted expected candidate is the sole pending row.
def test_partition_classification_chunk_holds_only_omitted_expected_candidate() -> None:
    units = [
        {"id": "c00s00000", "chapter": "01.txt", "quote": "Ren arrived."},
        {"id": "c00s00001", "chapter": "01.txt", "quote": "Zed arrived."},
        {"id": "c00s00002", "chapter": "01.txt", "quote": "Mara arrived."},
    ]
    candidates = [
        {"id": "p0000", "label": "Ren", "ref_ids": ["c00s00000n0000"]},
        {"id": "p0001", "label": "Zed", "ref_ids": ["c00s00001n0000"]},
        {"id": "p0002", "label": "Mara", "ref_ids": ["c00s00002n0000"]},
    ]
    records, rejected = partition_classification_chunk(
        {
            "classifications": [
                {"candidate_id": "p0000", "status": "new", "identity": "p0000", "evidence_unit_ids": ["c00s00000"]},
                {"candidate_id": "p0002", "status": "new", "identity": "p0002", "evidence_unit_ids": ["c00s00002"]},
            ]
        },
        candidates[:2],
        {},
        {},
        candidates,
        units,
        [],
    )
    assert [record["candidate_id"] for record in records] == ["p0000", "p0001"]
    assert records[0]["status"] == "new" and records[1]["status"] == "ambiguous"
    assert [(item["candidate"]["id"], item["reason"]) for item in rejected] == [
        ("p0001", "classification omitted this candidate")
    ]


# ##################################################################
# immutable-scope cache remapping
# a cache hit may remap only ephemeral ledger IDs; changed offered facts invalidate it.
def test_classification_cache_remaps_only_unchanged_immutable_scope() -> None:
    units = [
        {
            "id": "c00s00000",
            "chapter": "01.txt",
            "chapter_sha256": "a" * 64,
            "quote": "Ren arrived.",
        }
    ]
    old = {"id": "p0000", "label": "Ren", "ref_ids": ["c00s00000n0000"], "known_owner": None}
    new = {"id": "p0099", "label": "Ren", "ref_ids": ["c00s00000n0000"], "known_owner": None}
    old_key, old_provenance = classification_offer_fingerprint([old], [old], {}, {}, units)
    new_key, new_provenance = classification_offer_fingerprint([new], [new], {}, {}, units)
    assert old_key == new_key
    encoded = cache_model_records(
        [{"candidate_id": "p0000", "status": "new", "identity": "p0000", "evidence_unit_ids": ["c00s00000"]}],
        old_provenance,
    )
    assert restore_cached_model_records(encoded, new_provenance) == [
        {"candidate_id": "p0099", "status": "new", "identity": "p0099", "evidence_unit_ids": ["c00s00000"]}
    ]
    changed_key, _ = classification_offer_fingerprint([new], [new], {"ren": {"name": "Ren"}}, {}, units)
    assert changed_key != old_key


# ##################################################################
# test known-owner candidate route
# a chain targeting a candidate whose label already has an established owner resolves to that canonical (never a candidate ID) and defers an unapproved mention to adjudication, identically across repeated runs.
def test_known_owner_candidate_chain_resolves_to_canonical_and_adjudicates() -> None:
    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "Louu spoke. Lou answered. Louu left."}]
    registry = {"louu": {"name": "Louu"}}
    aliases = {"louu": "louu"}
    candidates = candidate_coverage_ledger(units, registry, aliases)
    owned = next(c for c in candidates if c["label"] == "Louu")
    other = next(c for c in candidates if c["label"] == "Lou")
    assert owned["known_owner"] == "louu"
    targets = json.dumps(discovery_schema(["louu"], [other], candidates))
    assert owned["id"] not in targets and '"louu"' in targets
    for _ in range(3):
        pending: list[dict] = []
        record = {
            "candidate_id": other["id"],
            "status": "known",
            "identity": owned["id"],
            "evidence_unit_ids": ["c00s00000"],
        }
        fixed = {"candidate_id": owned["id"], "status": "known", "identity": "louu", "evidence_unit_ids": ["c00s00000"]}
        out = validate_classification_chunk(
            {"classifications": [dict(record), fixed]}, [other, owned], registry, aliases, candidates, units, pending
        )
        assert out[0]["identity"] == "louu"
        assert [item["proposed"] for item in pending] == ["louu"] and pending[0]["candidate"] is other
    genuine = {
        "candidate_id": other["id"],
        "status": "known",
        "identity": other["id"],
        "evidence_unit_ids": ["c00s00000"],
    }
    with pytest.raises(ValueError):
        validate_classification_chunk(
            {"classifications": [genuine, fixed]}, [other, owned], registry, aliases, candidates, units, []
        )


# ##################################################################
# title vocative boundary
# preserves exact title and narrative subject references without fabricating a title-name actor from missing dialogue punctuation.
def test_title_vocative_followed_by_attribution_verb_is_not_full_name(tmp_path: Path) -> None:
    from src.cast_freeze import (
        candidate_coverage_ledger,
        immutable_evidence_units,
        immutable_name_references,
    )

    chapter = tmp_path / "28-part_28.txt"
    chapter.write_text("Professor Taro intervened, my beetle needs crystals.", encoding="utf-8")
    units = immutable_evidence_units([chapter])
    labels = {candidate["label"] for candidate in candidate_coverage_ledger(units, {}, {})}
    references = immutable_name_references(units).values()
    assert "Professor" in labels
    assert any(reference["label"] == "Taro" for reference in references)
    assert "Professor Taro" not in labels


# ##################################################################
# proposed-new identity review with a production-like scene
# a hesitation-merged name, a misspelled registered name, a verb-bearing fragment and a letter-prefixed vocative are redirected or refused before acceptance, while a genuinely new person survives; uncertainty fails closed and raw provenance is persisted.
def new_identity_ask(
    verdict_for, calls: list[str], primary_status: str | None = None, adjudication: dict | None = None
):
    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        calls.append(prompt)
        schema = response_schema or {}
        props = schema["properties"]
        if "entities" in props:
            ids = props["entities"]["items"]["properties"]["id"]["enum"]
            witnesses = {}
            for line in prompt.split("SOURCE:\n", 1)[1].splitlines():
                witnesses[line[1:10]] = line
            rows = []
            for identity in ids:
                name = next(
                    part.split("name='")[1].split("'")[0]
                    for part in prompt.split("Proposed identities: ")[1].splitlines()[0].split("; ")
                    if part.startswith(identity + " ")
                )
                unit = next(key for key, line in witnesses.items() if name in line)
                rows.append({"id": identity, "eligibility": "living", "evidence_unit_ids": [unit]})
            return json.dumps({"entities": rows})
        item = props["mentions"]["items"]["properties"] if "mentions" in props else {}
        if "verdict" in item:
            rows = []
            lines = prompt.split("MENTIONS:\n", 1)[1].splitlines()
            label = prompt.split("label ", 1)[1].split(" was proposed")[0].strip("'")
            owners = [
                option[len("existing:") :] for option in item["verdict"]["enum"] if option.startswith("existing:")
            ]
            for index, line in enumerate(lines):
                if "mention:" in line:
                    unit_id = line.split("unit ")[1].split()[0]
                    rows.append(
                        {
                            "mention_id": line.split()[0],
                            "verdict": verdict_for(label, owners, line),
                            "witness_unit_ids": [unit_id],
                            "confidence": 0.9,
                            "reason": "scene shows it",
                        }
                    )
            return json.dumps({"mentions": rows})
        if "refers_to_person" in item:
            return json.dumps(
                {
                    "mentions": [
                        {
                            "mention_id": line.split()[0],
                            "refers_to_person": "no",
                            "candidate_kind": "nonliving",
                            "decision": "non_character",
                            "canonical": "none",
                            "confidence": 0.9,
                            "reason": "none",
                            **(adjudication or {}),
                        }
                        for line in prompt.splitlines()
                        if line.startswith("m") and "mention:" in line
                    ]
                }
            )
        out = []
        for option in schema["properties"]["classifications"]["items"]["oneOf"]:
            branches = option.get("oneOf", [option])
            cid = branches[0]["properties"]["candidate_id"]["enum"][0]
            row = next(line for line in prompt.splitlines() if line.startswith(cid + " label="))
            witness = row.split("witnesses: [")[1].split("]")[0].split(",")[0].strip("' ")
            statuses = {st: b["properties"] for b in branches for st in b["properties"]["status"]["enum"]}
            status = (
                primary_status
                if primary_status in statuses
                else "new"
                if "new" in statuses
                else "known"
                if "known" in statuses
                else next(iter(statuses))
            )
            identity = statuses[status]["identity"]["enum"][0]
            out.append({"candidate_id": cid, "status": status, "identity": identity, "evidence_unit_ids": [witness]})
        return json.dumps({"classifications": out})

    return ask


def new_identity_fixture(tmp_path: Path) -> tuple[Path, list[dict], dict]:
    chapter = tmp_path / "ch1.txt"
    chapter.write_text(
        'Um Taro rubbed his eyes in the dim hall. Lou stood beside the window and Lu frowned. Lou Arched a brow at the door. "E Lou, wait for us!" shouted Taro. Han stepped from the crowd, a stranger who had never met them.',
        encoding="utf-8",
    )
    progress = {
        "registry": {
            "taro": {"name": "Taro", "bio": "A teacher.", "look": ""},
            "lu": {"name": "Lu", "bio": "Sister of Taro, sometimes called Lou by neighbours.", "look": ""},
        },
        "aliases": {"lou": "lu"},
    }
    return chapter, immutable_evidence_units([chapter]), progress


def structural_verdict(label: str, owners: list[str], line: str) -> str:
    words = label.split()
    if len(words) == 1 and re.search(
        rf"(?:\b[A-Z][a-z]* {label}\b|\b{label} [A-Z][a-z]*\b)", line.split("mention:", 1)[1].split("bounded scene")[0]
    ):
        return "nonidentity_fragment"
    if words[0] in {"Um", "E"} and len(words) == 2 and normalized_owner(words[1], owners):
        return f"existing:{normalized_owner(words[1], owners)}"
    if label == "Lou":
        return "existing:lu"
    if len(words) == 2 and words[1] == "Arched":
        return "nonidentity_fragment"
    return "distinct_living_identity"


def normalized_owner(word: str, owners: list[str]) -> str | None:
    return {"Taro": "taro", "Lou": "lu"}.get(word) if {"Taro": "taro", "Lou": "lu"}.get(word) in owners else None


def test_new_identity_review_redirects_fragments_and_keeps_genuine_new(tmp_path: Path) -> None:
    from src.cast_freeze import (
        NEW_IDENTITY_AUDIT_NAME,
        SCOPED_AUDIT_NAME,
        discover_batch,
    )

    chapter, units, progress = new_identity_fixture(tmp_path)
    calls: list[str] = []
    discoveries, classifications = discover_batch(
        tmp_path,
        0,
        [chapter],
        units,
        chapter.read_text(),
        progress,
        set(),
        ask=new_identity_ask(structural_verdict, calls),
    )
    assert "han" in [item["id"] for item in discoveries]
    assert not any(item["id"] == "lu" for item in discoveries)  # no unsupported spelling-based owner rewrite
    audit = json.loads((tmp_path / NEW_IDENTITY_AUDIT_NAME).read_text())["records"]
    by_label = {record["label"]: record for record in audit}
    assert by_label["Han"]["verdict"] == "distinct_living_identity"
    assert (
        by_label["Um Taro"]["verdict"] == "distinct_living_identity"
        and by_label["Lou Arched"]["verdict"] == "nonidentity_fragment"
    )
    for record in audit:
        assert (
            len(record["chapter_sha256"]) == 64
            and len(record["quote_sha256"]) == 64
            and record["witness_unit_ids"]
            and record["raw_review"]["reason"]
        )
    scoped = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert ("Lou Arched", "non_character", "none") in {(r["label"], r["decision"], r["canonical"]) for r in scoped}
    assert not any(r["decision"] == "alias" and r["canonical"] == "lu" for r in scoped)
    assert any("bounded scene" in call and "Canonical prior facts" in call for call in calls)
    assert {c["status"] for c in classifications} <= {"known", "new", "non_character"}


def pending_rows_of(project: Path, code: str) -> list[dict]:
    from src.data_recovery import RecoveryLedger

    return [row for row in RecoveryLedger(project).open_pending("cast") if row["code"] == code]


def test_new_identity_review_uncertain_is_typed_pending_not_fatal(tmp_path: Path) -> None:
    from src.cast_freeze import NEW_IDENTITY_AUDIT_NAME, PROPOSALS_NAME, discover_batch

    chapter, units, progress = new_identity_fixture(tmp_path)
    before = json.dumps(progress, sort_keys=True)
    discoveries, classifications = discover_batch(
        tmp_path,
        0,
        [chapter],
        units,
        chapter.read_text(),
        progress,
        set(),
        ask=new_identity_ask(lambda label, owners, line: "uncertain", []),
    )
    audit = json.loads((tmp_path / NEW_IDENTITY_AUDIT_NAME).read_text())["records"]
    assert audit and all(record["verdict"] == "uncertain" for record in audit)
    # nothing is registered, aliased or merged; every held candidate is non_character
    assert discoveries == [] and json.dumps(progress, sort_keys=True) == before
    assert "new" not in {item["status"] for item in classifications}
    assert all(item["identity"] in progress["registry"] for item in classifications if item["status"] == "known")
    rows = pending_rows_of(tmp_path, "pending_uncertain_living")
    assert rows and {row["severity"] for row in rows} == {"pending"}
    source = chapter.read_text()
    proposals = [json.loads(line) for line in (tmp_path / PROPOSALS_NAME).read_text().splitlines()]
    assert {entry["type"] for entry in proposals} == {"uncertain_living"}
    for entry in proposals:
        # literal mentions: the exact quote, label span and unit survive verbatim, with the recorded verdict
        for mention in entry["mentions"]:
            assert mention["quote"] in source
            assert mention["quote"][mention["span_start"] :].startswith(mention["label"])
            assert mention["verdict"] == "uncertain" and mention["unit_id"]
        assert entry["facts"] and all(fact in source for fact in entry["facts"])
    for row in rows:
        assert row["evidence"]["source"] == [chapter.name] and row["evidence"]["proposal_sha256"]


def test_ambiguous_named_person_is_new_identity_pending_not_forced_or_fatal(tmp_path: Path) -> None:
    from src.cast_freeze import PROPOSALS_NAME, SCOPED_AUDIT_NAME, discover_batch

    chapter = tmp_path / "ch1.txt"
    chapter.write_text("Zed stepped into the hall and greeted Lu. Lu smiled back.", encoding="utf-8")
    units = immutable_evidence_units([chapter])
    progress = {"registry": {"lu": {"name": "Lu", "bio": "A teacher.", "look": ""}}, "aliases": {}}
    before = json.dumps(progress, sort_keys=True)
    adjudication = {
        "refers_to_person": "yes",
        "candidate_kind": "individual_name",
        "decision": "ambiguous",
        "reason": "a named person no listed owner is shown to be",
    }
    prompts: list[str] = []
    discoveries, classifications = discover_batch(
        tmp_path,
        0,
        [chapter],
        units,
        chapter.read_text(),
        progress,
        set(),
        ask=new_identity_ask(
            lambda label, owners, line: "uncertain", prompts, primary_status="ambiguous", adjudication=adjudication
        ),
    )
    rows = pending_rows_of(tmp_path, "pending_new_identity")
    assert len(rows) == 1 and rows[0]["evidence"]["label"] == "Zed"
    assert discoveries == [] and json.dumps(progress, sort_keys=True) == before
    held = next(item for item in classifications if item["status"] == "non_character")
    assert held["identity"] == "none"
    # the scoped audit keeps the exact ambiguous decision (no alias, no owner forced)
    scoped = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert [(r["decision"], r["canonical"]) for r in scoped] == [("ambiguous", "none")]
    entry = json.loads((tmp_path / PROPOSALS_NAME).read_text().splitlines()[0])
    assert (
        entry["type"] == "new_identity" and entry["mentions"][0]["quote"] == "Zed stepped into the hall and greeted Lu."
    )
    assert any("never force an owner" in prompt for prompt in prompts)


def test_uncertain_after_invalid_existing_owner_is_unapproved_alias_pending(tmp_path: Path) -> None:
    from src.cast_freeze import discover_batch

    chapter = tmp_path / "ch1.txt"
    chapter.write_text("Lou frowned at the quiet door. Rin watched from the stair.", encoding="utf-8")
    units = immutable_evidence_units([chapter])
    progress = {"registry": {"lu": {"name": "Lu", "bio": "A teacher.", "look": ""}}, "aliases": {}}

    def verdict(label: str, owners: list[str], line: str) -> str:
        # claims the owner while it is offered (the source gives it no support), withdraws once it is removed
        return f"existing:{owners[0]}" if owners else "uncertain"

    calls: list[str] = []
    discoveries, classifications = discover_batch(
        tmp_path, 0, [chapter], units, chapter.read_text(), progress, set(), ask=new_identity_ask(verdict, calls)
    )
    assert not any(item["status"] == "known" for item in classifications)
    assert progress["aliases"] == {} and all(item["id"] != "lu" for item in discoveries)
    assert not any("existing:lu" in call for call in calls)  # Lu is not a source-grounded owner option for Lou


# ##################################################################
# same-provisional review sequencing
# a variant of another proposed-new name is linked to that provisional identity during review, before materialization could refuse it; a link to a target that is not independently introduced fails closed.
def provisional_fixture(tmp_path: Path) -> tuple[Path, list[dict], dict]:
    chapter = tmp_path / "ch1.txt"
    chapter.write_text(
        "Jun raised a shield beside the gate. Taro watched while Jun grinned. Jun that is June nodded at the gate, and soon June ran home.",
        encoding="utf-8",
    )
    progress = {"registry": {"taro": {"name": "Taro", "bio": "A teacher.", "look": ""}}, "aliases": {}}
    return chapter, immutable_evidence_units([chapter]), progress


def provisional_ask(verdicts: dict[str, str], calls: list[str]):
    base = new_identity_ask(lambda label, owners, line: verdicts.get(label, "distinct_living_identity"), calls)

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        response = json.loads(base(prompt, max_tokens, max_attempts, response_schema))
        if "mentions" in response and response["mentions"] and "verdict" in response["mentions"][0]:
            label = prompt.split("label ", 1)[1].split(" was proposed")[0].strip("'")
            options = response_schema["properties"]["mentions"]["items"]["properties"]["verdict"]["enum"]
            for row in response["mentions"]:
                if row["verdict"] == "same_provisional":
                    row["verdict"] = next(option for option in options if option.startswith("same_provisional:"))
                    row["witness_unit_ids"] = (
                        [
                            next(
                                line.split("unit ")[1].split()[0]
                                for line in prompt.splitlines()
                                if "mention:" in line
                                and label in line.split("mention:")[1]
                                and "June" in line
                                and "Jun " in line
                            )
                        ]
                        if label == "June"
                        else row["witness_unit_ids"]
                    )
        return json.dumps(response)

    return ask


def test_variant_resolves_via_same_provisional_before_materialization(tmp_path: Path) -> None:
    from src.cast_freeze import discover_batch

    chapter, units, progress = provisional_fixture(tmp_path)
    discoveries, classifications = discover_batch(
        tmp_path,
        0,
        [chapter],
        units,
        chapter.read_text(),
        progress,
        set(),
        ask=provisional_ask({"June": "same_provisional"}, []),
    )
    assert [item["id"] for item in discoveries] == ["jun"]
    by_label = {item["candidate_id"]: item for item in classifications}
    assert (
        by_label["p0001"]["status"] == "known"
        and by_label["p0001"]["identity"] == "p0000"
        and by_label["p0000"]["status"] == "new"
    )
    assert discoveries[0]["aliases"] == ["June"]


def provisional_review_inputs(tmp_path: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    from src.cast_freeze import candidate_coverage_ledger

    _chapter, units, progress = provisional_fixture(tmp_path)
    candidates = candidate_coverage_ledger(units, progress["registry"], progress["aliases"])
    records = [
        {
            "candidate_id": candidate["id"],
            "status": "new",
            "identity": candidate["id"],
            "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]],
        }
        for candidate in candidates
        if candidate["label"].startswith("Jun")
    ]
    return units, records, candidates, progress


def test_same_provisional_to_non_introduced_target_is_garbled_variant_pending(tmp_path: Path) -> None:
    from src.cast_freeze import (
        PENDING_IDENTITY_TYPES,
        demote_deferred,
        review_proposed_identities,
    )

    units, records, candidates, progress = provisional_review_inputs(tmp_path)
    ask = provisional_ask({"June": "same_provisional", "Jun": "nonidentity_fragment"}, [])
    # strict mode (no pending lane) still fails closed
    with pytest.raises(RuntimeError, match="independently introduced"):
        review_proposed_identities(tmp_path, units, [dict(r) for r in records], candidates, progress, ask)
    deferred: list[dict] = []
    review_proposed_identities(tmp_path, units, records, candidates, progress, ask, deferred)
    assert [entry["label"] for entry in deferred] == ["June"] and deferred[0]["type"] in PENDING_IDENTITY_TYPES
    entry = deferred[0]
    assert entry["type"] == "garbled_variant" and "independently introduced" in entry["reason"]
    source = " ".join(unit["quote"] for unit in units)
    for mention in entry["mentions"]:
        assert (
            mention["quote"] in source and mention["verdict"].startswith("same_provisional:") and mention["witnesses"]
        )
        assert all(w["quote"] in source for w in mention["witnesses"])
    # the held variant is never linked to a target or merged
    demote_deferred(records, candidates, units, deferred)
    june = next(item for item in records if item["candidate_id"] == entry["candidate_id"])
    assert (june["status"], june["identity"]) == ("non_character", "none") and progress["aliases"] == {}


def test_demote_deferred_cascades_to_candidates_linked_to_a_pending_one(tmp_path: Path) -> None:
    from src.cast_freeze import demote_deferred, pending_identity_entry
    from src.cast_freeze import immutable_name_references as references_of

    units, records, candidates, _ = provisional_review_inputs(tmp_path)
    by_label = {candidate["label"]: candidate for candidate in candidates}
    root, linked = by_label["June"], by_label["Jun"]
    for item in records:
        if item["candidate_id"] == linked["id"]:
            item["status"], item["identity"] = "known", root["id"]
    deferred = [pending_identity_entry(root, "uncertain_living", "open", units, references_of(units))]
    demote_deferred(records, candidates, units, deferred)
    assert {item["status"] for item in records} == {"non_character"}
    assert [entry["type"] for entry in deferred] == ["uncertain_living", "garbled_variant"]
    assert deferred[1]["label"] == linked["label"] and "pending identity" in deferred[1]["reason"]


def test_failed_provisional_link_converges_to_a_clean_batch(tmp_path: Path) -> None:
    from src.cast_freeze import discover_batch

    chapter, units, progress = provisional_fixture(tmp_path)
    base = provisional_ask({"June": "same_provisional", "Jun": "nonidentity_fragment"}, [])

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        # once the fragment is bound there is no provisional target to offer; the model then keeps June as its own identity
        offered = json.dumps(response_schema or {})
        if "label 'June' was proposed" in prompt and "same_provisional:" not in offered:
            return new_identity_ask(lambda label, owners, line: "distinct_living_identity", [])(
                prompt, max_tokens, max_attempts, response_schema
            )
        return base(prompt, max_tokens, max_attempts, response_schema)

    discoveries, _ = discover_batch(tmp_path, 0, [chapter], units, chapter.read_text(), progress, set(), ask=ask)
    assert [item["id"] for item in discoveries] == ["june"]
    assert pending_rows_of(tmp_path, "pending_garbled_variant") == []


# bounded correction of invalid proposed-new review results
# provisional options are limited to lexically/role-continuous names; an invalid existing target is re-asked once without that target, never repaired with manual aliases.
def test_provisional_plausibility_keeps_lou_lu_and_rejects_han_sora() -> None:
    from src.cast_freeze import provisional_plausible

    assert (
        provisional_plausible("Lou", "Lu")
        and provisional_plausible("Jun", "June")
        and provisional_plausible("Mom", "Mother")
    )
    assert not provisional_plausible("Mom", "Lu") and not provisional_plausible("Han", "Sora")
    # Han no Father/Mother
    assert not provisional_plausible("Han", "Father") and not provisional_plausible("Han", "Mother")
    # than not Han
    assert not provisional_plausible("than", "Han") and not provisional_plausible("Han", "than")


def test_existing_target_needs_own_mention_and_rejects_scene_participant() -> None:
    from src.cast_freeze import review_verdict_error

    registry = {"sora": {"name": "Sora"}}
    units = {"u1": {"id": "u1", "quote": "Han spoke to Sora."}, "u2": {"id": "u2", "quote": "Sora smiled."}}
    candidate = {"ref_ids": ["u1n0"]}
    assert review_verdict_error(
        "Han", "existing:sora", ["u1"], candidate, {}, registry, {}, units, units["u1"]
    ).startswith("new-identity review lacks independent owner proof for existing 'sora'")
    assert "own-label witness" in review_verdict_error(
        "Han", "existing:sora", ["u2"], candidate, {}, registry, {}, units, units["u1"]
    )

    # Jun/Ren own unit reject
    jun_ren_units = {"u1": {"id": "u1", "quote": "Jun stepped past Ren."}}
    ren_registry = {"ren": {"name": "Ren"}}
    assert review_verdict_error(
        "Jun", "existing:ren", ["u1"], candidate, {}, ren_registry, {}, jun_ren_units, jun_ren_units["u1"]
    ).startswith("new-identity review lacks independent owner proof for existing 'ren'")

    # disjoint Lou/Lu reject
    lone_disjoint = {"u3": {"id": "u3", "quote": "Lou smiled."}}
    lu = {
        "lu": {
            "name": "Lu",
            "source_facts": "Lou is Lu's recorded academy name.",
            "facts": {"look": ["Tall quiet student with a nightbat companion."]},
        }
    }
    assert (
        review_verdict_error("Lou", "existing:lu", ["u3"], candidate, {}, lu, {}, lone_disjoint, lone_disjoint["u3"])
        == "new-identity review lacks source/registry support for existing 'lu'"
    )

    # nightbat anchor accept
    lone = {"u3": {"id": "u3", "quote": "Lou's nightbat watched."}}
    assert review_verdict_error("Lou", "existing:lu", ["u3"], candidate, {}, lu, {}, lone, lone["u3"]) is None


def test_same_provisional_rejects_enumerated_distinct_actors_and_chaining() -> None:
    from src.cast_freeze import review_verdict_error

    cand_jun = {"id": "p0001", "label": "Jun", "ref_ids": ["u1n0"]}
    cand_june = {"id": "p0002", "label": "June", "ref_ids": ["u1n1"]}
    plausible = {"p0002": cand_june}
    enum_units = {"u1": {"id": "u1", "quote": "Jun and June nodded at the gate."}}

    # reject enumerated distinct actors
    err = review_verdict_error(
        "Jun", "same_provisional:p0002", ["u1"], cand_jun, plausible, {}, {}, enum_units, enum_units["u1"]
    )
    assert err is not None and "enumerated distinct actors" in err

    # no chain into existing
    cand_june_known = {"id": "p0002", "label": "June", "ref_ids": ["u1n1"], "known_owner": "taro"}
    err_chain = review_verdict_error(
        "Jun",
        "same_provisional:p0002",
        ["u1"],
        cand_jun,
        {"p0002": cand_june_known},
        {},
        {},
        enum_units,
        enum_units["u1"],
    )
    assert err_chain is not None and "cannot chain" in err_chain


def test_invalid_existing_target_is_reasked_without_it(tmp_path: Path) -> None:
    from src.cast_freeze import discover_batch

    chapter, units, progress = provisional_fixture(tmp_path)
    progress["registry"]["sora"] = {"name": "Sora", "bio": "A guard.", "look": ""}
    seen: list[list[str]] = []
    base = new_identity_ask(lambda label, owners, line: "distinct_living_identity", [])

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        response = json.loads(base(prompt, max_tokens, max_attempts, response_schema))
        if "mentions" in response and response["mentions"] and "verdict" in response["mentions"][0]:
            options = response_schema["properties"]["mentions"]["items"]["properties"]["verdict"]["enum"]
            seen.append(options)
            if "existing:sora" in options:
                for row in response["mentions"]:
                    row["verdict"] = "existing:sora"
        return json.dumps(response)

    discover_batch(tmp_path, 0, [chapter], units, chapter.read_text(), progress, set(), ask=ask)
    # An unrelated roster member is never offered as a target merely because a
    # generic reviewer might otherwise select it.
    assert seen and all("existing:sora" not in options for options in seen)


# ##################################################################
# immutable own-source witness
# the mention's own unit is deterministic input attached with provenance, never a model choice, so an exact nonidentity verdict needs no repeated own ID among all batch units.
def review_override_ask(base, decide):
    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        item = (
            response_schema["properties"].get("mentions", {}).get("items", {}).get("properties", {})
            if response_schema
            else {}
        )
        if "verdict" not in item:
            return base(prompt, max_tokens, max_attempts, response_schema)
        rows = []
        for line in prompt.split("MENTIONS:\n", 1)[1].splitlines():
            if "mention:" in line:
                verdict, witnesses = decide(prompt, line.split("unit ")[1].split()[0], item["verdict"]["enum"])
                rows.append(
                    {
                        "mention_id": line.split()[0],
                        "verdict": verdict,
                        "witness_unit_ids": witnesses,
                        "confidence": 0.9,
                        "reason": "scene shows it",
                    }
                )
        return json.dumps({"mentions": rows})

    return ask


def test_nonidentity_review_accepts_zero_cited_witnesses_and_records_own_source(tmp_path: Path) -> None:
    from src.cast_freeze import NEW_IDENTITY_AUDIT_NAME, discover_batch

    chapter = tmp_path / "ch1.txt"
    chapter.write_text(
        "The wind rose over the hills. Kyestus rubbed his hands near the fire. The kettle began to sing.",
        encoding="utf-8",
    )
    units = immutable_evidence_units([chapter])
    progress = {"registry": {"taro": {"name": "Taro", "bio": "A teacher.", "look": ""}}, "aliases": {}}
    ask = review_override_ask(
        new_identity_ask(lambda label, owners, line: "distinct_living_identity", []),
        lambda prompt, unit_id, options: ("nonidentity_fragment", []),
    )
    discoveries, _ = discover_batch(tmp_path, 0, [chapter], units, chapter.read_text(), progress, set(), ask=ask)
    assert discoveries == []
    record = next(
        r for r in json.loads((tmp_path / NEW_IDENTITY_AUDIT_NAME).read_text())["records"] if r["label"] == "Kyestus"
    )
    own_unit = next(u for u in units if "Kyestus" in u["quote"])
    own = own_unit["id"]
    assert record["verdict"] == "nonidentity_fragment" and record["witness_unit_ids"] == [own]
    assert record["own_source_witness"] == {
        "unit_id": own,
        "provenance": "immutable_candidate_reference",
        "label": "Kyestus",
        "span_start": own_unit["quote"].index("Kyestus"),
    }
    assert record["raw_review"]["witness_unit_ids"] == []


# ##################################################################
# one continuous consistency review
# a label with more conflicting scopes than one normal review chunk is reconciled in a single call over compacted episodes that render each unit once, with no scope dropped.
def test_many_conflicting_scopes_reviewed_once_over_compact_episodes(tmp_path: Path) -> None:
    from src.cast_freeze import (
        NEW_IDENTITY_AUDIT_NAME,
        NEW_IDENTITY_MENTIONS_PER_CALL,
        discover_batch,
    )

    chapter = tmp_path / "ch1.txt"
    count = NEW_IDENTITY_MENTIONS_PER_CALL * 2 + 1
    chapter.write_text(
        "Lu waited by the gate. " + " ".join(f"Lou nodded at post {index}." for index in range(count)), encoding="utf-8"
    )
    units = immutable_evidence_units([chapter])
    progress = {
        "registry": {
            "lu": {
                "name": "Lu",
                "bio": "Sister of Taro.",
                "source_facts": "Lou is Lu's recorded academy name.",
                "look": "",
            }
        },
        "aliases": {},
    }
    anchor = units[0]["id"]
    seen: list[str] = []

    def decide(prompt: str, unit_id: str, options: list[str]):
        if "CONSISTENCY REVIEW" in prompt:
            if prompt not in seen:
                seen.append(prompt)
            return "existing:lu", [anchor]
        return ("existing:lu", [anchor]) if int(unit_id[-5:]) % 2 else ("distinct_living_identity", [])

    ask = review_override_ask(new_identity_ask(lambda label, owners, line: "distinct_living_identity", []), decide)
    discoveries, _ = discover_batch(tmp_path, 0, [chapter], units, chapter.read_text(), progress, set(), ask=ask)
    assert discoveries == []
    assert len(seen) == 1 and seen[0].count("mention:") == count and "CONTINUOUS SOURCE EPISODES" in seen[0]
    episodes = seen[0].split("CONTINUOUS SOURCE EPISODES", 1)[1].split("MENTIONS:", 1)[0]
    assert all(episodes.count(f"[{unit['id']}]") == 1 for unit in units)
    records = json.loads((tmp_path / NEW_IDENTITY_AUDIT_NAME).read_text())["records"]
    assert len([r for r in records if r["label"] == "Lou"]) == count and all(
        r["verdict"] == "existing:lu" and r.get("reconciled") for r in records if r["label"] == "Lou"
    )


# ##################################################################
# bounded correction inside the consistency review
# an invalid same_provisional (no literal dual-label witnesses) in the consistency review gets the same bounded reask as an ordinary review, with the invalid option removed and the conflict context kept; the first answer survives in history.
def test_consistency_review_reasks_invalid_same_provisional_with_option_removed(tmp_path: Path) -> None:
    from src.cast_freeze import NEW_IDENTITY_AUDIT_NAME, discover_batch

    chapter, units, progress = provisional_fixture(tmp_path)
    first_unit = next(u["id"] for u in units if u["quote"].startswith("Jun raised"))
    both_unit = next(u["id"] for u in units if "Jun that is June" in u["quote"])
    reasks: list[tuple[str, list[str]]] = []

    def decide(prompt: str, unit_id: str, options: list[str]):
        label = prompt.split("label ", 1)[1].split(" was proposed")[0].strip("'")
        target = next((option for option in options if option.startswith("same_provisional:")), None)
        if label != "Jun":
            return "distinct_living_identity", []
        if "CONSISTENCY REVIEW" not in prompt:
            return (target, [both_unit]) if unit_id == both_unit else ("distinct_living_identity", [])
        if "previous answer for this exact mention was invalid" in prompt:
            reasks.append((prompt, options))
        # consistency review: the first-unit mention wrongly claims same_provisional with no dual-label witness
        return (target, []) if unit_id == first_unit and target else ("distinct_living_identity", [])

    ask = review_override_ask(new_identity_ask(lambda label, owners, line: "distinct_living_identity", []), decide)
    discover_batch(tmp_path, 0, [chapter], units, chapter.read_text(), progress, set(), ask=ask)
    assert (
        len(reasks) == 1
        and "CONSISTENCY REVIEW" in reasks[0][0]
        and not any(option.startswith("same_provisional:") for option in reasks[0][1])
    )
    records = {
        r["witness_unit_ids"][0]: r
        for r in json.loads((tmp_path / NEW_IDENTITY_AUDIT_NAME).read_text())["records"]
        if r["label"] == "Jun"
    }
    assert records[first_unit]["verdict"] == "distinct_living_identity" and records[first_unit]["reconciled"]
    assert records[first_unit]["history"][-1]["verdict"] == "distinct_living_identity"


# ##################################################################
# source-provisional variant refs in a new-root review
# a short form the primary classification already mapped known->new root is same-participant context for the root's review (literal and cycle guarded), never a target and never solely status=new candidates.
def variant_fixture(tmp_path: Path):
    from src.cast_freeze import candidate_coverage_ledger, immutable_evidence_units

    chapter = tmp_path / "ch1.txt"
    chapter.write_text(
        "Kyle raised a shield beside the gate. Taro watched while Ky grinned. Kyle nodded at the gate.",
        encoding="utf-8",
    )
    units = immutable_evidence_units([chapter])
    candidates = candidate_coverage_ledger(units, {"taro": {"name": "Taro", "bio": "A teacher.", "look": ""}}, {})
    by_label = {candidate["label"]: candidate for candidate in candidates}
    return units, candidates, by_label


def test_variant_refs_follow_known_to_new_root_with_guards(tmp_path: Path) -> None:
    from src.cast_freeze import immutable_name_references, provisional_variant_refs

    units, candidates, by_label = variant_fixture(tmp_path)
    units_by_id = {unit["id"]: unit for unit in units}
    references = immutable_name_references(units)
    root, short = by_label["Kyle"], by_label["Ky"]
    records = [
        {"candidate_id": root["id"], "status": "new", "identity": root["id"], "evidence_unit_ids": ["x"]},
        {"candidate_id": short["id"], "status": "known", "identity": root["id"], "evidence_unit_ids": ["x"]},
    ]
    variants = provisional_variant_refs(candidates, records, references, units_by_id)
    assert [v["candidate"]["id"] for v in variants[root["id"]]] == [short["id"]]
    # cycle: the root pointing back at its variant yields no root at all
    cyclic = [{**records[0], "status": "known", "identity": short["id"]}, records[1]]
    assert provisional_variant_refs(candidates, cyclic, references, units_by_id) == {}
    # a root that is not a living new identity (or a registry-bound short form) is no provisional root
    assert (
        provisional_variant_refs(
            candidates,
            [{**records[0], "status": "non_character", "identity": "none"}, records[1]],
            references,
            units_by_id,
        )
        == {}
    )
    assert (
        provisional_variant_refs(candidates, [records[0], {**records[1], "identity": "taro"}], references, units_by_id)
        == {}
    )
    # literal guard: a reference whose label bytes are not at its recorded span is dropped
    forged = {ref_id: {**reference, "start": reference["start"] + 1} for ref_id, reference in references.items()}
    assert provisional_variant_refs(candidates, records, forged, units_by_id) == {}


def test_new_root_review_prompt_carries_variant_refs_and_distinct_guidance(tmp_path: Path) -> None:
    from src.cast_freeze import review_proposed_identities

    units, candidates, by_label = variant_fixture(tmp_path)
    root, short, taro = by_label["Kyle"], by_label["Ky"], by_label["Taro"]
    records = [
        {"candidate_id": root["id"], "status": "new", "identity": root["id"], "evidence_unit_ids": [units[0]["id"]]},
        {"candidate_id": short["id"], "status": "known", "identity": root["id"], "evidence_unit_ids": [units[1]["id"]]},
        {"candidate_id": taro["id"], "status": "known", "identity": "taro", "evidence_unit_ids": [units[1]["id"]]},
    ]
    calls: list[str] = []
    progress = {"registry": {"taro": {"name": "Taro", "bio": "A teacher.", "look": ""}}, "aliases": {}}
    ask = new_identity_ask(lambda label, owners, line: "distinct_living_identity", calls)
    review_proposed_identities(tmp_path, units, records, candidates, progress, ask)
    review = next(call for call in calls if "was proposed as a NEW character identity" in call)
    assert "SOURCE-PROVISIONAL VARIANT REFS" in review and "label 'Ky' (" + short["id"] + ")" in review
    episodes = review.split("CONTINUOUS SOURCE EPISODES", 1)[1].split("SOURCE-PROVISIONAL VARIANT REFS", 1)[0]
    assert episodes.count(f"[{units[1]['id']}]") == 1 and "scene: episode E1" in review
    assert (
        "No existing match is NOT the same as nonliving or uncertain" in review
        and "supports distinct_living_identity" in review
    )


# ##################################################################
# episode owner anchor in consistency review
# a Lou->Lu verdict needs no literal owner name and no model-cited unit when the mention's own reviewed episode holds a registry anchor; the exact anchor unit is preserved as provenance and recorded as a witness.
def test_episode_anchor_supports_owner_without_literal_name_or_citation(tmp_path: Path) -> None:
    from src.cast_freeze import (
        NEW_IDENTITY_AUDIT_NAME,
        discover_batch,
        review_verdict_error,
    )

    units = {"u3": {"id": "u3", "quote": "Lou smiled."}, "u4": {"id": "u4", "quote": "A nightbat watched."}}
    lu = {
        "lu": {
            "name": "Lu",
            "source_facts": "Lou is Lu's recorded academy name.",
            "facts": {"look": ["Tall quiet student with a nightbat companion."]},
        }
    }
    candidate = {"ref_ids": ["u3n0"]}
    assert (
        review_verdict_error("Lou", "existing:lu", ["u3"], candidate, {}, lu, {}, units, units["u3"])
        == "new-identity review lacks source/registry support for existing 'lu'"
    )
    assert (
        review_verdict_error(
            "Lou", "existing:lu", ["u3"], candidate, {}, lu, {}, units, units["u3"], [units["u3"], units["u4"]]
        )
        is None
    )

    chapter = tmp_path / "ch1.txt"
    chapter.write_text(
        "The nightbat circled the gate. Lou nodded at the post. Lou grinned by the post later.", encoding="utf-8"
    )
    source_units = immutable_evidence_units([chapter])
    anchor = source_units[0]["id"]
    progress = {
        "registry": {
            "lu": {
                "name": "Lu",
                "bio": "A student.",
                "source_facts": "Lou is Lu's recorded academy name.",
                "look": "Owns a nightbat.",
            }
        },
        "aliases": {},
    }

    def decide(prompt: str, unit_id: str, options: list[str]):
        if "CONSISTENCY REVIEW" in prompt:
            return "existing:lu", []
        return ("existing:lu", [anchor]) if unit_id == source_units[1]["id"] else ("distinct_living_identity", [])

    ask = review_override_ask(new_identity_ask(lambda label, owners, line: "distinct_living_identity", []), decide)
    discoveries, _ = discover_batch(tmp_path, 0, [chapter], source_units, chapter.read_text(), progress, set(), ask=ask)
    assert discoveries == []
    records = [
        r for r in json.loads((tmp_path / NEW_IDENTITY_AUDIT_NAME).read_text())["records"] if r["label"] == "Lou"
    ]
    assert records and all(r["verdict"] == "existing:lu" for r in records)
    reconciled = next(r for r in records if r["witness_unit_ids"][0] == source_units[2]["id"])
    assert (
        reconciled["provenance"]["witness_unit_id"] == anchor == reconciled["episode_support_unit_id"]
        and anchor in reconciled["witness_unit_ids"]
    )
    assert (
        reconciled["own_source_witness"]["unit_id"] == source_units[2]["id"]
        and reconciled["raw_review"]["witness_unit_ids"] == []
    )


def test_root_is_offered_through_its_variant_refs_label(tmp_path: Path) -> None:
    from src.cast_freeze import candidate_coverage_ledger, review_proposed_identities

    chapter = tmp_path / "ch1.txt"
    chapter.write_text(
        "Sparrow Tail drew a bow beside the gate. Jun grinned at the gate. Junn cheered at the gate.", encoding="utf-8"
    )
    units = immutable_evidence_units([chapter])
    candidates = candidate_coverage_ledger(units, {}, {})
    by_label = {candidate["label"]: candidate for candidate in candidates}
    root, variant, typo = by_label["Sparrow Tail"], by_label["Jun"], by_label["Junn"]
    records = [
        {"candidate_id": root["id"], "status": "new", "identity": root["id"], "evidence_unit_ids": [units[0]["id"]]},
        {
            "candidate_id": variant["id"],
            "status": "known",
            "identity": root["id"],
            "evidence_unit_ids": [units[1]["id"]],
        },
        {"candidate_id": typo["id"], "status": "new", "identity": typo["id"], "evidence_unit_ids": [units[2]["id"]]},
    ]
    options: dict[str, list[str]] = {}
    base = new_identity_ask(lambda label, owners, line: "distinct_living_identity", [])

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        item = (
            response_schema["properties"].get("mentions", {}).get("items", {}).get("properties", {})
            if response_schema
            else {}
        )
        if "verdict" in item:
            options[prompt.split("label ", 1)[1].split(" was proposed")[0].strip("'")] = item["verdict"]["enum"]
        return base(prompt, max_tokens, max_attempts, response_schema)

    review_proposed_identities(tmp_path, units, records, candidates, {"registry": {}, "aliases": {}}, ask)
    assert f"same_provisional:{root['id']}" in options["Junn"]
    assert not any(option.startswith("same_provisional:") for option in options["Sparrow Tail"])


# ##################################################################
# scoped alias proof: an owner proposal never proves identity
# generic concepts (distinct speaker, title/role, another cast member's name word, missing continuity) decided only from the mention, its bounded scene and the owner's own profile; no name list or oracle.
def proof_scene(*sentences: str) -> tuple[list[dict], dict]:
    units = [
        {"id": f"c00s{index:05d}", "chapter": "ch.txt", "chapter_sha256": "a" * 64, "quote": sentence}
        for index, sentence in enumerate(sentences)
    ]
    return units, units[-1]


def test_alias_proof_requires_literal_tie_continuity_and_no_contradiction() -> None:
    from src.cast_freeze import scoped_alias_proof

    registry = {
        "mother": {"name": "Mother", "bio": "Ren's parent, whom Ren calls Mom"},
        "ren": {"name": "Ren", "bio": "a ten year old boy"},
        "professor_orr": {"name": "Professor Orr", "bio": "Professor at the academy"},
        "professor_vale": {"name": "Professor Vale", "bio": "Professor at the academy"},
        "ana": {"name": "Ana", "bio": "a scout"},
        "bell": {"name": "Bell", "bio": "a smith"},
    }

    def proof(label: str, owner: str, *sentences: str) -> tuple[dict | None, str | None]:
        scene, mention = proof_scene(*sentences)
        return scoped_alias_proof(label, owner, mention, scene, registry, {})

    # literal profile tie + a related cast member present in the scene is continuity
    assert proof("Mom", "mother", "Ren sat down.", "Ren whispered, I love you, Mom.")[1] is None
    # the same label with no source continuity is not proven, whoever proposed the owner
    assert "continuity" in proof("Mom", "mother", "Rain fell on the roof.", "Thanks, Mom.")[1]
    # a label the owner's names and profile never carry cannot be forced onto a roster member
    assert "profile" in proof("Darling", "ren", "Ren sat down.", "Darling, come here.")[1]
    # distinct speaker: the owner is the other participant of this very mention
    assert "distinct participant" in proof("Mom", "mother", "Ren sat down.", "Mom, said Mother.")[1]
    assert "profile" in proof("Mom", "ren", "Ren sat down.", "Ren whispered, I love you, Mom.")[1]
    # a title that fits two cast members is not an identity when the scene names both
    assert (
        "equally"
        in proof("Professor", "professor_orr", "Professor Orr and Professor Vale met.", "Professor, said Ren.")[1]
    )
    assert proof("Professor", "professor_orr", "Professor Orr waved.", "Professor, said Ren.")[1] is None
    # title plus a different name word contradicts the owner
    assert (
        "shares no name word"
        in proof("Professor Quill", "professor_orr", "Professor Orr waved.", "Professor Quill spoke.")[1]
    )
    # a full name extending the owner's name must not carry another cast member's name word
    assert "another cast member" in proof("Ana Bell", "ana", "Ana ran.", "Ana Bell shouted.")[1]
    assert proof("Ana Reed", "ana", "Ana ran.", "Ana Reed shouted.")[0]["literal"] == "shared_name_word"
    # the mention itself is never continuity: the owner name inside the label does not count
    assert "continuity" in proof("Ana Reed", "ana", "Ana Reed shouted.")[1]


def test_adjudication_withholds_unproven_alias_and_keeps_raw_binding(tmp_path: Path) -> None:
    from src.cast_freeze import (
        SCOPED_AUDIT_NAME,
        adjudicate_pending_mentions,
        immutable_name_references,
    )

    chapter = tmp_path / "ch1.txt"
    chapter.write_text("Mother smiled. Darling, said Mother softly.", encoding="utf-8")
    units = immutable_evidence_units([chapter])
    registry = {"mother": {"name": "Mother", "bio": "Ren's parent who calls the child Darling"}}
    candidate = {
        "id": "p0000",
        "label": "Darling",
        "ref_ids": [r for r, ref in immutable_name_references(units).items() if ref["label"] == "Darling"],
    }

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        return json.dumps(
            {
                "mentions": [
                    {
                        "mention_id": "m0",
                        "refers_to_person": "yes",
                        "candidate_kind": "individual_name",
                        "decision": "alias",
                        "canonical": "mother",
                        "confidence": 0.99,
                        "reason": "the proposed owner fits",
                    }
                ]
            }
        )

    adjudicate_pending_mentions(tmp_path, [{"candidate": candidate, "proposed": "mother"}], units, registry, ask)
    (record,) = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert (record["decision"], record["canonical"]) == ("ambiguous", "none")
    assert record["reason"].startswith("[alias proof rejected:") and "distinct participant" in record["reason"]
    assert record["raw_adjudication"]["decision"] == "alias" and record["raw_adjudication"]["canonical"] == "mother"
    # final for these owners: the ledger neither binds an owner nor re-asks
    (scoped,) = [c for c in candidate_coverage_ledger(units, registry, {}, [record]) if c["label"] == "Darling"]
    assert scoped["known_owner"] is None and not scoped.get("scoped_stale")


def test_legacy_unproven_alias_is_withdrawn_without_a_model_call(tmp_path: Path) -> None:
    from src.cast_freeze import (
        SCOPED_AUDIT_NAME,
        adjudicate_pending_mentions,
        immutable_name_references,
    )

    chapter = tmp_path / "ch1.txt"
    chapter.write_text("Rain fell. Mom, said Ren.", encoding="utf-8")
    units = immutable_evidence_units([chapter])
    registry = {"ren": {"name": "Ren"}, "mother": {"name": "Mother", "bio": "Ren's parent"}}
    quote = units[1]["quote"]
    legacy = {
        "chapter_sha256": units[1]["chapter_sha256"],
        "quote_sha256": hashlib.sha256(quote.encode()).hexdigest(),
        "label": "Mom",
        "span_start": quote.index("Mom"),
        "canonical": "mother",
        "decision": "alias",
        "confidence": 0.95,
        "reason": "scene supports it",
    }
    (tmp_path / SCOPED_AUDIT_NAME).write_text(json.dumps({"records": [legacy]}))
    (scoped,) = [c for c in candidate_coverage_ledger(units, registry, {}, [legacy]) if c["label"] == "Mom"]
    assert scoped["known_owner"] is None and scoped["scoped_stale"] and scoped["scoped_unproven"]
    candidate = {
        "id": "p0000",
        "label": "Mom",
        "ref_ids": [r for r, ref in immutable_name_references(units).items() if ref["label"] == "Mom"],
    }

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        raise AssertionError("a mechanical withdrawal must not call the model")

    assert adjudicate_pending_mentions(tmp_path, [{"candidate": candidate, "proposed": "mother"}], units, registry, ask)
    (record,) = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert record["decision"] == "ambiguous" and record["canonical"] == "none"
    assert record["history"][0]["decision"] == "alias" and record["history"][0]["canonical"] == "mother"
    assert record["raw_adjudication"]["canonical"] == "mother"
    (settled,) = [c for c in candidate_coverage_ledger(units, registry, {}, [record]) if c["label"] == "Mom"]
    assert not settled.get("scoped_stale") and not settled.get("scoped_unproven")


def test_pending_proposal_is_stored_once_per_exact_proposal(tmp_path: Path) -> None:
    from src.cast_freeze import PROPOSALS_NAME, save_pending_proposal

    proposal = {"type": "uncertain_living", "id": "kit", "mentions": [{"unit_id": "c00s00000", "span_start": 3}]}
    first = save_pending_proposal(tmp_path, proposal)
    assert save_pending_proposal(tmp_path, dict(reversed(list(proposal.items())))) == first
    other = save_pending_proposal(tmp_path, {**proposal, "mentions": [{"unit_id": "c00s00001", "span_start": 3}]})
    assert other != first
    assert len((tmp_path / PROPOSALS_NAME).read_text().splitlines()) == 2


# ##################################################################
# open-world owner retrieval
# every actor with literal profile/source-fact evidence is ranked before any model sees options, while an unsupported model proposal cannot turn an absent identity into an existing owner.
def test_owner_retrieval_ranks_all_evidenced_actors_and_keeps_absent_owner_open() -> None:
    from src.cast_freeze import adjudication_owners

    registry = {
        **{f"filler_{index:02d}": {"name": f"Filler {index:02d}", "bio": "unrelated resident"} for index in range(48)},
        "foam_xiao": {"name": "Foam Xiao", "source_facts": "Fang's cobra carried poison daggers."},
    }
    assert adjudication_owners("Fang", registry) == ["foam_xiao"]
    assert adjudication_owners("Lynn", registry, proposed="filler_00") == []


# ##################################################################
# caretaker source-reviewed retirement of prepared prose actors
# Real temp chapters and a real progress file: a prepared actor leaves the registry only through exact source evidence plus
# its complete own non-character references; anchors/profiles/referenced actors are never dropped; history is append-only.
RETIRE_SHA = "c" * 64


def retire_fixture(tmp_path: Path) -> tuple[list[Path], dict, RecoveryLedger]:
    first = tmp_path / "01-part_01.txt"
    second = tmp_path / "02-part_02.txt"
    first.write_text(
        "Ana ran across the yard. Zephyr howled over the wall. Later Ana saw Zephyr again. Ana waited.",
        encoding="utf-8",
    )
    second.write_text("Zephyr faded. Quill wrote a note. Ana left.", encoding="utf-8")
    prepared = {"name": "Zephyr", "bio": "", "look": "", "origin": "prepared", "facts": {"voice": ["Zephyr howled."], "look": []}}
    registry = {
        "ana": {"name": "Ana", "bio": "a scout", "look": "", "origin": "existing", "facts": {"voice": [], "look": []}},
        "narrator": {"name": "Narrator", "bio": "", "look": "", "origin": "existing", "facts": {"voice": [], "look": []}},
        "zephyr": prepared,
        "quill": {**prepared, "name": "Quill", "facts": {"voice": ["Quill wrote."], "look": []}},
    }
    aliases = {key: key for key in registry}
    progress = {"registry": registry, "aliases": aliases}
    (tmp_path / PROGRESS_NAME).write_text(json.dumps(progress), encoding="utf-8")
    (tmp_path / "characters.json").write_text(json.dumps({"ana": {"name": "Ana"}, "narrator": {"name": "Narrator"}}), encoding="utf-8")
    (tmp_path / "voices.json").write_text(json.dumps({"ana": {"description": "calm"}}), encoding="utf-8")
    return [first, second], progress, RecoveryLedger(tmp_path)


def own_refs_of(chapters: list[Path], word: str) -> list[dict]:
    """Independent (regex) enumeration of every exact own mention of a name in the whole book."""
    refs = []
    for chapter in chapters:
        for unit in immutable_evidence_units([chapter]):
            for match in re.finditer(rf"\b{word}\b", unit["quote"]):
                refs.append(
                    {
                        "chapter": chapter.name,
                        "chapter_sha256": unit["chapter_sha256"],
                        "unit_id": unit["id"],
                        "quote_sha256": hashlib.sha256(unit["quote"].encode()).hexdigest(),
                        "label": word,
                        "span_start": match.start(),
                    }
                )
    return refs


def retire_entry(chapters: list[Path], actor_id: str = "zephyr", word: str = "Zephyr", **over) -> dict:
    index = {"zephyr": (0, 1), "quill": (1, 1)}[actor_id]
    entry = {
        "action": "retire_actor",
        "actor_id": actor_id,
        "reviewer_role": "caretaker",
        "factual_basis": f"the cited unit uses {word} as weather prose, never a living person",
        "evidence": [v2_witness(chapters, *index)],
        "own_refs": own_refs_of(chapters, word),
    }
    return {**entry, **over}


def write_retirement(tmp_path: Path, progress: dict, entries: list[dict], **over) -> None:
    payload = {
        "contract": SOURCE_RETIREMENT_CONTRACT,
        "version": 1,
        "source_sha256": RETIRE_SHA,
        "registry_sha256": registry_digest(progress["registry"]),
        "retirements": entries,
        **over,
    }
    (tmp_path / SOURCE_RETIREMENT_NAME).write_text(json.dumps(payload), encoding="utf-8")


def retire(tmp_path: Path, chapters: list[Path], progress: dict, recovery: RecoveryLedger) -> dict:
    return ingest_source_reviewed_retirements(tmp_path, RETIRE_SHA, chapters, progress, recovery)


def retire_state(tmp_path: Path) -> dict:
    names = (PROGRESS_NAME, "characters.json", "voices.json", SCOPED_AUDIT_NAME, "recovery_ledger.jsonl")
    return {name: (tmp_path / name).read_bytes() if (tmp_path / name).exists() else None for name in names}


def blocked_rows(recovery: RecoveryLedger) -> list[dict]:
    return recovery.open_pending(SOURCE_RETIREMENT_STAGE)


def test_retirement_removes_only_the_named_prepared_actor_and_keeps_append_only_history(tmp_path: Path) -> None:
    chapters, progress, recovery = retire_fixture(tmp_path)
    refs = own_refs_of(chapters, "Zephyr")
    assert len(refs) == 3
    originals = {name: (tmp_path / name).read_bytes() for name in ("characters.json", "voices.json")}
    ana, quill = json.loads(json.dumps(progress["registry"]["ana"])), json.loads(json.dumps(progress["registry"]["quill"]))
    snapshot = json.loads(json.dumps(progress["registry"]["zephyr"]))
    digest = registry_digest(progress["registry"])
    write_retirement(tmp_path, progress, [retire_entry(chapters)])

    outcome = retire(tmp_path, chapters, progress, recovery)

    assert outcome == {"retired": ["zephyr"], "blocked": [], "already": False}
    saved = json.loads((tmp_path / PROGRESS_NAME).read_text())
    assert saved["registry"] == progress["registry"] and "zephyr" not in saved["registry"] and "zephyr" not in saved["aliases"]
    assert saved["registry"]["ana"] == ana and saved["registry"]["quill"] == quill and "narrator" in saved["registry"]
    assert all(target in saved["registry"] for target in saved["aliases"].values())
    (record,) = saved["source_reviewed_retirements"]
    assert record["retired_entry"] == snapshot and record["registry_sha256_before"] == digest
    assert record["removed_aliases"] == ["zephyr"] and len(record["own_refs"]) == 3 and record["evidence"]
    scoped = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert len(scoped) == 3 and {(r["decision"], r["canonical"], r["label"]) for r in scoped} == {("non_character", "none", "Zephyr")}
    assert {(r["chapter_sha256"], r["quote_sha256"], r["span_start"]) for r in scoped} == {
        (r["chapter_sha256"], r["quote_sha256"], r["span_start"]) for r in refs
    }
    assert {name: (tmp_path / name).read_bytes() for name in originals} == originals
    # a file left in place is processed once: replay is a byte-identical no-op even though the registry moved on
    before = retire_state(tmp_path)
    assert retire(tmp_path, chapters, json.loads((tmp_path / PROGRESS_NAME).read_text()), recovery)["already"] is True
    assert retire_state(tmp_path) == before
    # a later distinct input appends; the earlier record is preserved verbatim
    write_retirement(tmp_path, progress, [retire_entry(chapters, "quill", "Quill")])
    assert retire(tmp_path, chapters, progress, recovery)["retired"] == ["quill"]
    again = json.loads((tmp_path / PROGRESS_NAME).read_text())
    assert [r["actor_id"] for r in again["source_reviewed_retirements"]] == ["zephyr", "quill"]
    assert again["source_reviewed_retirements"][0] == record and len(again["source_retirement_inputs"]) == 2


def test_retirement_whole_file_mismatches_fail_closed_before_any_write(tmp_path: Path) -> None:
    chapters, progress, recovery = retire_fixture(tmp_path)
    good = retire_entry(chapters)
    cases = [
        {"source_sha256": "d" * 64},
        {"registry_sha256": "e" * 64},
        {"contract": "cast_variant_root_approval"},
        {"version": 2},
        {"extra": 1},
        {"retirements": [good, retire_entry(chapters)]},
        {"retirements": [{**good, "action": "approve_root"}]},
        {"retirements": [{**good, "extra": 1}]},
        {"retirements": [{**good, "actor_id": "Not An Id"}]},
        {"retirements": [{**good, "factual_basis": "human reviewed the source"}]},
        {"retirements": "zephyr"},
    ]
    for over in cases:
        write_retirement(tmp_path, progress, [good], **over)
        before = retire_state(tmp_path)
        with pytest.raises(OperationalError):
            retire(tmp_path, chapters, progress, recovery)
        assert retire_state(tmp_path) == before and "zephyr" in progress["registry"]
    # a registry that moved after review is a whole-file mismatch, even when the actor itself is unchanged
    write_retirement(tmp_path, progress, [good])
    progress["registry"]["quill"]["facts"]["voice"].append("a later fact")
    with pytest.raises(OperationalError):
        retire(tmp_path, chapters, progress, recovery)
    assert "source_reviewed_retirements" not in progress


def test_retirement_never_retires_anchors_original_profiles_or_non_prepared_actors(tmp_path: Path) -> None:
    chapters, progress, recovery = retire_fixture(tmp_path)
    progress["registry"]["ren"] = {**progress["registry"]["zephyr"], "name": "Ren", "origin": "existing"}
    progress["aliases"]["ren"] = "ren"
    progress["registry"]["quill"]["origin"] = "approved_root"
    (tmp_path / "characters.json").write_text(json.dumps({"ana": {"name": "Ana"}, "zephyr": {"name": "Zephyr"}}), encoding="utf-8")
    entries = [
        retire_entry(chapters, "zephyr"),
        retire_entry(chapters, "quill", "Quill"),
        {**retire_entry(chapters), "actor_id": "ana"},
        {**retire_entry(chapters), "actor_id": "ren"},
        {**retire_entry(chapters), "actor_id": "narrator"},
    ]
    write_retirement(tmp_path, progress, entries)
    registry_before = json.loads(json.dumps(progress["registry"]))

    outcome = retire(tmp_path, chapters, progress, recovery)

    assert outcome["retired"] == [] and len(outcome["blocked"]) == 5
    assert progress["registry"] == registry_before
    reasons = {b["actor_id"]: " ".join(b["blockers"]) for b in outcome["blocked"]}
    assert "original character/voice profile" in reasons["zephyr"] and "not a prepared prose actor" in reasons["quill"]
    assert "not a prepared prose actor" in reasons["ana"] and "original anchor" in reasons["ren"] and "original anchor" in reasons["narrator"]
    assert len(blocked_rows(recovery)) == 5 and {r["code"] for r in blocked_rows(recovery)} == {"source_retirement_blocked"}
    assert not (tmp_path / SCOPED_AUDIT_NAME).exists()


def test_retirement_never_drops_a_referenced_actor(tmp_path: Path) -> None:
    chapters, progress, recovery = retire_fixture(tmp_path)
    progress["aliases"]["the_gale"] = "zephyr"
    write_retirement(tmp_path, progress, [retire_entry(chapters)])
    outcome = retire(tmp_path, chapters, progress, recovery)
    assert outcome["retired"] == [] and "still resolve to the actor" in outcome["blocked"][0]["blockers"][0]
    assert progress["aliases"]["the_gale"] == "zephyr" and "zephyr" in progress["registry"]
    del progress["aliases"]["the_gale"]
    # a mention-scoped alias to the actor, an alias-audit record and an open context row are each references too
    ref = own_refs_of(chapters, "Zephyr")[0]
    scoped = {
        "chapter_sha256": "f" * 64, "quote_sha256": "a" * 64, "label": "Gale", "span_start": 0, "canonical": "zephyr",
        "decision": "alias", "confidence": 1.0, "reason": "x",
    }
    (tmp_path / SCOPED_AUDIT_NAME).write_text(json.dumps({"records": [scoped]}), encoding="utf-8")
    (tmp_path / "cast_alias_audit.json").write_text(
        json.dumps({"records": [{"alias": "Gale", "canonical": "Zephyr", "evidence": ["x"], "decision": "merge"}]}), encoding="utf-8"
    )
    recovery.record(CONTEXT_V2_STAGE, "context row", "context_new_actor", "m", severity="pending", evidence={"canonical_target": "zephyr", "source": ["01-part_01.txt"]})
    write_retirement(tmp_path, progress, [retire_entry(chapters, factual_basis="second review: weather prose, no person")])
    outcome = retire(tmp_path, chapters, progress, recovery)
    text = " ".join(outcome["blocked"][0]["blockers"])
    assert outcome["retired"] == [] and "scoped to the actor as an alias" in text and "alias audit" in text and "pending row" in text
    # an own reference that already carries a scoped alias to another actor is also a hard conflict
    (tmp_path / SCOPED_AUDIT_NAME).write_text(
        json.dumps({"records": [{**scoped, "chapter_sha256": ref["chapter_sha256"], "quote_sha256": ref["quote_sha256"], "label": "Zephyr", "span_start": ref["span_start"], "canonical": "ana"}]}),
        encoding="utf-8",
    )
    write_retirement(tmp_path, progress, [retire_entry(chapters, factual_basis="third review: weather prose, no person")])
    assert "already has scoped alias decision" in " ".join(retire(tmp_path, chapters, progress, recovery)["blocked"][0]["blockers"])
    assert "zephyr" in progress["registry"] and progress["source_reviewed_retirements"] == []


def test_retirement_needs_exact_evidence_and_the_complete_own_non_character_refs(tmp_path: Path) -> None:
    chapters, progress, recovery = retire_fixture(tmp_path)
    forged = {**v2_witness(chapters, 0, 1), "unit_quote": "Zephyr forged."}
    other_ref = own_refs_of(chapters, "Ana")[0]
    entries = [
        retire_entry(chapters, own_refs=own_refs_of(chapters, "Zephyr")[:-1]),
        retire_entry(chapters, own_refs=[*own_refs_of(chapters, "Zephyr"), other_ref]),
        retire_entry(chapters, own_refs=[]),
        retire_entry(chapters, evidence=[forged]),
        retire_entry(chapters, evidence=[v2_witness(chapters, 0, 0)]),
        retire_entry(chapters, evidence=[]),
        retire_entry(chapters, reviewer_role="human"),
        retire_entry(chapters, factual_basis="  "),
    ]
    for entry in entries:
        write_retirement(tmp_path, progress, [entry])
        before = retire_state(tmp_path)
        outcome = retire(tmp_path, chapters, progress, recovery)
        assert outcome["retired"] == [] and len(outcome["blocked"]) == 1, entry
        assert "zephyr" in progress["registry"] and not (tmp_path / SCOPED_AUDIT_NAME).exists()
        saved = json.loads((tmp_path / PROGRESS_NAME).read_text())
        assert "zephyr" in saved["registry"] and saved["source_reviewed_retirements"] == []
        assert retire_state(tmp_path)["characters.json"] == before["characters.json"]
    assert all(row["code"] == "source_retirement_blocked" and row["evidence"]["actor_id"] == "zephyr" for row in blocked_rows(recovery))


def test_retirement_blocks_one_bad_entry_only_and_a_corrected_input_resolves_its_pending_row(tmp_path: Path) -> None:
    chapters, progress, recovery = retire_fixture(tmp_path)
    bad = retire_entry(chapters, own_refs=own_refs_of(chapters, "Zephyr")[:1])
    write_retirement(tmp_path, progress, [bad, retire_entry(chapters, "quill", "Quill")])

    outcome = retire(tmp_path, chapters, progress, recovery)

    assert outcome["retired"] == ["quill"] and [b["actor_id"] for b in outcome["blocked"]] == ["zephyr"]
    assert "zephyr" in progress["registry"] and "quill" not in progress["registry"]
    (row,) = blocked_rows(recovery)
    assert row["evidence"]["actor_id"] == "zephyr" and row["severity"] == "pending"
    # the pending row is visible to the producer's quality gate and the same bad input is not re-applied or duplicated
    assert retire(tmp_path, chapters, progress, recovery)["already"] is True and len(blocked_rows(recovery)) == 1
    write_retirement(tmp_path, progress, [retire_entry(chapters)])
    assert retire(tmp_path, chapters, progress, recovery)["retired"] == ["zephyr"]
    assert blocked_rows(recovery) == [] and "zephyr" not in progress["registry"]
    assert [r["actor_id"] for r in progress["source_reviewed_retirements"]] == ["quill", "zephyr"]


def test_retirement_answers_the_actors_quality_flag_and_leaves_no_dangling_speaker(tmp_path: Path) -> None:
    chapters, progress, recovery = retire_fixture(tmp_path)
    scope = {"chapter": chapters[0].name, **{k: v for k, v in own_refs_of(chapters, "Zephyr")[0].items() if k in {"chapter_sha256", "unit_id", "label", "span_start"}}}
    unit = immutable_evidence_units([chapters[0]])[1]
    scope.update(quote=unit["quote"], quote_sha256=hashlib.sha256(unit["quote"].encode()).hexdigest(), span_start=unit["quote"].index("Zephyr"))
    recovery.record(
        "cast_quality", "quality:zephyr:p1", "quality_garble", "garbled name", severity="pending",
        evidence={"registry_id": "zephyr", "proposal_id": "p1", "source": [chapters[0].name], "scope": scope},
    )
    assert [r["evidence"]["registry_id"] for r in active_quality_pending(recovery, progress["registry"])] == ["zephyr"]
    write_retirement(tmp_path, progress, [retire_entry(chapters)])

    assert retire(tmp_path, chapters, progress, recovery)["retired"] == ["zephyr"]

    assert recovery.open_pending("cast_quality") == [] and active_quality_pending(recovery, progress["registry"]) == []
    resolved = [r for r in recovery.entries() if r["severity"] == "resolved"]
    assert [r["evidence"]["outcome"] for r in resolved] == ["actor_retired_source_reviewed"]
    # every remaining alias still lands on an active actor: no speaker can fall to narrator or an inactive alias
    assert progress["aliases"] and all(target in progress["registry"] for target in progress["aliases"].values())
    assert "narrator" in progress["registry"] and progress["aliases"]["narrator"] == "narrator"


def test_prepare_cast_reads_the_retirement_input_at_the_batch_boundary_and_fails_closed(tmp_path: Path) -> None:
    from src.cast_freeze import prepare_cast

    source = tmp_path / "book.txt"
    source.write_text("source", encoding="utf-8")
    project = get_output_dir(source)
    try:
        (project / "chapters").mkdir(parents=True)
        (project / "chapters" / "00-intro.txt").write_text("Book by Tester, narrated by Narrator", encoding="utf-8")
        (project / "chapters" / "01-plain.txt").write_text("it rained all day.", encoding="utf-8")
        (project / "characters.json").write_text(
            json.dumps({actor_id: {"name": actor_id, "bio": "", "look": ""} for actor_id in ANCHOR_IDS}), encoding="utf-8"
        )
        payload = {
            "contract": SOURCE_RETIREMENT_CONTRACT,
            "version": 1,
            "source_sha256": "0" * 64,
            "registry_sha256": "0" * 64,
            "retirements": [],
        }
        (project / SOURCE_RETIREMENT_NAME).write_text(json.dumps(payload), encoding="utf-8")

        def never(*args, **kwargs):  # the input is rejected before any model call
            raise AssertionError("no model call before the retirement input is validated")

        with pytest.raises(OperationalError, match="different source book"):
            prepare_cast(source, ask=never)
        assert not (project / MANIFEST_NAME).exists()
    finally:
        shutil.rmtree(project, ignore_errors=True)
