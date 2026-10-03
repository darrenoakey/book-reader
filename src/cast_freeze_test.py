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
    MANIFEST_NAME,
    REJECTIONS_NAME,
    apply_alias_audit,
    asset_hashes,
    candidate_coverage_ledger,
    context_safe_batch,
    discovery_schema,
    immutable_evidence_units,
    materialize_classifications,
    record_rejected_discovery,
    refresh_alias_audit,
    source_label_present,
    validate_classification_chunk,
    validate_preparation_coverage,
    verify_frozen_cast,
)
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
    assert ("Lou Arched", "non_character", "none") in {
        (r["label"], r["decision"], r["canonical"]) for r in scoped
    }
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
    assert (
        review_verdict_error("Han", "existing:sora", ["u1"], candidate, {}, registry, {}, units, units["u1"])
        .startswith("new-identity review lacks independent owner proof for existing 'sora'")
    )
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
        "registry": {"lu": {"name": "Lu", "bio": "Sister of Taro.", "source_facts": "Lou is Lu's recorded academy name.", "look": ""}},
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
