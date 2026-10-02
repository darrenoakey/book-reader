"""Filesystem and source-evidence tests for frozen whole-book cast preparation."""

import hashlib
import json
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
    candidates = [{"id": "p0000", "label": "Han", "ref_ids": ["c00s00000n000"]}, {"id": "p0001", "label": "The", "ref_ids": ["c00s00001n000"]}]
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
        response = {"classifications": [
            {"candidate_id": candidate["id"], "status": "new", "identity": candidate["id"], "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]]}
            if candidate["label"] in {"Foam Xiao", "Aster Blackwood"}
            else {"candidate_id": candidate["id"], "status": "non_character", "identity": "none", "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]]}
            for candidate in candidates
        ]}
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
        with pytest.raises(RuntimeError, match="refusing to truncate or skip"):
            context_safe_batch(chapters, 0, {"narrator": {"name": "Narrator"}}, {})


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
        json.dumps({"k_goldest": {"name": "K Goldest"}, "kai": {"name": "Kai"}, "lizard_boy": {"name": "Lizard Boy"}}), encoding="utf-8"
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
        progress = {"next_chapter": 2, "completed_batches": [{"start": 0, "end": 2, "chapter_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in chapters}}]}
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

    candidates = [{"id": f"p{index:04d}", "label": f"Name{index}", "ref_ids": [f"c00s{index:05d}n000"]} for index in range(CLASSIFICATION_CHUNK_SIZE * 2 + 1)]
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
            status, identity = ("new", june["id"]) if candidate == june else (("known", june["id"]) if candidate == jun else ("non_character", "none"))
            evidence = [candidate["ref_ids"][0].rsplit("n", 1)[0]]
            if candidate == jun:
                evidence.append(june["ref_ids"][0].rsplit("n", 1)[0])
            records.append({"candidate_id": candidate["id"], "status": status, "identity": identity, "evidence_unit_ids": evidence})
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
    response = {"classifications": [{"candidate_id": candidate["id"], "status": "known", "identity": "k_goldest", "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]]} if candidate == gold_crest else {"candidate_id": candidate["id"], "status": "non_character", "identity": "none", "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]]} for candidate in candidates]}
    with pytest.raises(ValueError, match="unapproved alias"):
        materialize_classifications(response, units, candidates, {"k_goldest": {"name": "Klein Goldest"}}, {})

# ##################################################################
# test full names lead candidate ownership
# orders source-qualified full names before their components so new actor IDs remain Foam Xiao and Aster Blackwood rather than shortened fragments.
def test_full_name_candidates_precede_short_components() -> None:
    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "Foam Xiao and Aster Blackwood arrived. Foam and Aster followed."}]
    labels = [candidate["label"] for candidate in candidate_coverage_ledger(units, {}, {})]
    assert labels.index("Foam Xiao") < labels.index("Foam")
    assert labels.index("Aster Blackwood") < labels.index("Aster")

# ##################################################################
# test approved full names carry fixed owner
# labels already audited to an established actor are visibly fixed in the native ledger rather than left eligible for a spurious new identity.
def test_candidate_ledger_marks_approved_full_name_owner() -> None:
    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "Klein Goldest confronted Ren."}]
    candidates = candidate_coverage_ledger(units, {"k_goldest": {"name": "Klein Goldest"}}, {"klein_goldest": "k_goldest"})
    assert next(candidate for candidate in candidates if candidate["label"] == "Klein Goldest")["known_owner"] == "k_goldest"

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

    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "The Ceremony Master greeted Ren. In Luna Starwaver's hall, Cass waited."}]
    labels = {reference["label"] for reference in immutable_name_references(units).values()}
    assert not {"The Ceremony Master", "In Luna", "In Luna Starwaver"}.intersection(labels)
    assert {"Ceremony Master", "Luna Starwaver", "Cass"} <= labels

# ##################################################################
# test lowercase lexical usage is not a standalone name
# excludes a sentence-initial ordinary word when the exact lower-case lexical form appears in source, while preserving a name component of a full label.
def test_lowercase_lexical_usage_excludes_single_word_candidate() -> None:
    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "Keep walking. Please keep walking. Foam Xiao arrived. Foam stayed."}]
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

    progress = json.loads((project / "cast_preparation_progress.json").read_text(encoding="utf-8"))
    _, _, chapters = source_chapters(source, project)
    batch, _ = context_safe_batch(chapters, 59, progress["registry"], progress["aliases"])
    candidates = candidate_coverage_ledger(immutable_evidence_units(batch), progress["registry"], progress["aliases"])
    labels = {candidate["label"] for candidate in candidates}
    assert {"Han", "Sora", "Jun", "June"} <= labels
    assert not {"han", "sora", "june"}.intersection(progress["registry"])
    schema = discovery_schema(list(progress["registry"]), candidates, allow_new=True)
    han_id = next(candidate["id"] for candidate in candidates if candidate["label"] == "Han")
    han = next(branch for branch in schema["properties"]["classifications"]["items"]["oneOf"] if "oneOf" in branch and branch["oneOf"][0]["properties"]["candidate_id"]["enum"] == [han_id])
    assert han["oneOf"][0]["properties"]["status"]["enum"] == ["new"]

# ##################################################################
# test full-name component link is local source proof
# permits Aster to target the source-qualified Aster Blackwood candidate without requiring the model to duplicate a second witness already encoded by the full lexical span.
def test_full_name_component_alias_uses_local_full_span_proof() -> None:
    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "Aster Blackwood arrived. Aster charged."}]
    candidates = candidate_coverage_ledger(units, {}, {})
    full = next(candidate for candidate in candidates if candidate["label"] == "Aster Blackwood")
    short = next(candidate for candidate in candidates if candidate["label"] == "Aster")
    response = {"classifications": [{"candidate_id": candidate["id"], "status": "new", "identity": candidate["id"], "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]]} if candidate == full else ({"candidate_id": candidate["id"], "status": "known", "identity": full["id"], "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]]} if candidate == short else {"candidate_id": candidate["id"], "status": "non_character", "identity": "none", "evidence_unit_ids": [candidate["ref_ids"][0].rsplit("n", 1)[0]]}) for candidate in candidates]}
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

    (tmp_path / "qa-klein-team-alias-proposal.json").write_text(json.dumps({"decisions": [{"alias": ["Foam", "Fong", "Fo", "Fang", "Foam Xiao"], "note": "red cobra"}]}), encoding="utf-8")
    candidates = [{"id": "p0000", "label": label, "ref_ids": ["c00s00000n000"]} for label in ["Foam", "Fong", "Fo", "Fang", "Foam Xiao"]]
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
    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "Foam Xiao arrived."}, {"id": "c00s00001", "chapter": "01.txt", "quote": "Foam's scarlet cobra marked his neck."}]
    candidates = [{"id": "p0000", "label": "Foam Xiao", "ref_ids": ["c00s00000n000"]}, {"id": "p0001", "label": "Foam", "ref_ids": ["c00s00001n000"], "audited_target": "p0000"}]
    response = {"classifications": [{"candidate_id": "p0000", "status": "new", "identity": "p0000", "evidence_unit_ids": ["c00s00000"]}, {"candidate_id": "p0001", "status": "known", "identity": "p0000", "evidence_unit_ids": ["c00s00001"]}]}
    discoveries, _ = materialize_classifications(response, units, candidates, {}, {})
    foam = discoveries[0]
    assert foam["aliases"] == ["Foam"]
    assert "scarlet cobra" in foam["look_facts"]

# ##################################################################
# test action and group clues do not force nonentity
# a living actor may be named in an impact or team phrase, so those source clues remain native eligibility questions rather than a blanket negative schema force.
def test_action_and_group_clues_do_not_force_nonentity() -> None:
    units = [{"id": "c00s00000", "chapter": "01.txt", "quote": "The impact of Han and Hammer stopped the beast. Mira team advanced."}]
    candidates = candidate_coverage_ledger(units, {}, {})
    schema = discovery_schema([], candidates)
    branches = schema["properties"]["classifications"]["items"]["oneOf"]
    for label in ("Han", "Mira"):
        candidate = next(item for item in candidates if item["label"] == label)
        branch = next(item for item in branches if "oneOf" in item and item["oneOf"][0]["properties"]["candidate_id"]["enum"] == [candidate["id"]])
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
        {"id": "c00s00002", "chapter": "ch2.txt", "chapter_sha256": hashlib.sha256(b"chapter two body").hexdigest(), "quote": "Young stood up. Young left."},
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
    plain = candidate_coverage_ledger(units, {}, {})
    assert next(c for c in plain if c["label"] == "Young").get("scoped_audit") is None
    ledger = candidate_coverage_ledger(units, {}, {}, [record])
    young = [c for c in ledger if c["label"] == "Young"]
    scoped = [c for c in young if c.get("scoped_audit")]
    other = [c for c in young if not c.get("scoped_audit")]
    assert len(scoped) == 1 and len(other) == 1
    assert scoped[0]["known_owner"] == "young_ren" and other[0]["known_owner"] is None
    assert all(ref.startswith("c00s00001") for ref in scoped[0]["ref_ids"])
    assert len(scoped[0]["ref_ids"]) == 1  # only the offset-0 Young in the quote, not the second Young
    assert next(c for c in ledger if c["label"] == "Young" and not c.get("scoped_audit"))["known_owner"] is None
    with pytest.raises(ValueError):
        candidate_coverage_ledger(units, {}, {}, [{k: v for k, v in record.items() if k != "span_start"}])
    renamed = [{**u, "chapter": "other.txt"} for u in units]
    assert any(c.get("scoped_audit") for c in candidate_coverage_ledger(renamed, {}, {}, [record]))
    changed = [{**u, "chapter_sha256": "0" * 64} if u["id"] == "c00s00001" else u for u in units]
    assert not any(c.get("scoped_audit") for c in candidate_coverage_ledger(changed, {}, {}, [record]))
    assert not set(scoped[0]["ref_ids"]) & set(other[0]["ref_ids"])
    assert any(c["label"] == "Young Ren" and c["known_owner"] is None for c in ledger)
    with pytest.raises(ValueError):
        candidate_coverage_ledger(units, {}, {}, [{**record, "reason": ""}])


# ##################################################################
# scoped adjudication end to end
# an unapproved existing-owner link is decided per mention, persisted with hashes first, and never becomes a global Young alias.
def test_discover_batch_adjudicates_each_mention_and_persists_before_mapping(tmp_path: Path) -> None:
    from src.cast_freeze import SCOPED_AUDIT_NAME, discover_batch

    chapter = tmp_path / "ch1.txt"
    chapter.write_text("Young Ren smiled. Young bowed to the king. The young fox ran. Young sold the fruit.", encoding="utf-8")
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
                    rows.append({"mention_id": mention_id, "refers_to_person": "yes" if bows else "no", "candidate_kind": "individual_name" if bows else "nonliving", "decision": "alias" if bows else "non_character", "canonical": "young_ren" if bows else "none", "confidence": 0.9, "reason": "context"})
            return json.dumps({"mentions": rows})
        out = []
        for option in schema["properties"]["classifications"]["items"]["oneOf"]:
            branches = option.get("oneOf", [option])
            cid = branches[0]["properties"]["candidate_id"]["enum"][0]
            row = next(line for line in prompt.splitlines() if line.startswith(cid + " label="))
            witness = row.split("witnesses: [")[1].split("]")[0]
            label = row.split("label='")[1].split("'")[0]
            statuses = {b["properties"]["status"]["enum"][0]: b["properties"] for b in branches}
            if len(branches) == 1:
                status = next(iter(statuses))
            elif label == "Young":
                status = "known" if "known" in statuses else "non_character"
            else:
                status = "new"
            identity = statuses[status]["identity"]["enum"]
            out.append({"candidate_id": cid, "status": status, "identity": "young_ren" if "young_ren" in identity else identity[0], "evidence_unit_ids": ["c00s00001" if label == "Young" and len(branches) > 1 else witness]})
        return json.dumps({"classifications": out})

    def cid_label(prompt: str, cid: str) -> str:
        for line in prompt.splitlines():
            if line.startswith(cid + " label="):
                return line.split("label=")[1].split()[0].strip("'")
        return ""

    discoveries, classifications = discover_batch(tmp_path, 0, [chapter], units, chapter.read_text(), progress, set(), ask=ask)
    assert discoveries == []
    records = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert {r["label"] for r in records} == {"Young"}
    assert {r["decision"] for r in records} == {"alias", "non_character"}
    for r in records:
        assert len(r["chapter_sha256"]) == 64 and len(r["quote_sha256"]) == 64 and r["reason"] and r["confidence"] == 0.9
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
    chapter.write_text("Young bowed to the king. Darling, said Mother softly.", encoding="utf-8")
    units = immutable_evidence_units([chapter])
    progress = {"registry": {"young_ren": {"name": "Young Ren"}, "mother": {"name": "Mother", "voice_facts": "Darling is what Mother calls her child."}}, "aliases": {}}
    darling_rounds = {"count": 0}

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        schema = response_schema or {}
        if "mentions" in schema["properties"]:
            rows = []
            for line in prompt.splitlines():
                if line.startswith("m") and "mention:" in line:
                    darling = "Darling" in line
                    rows.append({"mention_id": line.split()[0], "refers_to_person": "yes", "candidate_kind": "individual_name", "decision": "alias", "canonical": "mother" if darling else "young_ren", "confidence": 0.9, "reason": "context"})
            return json.dumps({"mentions": rows})
        out = []
        for option in schema["properties"]["classifications"]["items"]["oneOf"]:
            branches = option.get("oneOf", [option])
            cid = branches[0]["properties"]["candidate_id"]["enum"][0]
            row = next(line for line in prompt.splitlines() if line.startswith(cid + " label="))
            witness = row.split("witnesses: [")[1].split("]")[0]
            label = row.split("label='")[1].split("'")[0]
            statuses = {b["properties"]["status"]["enum"][0]: b["properties"] for b in branches}
            if len(branches) == 1:
                status = next(iter(statuses))
            elif label == "Darling":
                darling_rounds["count"] += 1
                status = "known" if darling_rounds["count"] > 1 and "known" in statuses else "non_character" if "non_character" in statuses else next(iter(statuses))
            elif label == "Young":
                status = "known" if "known" in statuses else "non_character"
            else:
                status = "new"
            identity = statuses[status]["identity"]["enum"]
            target = "mother" if label == "Darling" and "mother" in identity else "young_ren" if "young_ren" in identity else identity[0]
            out.append({"candidate_id": cid, "status": status, "identity": target, "evidence_unit_ids": [witness]})
        return json.dumps({"classifications": out})

    discover_batch(tmp_path, 0, [chapter], units, chapter.read_text(), progress, set(), ask=ask)
    after = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert all(record in after for record in records_before)
    assert any(record["label"] == "Darling" and record["decision"] == "alias" and record["canonical"] == "mother" for record in after)


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
    assert references == [{k: records[0][k] for k in ("chapter_sha256", "quote_sha256", "label", "span_start", "canonical")}]
    assert json.loads(manifest_path.read_text())["approved_aliases"] == legacy["approved_aliases"] and "ron" not in legacy["approved_aliases"]
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
    chapter.write_text(f"Opening remark about the harbour. Elena of the north was proud. Marta wept. Young bowed low. Later Marta said farewell. {filler}", encoding="utf-8")
    units = immutable_evidence_units([chapter])
    registry = {"young_ren": {"name": "Young Ren", "bio": "a boy raised by Elena of the north", "look": "freckled", "facts": {"voice": ["soft tenor"], "look": ["scar on chin"]}}}
    candidate = {"id": "p0000", "label": "Young", "ref_ids": [r for r, ref in immutable_name_references(units).items() if ref["label"] == "Young"]}
    prompts: list[str] = []

    def ask(prompt: str, max_tokens: int = 0, max_attempts: int = 1, response_schema: dict | None = None) -> str:
        prompts.append(prompt)
        return json.dumps({"mentions": [{"mention_id": "m0", "refers_to_person": "unclear", "candidate_kind": "unclear", "decision": "ambiguous", "canonical": "none", "confidence": 0.5, "reason": "x"}]})

    adjudicate_pending_mentions(tmp_path, [{"candidate": candidate, "proposed": "young_ren"}], units, registry, ask)
    prompt = prompts[0]
    assert "a boy raised by Elena of the north" in prompt and "soft tenor" in prompt and "scar on chin" in prompt and "freckled" in prompt
    scene = prompt.split("bounded scene: ")[1].splitlines()[0]
    assert "Opening remark about the harbour." in scene and "Marta wept." in scene and "Later Marta said farewell." in scene
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
    assert adjudication_owners("Mom", registry) == ["mother", "ren"]  # no lexical match: bounded roster, label-mentioning owner first, never the narrator
    assert adjudication_owners("Mom", registry, proposed="ren")[0] == "ren"
    chapter_one, chapter_two = tmp_path / "ch1.txt", tmp_path / "ch2.txt"
    chapter_one.write_text("Ren Dove hid under the covers. Ren whispered, I love you, Mom.", encoding="utf-8")
    chapter_two.write_text("Someone shouted at Mom about the broken cart.", encoding="utf-8")
    units = immutable_evidence_units([chapter_one, chapter_two])
    stale_reason = "only the full name is shown; no owner was offered"

    def cached(label: str, quote: str) -> dict:
        unit = next(u for u in units if u["quote"] == quote)
        return {"chapter_sha256": unit["chapter_sha256"], "quote_sha256": hashlib.sha256(quote.encode()).hexdigest(), "label": label, "span_start": quote.index(label), "canonical": "none", "decision": "ambiguous", "confidence": 0.5, "reason": stale_reason}

    legacy = [cached("Ren Dove", units[0]["quote"]), cached("Mom", units[1]["quote"]), cached("Mom", units[2]["quote"])]
    (tmp_path / SCOPED_AUDIT_NAME).write_text(json.dumps({"records": legacy}))
    ledger = candidate_coverage_ledger(units, registry, {}, legacy)
    assert {c["label"] for c in ledger if c.get("scoped_stale")} == {"Ren Dove", "Mom"}  # cached ambiguity is stale, not final
    # a record bound to other bytes or another offset is not applied at all: the source guards still decide scope
    moved = [{**legacy[1], "span_start": legacy[1]["span_start"] + 1}, {**legacy[2], "chapter_sha256": "0" * 64}]
    assert not any(c.get("scoped_audit") for c in candidate_coverage_ledger(units, registry, {}, moved) if c["label"] == "Mom")

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
                        rows.append({"mention_id": line.split()[0], "refers_to_person": "no", "candidate_kind": "nonliving", "decision": "non_character", "canonical": "none", "confidence": 0.9, "reason": "a stranger's mom, not the cast"})
                    else:
                        target = "ren" if "Dove" in line else "mother"
                        assert target in allowed
                        rows.append({"mention_id": line.split()[0], "refers_to_person": "yes", "candidate_kind": "individual_name", "decision": "alias", "canonical": target, "confidence": 0.9, "reason": "scene and prior facts support it"})
            return json.dumps({"mentions": rows})
        out = []
        for option in schema["properties"]["classifications"]["items"]["oneOf"]:
            branches = option.get("oneOf", [option])
            cid = branches[0]["properties"]["candidate_id"]["enum"][0]
            row = next(line for line in prompt.splitlines() if line.startswith(cid + " label="))
            label = row.split("label='")[1].split("'")[0]
            witness = row.split("witnesses: [")[1].split("]")[0]
            statuses = {b["properties"]["status"]["enum"][0]: b["properties"] for b in branches}
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
    discoveries, classifications = discover_batch(tmp_path, 0, [chapter_one, chapter_two], units, "", progress, set(), ask=ask)
    assert discoveries == [] and adjudications
    records = json.loads((tmp_path / SCOPED_AUDIT_NAME).read_text())["records"]
    assert len(records) == 3  # replaced in place of the cached records, never duplicated
    by_quote = {(r["label"], r["quote_sha256"]): r for r in records}
    dove = by_quote[("Ren Dove", hashlib.sha256(units[0]["quote"].encode()).hexdigest())]
    mom_vocative = by_quote[("Mom", hashlib.sha256(units[1]["quote"].encode()).hexdigest())]
    mom_third = by_quote[("Mom", hashlib.sha256(units[2]["quote"].encode()).hexdigest())]
    assert (dove["decision"], dove["canonical"], dove["owners"]) == ("alias", "ren", ["ren"])
    assert (mom_vocative["decision"], mom_vocative["canonical"]) == ("alias", "mother") and set(mom_vocative["owners"]) == {"mother", "ren"}
    assert (mom_third["decision"], mom_third["canonical"]) == ("non_character", "none")  # the same label elsewhere is decided on its own scene
    for record in (dove, mom_vocative, mom_third):
        assert record["history"] == [{"decision": "ambiguous", "canonical": "none", "confidence": 0.5, "reason": stale_reason}]
    assert any(c["status"] == "known" and c["identity"] == "ren" for c in classifications) and any(c["identity"] == "mother" for c in classifications)
    # idempotent: decisions that already saw every plausible owner are never re-asked
    calls = len(adjudications)
    candidates = candidate_coverage_ledger(units, registry, {}, records)
    assert not any(c.get("scoped_stale") for c in candidates)
    adjudicate_pending_mentions(tmp_path, [{"candidate": next(c for c in candidates if c["label"] == "Mom" and c["scoped_audit"]["decision"] == "alias"), "proposed": "mother"}], units, registry, ask)
    assert len(adjudications) == calls
    # an ambiguous answer that already saw every owner stays final with its reason; it is not re-asked
    stuck = {**records[0], "decision": "ambiguous", "canonical": "none", "owners": ["ren", "mother"]}
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
        person, decision, canonical = next(answers)
        return json.dumps({"mentions": [{"mention_id": "m0", "refers_to_person": person, "candidate_kind": "individual_name", "decision": decision, "canonical": canonical, "confidence": 0.95, "reason": "he is Ren's parent"}]})

    for label in ("Ren Dove", "Mom"):
        candidate = next(c for c in ledger if c["label"] == label)
        candidate = {**candidate, "ref_ids": candidate["ref_ids"][:1]}
        adjudicate_pending_mentions(tmp_path, [{"candidate": candidate, "proposed": None}], units, registry, ask)
    records = json.loads((tmp_path / "mention-scoped-audit.json").read_text())["records"]
    assert [(r["label"], r["decision"], r["canonical"]) for r in records] == [("Ren Dove", "ambiguous", "none"), ("Mom", "ambiguous", "none")]
    assert records[0]["raw_adjudication"]["decision"] == "non_character" and records[0]["raw_adjudication"]["refers_to_person"] == "yes"
    assert all("refers_to_person" in prompt and "candidate_kind" in prompt and "endearment" in prompt for prompt in prompts)
