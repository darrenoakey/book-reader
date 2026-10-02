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
    context_safe_batch,
    discovery_schema,
    immutable_evidence_units,
    immutable_name_references,
    materialize_evidence_discovery,
    record_rejected_discovery,
    refresh_alias_audit,
    source_label_present,
    validate_discovery,
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
# test source reference schemas
# existing actors select only canonical IDs plus nonempty source references; new actors select one local name reference and optional aliases.
def test_discovery_schema_has_mode_specific_name_references() -> None:
    ref_ids = ["c00s00000n000", "c00s00000n001"]
    existing = discovery_schema(["ren", "ron_blackfire"], ref_ids, "existing")["properties"]["characters"]["items"]
    assert existing["required"] == ["canonical_id", "alias_refs", "voice_facts", "look_facts"]
    assert "name_ref" not in existing["properties"]
    assert existing["properties"]["alias_refs"]["minItems"] == 1
    new = discovery_schema(["ren"], ref_ids, "new")["properties"]["characters"]["items"]
    assert new["properties"]["name_ref"]["enum"] == ref_ids


# ##################################################################
# test source references materialize exact labels
# local reference lookup prevents Kloene invention and bare Xiao selection without offset arithmetic or copied strings.
def test_source_references_materialize_exact_labels_and_reject_bare_xiao() -> None:
    with tempfile.TemporaryDirectory() as directory:
        chapter = Path(directory) / "01-part.txt"
        chapter.write_text("Klene Goldest arrived. Foam Xiao waved. Xiao left.", encoding="utf-8")
        units = immutable_evidence_units([chapter])
        refs = immutable_name_references(units)
        assert all(" and " not in ref["label"].casefold() for ref in refs.values())
        find = lambda label: next(ref_id for ref_id, ref in refs.items() if ref["label"] == label)
        raw = {"characters": [{"canonical_id": "k_goldest", "alias_refs": [find("Klene Goldest")], "voice_facts": "", "look_facts": ""}]}
        materialized, citations = materialize_evidence_discovery(raw, units, {"klene_goldest": "k_goldest"}, {"k_goldest": {"name": "K Goldest"}})
        assert materialized["characters"][0]["name"] == "K Goldest"
        assert materialized["characters"][0]["aliases"] == ["Klene Goldest"]
        assert "Kloene" not in json.dumps(materialized)
        raw["characters"][0]["alias_refs"] = [find("Xiao")]
        with pytest.raises(ValueError, match="bare Xiao"):
            materialize_evidence_discovery(raw, units, {}, {"k_goldest": {"name": "K Goldest"}})
        assert citations[0][0] in refs


# ##################################################################
# test new identity full source references
# derives full canonical IDs locally from exact enumerated source spans.
def test_new_identity_uses_full_exact_source_name_reference() -> None:
    with tempfile.TemporaryDirectory() as directory:
        chapter = Path(directory) / "01-part.txt"
        chapter.write_text("Foam Xiao arrived. Aster Blackwood spoke.", encoding="utf-8")
        units = immutable_evidence_units([chapter])
        refs = immutable_name_references(units)
        find = lambda label: next(ref_id for ref_id, ref in refs.items() if ref["label"] == label)
        raw = {"characters": [
            {"canonical_id": "new", "name_ref": find("Foam Xiao"), "alias_refs": [], "voice_facts": "", "look_facts": ""},
            {"canonical_id": "new", "name_ref": find("Aster Blackwood"), "alias_refs": [], "voice_facts": "", "look_facts": ""},
        ]}
        materialized, _ = materialize_evidence_discovery(raw, units)
        assert [entry["id"] for entry in materialized["characters"]] == ["foam_xiao", "aster_blackwood"]


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
            chapter.write_text("x" * 24_000, encoding="utf-8")
            chapters.append(chapter)
        batch, prompt = context_safe_batch(chapters, 0, {"narrator": {"name": "Narrator"}}, {})
        assert [path.name for path in batch] == ["01-part.txt", "02-part.txt"]
        assert len(prompt) <= 53_536
        chapters[0].write_text("x" * 60_000, encoding="utf-8")
        with pytest.raises(RuntimeError, match="refusing to truncate or skip"):
            context_safe_batch(chapters, 0, {"narrator": {"name": "Narrator"}}, {})


# ##################################################################
# test source evidence rejection
# rejects a locally plausible but non-verbatim character claim before preparation can create an identity.
def test_discovery_requires_verbatim_source_evidence() -> None:
    good = {
        "characters": [
            {
                "canonical_id": "new",
                "id": "ren",
                "name": "Ren",
                "aliases": [],
                "voice_facts": "",
                "look_facts": "",
                "evidence": ["Ren said"],
            }
        ]
    }
    assert validate_discovery(good, "Ren said hello.", set()) == good["characters"]
    with pytest.raises(ValueError, match="source-ambiguous"):
        validate_discovery(good, "Ren said hello.", set(), {"ren"})
    bad = {"characters": [{**good["characters"][0], "evidence": ["Ren has blue eyes"]}]}
    with pytest.raises(ValueError, match="non-source evidence"):
        validate_discovery(bad, "Ren said hello.", set())


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
