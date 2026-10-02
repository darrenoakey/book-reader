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
    apply_alias_audit,
    asset_hashes,
    context_safe_batch,
    discovery_schema,
    immutable_evidence_units,
    materialize_evidence_discovery,
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
# test schema is closed
# proves native Ollama receives a bounded response schema and cannot add arbitrary top-level response fields.
def test_discovery_schema_is_closed_and_known_canonical_only() -> None:
    schema = discovery_schema(["ren", "ron_blackfire"])
    assert schema["additionalProperties"] is False
    item = schema["properties"]["characters"]["items"]
    assert item["additionalProperties"] is False
    assert item["properties"]["canonical_id"]["enum"] == ["new", "ren", "ron_blackfire"]
    source_schema = discovery_schema(["ren"], ["c00s00000", "c00s00001"])
    source_item = source_schema["properties"]["characters"]["items"]
    assert source_item["required"][-1] == "evidence_unit_ids"
    assert source_item["properties"]["evidence_unit_ids"]["items"]["enum"] == ["c00s00000", "c00s00001"]


# ##################################################################
# test immutable evidence materialization
# proves the native schema can only select source IDs and local code writes the byte-exact source sentence rather than LLM-provided prose.
def test_evidence_ids_materialize_exact_source_and_ground_names() -> None:
    with tempfile.TemporaryDirectory() as directory:
        chapter = Path(directory) / "01-part.txt"
        chapter.write_text("Ren said, Hello. Ron watched Ren.", encoding="utf-8")
        units = immutable_evidence_units([chapter])
        raw = {
            "characters": [
                {
                    "canonical_id": "new",
                    "id": "ren",
                    "name": "Ren",
                    "aliases": [],
                    "voice_facts": "speaks",
                    "look_facts": "",
                    "evidence_unit_ids": [units[0]["id"]],
                }
            ]
        }
        materialized, citations = materialize_evidence_discovery(raw, units)
        assert citations == [[units[0]["id"]]]
        assert materialized["characters"][0]["evidence"] == ["Ren said, Hello."]
        assert validate_discovery(materialized, chapter.read_text(encoding="utf-8"), set()) == materialized["characters"]
        raw["characters"][0]["aliases"] = ["Rin"]
        with pytest.raises(ValueError, match="lacks selected source evidence"):
            materialize_evidence_discovery(raw, units)
        raw["characters"][0]["aliases"] = []
        raw["characters"][0]["evidence_unit_ids"] = ["made_up"]
        with pytest.raises(ValueError, match="invalid immutable evidence ID"):
            materialize_evidence_discovery(raw, units)


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
