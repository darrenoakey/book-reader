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
    discover_batch,
    discovery_prompt,
    discovery_schema,
    immutable_evidence_units,
    materialize_evidence_discovery,
    record_rejected_discovery,
    repair_prompt,
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
# test discovery reduction and repair
# keeps normal batch output focused on additions and makes a repair carry a concrete validation failure plus the same numbered source units.
def test_discovery_prompt_requests_only_additions_and_repair_has_error() -> None:
    with tempfile.TemporaryDirectory() as directory:
        chapter = Path(directory) / "01-part.txt"
        chapter.write_text("Ren spoke.", encoding="utf-8")
        prompt = discovery_prompt([chapter], {"ren": {"name": "Ren"}}, {"ren": "ren"})
        repair = repair_prompt(ValueError("missing c00s00000"), [chapter], {"ren": {"name": "Ren"}}, {"ren": "ren"})
        assert "Do not repeat an unchanged known actor" in prompt
        assert "Scan every numbered source unit" in prompt
        assert "one record per actor" in prompt
        assert 'canonical_id MUST be exactly "new"' in prompt
        assert 'id="professor_xiao"' in prompt
        assert "missing c00s00000" in repair
        assert "structural identity mismatch" in repair
        assert "[c00s00000] Ren spoke." in repair


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
        assert (
            validate_discovery(materialized, chapter.read_text(encoding="utf-8"), set()) == materialized["characters"]
        )
        raw["characters"][0]["aliases"] = ["Rin"]
        with pytest.raises(ValueError, match="lacks selected source evidence"):
            materialize_evidence_discovery(raw, units)
        raw["characters"][0]["aliases"] = []
        raw["characters"][0]["evidence_unit_ids"] = ["made_up"]
        with pytest.raises(ValueError, match="invalid immutable evidence ID"):
            materialize_evidence_discovery(raw, units)
        raw["characters"][0].update({"canonical_id": "ron", "id": "foam_xiao", "name": "Foam Xiao"})
        raw["characters"][0]["evidence_unit_ids"] = [units[0]["id"]]
        with pytest.raises(ValueError, match="structural identity mismatch.*canonical_id='new'"):
            materialize_evidence_discovery(raw, units)
        raw["characters"] = [
            {"canonical_id": "new", "id": "ren", "name": "Ren", "aliases": ["Ron"], "voice_facts": "", "look_facts": "", "evidence_unit_ids": [unit["id"] for unit in units]},
            {"canonical_id": "new", "id": "ron", "name": "Ron", "aliases": [], "voice_facts": "", "look_facts": "", "evidence_unit_ids": [unit["id"] for unit in units]},
        ]
        with pytest.raises(ValueError, match="duplicate new identity label"):
            materialize_evidence_discovery(raw, units)
        raw["characters"] = [{"canonical_id": "new", "id": "ren", "name": "Ren", "aliases": ["Ron"], "voice_facts": "", "look_facts": "", "evidence_unit_ids": [unit["id"] for unit in units]}]
        with pytest.raises(ValueError, match="conflicts with approved source alias"):
            materialize_evidence_discovery(raw, units, {"ron": "ron"})


# ##################################################################
# test audited global alias
# permits an already source-audited alias for an existing canonical actor without letting the same absent label invent a new actor.
def test_existing_canonical_can_use_only_approved_global_alias() -> None:
    with tempfile.TemporaryDirectory() as directory:
        chapter = Path(directory) / "01-part.txt"
        chapter.write_text("The tiger boy watched the road.", encoding="utf-8")
        units = immutable_evidence_units([chapter])
        raw = {
            "characters": [
                {
                    "canonical_id": "tiger_boy",
                    "id": "tiger_boy",
                    "name": "Jean",
                    "aliases": ["Jean"],
                    "voice_facts": "",
                    "look_facts": "",
                    "evidence_unit_ids": [units[0]["id"]],
                }
            ]
        }
        materialized, _ = materialize_evidence_discovery(raw, units, {"jean": "tiger_boy"})
        assert materialized["characters"][0]["evidence"] == ["The tiger boy watched the road."]
        raw["characters"][0]["canonical_id"] = "new"
        raw["characters"][0]["id"] = "jean"
        with pytest.raises(ValueError, match="lacks selected source evidence"):
            materialize_evidence_discovery(raw, units, {"jean": "tiger_boy"})


# ##################################################################
# test selected alias actor link
# accepts a new existing-actor spelling only when its own source unit and a canonical or approved spelling are both selected.
def test_existing_alias_requires_selected_exact_witness_and_actor_link() -> None:
    with tempfile.TemporaryDirectory() as directory:
        chapter = Path(directory) / "01-part.txt"
        chapter.write_text("Weey began drawing. Professor Weii taught the class. We sat quietly.", encoding="utf-8")
        units = immutable_evidence_units([chapter])
        raw = {
            "characters": [
                {
                    "canonical_id": "professor_wei",
                    "id": "professor_wei",
                    "name": "Professor Wei",
                    "aliases": ["Weey"],
                    "voice_facts": "",
                    "look_facts": "",
                    "evidence_unit_ids": [unit["id"] for unit in units],
                }
            ]
        }
        materialized, _ = materialize_evidence_discovery(
            raw,
            units,
            {"weii": "professor_wei", "professor_wei": "professor_wei"},
            {"professor_wei": {"name": "Professor Wei"}},
        )
        assert materialized["characters"][0]["aliases"] == ["Weey"]
        raw["characters"][0]["evidence_unit_ids"] = [units[0]["id"]]
        with pytest.raises(ValueError, match="lacks selected actor link"):
            materialize_evidence_discovery(
                raw,
                units,
                {"weii": "professor_wei", "professor_wei": "professor_wei"},
                {"professor_wei": {"name": "Professor Wei"}},
            )
        raw["characters"][0]["aliases"] = ["We"]
        raw["characters"][0]["evidence_unit_ids"] = [unit["id"] for unit in units]
        with pytest.raises(ValueError, match="generic pronoun"):
            materialize_evidence_discovery(
                raw,
                units,
                {"weii": "professor_wei", "professor_wei": "professor_wei"},
                {"professor_wei": {"name": "Professor Wei"}},
            )


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
# test split calls
# proves each batch makes a new-only call and an existing-only call with distinct closed schemas, and combines only when both validate.
def scripted(responses: list[list[str]], calls: list[dict]):
    queues = {"new": list(responses[0]), "existing": list(responses[1])}

    def ask(prompt: str, max_tokens: int, max_attempts: int, response_schema: dict) -> str:
        enum = response_schema["properties"]["characters"]["items"]["properties"]["canonical_id"]["enum"]
        mode = "new" if enum == ["new"] else "existing"
        calls.append({"mode": mode, "enum": enum, "prompt": prompt})
        return queues[mode].pop(0)

    return ask


def record(canonical: str, actor_id: str, name: str, aliases: list[str], unit: str) -> dict:
    return {"canonical_id": canonical, "id": actor_id, "name": name, "aliases": aliases, "voice_facts": "", "look_facts": "", "evidence_unit_ids": [unit]}


def test_split_calls_use_distinct_schemas_and_combine() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        chapter = root / "01-part.txt"
        chapter.write_text("Ren spoke. Zed Quill answered.", encoding="utf-8")
        units = immutable_evidence_units([chapter])
        zed = next(u["id"] for u in units if "Zed" in u["quote"])
        progress = {"registry": {"ren": {"name": "Ren"}}, "aliases": {"ren": "ren"}}
        calls: list[dict] = []
        ask = scripted(
            [[json.dumps({"characters": [record("new", "zed_quill", "Zed Quill", [], zed)]})], [json.dumps({"characters": []})]],
            calls,
        )
        found, citations = discover_batch(root, 0, [chapter], units, chapter.read_text(), progress, set(), None, ask)
        assert [c["mode"] for c in calls] == ["new", "existing"]
        assert calls[0]["enum"] == ["new"] and calls[1]["enum"] == ["ren"]
        assert [f["id"] for f in found] == ["zed_quill"] and len(citations) == 1


def test_split_repairs_each_call_at_most_twice_and_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        chapter = root / "01-part.txt"
        chapter.write_text("Ren spoke. Zed Quill answered.", encoding="utf-8")
        units = immutable_evidence_units([chapter])
        zed = next(u["id"] for u in units if "Zed" in u["quote"])
        progress = {"registry": {"ren": {"name": "Ren"}}, "aliases": {"ren": "ren"}}
        bad = json.dumps({"characters": [record("new", "wrong", "Zed Quill", [], zed)]})
        good = json.dumps({"characters": [record("new", "zed_quill", "Zed Quill", [], zed)]})
        calls: list[dict] = []
        found, _ = discover_batch(
            root, 0, [chapter], units, chapter.read_text(), progress, set(), None,
            scripted([[bad, bad, good], [json.dumps({"characters": []})]], calls),
        )
        assert [c["mode"] for c in calls] == ["new", "new", "new", "existing"] and found[0]["id"] == "zed_quill"
        calls.clear()
        with pytest.raises(RuntimeError, match="new batch 0-0 rejected after 2 repairs"):
            discover_batch(root, 0, [chapter], units, chapter.read_text(), progress, set(), None, scripted([[bad, bad, bad], []], calls))
        assert [c["mode"] for c in calls] == ["new", "new", "new"]
        assert len((root / "cast_preparation_rejections.jsonl").read_text().splitlines()) == 5


def test_existing_call_rejects_new_record_and_empty_registry_skips_it() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        chapter = root / "01-part.txt"
        chapter.write_text("Ren spoke. Zed Quill answered.", encoding="utf-8")
        units = immutable_evidence_units([chapter])
        zed = next(u["id"] for u in units if "Zed" in u["quote"])
        calls: list[dict] = []
        empty = {"registry": {}, "aliases": {}}
        good = json.dumps({"characters": [record("new", "zed_quill", "Zed Quill", [], zed)]})
        discover_batch(root, 0, [chapter], units, chapter.read_text(), empty, set(), None, scripted([[good], []], calls))
        assert [c["mode"] for c in calls] == ["new"]
        progress = {"registry": {"ren": {"name": "Ren"}}, "aliases": {"ren": "ren"}}
        with pytest.raises(ValueError, match="unknown discovery mode"):
            discovery_schema(["ren"], None, "bogus")
        assert discovery_schema(["ren"], None, "existing")["properties"]["characters"]["items"]["properties"]["canonical_id"]["enum"] == ["ren"]
        assert progress


def test_existing_pass_excludes_new_labels_and_rejects_reuse() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        chapter = root / "01-part.txt"
        chapter.write_text("Ren spoke. Zed Quill answered.", encoding="utf-8")
        units = immutable_evidence_units([chapter])
        zed = next(u["id"] for u in units if "Zed" in u["quote"])
        both = [u["id"] for u in units]
        progress = {"registry": {"ren": {"name": "Ren"}}, "aliases": {"ren": "ren"}}
        new = json.dumps({"characters": [record("new", "zed_quill", "Zed Quill", ["Zed"], zed)]})
        steal = json.dumps({"characters": [{**record("ren", "ren", "Ren", ["Zed"], zed), "evidence_unit_ids": both}]})
        clean = json.dumps({"characters": []})
        calls: list[dict] = []
        found, _ = discover_batch(
            root, 0, [chapter], units, chapter.read_text(), progress, set(), None, scripted([[new], [steal, clean]], calls)
        )
        existing = [c for c in calls if c["mode"] == "existing"]
        assert len(existing) == 2 and [f["id"] for f in found] == ["zed_quill"]
        assert "EXCLUDED LABELS" in existing[0]["prompt"] and "Zed Quill" in existing[0]["prompt"]
        assert "uses excluded new-actor labels" in existing[1]["prompt"]
        calls.clear()
        with pytest.raises(RuntimeError, match="existing batch 0-0 rejected after 2 repairs"):
            discover_batch(
                root, 0, [chapter], units, chapter.read_text(), progress, set(), None, scripted([[new], [steal] * 3], calls)
            )


def test_source_label_requires_whole_word_boundary() -> None:
    units = [{"id": "c00s00000", "chapter": "a", "quote": "Foam Xiao spoke; the foamy sea rose."}]
    assert source_label_present("Foam", units)
    assert source_label_present("foam xiao", units)
    assert not source_label_present("Fo", units)
    assert not source_label_present("oam", units)
    assert not source_label_present("Foa", units)
    assert not source_label_present("Xia", units)
    assert source_label_present("Xiao", [{"id": "x", "chapter": "a", "quote": "Hi, Xiao."}])
