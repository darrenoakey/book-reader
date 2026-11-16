import hashlib
import json
import wave
from pathlib import Path

import pytest

from src.cast_plan import (
    SIDECAR_VOICES,
    artifact_matches_plan,
    plan_fingerprint,
    plan_is_stale,
    write_plan_sidecar,
)
from src.voice_acoustic import description_violations
from src.voice_assets import (
    remember_reference,
    require_audio_binding,
    require_current_descriptions,
    require_materialized_voices,
)
from src.voice_clone import clone_voice
from src.voice_description import _voices_fresh

DESCRIPTION = "An adult American female voice in her thirties, warm and clear."


# ##################################################################
# typed filesystem contract fixture
# These explicitly authored profile inputs test storage contracts, not model
# output. No speech or model service is substituted.
def project_at(path: Path) -> Path:
    characters = {"a": {"name": "A", "bio": "An American woman in her thirties."}}
    profile = {"gender": "female", "age_band": "adult", "accent": "American English", "vocal_size": "unknown", "approx_age": "34", "distinct_from": [], "evidence": []}
    plan = {"fingerprint": plan_fingerprint(characters, {}), "characters": {"a": profile}, "cohorts": []}
    path.mkdir(parents=True, exist_ok=True)
    (path / "characters.json").write_text(json.dumps(characters))
    (path / "cast_plan.json").write_text(json.dumps(plan))
    (path / "voices.json").write_text(json.dumps({"a": {"description": DESCRIPTION}}))
    write_plan_sidecar(path, plan["fingerprint"], SIDECAR_VOICES)
    return path


# ##################################################################
# complete reference fixture
# Real WAV/container bytes with explicit typed cast bindings for orchestration
# tests that assert completed generation is skipped, not service simulation.
def prepared_project_at(path: Path) -> Path:
    project_at(path)
    (path / "voices").mkdir()
    with wave.open(str(path / "voices" / "a.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(bytes(200))
    remember_reference(path, "breeze", "a", DESCRIPTION)
    require_audio_binding(path, path / "audio", "breeze")
    return path


def test_cache_binding_detects_content_and_plan_tampering(tmp_path: Path) -> None:
    project = project_at(tmp_path / "project")
    plan = json.loads((project / "cast_plan.json").read_text())
    assert _voices_fresh(project, plan["fingerprint"])
    (project / "voices.json").write_text("{}")
    assert not _voices_fresh(project, plan["fingerprint"])
    (project / "voices.json").write_text(json.dumps({"a": {"description": DESCRIPTION}}))
    write_plan_sidecar(project, plan["fingerprint"], SIDECAR_VOICES)
    plan["characters"]["a"]["accent"] = "French-accented English"
    (project / "cast_plan.json").write_text(json.dumps(plan))
    assert not artifact_matches_plan(project, SIDECAR_VOICES, plan)


def test_changed_source_story_invalidates_plan(tmp_path: Path) -> None:
    project = project_at(tmp_path / "project")
    characters = json.loads((project / "characters.json").read_text())
    (project / "source_context.json").write_text(json.dumps({"description": "Raised in Lyon."}))
    assert plan_is_stale(project, characters)


def test_qwen_reference_cannot_be_silently_reused(tmp_path: Path) -> None:
    project = project_at(tmp_path / "project")
    (project / "voices").mkdir()
    wav = project / "voices" / "a.wav"
    wav.write_bytes(bytes(200))
    with pytest.raises(ValueError, match="unbound qwen"):
        clone_voice("a", DESCRIPTION, project)
    remember_reference(project, "qwen", "a", DESCRIPTION)
    assert clone_voice("a", DESCRIPTION, project) == wav
    before = hashlib.sha256(wav.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="stale qwen"):
        clone_voice("a", "A different voice", project)
    assert hashlib.sha256(wav.read_bytes()).hexdigest() == before


def test_rendered_audio_binding_refuses_changed_reference(tmp_path: Path) -> None:
    project = project_at(tmp_path / "project")
    (project / "voices").mkdir()
    wav = project / "voices" / "a.wav"
    wav.write_bytes(bytes(200))
    remember_reference(project, "qwen", "a", DESCRIPTION)
    audio = project / "audio"
    require_audio_binding(project, audio, "qwen")
    (audio / "existing.wav").write_bytes(bytes(200))
    wav.write_bytes(bytes(201))
    remember_reference(project, "qwen", "a", DESCRIPTION)
    with pytest.raises(ValueError, match="voice inputs changed"):
        require_audio_binding(project, audio, "qwen")


def test_missing_reference_rejected_even_with_complete_state(tmp_path: Path) -> None:
    project = project_at(tmp_path / "project")
    assert require_current_descriptions(project)
    with pytest.raises(ValueError, match="missing breeze"):
        require_materialized_voices(project, "breeze")


def test_description_contradictions_and_age_boundaries() -> None:
    middle = {"gender": "female", "age_band": "middle_aged", "accent": "American English"}
    assert description_violations("Female, middle-aged (52), American English Midwestern accent.", middle) == []
    adult = {"gender": "male", "age_band": "adult", "accent": "British English"}
    assert description_violations("A 38-year-old male with a British English accent.", adult) == []
    assert description_violations("An adult male British voice with a French accent.", adult)
    assert description_violations("An adult British male and female voice.", adult)


def test_other_characters_accents_are_not_cast_as_this_voice() -> None:
    from src.voice_acoustic import self_description

    june = {"gender": "female", "age_band": "adult", "accent": "American English"}
    text = "A 41-year-old female with an American Midwestern accent. Her resonance distinguishes her from the French-accented characters."
    assert description_violations(text, june) == []
    cleaned = self_description(text)
    assert "American" in cleaned and "French" not in cleaned
    assert description_violations("An adult American female with a French accent.", june)


def test_missing_casting_identity_fails_closed(tmp_path: Path) -> None:
    project = project_at(tmp_path / "project")
    (project / "cast_plan.json").unlink()
    with pytest.raises(ValueError, match="identity plan missing"):
        require_current_descriptions(project)


def test_job_seniority_is_not_elderly_voice_age() -> None:
    profile = {"gender": "female", "age_band": "adult", "accent": "American English"}
    assert description_violations("An American woman in her thirties with the assured delivery of a senior architect.", profile) == []
    assert description_violations("An adult American female voice, elderly and wavering.", profile)
