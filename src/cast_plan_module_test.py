"""Focused deterministic tests for cast_plan + kokoro plan validation.

No model calls here: these exercise the identity, cohort, staleness and
repair contracts the real-LLM tests (cast_plan_test.py) then rely on.
"""

import json
from pathlib import Path

import pytest

from src.cast_plan import (
    PLAN_VERSION,
    build_prompt,
    cast_input,
    normalize_plan,
    plan_fingerprint,
    plan_is_stale,
    plan_missing_characters,
    resolve_cohort_accents,
    validate_plan,
)
from src.kokoro_f0_table import ATLAS_PROVENANCE, F0_MEDIANS
from src.kokoro_voices import (
    ENGLISH_VOICES,
    blend_f0,
    validate_against_plan,
    voice_parts,
)


def _rows(ids: list[str]) -> list[dict]:
    return cast_input({i: {"name": i.title(), "bio": f"bio of {i}"} for i in ids}, {})


def _raw_full(ids: list[str], **overrides) -> dict:
    characters = {}
    for char_id in ids:
        characters[char_id] = {
            "gender": "unknown",
            "age_band": "unknown",
            "approx_age": None,
            "accent": "unknown",
            "vocal_size": "unknown",
            "distinct_from": [],
            "accent_source": "individual",
            "evidence": [],
            **overrides.get(char_id, {}),
        }
    return {"characters": characters, "cohorts": []}


# ##################################################################
# identity set is exact
def test_normalize_missing_identity_fails() -> None:
    with pytest.raises(ValueError, match="missing identities.*bob"):
        normalize_plan(_raw_full(["alice"]), {"alice", "bob"}, _rows(["alice", "bob"]))


def test_normalize_extra_identity_fails() -> None:
    with pytest.raises(ValueError, match="unknown identities.*zed"):
        normalize_plan(_raw_full(["alice", "bob", "zed"]), {"alice", "bob"}, _rows(["alice", "bob"]))


def test_normalize_full_identity_set_passes() -> None:
    plan = normalize_plan(_raw_full(["alice", "bob"]), {"alice", "bob"}, _rows(["alice", "bob"]))
    assert set(plan["characters"]) == {"alice", "bob"}


# ##################################################################
# cohort accent resolution + validation
def test_cohort_resolves_shared_accent_but_preserves_evidence_backed_exception() -> None:
    raw = _raw_full(
        ["kate", "sam", "marco"],
        kate={"accent": "British English (South London)", "accent_source": "cohort"},
        sam={"accent": "unknown", "accent_source": "cohort"},
        marco={
            "accent": "Italian-accented English",
            "accent_source": "individual",
            "evidence": [{"fact": "grew up in Naples until age 20", "source": "bio"}],
        },
    )
    raw["cohorts"] = [
        {
            "id": "london_household",
            "kind": "household",
            "members": ["kate", "sam", "marco"],
            "accent": "British English (South London)",
            "basis": "raised together",
        }
    ]
    plan = normalize_plan(raw, {"kate", "sam", "marco"}, _rows(["kate", "sam", "marco"]))
    resolve_cohort_accents(plan)
    assert plan["characters"]["sam"]["accent"] == "British English (South London)"
    assert plan["characters"]["marco"]["accent"] == "Italian-accented English"
    assert plan["cohorts"][0]["accent_exceptions"] == ["marco"]
    assert validate_plan(plan) == []


def test_cohort_unexplained_accent_mismatch_is_violation() -> None:
    raw = _raw_full(
        ["kate", "sam", "rogue"],
        kate={"accent": "British English", "accent_source": "cohort"},
        sam={"accent": "unknown", "accent_source": "cohort"},
        rogue={"accent": "American English", "accent_source": "individual", "evidence": []},
    )
    raw["cohorts"] = [
        {
            "id": "h",
            "kind": "upbringing",
            "members": ["kate", "sam", "rogue"],
            "accent": "British English",
            "basis": "same school",
        }
    ]
    plan = normalize_plan(raw, {"kate", "sam", "rogue"}, _rows(["kate", "sam", "rogue"]))
    resolve_cohort_accents(plan)
    # rogue had no accent/migration evidence → cohort accent applied
    assert plan["characters"]["rogue"]["accent"] == "British English"
    assert validate_plan(plan) == []


def test_cohort_genuine_mismatch_after_resolution_is_caught() -> None:
    raw = _raw_full(
        ["kate", "marco"],
        marco={
            "accent": "Italian-accented English",
            "accent_source": "individual",
            "evidence": [{"fact": "grew up in Naples", "source": "bio"}],
        },
    )
    raw["cohorts"] = [
        {"id": "h", "kind": "household", "members": ["kate", "marco"], "accent": "British English", "basis": "b"}
    ]
    plan = normalize_plan(raw, {"kate", "marco"}, _rows(["kate", "marco"]))
    resolve_cohort_accents(plan)
    problems = validate_plan(plan)
    assert not problems  # marco is a recorded exception → not a violation
    assert plan["cohorts"][0]["accent_exceptions"] == ["marco"]


# ##################################################################
# staleness / fingerprints
def test_plan_staleness_follows_characters_and_context(tmp_path: Path) -> None:
    characters = {"a": {"name": "A", "bio": "x"}}
    (tmp_path / "source_context").mkdir()
    (tmp_path / "source_context" / "a.txt").write_text("verbatim", encoding="utf-8")
    fingerprint = plan_fingerprint(characters, {"a": "verbatim"})
    document = {"fingerprint": fingerprint, "characters": {}, "cohorts": []}
    (tmp_path / "cast_plan.json").write_text(json.dumps(document), encoding="utf-8")
    assert plan_is_stale(tmp_path, characters) is False
    changed = {"a": {"name": "A", "bio": "y"}}
    assert plan_is_stale(tmp_path, changed) is True
    # changed importer context also invalidates
    (tmp_path / "source_context" / "a.txt").write_text("verbatim 2", encoding="utf-8")
    assert plan_is_stale(tmp_path, characters) is True


def test_plan_missing_characters_and_version(tmp_path: Path) -> None:
    characters = {"a": {"name": "A"}, "b": {"name": "B"}}
    assert plan_missing_characters(tmp_path, characters) == ["a", "b"]
    (tmp_path / "cast_plan.json").write_text(
        json.dumps({"fingerprint": {"plan_version": PLAN_VERSION}, "characters": {"a": {}}, "cohorts": []}),
        encoding="utf-8",
    )
    assert plan_missing_characters(tmp_path, characters) == ["b"]
    (tmp_path / "cast_plan.json").write_text(
        json.dumps({"fingerprint": {"plan_version": 1}, "characters": {"a": {}, "b": {}}, "cohorts": []}),
        encoding="utf-8",
    )
    assert plan_is_stale(tmp_path, characters) is True


def test_prompt_carries_verbatim_context_and_cohort_rules() -> None:
    rows = cast_input(
        {"a": {"name": "A", "bio": "b"}}, {"a": "Elena, 34, Italian-American architect in Chicago"}
    )
    prompt = build_prompt(rows, "premise")
    assert "Elena, 34, Italian-American architect in Chicago" in prompt
    assert "FORMATIVE speech community" in prompt
    assert "RETAIN the accent of where they grew up" in prompt
    assert "infer any demographic from a character's NAME" in prompt


# ##################################################################
# kokoro deterministic validation
def _plan(gender: str, accent: str) -> dict:
    return {
        "characters": {
            "a": {"gender": gender, "age_band": "adult", "accent": accent, "vocal_size": "medium", "distinct_from": ["b"]},
            "b": {"gender": gender, "age_band": "adult", "accent": accent, "vocal_size": "medium", "distinct_from": ["a"]},
        },
        "cohorts": [],
    }


def test_gender_mismatch_is_repaired_deterministically() -> None:
    mapping = {
        "a": {"voice": "am_michael", "speed": 1.0},
        "b": {"voice": "am_adam", "speed": 1.0},
    }
    plan = _plan("female", "American English")
    repaired, problems = validate_against_plan(mapping, plan, ENGLISH_VOICES)
    assert repaired["a"]["voice"].startswith("af_")
    assert repaired["b"]["voice"].startswith("af_")
    specs = {repaired["a"]["voice"], repaired["b"]["voice"]}
    assert len(specs) == 2, f"repair gave both characters the same voice: {specs}"
    assert any("repaired" in p for p in problems)


def test_mixed_gender_blend_rejected() -> None:
    mapping = {"a": {"voice": "af_heart*0.6+am_michael*0.4", "speed": 1.0}, "b": {"voice": "bf_emma", "speed": 1.0}}
    repaired, problems = validate_against_plan(mapping, _plan("female", "American English"), ENGLISH_VOICES)
    assert any("mixes genders" in p for p in problems)
    assert repaired["a"]["voice"].startswith("af_")


def test_mixed_accent_blend_rejected() -> None:
    mapping = {"a": {"voice": "af_heart*0.5+bf_emma*0.5", "speed": 1.0}, "b": {"voice": "af_bella", "speed": 1.0}}
    repaired, problems = validate_against_plan(mapping, _plan("female", "American English"), ENGLISH_VOICES)
    assert any("mixes accents" in p for p in problems)
    assert len({name[0] for name, _ in voice_parts(repaired["a"]["voice"])}) == 1


def test_cohort_accent_coherence_enforced() -> None:
    plan = {
        "characters": {
            "a": {"gender": "male", "age_band": "adult", "accent": "British English (South London)", "vocal_size": "rich", "distinct_from": ["b"]},
            "b": {"gender": "male", "age_band": "adult", "accent": "British English (South London)", "vocal_size": "light", "distinct_from": ["a"]},
        },
        "cohorts": [
            {"id": "lads", "kind": "upbringing", "members": ["a", "b"], "accent": "British English (South London)", "basis": ""}
        ],
    }
    ok = {"a": {"voice": "bm_fable", "speed": 1.0}, "b": {"voice": "bm_george", "speed": 0.95}}
    _, problems = validate_against_plan(dict(ok), plan, ENGLISH_VOICES)
    assert problems == []
    # wrong-accent member: per-profile check repairs it into the b pool first
    bad = {"a": {"voice": "bm_george", "speed": 1.0}, "b": {"voice": "am_adam", "speed": 1.0}}
    repaired, problems = validate_against_plan(dict(bad), plan, ENGLISH_VOICES)
    assert repaired["b"]["voice"].startswith("bm_")
    assert any("needs prefix 'b'" in p for p in problems)
    # unknown individual accents: only the cohort check can catch a split pool
    unknown = {
        "characters": {
            "a": {"gender": "male", "age_band": "adult", "accent": "unknown", "vocal_size": "medium", "distinct_from": ["b"]},
            "b": {"gender": "male", "age_band": "adult", "accent": "unknown", "vocal_size": "medium", "distinct_from": ["a"]},
        },
        "cohorts": [
            {"id": "lads", "kind": "upbringing", "members": ["a", "b"], "accent": "British English", "basis": ""}
        ],
    }
    split = {"a": {"voice": "bm_george", "speed": 1.0}, "b": {"voice": "am_adam", "speed": 1.0}}
    repaired, problems = validate_against_plan(dict(split), unknown, ENGLISH_VOICES)
    assert any("cohort lads" in p for p in problems)
    prefixes = {
        name for member in repaired.values() for name, _ in voice_parts(member["voice"])
    }
    assert all(name.startswith("bm_") for name in prefixes), prefixes


def test_duplicate_spec_separated() -> None:
    mapping = {"a": {"voice": "af_heart", "speed": 1.0}, "b": {"voice": "af_heart", "speed": 1.0}}
    repaired, problems = validate_against_plan(mapping, _plan("female", "American English"), ENGLISH_VOICES)
    assert repaired["a"]["voice"] != repaired["b"]["voice"]
    assert any("duplicate voice" in problem for problem in problems)


# ##################################################################
# measured f0 table
def test_f0_table_complete_for_english_bank() -> None:
    english = [v for v in F0_MEDIANS]
    assert len(english) == 28
    assert all(isinstance(v, float) and 50 < v < 400 for v in F0_MEDIANS.values())
    assert ATLAS_PROVENANCE["measurement"].startswith("f0_median")


def test_blend_f0_weighted_mean() -> None:
    assert blend_f0("af_heart") == pytest.approx(F0_MEDIANS["af_heart"])
    assert blend_f0("am_onyx*1.0") == pytest.approx(F0_MEDIANS["am_onyx"])
    mixed = blend_f0("af_heart*0.6+af_nicole*0.4")
    expected = 0.6 * F0_MEDIANS["af_heart"] + 0.4 * F0_MEDIANS["af_nicole"]
    assert mixed == pytest.approx(expected)
    assert blend_f0("ef_dora*0.5+af_heart*0.5") is None  # unmeasured voice → None, never a guess


def test_voice_parts_parsing() -> None:
    assert voice_parts("af_heart") == [("af_heart", 1.0)]
    assert voice_parts("af_heart*0.6+am_michael*0.4") == [("af_heart", 0.6), ("am_michael", 0.4)]
    assert voice_parts("") == []


# ##################################################################
# voice_acoustic deterministic validation
def test_acoustic_violations_detect_wrong_gender() -> None:
    from src.voice_acoustic import description_violations

    profile = {"gender": "female", "age_band": "adult", "accent": "American English"}
    assert description_violations("A warm middle-aged American man's voice.", profile) != []
    good = "A warm American woman in her thirties, steady pace."
    assert description_violations(good, profile) == []


def test_acoustic_violations_missing_accent() -> None:
    from src.voice_acoustic import description_violations

    profile = {"gender": "male", "age_band": "older", "accent": "British English"}
    problems = description_violations("An elderly man, deep and gravelly.", profile)
    assert any("accent" in p for p in problems)
