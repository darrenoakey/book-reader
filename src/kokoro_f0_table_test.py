import math

from src.kokoro_f0_table import ATLAS_PROVENANCE, F0_MEDIANS
from src.kokoro_voices import (
    ENGLISH_VOICES,
    accent_prefix,
    blend_f0,
    validate_against_plan,
    voice_parts,
)


# ##################################################################
# measured bank provenance
# These are real narration measurements, not estimates from names or gender.
def test_measured_bank_is_complete() -> None:
    assert set(F0_MEDIANS) == set(ENGLISH_VOICES)
    assert all(math.isfinite(value) and 50 < value < 400 for value in F0_MEDIANS.values())
    assert ATLAS_PROVENANCE["atlas_generated"] == "2026-10-28"
    assert blend_f0("am_onyx") < blend_f0("am_santa")


def test_rich_voice_is_repaired_toward_measured_depth() -> None:
    plan = {"characters": {"a": {"gender": "male", "age_band": "adult", "accent": "American English", "vocal_size": "rich"}}, "cohorts": []}
    mapping, problems = validate_against_plan({"a": {"voice": "am_santa", "speed": 1}}, plan, ENGLISH_VOICES)
    assert any("pitch" in problem for problem in problems)
    assert blend_f0(mapping["a"]["voice"]) < F0_MEDIANS["am_santa"]
    assert mapping["a"]["voice"].startswith("am_")


def test_foreign_accents_do_not_match_american_substrings() -> None:
    assert accent_prefix("Australian English") is None
    assert accent_prefix("Russian-accented English") is None
    assert accent_prefix("French-accented English") == "f"


def test_invalid_blends_are_not_silently_accepted() -> None:
    for value in ("af_heart*nan", "af_heart*inf", "af_heart*-1", "af_heart*nope"):
        assert voice_parts(value) == []
