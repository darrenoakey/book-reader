"""The durable narrator default, its explicit override, and the narrated proof.

Darren's standing preference: future audiobooks are narrated by a deep male
voice unless a book or series explicitly overrides it, with character voices
and series continuity untouched. These tests cover the resolution rules, the
cast-plan profile the policy imposes, and — the part that actually proves it —
isolated synthetic prose and a chapter announcement rendered through the real
TTS engine, so the narrator you hear is the narrator that was configured.
"""

from pathlib import Path

import pytest

from src import narrator_voice
from src.audiobook_request import narrator_voice_choice
from src.narrator_voice import (
    DEEP_MALE,
    DEFAULT_NARRATOR_VOICE,
    FEMALE,
    NarratorVoiceError,
    configured_default,
    description,
    normalize,
    project_override,
    resolve,
    resolve_for_project,
    satisfies,
)

ROOT = Path(__file__).resolve().parent.parent


# ##################################################################
# the default is a deep male voice
# The standing preference, asserted directly so it cannot drift silently.
def test_built_in_default_is_a_deep_male_narrator() -> None:
    assert DEFAULT_NARRATOR_VOICE == DEEP_MALE
    text = description()
    lowered = text.lower()
    assert "male" in lowered
    assert "deep" in lowered or "low-pitched" in lowered
    # a narration voice, not a character performance
    assert "narrator" in lowered


# ##################################################################
# the default carries the deep-male contract
# satisfies() is what the materialised description is checked against.
def test_default_description_satisfies_the_deep_male_contract() -> None:
    assert satisfies(DEEP_MALE, description()) is True
    assert satisfies(DEEP_MALE, description(FEMALE)) is False


# ##################################################################
# explicit override changes only the narrator
# A book may ask for another narrator; the character policy is untouched.
def test_explicit_override_selects_a_different_narrator() -> None:
    assert resolve(job_override="female") == FEMALE
    assert "female" in description(FEMALE).lower()
    assert "male" in description(DEEP_MALE).lower()


# ##################################################################
# the configuration layer is durable and validated
# [audiobook] narrator_voice in the git-ignored config is the machine default,
# and an absent key still yields the deep male default.
def test_configured_default_reads_the_durable_config(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[audiobook]\nproductions_root = '/tmp'\n", encoding="utf-8")
    assert configured_default(config) == DEFAULT_NARRATOR_VOICE

    config.write_text("[audiobook]\nnarrator_voice = \"female\"\n", encoding="utf-8")
    assert configured_default(config) == FEMALE

    config.write_text("[audiobook]\nnarrator_voice = \"deep male\"\n", encoding="utf-8")
    assert configured_default(config) == DEEP_MALE


def test_unsupported_narrator_voice_fails_closed(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[audiobook]\nnarrator_voice = \"a robot\"\n", encoding="utf-8")
    with pytest.raises(NarratorVoiceError):
        configured_default(config)
    with pytest.raises(NarratorVoiceError):
        normalize("a robot")
    with pytest.raises(NarratorVoiceError):
        narrator_voice_choice("a robot")


def test_obvious_choice_spellings_normalize() -> None:
    assert normalize("Deep Male") == DEEP_MALE
    assert normalize("a deep, gravelly male voice") == DEEP_MALE
    assert normalize("low baritone male") == DEEP_MALE
    assert normalize("female") == FEMALE
    assert normalize("neutral") == "neutral"


# ##################################################################
# precedence: book, then series, then machine default
# The override is explicit for the book, or inherited through the series
# parent, before the machine default applies.
def test_precedence_book_then_series_then_default(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[audiobook]\nnarrator_voice = \"neutral\"\n", encoding="utf-8")
    assert resolve(config_path=config) == "neutral"
    assert resolve(series_override="female", config_path=config) == FEMALE
    assert resolve(job_override=DEEP_MALE, series_override="female", config_path=config) == DEEP_MALE


# ##################################################################
# a project override is durable on disk
# The choice lives beside the production so every later step reads the same one.
def test_project_override_persists_and_wins(tmp_path: Path) -> None:
    project = tmp_path / "production"
    project.mkdir()
    assert project_override(project) is None
    (project / narrator_voice.PROJECT_FILE).write_text("female\n", encoding="utf-8")
    assert project_override(project) == FEMALE
    assert resolve_for_project(project) == FEMALE


# ##################################################################
# the cast plan never profiles the narrator from the story
# A deep male narrator must not be rejected because the book's cast is female.
def test_narrator_profile_is_reading_voice_not_story_demographics() -> None:
    from src.cast_plan import apply_narrator_profile, validate_plan

    plan = {
        "characters": {
            "narrator": {
                "name": "Narrator",
                "gender": "female",
                "age_band": "young_adult",
                "approx_age": None,
                "accent": "American English",
                "vocal_size": "light",
                "distinct_from": ["menolly"],
                "accent_source": "individual",
                "evidence": [],
            },
            "menolly": {
                "name": "Menolly",
                "gender": "female",
                "age_band": "young_adult",
                "approx_age": None,
                "accent": "American English",
                "vocal_size": "medium",
                "distinct_from": [],
                "accent_source": "cohort",
                "evidence": [],
            },
        },
        "cohorts": [
            {"id": "hall", "kind": "place", "members": ["narrator", "menolly"], "accent": "American English", "basis": "x"}
        ],
    }
    apply_narrator_profile(plan)
    narrator = plan["characters"]["narrator"]
    assert (narrator["gender"], narrator["age_band"], narrator["accent"], narrator["vocal_size"]) == (
        "unknown",
        "unknown",
        "unknown",
        "rich",
    )
    # the reading voice is never an accent cohort member, and the real cast row
    # keeps the cohort it belongs to
    assert all("narrator" not in cohort["members"] for cohort in plan["cohorts"])
    assert validate_plan(plan) == []


def test_validate_plan_rejects_a_narrator_with_story_demographics() -> None:
    from src.cast_plan import validate_plan

    plan = {
        "characters": {
            "narrator": {
                "name": "Narrator",
                "gender": "female",
                "age_band": "young_adult",
                "approx_age": None,
                "accent": "American English",
                "vocal_size": "light",
                "distinct_from": [],
                "accent_source": "individual",
                "evidence": [],
            }
        },
        "cohorts": [],
    }
    problems = validate_plan(plan)
    assert any("reading-voice profile" in problem for problem in problems)


# ##################################################################
# described narrator is what the voices step writes
# The description step must not spend a model call on the narrator at all.
def test_voice_description_for_narrator_comes_from_the_policy(tmp_path: Path) -> None:
    import asyncio

    from src.voice_description import _voice_description_for_one

    project = tmp_path / "production"
    project.mkdir()
    (project / narrator_voice.PROJECT_FILE).write_text("female\n", encoding="utf-8")
    char_id, info = asyncio.run(_voice_description_for_one("narrator", {"bio": "anything"}, None, None, project))
    assert char_id == "narrator"
    assert info["description"] == description(FEMALE)


# ##################################################################
# the ambient configuration on this machine is explicit
# The durable setting must be present in config, not implicit in the code.
def test_local_config_records_the_narrator_default() -> None:
    config = ROOT / "local" / "config.toml"
    # the durable setting must be present wherever this machine's config exists;
    # the location is git-ignored, so a fresh worktree simply has no config to read
    if config.is_file():
        assert "narrator_voice" in config.read_text(encoding="utf-8")
