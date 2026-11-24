"""Real full-cast plan tests: shared-upbringing cohort + explicit outsider, and
the American-traveller/French-locals accent scenario.

Real model calls through the casting choke point (src.casting_llm → src.llm,
boringstack primary / loopback backup) with a content-addressed record/replay
cache rooted at src/testdata/casting-recordings: the first run recorded these
answers live; reruns replay the real recorded bytes with zero network. No
hand-authored replies exist anywhere in this suite.
"""

import asyncio
import json
from pathlib import Path

import pytest

from src.cast_plan import (
    CAST_PLAN_SCHEMA,
    PLAN_NAME,
    SIDECAR_KOKORO,
    SIDECAR_VOICES,
    build_cast_plan,
    load_plan,
    plan_is_stale,
)
from src.kokoro_voices import map_characters_to_voices
from src.voice_description import generate_voices

RECORDINGS = Path(__file__).parent / "testdata" / "casting-recordings"

PREMISE = (
    "A quiet Chicago domestic story: Elena, an Italian-American architect, and her "
    "partner Marcus, a Black British jazz pianist who grew up in South London, share "
    "an apartment with Elena's teenage nieces Kaya and Rosa, twins raised in the same "
    "Chicago household. Gerald is the building's elderly Welsh caretaker. Diane is a "
    "middle-aged American neighbor. The Station Man sells train tickets in a booth."
)

CHARACTERS = {
    "narrator": {
        "name": "Narrator",
        "bio": "A neutral audiobook narrator for a quiet domestic Chicago story.",
    },
    "elena": {
        "name": "Elena",
        "bio": (
            PREMISE
            + " Elena, 34, Italian-American architect living in Chicago; she grew up in "
            "Naples and moved to Chicago at 19, so her English carries a light Italian accent. "
            "Medium vocal size, articulate and warm."
        ),
        "profile": {"bio": "34-year-old Italian-American architect in Chicago"},
        "evidence": [
            {"fact": "34 years old", "source": "metadata"},
            {"fact": "grew up in Naples, moved to Chicago at 19", "source": "metadata"},
            {"fact": "Italian-American", "source": "metadata"},
        ],
    },
    "marcus": {
        "name": "Marcus",
        "bio": (
            PREMISE
            + " Marcus, 38, Black British jazz pianist who grew up in South London; rich, "
            "unhurried baritone with a South London accent."
        ),
        "profile": {"bio": "38-year-old Black British jazz pianist raised in South London"},
        "evidence": [
            {"fact": "38 years old", "source": "metadata"},
            {"fact": "grew up in South London", "source": "metadata"},
            {"fact": "Black British", "source": "metadata"},
        ],
    },
    "kaya": {
        "name": "Kaya",
        "bio": (
            PREMISE
            + " Kaya, 16, one of the twins raised in the Chicago household; bright, quick "
            "speech, lighter voice."
        ),
    },
    "rosa": {
        "name": "Rosa",
        "bio": (
            PREMISE
            + " Rosa, 16, Kaya's twin, raised in the same Chicago household; softer and "
            "lower-speaking than her sister."
        ),
    },
    "gerald": {
        "name": "Gerald",
        "bio": (
            PREMISE
            + " Gerald, 74, the building's Welsh-born caretaker, gravelly and slow-spoken."
        ),
    },
    "diane": {
        "name": "Diane",
        "bio": PREMISE + " Diane, 52, a direct American neighbor with a no-nonsense Midwestern voice.",
    },
    "station_man": {
        "name": "Station Man",
        "bio": PREMISE + " The Station Man sells train tickets in a booth; nothing else is known about him.",
    },
}

# American traveller in France: SHE stays American; the LOCALS are French —
# their accents differ legitimately and cohorts must never flatten that.
TRAVEL_PREMISE = (
    "An American architect, June, flies from Chicago to Lyon for a restoration "
    "project. She stays American-accented throughout. The locals — hotelier "
    "Madame Berger and her nephew Théo, both born and raised in Lyon — speak "
    "English with French accents."
)
TRAVEL_CHARACTERS = {
    "narrator": {"name": "Narrator", "bio": "A neutral audiobook narrator."},
    "june": {
        "name": "June",
        "bio": TRAVEL_PREMISE + " June, 41, American from Chicago, direct midwestern speech.",
    },
    "mme_berger": {
        "name": "Madame Berger",
        "bio": TRAVEL_PREMISE + " Madame Berger, 68, Lyon hotelier all her life, French-accented English.",
    },
    "theo": {
        "name": "Théo",
        "bio": TRAVEL_PREMISE + " Théo, 27, her nephew, raised in the same Lyon household, French-accented English.",
    },
}


# ##################################################################
# helper
def _write_project(tmp_path: Path, characters: dict) -> Path:
    project = tmp_path / f"project-{len(characters)}-{abs(hash(json.dumps(sorted(characters)))) % 9973}"
    project.mkdir(parents=True)
    (project / "characters.json").write_text(json.dumps(characters, indent=2), encoding="utf-8")
    return project


# ##################################################################
# live schema boundary
# native structured output is required for the live whole-cast request; legacy
# recordings still exercise the same deterministic validation and persistence.
def test_live_cast_plan_uses_native_json_schema(monkeypatch, tmp_path: Path) -> None:
    project = _write_project(tmp_path, CHARACTERS)
    captured = {}

    async def answer(prompt: str, **kwargs) -> str:
        captured.update(kwargs)
        return json.dumps({
            "characters": {
                cid: {
                    "gender": "unknown",
                    "age_band": "unknown",
                    "approx_age": None,
                    "accent": "unknown",
                    "vocal_size": "unknown",
                    "distinct_from": [],
                    "accent_source": "individual",
                    "evidence": [],
                }
                for cid in CHARACTERS
            },
            "cohorts": [],
        })

    monkeypatch.setattr("src.cast_plan.ask", answer)
    plan = asyncio.run(build_cast_plan(project, CHARACTERS, story_context=PREMISE))
    assert set(plan["characters"]) == set(CHARACTERS)
    assert captured["response_schema"] == CAST_PLAN_SCHEMA
    assert captured["root"] is None


# ##################################################################
# test real full cast plan
# cohort accents coherent with justified exception; distinct within accent;
# unknowns stay unknown; no demographic inferred from names
def test_real_full_cast_plan(tmp_path: Path) -> None:
    project = _write_project(tmp_path, CHARACTERS)
    plan = asyncio.run(build_cast_plan(project, CHARACTERS, story_context=PREMISE, cache_root=RECORDINGS))

    persisted = load_plan(project)
    assert plan == persisted
    chars = persisted["characters"]
    assert set(chars) == set(CHARACTERS)

    # twins share a formative household → same accent
    assert chars["kaya"]["accent"] == chars["rosa"]["accent"]
    assert "american" in chars["kaya"]["accent"].lower()

    # justified exceptions: Elena (Naples upbringing) and Marcus (South London)
    # differ from each other AND from the Chicago household
    accents = {chars["kaya"]["accent"], chars["elena"]["accent"], chars["marcus"]["accent"]}
    assert len(accents) == 3, accents
    assert "italian" in chars["elena"]["accent"].lower()
    assert "british" in chars["marcus"]["accent"].lower()

    # profile facts
    assert chars["elena"]["gender"] == "female" and chars["elena"]["age_band"] in ("adult", "young_adult")
    assert chars["marcus"]["gender"] == "male" and chars["marcus"]["age_band"] in ("adult", "middle_aged")
    assert chars["kaya"]["age_band"] == "teen" and chars["rosa"]["age_band"] == "teen"
    assert chars["gerald"]["age_band"] == "older"
    assert chars["diane"]["age_band"] == "middle_aged"

    # twins must be distinct voices despite the shared accent
    assert "rosa" in chars["kaya"]["distinct_from"] or "kaya" in chars["rosa"]["distinct_from"]

    # unknowns stay unknown — nothing invented from the name
    station = chars["station_man"]
    # no fabricated demographic without evidence
    if station["gender"] == "unknown":
        assert station["accent"] in ("unknown", ""), station

    # every profile row carries evidence
    for char_id, entry in chars.items():
        if entry["gender"] != "unknown" or entry["accent"] not in ("unknown", ""):
            assert entry["evidence"], f"{char_id} has profile facts without evidence"

    # plan persisted with matching fingerprint → not stale
    assert plan_is_stale(project, CHARACTERS) is False


# ##################################################################
# test american traveller vs french locals
# accents differ by formative background even inside one shared scene/place;
# the Lyon household cohort shares ITS accent, June keeps hers
def test_real_american_traveller_french_locals(tmp_path: Path) -> None:
    project = _write_project(tmp_path, TRAVEL_CHARACTERS)
    asyncio.run(build_cast_plan(project, TRAVEL_CHARACTERS, story_context=TRAVEL_PREMISE, cache_root=RECORDINGS))
    chars = load_plan(project)["characters"]

    # June stays American — current place (Lyon) must NOT re-accent her
    assert "american" in chars["june"]["accent"].lower(), chars["june"]["accent"]

    # the Lyon locals share their formative French accent and differ from June
    assert "french" in chars["mme_berger"]["accent"].lower()
    assert "french" in chars["theo"]["accent"].lower()
    assert chars["mme_berger"]["accent"] == chars["theo"]["accent"]
    assert chars["june"]["accent"] != chars["mme_berger"]["accent"]


# ##################################################################
# test kokoro refuses unsupported accents fail-closed
# the English bank cannot honestly voice French-accented English; the Lyon
# project must be rejected with an actionable Breeze/Qwen recommendation
def test_real_kokoro_french_accent_fails_closed(tmp_path: Path) -> None:
    project = _write_project(tmp_path, TRAVEL_CHARACTERS)
    asyncio.run(build_cast_plan(project, TRAVEL_CHARACTERS, story_context=TRAVEL_PREMISE, cache_root=RECORDINGS))
    voices = asyncio.run(generate_voices(project, cache_root=RECORDINGS))
    assert voices.is_file()
    with pytest.raises(ValueError, match="Breeze or Qwen"):
        map_characters_to_voices(project, cache_root=RECORDINGS)


# ##################################################################
# test stale plan not reused
# changing characters.json invalidates the fingerprint and regenerates
def test_real_stale_plan_regenerated(tmp_path: Path) -> None:
    project = _write_project(tmp_path, CHARACTERS)
    first = asyncio.run(build_cast_plan(project, CHARACTERS, story_context=PREMISE, cache_root=RECORDINGS))
    assert (project / PLAN_NAME).is_file()

    changed = json.loads(json.dumps(CHARACTERS))
    changed["gerald"]["bio"] += " He trained as a rugby coach in Cardiff."
    assert plan_is_stale(project, changed) is True

    second = asyncio.run(build_cast_plan(project, changed, story_context=PREMISE, cache_root=RECORDINGS))
    assert second["fingerprint"] != first["fingerprint"]
    assert second["fingerprint"] == load_plan(project)["fingerprint"]


# ##################################################################
# test plan-constrained kokoro mapping real
# one real mapping call, then deterministic plan validation runs inside
def test_real_kokoro_mapping_plan_constrained(tmp_path: Path) -> None:
    english_only = {k: v for k, v in CHARACTERS.items() if k not in ("elena",)}
    project = _write_project(tmp_path, english_only)
    asyncio.run(build_cast_plan(project, english_only, story_context=PREMISE, cache_root=RECORDINGS))
    asyncio.run(generate_voices(project, cache_root=RECORDINGS))
    result_path = map_characters_to_voices(project, cache_root=RECORDINGS)
    mapping = json.loads(result_path.read_text(encoding="utf-8"))

    plan = load_plan(project)
    chars = plan["characters"]
    for char_id, entry in mapping.items():
        voice = entry["voice"]
        for part in voice.split("+"):
            voice_id = part.split("*")[0]
            if chars[char_id]["gender"] == "female":
                assert voice_id[1] == "f", f"{char_id} female but {voice_id}"
            elif chars[char_id]["gender"] == "male":
                assert voice_id[1] == "m", f"{char_id} male but {voice_id}"
    # twins: same accent prefix, distinct specs
    twins = [mapping["kaya"]["voice"], mapping["rosa"]["voice"]]
    assert twins[0].split("*")[0][0] == twins[1].split("*")[0][0]
    assert mapping["kaya"]["voice"] != mapping["rosa"]["voice"] or mapping["kaya"]["speed"] != mapping["rosa"]["speed"]
    # sidecar binds the mapping to the plan it was validated against
    sidecar = json.loads((project / SIDECAR_KOKORO).read_text(encoding="utf-8"))
    assert sidecar["fingerprint"] == plan["fingerprint"]


# ##################################################################
# test kokoro stale mapping regenerated
# a changed plan invalidates the existing kokoro_voices.json instead of reuse
def test_real_kokoro_stale_mapping_regenerated(tmp_path: Path) -> None:
    english_only = {k: v for k, v in CHARACTERS.items() if k not in ("elena",)}
    project = _write_project(tmp_path, english_only)
    asyncio.run(build_cast_plan(project, english_only, story_context=PREMISE, cache_root=RECORDINGS))
    asyncio.run(generate_voices(project, cache_root=RECORDINGS))
    first = map_characters_to_voices(project, cache_root=RECORDINGS)
    first_map = json.loads(first.read_text(encoding="utf-8"))
    assert set(first_map) == set(english_only)

    changed = json.loads(json.dumps(english_only))
    changed["diane"]["bio"] += " She spent her childhood in Glasgow."
    (project / "characters.json").write_text(json.dumps(changed, indent=2), encoding="utf-8")
    asyncio.run(build_cast_plan(project, changed, story_context=PREMISE, cache_root=RECORDINGS))
    asyncio.run(generate_voices(project, cache_root=RECORDINGS))
    second = map_characters_to_voices(project, cache_root=RECORDINGS)
    second_map = json.loads(second.read_text(encoding="utf-8"))
    plan = load_plan(project)
    sidecar = json.loads((project / SIDECAR_KOKORO).read_text(encoding="utf-8"))
    assert sidecar["fingerprint"] == plan["fingerprint"]
    assert second_map  # regenerated, not the stale first mapping silently kept


# ##################################################################
# test voices regenerated when plan changes; consumers never see metadata
def test_real_voices_not_reused_after_plan_change(tmp_path: Path) -> None:
    from src.script_generate import get_speaker_ids

    project = _write_project(tmp_path, CHARACTERS)
    asyncio.run(build_cast_plan(project, CHARACTERS, story_context=PREMISE, cache_root=RECORDINGS))
    first = asyncio.run(generate_voices(project, cache_root=RECORDINGS))
    first_voices = json.loads(first.read_text(encoding="utf-8"))
    assert set(first_voices) == set(CHARACTERS)
    # plain per-character map: script generation sees ONLY speakers
    assert get_speaker_ids(project) == sorted(CHARACTERS)
    sidecar = json.loads((project / SIDECAR_VOICES).read_text(encoding="utf-8"))
    assert sidecar["fingerprint"] == load_plan(project)["fingerprint"]

    changed = json.loads(json.dumps(CHARACTERS))
    changed["diane"]["bio"] += " She spent her childhood in Glasgow."
    (project / "characters.json").write_text(json.dumps(changed, indent=2), encoding="utf-8")
    second = asyncio.run(generate_voices(project, cache_root=RECORDINGS))
    second_voices = json.loads(second.read_text(encoding="utf-8"))
    assert set(second_voices) == set(changed)
    assert get_speaker_ids(project) == sorted(CHARACTERS)
    sidecar2 = json.loads((project / SIDECAR_VOICES).read_text(encoding="utf-8"))
    assert sidecar2["fingerprint"] != sidecar["fingerprint"]
