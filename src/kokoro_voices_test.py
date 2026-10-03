import json
import tempfile
from pathlib import Path

import pytest

from src.kokoro_voices import (
    ALL_VOICES,
    DEFAULT_NARRATOR,
    DEFAULT_VOICE,
    ENGLISH_VOICES,
    _build_prompt,
    _catalog_text,
    _parse_mapping,
    _sanitize,
    describe_voice,
    load_voice_map,
    map_characters_to_voices,
)


# ##################################################################
# test voice bank shape
# every voice id is two prefix letters + underscore + name, and the English pool is only a/b prefixes
def test_voice_bank_shape() -> None:
    assert len(ALL_VOICES) == len(set(ALL_VOICES))
    for voice in ALL_VOICES:
        assert voice[1] in ("f", "m") and voice[2] == "_"
    assert ENGLISH_VOICES and all(v[0] in ("a", "b") for v in ENGLISH_VOICES)
    assert DEFAULT_VOICE in ENGLISH_VOICES
    assert DEFAULT_NARRATOR in ENGLISH_VOICES


# ##################################################################
# test describe voice
# accent and gender are derived from the id prefix
def test_describe_voice() -> None:
    assert describe_voice("af_heart") == "American English female"
    assert describe_voice("bm_george") == "British English male"
    assert describe_voice("zf_xiaoxiao") == "Mandarin Chinese female"
    assert describe_voice("qm_unknown") == "English male"


# ##################################################################
# test prompt contains cast and catalog
# the prompt lists every candidate voice and every character description (bio as fallback)
def test_build_prompt_contains_cast_and_catalog() -> None:
    pool = ["af_heart", "bm_george"]
    prompt = _build_prompt(
        {
            "ann": {"description": "bright young woman"},
            "bob": {"bio": "gruff old sailor"},
        },
        pool,
    )
    assert _catalog_text(pool) in prompt
    assert "- af_heart: American English female" in prompt
    assert "- ann: bright young woman" in prompt
    assert "- bob: gruff old sailor" in prompt
    assert "bm_adam" not in prompt


# ##################################################################
# test parse mapping tolerant
# plain json, fenced json and json wrapped in chatter all parse; garbage raises
def test_parse_mapping_tolerant() -> None:
    expected = {"a": {"voice": "af_heart", "speed": 1.0}}
    raw = json.dumps(expected)
    assert _parse_mapping(raw) == expected
    assert _parse_mapping(f"```json\n{raw}\n```") == expected
    assert _parse_mapping(f"Here you go: {raw} hope it helps") == expected
    with pytest.raises(ValueError):
        _parse_mapping("no json at all")


# ##################################################################
# test sanitize voice spec
# valid ids and blends survive, anything unknown or empty collapses to the fallback
def test_sanitize_voice_spec() -> None:
    assert _sanitize("af_heart", "fb") == "af_heart"
    assert _sanitize(" af_heart * 0.6 + am_michael*0.4 ", "fb") == "af_heart*0.6+am_michael*0.4"
    assert _sanitize("af_heart+nonsense_voice", "fb") == "fb"
    assert _sanitize("", "fb") == "fb"
    assert _sanitize(None, "fb") == "fb"  # type: ignore[arg-type]


# ##################################################################
# test load voice map
# reads the saved file into (voice, speed) tuples with a default speed of 1.0
def test_load_voice_map() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "kokoro_voices.json").write_text(
            json.dumps(
                {
                    "narrator": {"voice": "bm_george", "speed": 0.9},
                    "ann": {"voice": "af_heart"},
                }
            ),
            encoding="utf-8",
        )
        assert load_voice_map(out) == {
            "narrator": ("bm_george", 0.9),
            "ann": ("af_heart", 1.0),
        }


# ##################################################################
# test load voice map missing
# a missing mapping fails closed with a clear error
def test_load_voice_map_missing() -> None:
    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.raises(ValueError, match="kokoro_voices.json not found"),
    ):
        load_voice_map(Path(tmp))


# ##################################################################
# test map characters missing voices json
# without voices.json there is nothing to map and the call fails closed
def test_map_characters_requires_voices_json() -> None:
    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.raises(ValueError, match="voices.json not found"),
    ):
        map_characters_to_voices(Path(tmp))


# ##################################################################
# test map characters existing output kept
# an existing mapping is authoritative and never regenerated or overwritten
def test_map_characters_existing_output_kept() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        existing = out / "kokoro_voices.json"
        existing.write_text('{"keep": {"voice": "af_sky", "speed": 1.0}}', encoding="utf-8")
        assert map_characters_to_voices(out) == existing
        assert json.loads(existing.read_text(encoding="utf-8")) == {"keep": {"voice": "af_sky", "speed": 1.0}}


# ##################################################################
# test map characters real llm
# real LLM mapping of a two-character cast: every character is mapped to real voice ids with a clamped speed
def test_map_characters_real_llm() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        cast = {
            "narrator": {"description": "Calm, authoritative British male narrator."},
            "ann": {"description": "Bright young American woman, fast and energetic."},
        }
        (out / "voices.json").write_text(json.dumps(cast), encoding="utf-8")
        path = map_characters_to_voices(out)
        mapping = json.loads(path.read_text(encoding="utf-8"))
        assert set(mapping) == {"narrator", "ann"}
        for cid, entry in mapping.items():
            assert entry["description"] == cast[cid]["description"]
            assert 0.7 <= entry["speed"] <= 1.3
            for part in entry["voice"].split("+"):
                assert part.partition("*")[0] in ALL_VOICES
        assert set(load_voice_map(out)) == {"narrator", "ann"}
