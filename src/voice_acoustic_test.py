import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from src.voice_acoustic import (
    SYSTEM_PROMPT,
    generate_acoustic_descriptions,
    generate_one,
)


# ##################################################################
# test system prompt rules
# the prompt carries the acoustic-only rules the voice-design model depends on
def test_system_prompt_rules() -> None:
    assert "ACOUSTIC" in SYSTEM_PROMPT
    assert "DO NOT include personality" in SYSTEM_PROMPT
    assert "No celebrity names" in SYSTEM_PROMPT


# ##################################################################
# test generate one unknown character
# asking for a character that is not in voices.json fails closed before any model call
def test_generate_one_unknown_character() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "voices.json").write_text(json.dumps({"ann": {"description": "bright"}}), encoding="utf-8")
        with pytest.raises(ValueError, match="ghost not in voices.json"):
            asyncio.run(generate_one(out, "ghost"))


# ##################################################################
# test generate descriptions missing file
# a missing voices.json is an error, not an empty result
def test_generate_descriptions_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(FileNotFoundError):
        asyncio.run(generate_acoustic_descriptions(Path(tmp)))


# ##################################################################
# test generate descriptions empty cast
# an empty cast makes no model calls and leaves voices.json an empty object
def test_generate_descriptions_empty_cast() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "voices.json").write_text("{}", encoding="utf-8")
        assert asyncio.run(generate_acoustic_descriptions(out)) == {}
        assert json.loads((out / "voices.json").read_text(encoding="utf-8")) == {}
