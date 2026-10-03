import json
import tempfile
import wave
from pathlib import Path

import pytest

from src.voice_clone import (
    REF_SAMPLE_TEXT,
    SPARK_HOST,
    SPARK_REFS_DIR,
    clone_all_voices,
    clone_voice,
    voice_path,
)


# ##################################################################
# write real wav
# write a real 0.2s 16 kHz mono silence wav (well over the 100 byte reuse threshold)
def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 3200)


# ##################################################################
# test constants
# the reference text is a pangram-style multi-sentence sample and the spark registry location is absolute
def test_reference_constants() -> None:
    assert REF_SAMPLE_TEXT.count(".") >= 3
    assert "@" in SPARK_HOST
    assert SPARK_REFS_DIR.startswith("/")


# ##################################################################
# test voice path existing
# an existing character wav resolves to its path
def test_voice_path_existing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        voices = Path(tmp)
        _write_wav(voices / "ann.wav")
        assert voice_path(voices, "ann") == voices / "ann.wav"


# ##################################################################
# test voice path missing
# an absent voice fails closed with the character name in the error
def test_voice_path_missing() -> None:
    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.raises(ValueError, match="voice file not found for ghost"),
    ):
        voice_path(Path(tmp), "ghost")


# ##################################################################
# test clone voice reuses existing wav
# a finished reference wav is returned untouched with no synthesis, and the voices dir is created
def test_clone_voice_reuses_existing_wav() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        wav = out / "voices" / "ann.wav"
        _write_wav(wav)
        before = wav.read_bytes()
        assert clone_voice("ann", "bright young woman", out) == wav
        assert wav.read_bytes() == before


# ##################################################################
# test clone all requires voices json
# bulk cloning fails closed without character descriptions
def test_clone_all_requires_voices_json() -> None:
    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.raises(ValueError, match="voices.json not found"),
    ):
        clone_all_voices(Path(tmp))


# ##################################################################
# test clone all rejects entry without description
# a malformed voices.json entry raises instead of silently synthesizing a default voice
def test_clone_all_rejects_missing_description() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "voices.json").write_text(json.dumps({"ann": {"bio": "no description key"}}), encoding="utf-8")
        with pytest.raises(KeyError, match="description"):
            clone_all_voices(out)
