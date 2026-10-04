import json
import tempfile
import wave
from pathlib import Path

import pytest

from src.breeze_voices import (
    REFERENCE_TEXT,
    load_breeze_manifest,
    prepare_breeze_voices,
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
# test reference text
# the shared reference line is two sentences, short enough to clone well
def test_reference_text_is_short_two_sentences() -> None:
    assert REFERENCE_TEXT.count(".") == 2
    assert len(REFERENCE_TEXT.split()) < 25


# ##################################################################
# test load manifest absolutizes
# ref_wav paths become absolute under the output dir and other fields are untouched
def test_load_manifest_makes_ref_wav_absolute() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "breeze_voices.json").write_text(
            json.dumps(
                {
                    "ann": {
                        "ref_wav": "voices/ann.wav",
                        "ref_text": "hi",
                        "description": "d",
                    }
                }
            ),
            encoding="utf-8",
        )
        manifest = load_breeze_manifest(out)
        assert manifest["ann"]["ref_wav"] == str(out / "voices/ann.wav")
        assert manifest["ann"]["ref_text"] == "hi"


# ##################################################################
# test load manifest missing
# a missing manifest fails closed
def test_load_manifest_missing() -> None:
    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.raises(ValueError, match="breeze_voices.json not found"),
    ):
        load_breeze_manifest(Path(tmp))


# ##################################################################
# test prepare requires voices json
# preparation cannot start without character descriptions
def test_prepare_requires_voices_json() -> None:
    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.raises(ValueError, match="voices.json not found"),
    ):
        prepare_breeze_voices(Path(tmp))


# ##################################################################
# test prepare empty cast fails
# an empty cast yields no manifest entries and must not report success
def test_prepare_empty_cast_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "voices.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="no voices prepared"):
            prepare_breeze_voices(out)


# ##################################################################
# test prepare reuses finished voices
# characters that already have a manifest entry and a real wav are skipped with no synthesis, and the manifest is preserved
def test_prepare_reuses_finished_voices() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "voices.json").write_text(
            json.dumps({"ann": {"description": "bright"}, "bob": {"description": "gruff"}}),
            encoding="utf-8",
        )
        manifest = {
            cid: {
                "ref_wav": f"voices/{cid}.wav",
                "ref_text": REFERENCE_TEXT,
                "description": cid,
            }
            for cid in ("ann", "bob")
        }
        for cid in manifest:
            _write_wav(out / "voices" / f"{cid}.wav")
        (out / "breeze_voices.json").write_text(json.dumps(manifest), encoding="utf-8")
        path = prepare_breeze_voices(out)
        assert path == out / "breeze_voices.json"
        assert json.loads(path.read_text(encoding="utf-8")) == manifest
        loaded = load_breeze_manifest(out)
        assert Path(loaded["ann"]["ref_wav"]).exists()


# ##################################################################
# test prepare active only skips unrelated missing voices
# an `only` filter never synthesizes or lists a described voice outside the active set
def test_prepare_only_skips_non_active_missing_voice() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "voices.json").write_text(
            json.dumps({"ann": {"description": "bright"}, "bob": {"description": "gruff"}}),
            encoding="utf-8",
        )
        manifest = {"ann": {"ref_wav": "voices/ann.wav", "ref_text": REFERENCE_TEXT, "description": "ann"}}
        _write_wav(out / "voices" / "ann.wav")
        (out / "breeze_voices.json").write_text(json.dumps(manifest), encoding="utf-8")
        path = prepare_breeze_voices(out, only={"ann"})
        assert json.loads(path.read_text(encoding="utf-8")) == manifest
        assert not (out / "voices" / "bob.wav").exists()
