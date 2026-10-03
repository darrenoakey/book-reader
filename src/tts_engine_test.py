import json
import tempfile
import wave
from pathlib import Path

import pytest

from src import tts_engine


# ##################################################################
# using engine
# run a block with the module's selected engine switched (the engine is read at call time) and restore afterwards
class _UsingEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.previous = tts_engine.ENGINE

    def __enter__(self) -> None:
        tts_engine.ENGINE = self.name

    def __exit__(self, *exc: object) -> None:
        tts_engine.ENGINE = self.previous


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
# test engine name
# the reported engine is the lower-cased module selection and one of the three supported engines
def test_engine_name_supported() -> None:
    assert tts_engine.engine_name() == tts_engine.ENGINE
    assert tts_engine.ENGINE == tts_engine.ENGINE.lower()
    with _UsingEngine("kokoro"):
        assert tts_engine.engine_name() == "kokoro"


# ##################################################################
# test voices ready per engine
# readiness looks at the artifact owned by the selected engine
def test_voices_ready_per_engine() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for engine in ("breeze", "kokoro", "qwen"):
            with _UsingEngine(engine):
                assert not tts_engine.voices_ready(out)
        (out / "breeze_voices.json").write_text("{}", encoding="utf-8")
        with _UsingEngine("breeze"):
            assert tts_engine.voices_ready(out)
        with _UsingEngine("kokoro"):
            assert not tts_engine.voices_ready(out)
        (out / "kokoro_voices.json").write_text("{}", encoding="utf-8")
        with _UsingEngine("kokoro"):
            assert tts_engine.voices_ready(out)
        (out / "voices").mkdir()
        with _UsingEngine("qwen"):
            assert tts_engine.voices_ready(out)


# ##################################################################
# test speaker set per engine
# valid speakers come from the selected engine's own manifest; a missing manifest fails closed
def test_speaker_set_per_engine() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "breeze_voices.json").write_text(json.dumps({"a": {}, "b": {}}), encoding="utf-8")
        (out / "kokoro_voices.json").write_text(json.dumps({"k": {}}), encoding="utf-8")
        (out / "voices.json").write_text(json.dumps({"q": {}, "r": {}, "s": {}}), encoding="utf-8")
        with _UsingEngine("breeze"):
            assert tts_engine.speaker_set(out) == {"a", "b"}
        with _UsingEngine("kokoro"):
            assert tts_engine.speaker_set(out) == {"k"}
        with _UsingEngine("qwen"):
            assert tts_engine.speaker_set(out) == {"q", "r", "s"}
    with (
        tempfile.TemporaryDirectory() as tmp,
        _UsingEngine("kokoro"),
        pytest.raises(ValueError, match="kokoro_voices.json not found"),
    ):
        tts_engine.speaker_set(Path(tmp))


# ##################################################################
# test prepare voices reuses prepared artifacts
# preparation counts voices from the finished artifact without regenerating when outputs already exist
def test_prepare_voices_reuses_prepared_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "voices.json").write_text(json.dumps({"ann": {"description": "d"}}), encoding="utf-8")
        (out / "kokoro_voices.json").write_text(
            json.dumps(
                {
                    "ann": {"voice": "af_heart", "speed": 1.0},
                    "bob": {"voice": "am_adam", "speed": 1.0},
                }
            ),
            encoding="utf-8",
        )
        with _UsingEngine("kokoro"):
            assert tts_engine.prepare_voices(out) == 2
        _write_wav(out / "voices" / "ann.wav")
        (out / "breeze_voices.json").write_text(
            json.dumps(
                {
                    "ann": {
                        "ref_wav": "voices/ann.wav",
                        "ref_text": "t",
                        "description": "d",
                    }
                }
            ),
            encoding="utf-8",
        )
        with _UsingEngine("breeze"):
            assert tts_engine.prepare_voices(out) == 1


# ##################################################################
# test prepare voices missing descriptions
# the qwen path fails closed when voices.json is absent
def test_prepare_voices_qwen_requires_voices_json() -> None:
    with (
        tempfile.TemporaryDirectory() as tmp,
        _UsingEngine("qwen"),
        pytest.raises(ValueError, match="voices.json not found"),
    ):
        tts_engine.prepare_voices(Path(tmp))


# ##################################################################
# test synthesize no jobs
# an empty job list is a no-op for every engine, even with no voice artifacts present
def test_synthesize_no_jobs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for engine in ("breeze", "kokoro", "qwen"):
            with _UsingEngine(engine):
                assert tts_engine.synthesize_jobs([], Path(tmp)) is None


# ##################################################################
# test synthesize cached jobs
# lines whose wavs already exist are left byte-identical through the kokoro and breeze engines
def test_synthesize_cached_jobs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "kokoro_voices.json").write_text(
            json.dumps({"narrator": {"voice": "bm_george", "speed": 0.95}}),
            encoding="utf-8",
        )
        wav = out / "line.wav"
        _write_wav(wav)
        before = wav.read_bytes()
        jobs = [{"speaker": "stranger", "text": "Hello there.", "output_path": wav}]
        for engine in ("kokoro", "breeze"):
            with _UsingEngine(engine):
                tts_engine.synthesize_jobs(jobs, out)
            assert wav.read_bytes() == before


# ##################################################################
# test synthesize kokoro needs voice map
# with real work to do, the kokoro engine fails closed when the voice map is missing
def test_synthesize_kokoro_requires_voice_map() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        jobs = [{"speaker": "narrator", "text": "Hi.", "output_path": out / "missing.wav"}]
        with (
            _UsingEngine("kokoro"),
            pytest.raises(ValueError, match="kokoro_voices.json not found"),
        ):
            tts_engine.synthesize_jobs(jobs, out)
        assert not (out / "missing.wav").exists()


# ##################################################################
# test synthesize breeze needs manifest
# with real work to do, the breeze engine fails closed when the voice manifest is missing
def test_synthesize_breeze_requires_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        jobs = [{"speaker": "narrator", "text": "Hi.", "output_path": out / "missing.wav"}]
        with (
            _UsingEngine("breeze"),
            pytest.raises(ValueError, match="breeze_voices.json not found"),
        ):
            tts_engine.synthesize_jobs(jobs, out)
