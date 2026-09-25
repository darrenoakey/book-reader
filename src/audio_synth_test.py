"""Real integration tests for chapter audio assembly.

The GPU voice-cloning / TTS path (tts_engine.synthesize_jobs) is exercised by
its own module; here we cover audio_synth's own logic against the CURRENT API:
line splitting, real ffmpeg concatenation of real wavs, idempotent skipping,
and the speaker-fallback validation. No mocks — ffmpeg is invoked for real.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from src.audio_synth import concat_wavs, split_long_text, synthesize_chapter


# ##################################################################
# wav duration
# return duration of a wav file in seconds via ffprobe
def wav_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return float(out.strip())


# ##################################################################
# make tone wav
# generate a real short sine-tone wav via ffmpeg for concat tests
def make_tone_wav(path: Path, seconds: float = 0.5) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={seconds}",
        "-ar",
        "24000",
        "-ac",
        "1",
        str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


# ##################################################################
# test split long text
# long lines split into sentence chunks each within the word budget
def test_split_long_text() -> None:
    assert split_long_text("Short line.") == ["Short line."]
    long = " ".join(f"Sentence number {i} here." for i in range(20))
    chunks = split_long_text(long, max_words=35)
    assert len(chunks) > 1
    for c in chunks:
        # each chunk is within budget, unless it is a single overlong sentence
        assert len(c.split()) <= 35 or c.count(".") == 1


# ##################################################################
# test concat wavs real
# real ffmpeg concatenation of two real tone wavs into one longer wav
def test_concat_wavs_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        a, b = tmp / "a.wav", tmp / "b.wav"
        make_tone_wav(a, 0.5)
        make_tone_wav(b, 0.5)
        out = tmp / "out.wav"
        concat_wavs([a, b], out)
        assert out.exists()
        assert wav_duration(out) > 0.9  # ~1.0s combined


# ##################################################################
# test concat wavs empty
# concatenating nothing is a hard error, not a silent empty file
def test_concat_wavs_empty() -> None:
    with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValueError):
        concat_wavs([], Path(tmpdir) / "out.wav")


# ##################################################################
# test synthesize chapter idempotent
# an already-synthesized chapter wav is never overwritten
def test_synthesize_chapter_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        audio_dir = tmp / "audio"
        voices_dir = tmp / "voices"
        audio_dir.mkdir()
        script_path = tmp / "01-test.jsonl"
        script_path.write_text('{"narrator": "Hello"}\n')
        existing = audio_dir / "01-test.wav"
        existing.write_text("PRESERVED")
        result_path = synthesize_chapter(script_path, audio_dir, voices_dir, {"narrator"})
        assert existing.read_text() == "PRESERVED"
        assert result_path == existing


# ##################################################################
# test synthesize chapter unknown speaker
# a line for an unknown speaker with no narrator fallback is a hard error
def test_synthesize_chapter_unknown_speaker() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        audio_dir = tmp / "audio"
        voices_dir = tmp / "voices"
        audio_dir.mkdir()
        script_path = tmp / "02-test.jsonl"
        script_path.write_text('{"ghost": "boo"}\n')
        with pytest.raises(ValueError):
            synthesize_chapter(script_path, audio_dir, voices_dir, {"alice"})
