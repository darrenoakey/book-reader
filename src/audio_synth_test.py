import json
import subprocess
import tempfile
from pathlib import Path

from src.audio_synth import load_voices, synthesize_chapter


# ##################################################################
# wav duration
# return duration of a wav file in seconds via ffprobe
def wav_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return float(out.strip())


# ##################################################################
# test synthesize chapter real
# synthesizes a chapter using a single tts-design voice
def test_synthesize_chapter_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True)
        voices = {
            "narrator": "A warm male voice in his thirties. Clear and articulate with slight British accent.",
        }
        script_path = tmpdir / "01-test_chapter.jsonl"
        lines = [
            {"narrator": "Chapter One. The Beginning."},
            {"narrator": "It was a dark and stormy night."},
        ]
        with open(script_path, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        result_path = synthesize_chapter(script_path, audio_dir, voices)
        assert result_path.exists()
        assert result_path.stat().st_size > 1000
        assert result_path.name == "01-test_chapter.wav"
        assert wav_duration(result_path) > 1.0


# ##################################################################
# test synthesize chapter multi voice
# synthesizes a chapter with multiple distinct character voices via tts-design
def test_synthesize_chapter_multi_voice() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True)
        voices = {
            "narrator": "A neutral male narrator in his thirties with clear articulation.",
            "old_woman": "An elderly female voice in her seventies, raspy and warm with a slight tremor.",
            "young_boy": "A bright energetic young boy around ten years old, high pitched and eager.",
        }
        script_path = tmpdir / "02-multi.jsonl"
        lines = [
            {"narrator": "The old woman beckoned the boy closer."},
            {"old_woman": "Come here, child. Let me see your face."},
            {"young_boy": "Yes, grandma! I brought you the bread you asked for."},
            {"narrator": "She smiled and patted his head."},
        ]
        with open(script_path, "w") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        result_path = synthesize_chapter(script_path, audio_dir, voices)
        assert result_path.exists()
        assert result_path.stat().st_size > 1000
        # verify all per-line wavs were produced (one per line)
        line_dir = audio_dir / ".lines_02-multi"
        line_wavs = sorted(line_dir.glob("*.wav"))
        assert len(line_wavs) == 4
        for w in line_wavs:
            assert w.stat().st_size > 500
            assert wav_duration(w) > 0.3
        assert wav_duration(result_path) > 3.0


# ##################################################################
# test synthesize chapter idempotent
# verify existing audio is not overwritten
def test_synthesize_chapter_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        audio_dir = tmpdir / "audio"
        audio_dir.mkdir()
        script_path = tmpdir / "01-test.jsonl"
        script_path.write_text('{"narrator": "Hello"}\n')
        existing = audio_dir / "01-test.wav"
        existing.write_text("PRESERVED")
        result_path = synthesize_chapter(script_path, audio_dir, {"narrator": "any"})
        assert existing.read_text() == "PRESERVED"
        assert result_path == existing


# ##################################################################
# test load voices
# verify voices.json parses into a description map
def test_load_voices() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(parents=True)
        (output_dir / "voices.json").write_text(json.dumps({
            "alice": {"description": "Alice description"},
            "bob": {"description": "Bob description"},
        }))
        v = load_voices(output_dir)
        assert v == {"alice": "Alice description", "bob": "Bob description"}
