import json
import tempfile
from pathlib import Path

from src.voice_description import generate_voices_sync

RECORDINGS = Path(__file__).parent / "testdata" / "casting-recordings"


# ##################################################################
# test generate voices real
# calls claude haiku to generate voice descriptions
def test_generate_voices_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True)
        characters = {
            "john": {
                "name": "John",
                "bio": "A tall man in his forties with dark hair. Speaks slowly and thoughtfully.",
            },
            "narrator": {"name": "Narrator", "bio": "An authoritative but warm narrator for a family drama novel."},
        }
        characters_path = output_dir / "characters.json"
        characters_path.write_text(json.dumps(characters), encoding="utf-8")
        result_path = generate_voices_sync(output_dir, cache_root=RECORDINGS)
        assert result_path.exists()
        voices = json.loads(result_path.read_text())
        assert "john" in voices
        assert "narrator" in voices
        assert "description" in voices["john"]
        assert "description" in voices["narrator"]
        assert len(voices["john"]["description"]) > 20
        assert len(voices["narrator"]["description"]) > 20


# ##################################################################
# test generate voices idempotent
# verify a genuinely fresh, plan-bound voices.json is not overwritten
def test_generate_voices_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True)
        characters = {"test": {"name": "Test", "bio": "Test character"}}
        characters_path = output_dir / "characters.json"
        characters_path.write_text(json.dumps(characters), encoding="utf-8")
        voices_path = generate_voices_sync(output_dir, cache_root=RECORDINGS)
        before = voices_path.read_bytes()
        modified = voices_path.stat().st_mtime_ns
        generate_voices_sync(output_dir, cache_root=RECORDINGS)
        assert voices_path.read_bytes() == before
        assert voices_path.stat().st_mtime_ns == modified
        assert set(json.loads(before)) == set(characters)
