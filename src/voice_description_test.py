import json
import tempfile
from pathlib import Path

from src.voice_description import generate_voices_sync


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
                "bio": "A tall man in his forties with dark hair. Speaks slowly and thoughtfully."
            },
            "narrator": {
                "name": "Narrator",
                "bio": "An authoritative but warm narrator for a family drama novel."
            }
        }
        characters_path = output_dir / "characters.json"
        characters_path.write_text(json.dumps(characters), encoding="utf-8")
        result_path = generate_voices_sync(output_dir)
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
# verify existing voices.json is not overwritten
def test_generate_voices_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True)
        characters = {"test": {"name": "Test", "bio": "Test character"}}
        characters_path = output_dir / "characters.json"
        characters_path.write_text(json.dumps(characters), encoding="utf-8")
        voices_path = output_dir / "voices.json"
        voices_path.write_text('{"preserved": true}', encoding="utf-8")
        generate_voices_sync(output_dir)
        result = json.loads(voices_path.read_text())
        assert result == {"preserved": True}
