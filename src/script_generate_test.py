import json
import tempfile
from pathlib import Path

from src.script_generate import generate_scripts_sync, parse_jsonl_response


# ##################################################################
# test parse jsonl response
# verify jsonl parsing from plain and markdown responses
def test_parse_jsonl_response() -> None:
    plain = '{"narrator": "Hello"}\n{"john": "Hi"}'
    result = parse_jsonl_response(plain)
    assert len(result) == 2
    assert result[0] == {"narrator": "Hello"}
    assert result[1] == {"john": "Hi"}
    markdown = '```jsonl\n{"narrator": "Hello"}\n{"john": "Hi"}\n```'
    result = parse_jsonl_response(markdown)
    assert len(result) == 2


# ##################################################################
# test generate scripts real
# calls claude haiku to generate scripts
def test_generate_scripts_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        chapters_dir = output_dir / "chapters"
        chapters_dir.mkdir(parents=True)
        intro = chapters_dir / "00-intro.txt"
        intro.write_text("Test Book by Author.")
        chapter1 = chapters_dir / "01-chapter_one.txt"
        chapter1.write_text("""
John walked into the room.

"Hello," said Mary.

John nodded. "Good to see you."
        """.strip())
        voices = {
            "narrator": {"description": "The narrator voice"},
            "john": {"description": "A male voice"},
            "mary": {"description": "A female voice"},
        }
        voices_path = output_dir / "voices.json"
        voices_path.write_text(json.dumps(voices), encoding="utf-8")
        scripts = generate_scripts_sync(output_dir)
        assert len(scripts) == 2
        intro_script = scripts[0]
        assert intro_script.exists()
        lines = intro_script.read_text().strip().split("\n")
        assert len(lines) >= 1
        first = json.loads(lines[0])
        assert "narrator" in first
        chapter_script = scripts[1]
        assert chapter_script.exists()
        lines = chapter_script.read_text().strip().split("\n")
        assert len(lines) >= 2
        speakers = set()
        for line in lines:
            entry = json.loads(line)
            speakers.update(entry.keys())
        assert "narrator" in speakers


# ##################################################################
# test generate scripts idempotent
# verify existing scripts are not overwritten
def test_generate_scripts_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        chapters_dir = output_dir / "chapters"
        script_dir = output_dir / "script"
        chapters_dir.mkdir(parents=True)
        script_dir.mkdir(parents=True)
        intro = chapters_dir / "00-intro.txt"
        intro.write_text("Test")
        existing = script_dir / "00-intro.jsonl"
        existing.write_text("PRESERVED")
        voices = {"narrator": {"description": "Test narrator voice"}}
        voices_path = output_dir / "voices.json"
        voices_path.write_text(json.dumps(voices), encoding="utf-8")
        generate_scripts_sync(output_dir)
        assert existing.read_text() == "PRESERVED"
