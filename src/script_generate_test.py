import json
import tempfile
from pathlib import Path

import pytest

from src.script_generate import (
    ScriptGenerationError,
    generate_scripts_sync,
    load_cached_script,
    meta_path_for,
    parse_jsonl_response,
    parse_jsonl_strict,
    script_fingerprint,
    validate_chunk,
    validate_script_lines,
    write_canonical_script,
)


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
        chapter1.write_text(
            """
John walked into the room.

"Hello," said Mary.

John nodded. "Good to see you."
        """.strip()
        )
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
        voices = {"narrator": {"description": "Test narrator voice"}}
        (output_dir / "voices.json").write_text(json.dumps(voices), encoding="utf-8")
        fp = script_fingerprint("Test", "Intro", ["narrator"], True)
        write_canonical_script(existing, [{"narrator": "PRESERVED"}], fp)
        generate_scripts_sync(output_dir)
        assert "PRESERVED" in existing.read_text()


def test_unfingerprinted_script_is_regenerated() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "00-intro.jsonl"
        script.write_text('{"narrator": "x"}\n')
        assert load_cached_script(script, "fp") is None


def test_cache_requires_matching_fingerprint_and_content() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "01-a.jsonl"
        write_canonical_script(script, [{"narrator": "T"}, {"a": "hello there"}], "fp1")
        assert load_cached_script(script, "fp1") == [
            {"narrator": "T"},
            {"a": "hello there"},
        ]
        assert load_cached_script(script, "fp2") is None
        script.write_text('{"narrator": "tampered"}\n')
        assert load_cached_script(script, "fp1") is None
        assert not list(Path(tmpdir).glob("*.tmp"))
        meta_path_for(script).unlink()
        assert load_cached_script(script, "fp1") is None


def test_fingerprint_changes_with_inputs() -> None:
    base = script_fingerprint("text", "T", ["narrator", "a"])
    assert base == script_fingerprint("text", "T", ["a", "narrator"])
    assert base != script_fingerprint("text2", "T", ["narrator", "a"])
    assert base != script_fingerprint("text", "T", ["narrator"])
    assert base != script_fingerprint("text", "T", ["narrator", "a"], True)


def test_strict_parse_rejects_partial_output() -> None:
    with pytest.raises(ScriptGenerationError):
        parse_jsonl_strict('{"narrator": "ok"}\nsorry, I cannot continue')
    with pytest.raises(ScriptGenerationError):
        parse_jsonl_strict("no json at all")
    with pytest.raises(ScriptGenerationError):
        parse_jsonl_strict('{"narrator": "ok"}\n{"narrator": "trunc')
    ok = parse_jsonl_strict('```jsonl\n{"speaker_id": "a", "text": "hi"}\n```')
    assert ok == [{"a": "hi"}]


def test_validation_rejects_bad_speakers_and_short_coverage() -> None:
    chunk = "John walked into the room and sat down beside the old fireplace."
    with pytest.raises(ScriptGenerationError):
        validate_chunk([{"zed": chunk}], chunk, ["narrator"])
    with pytest.raises(ScriptGenerationError):
        validate_chunk([{"narrator": "John walked."}], chunk, ["narrator"])
    validate_chunk([{"narrator": chunk}], chunk, ["narrator"])
    with pytest.raises(ScriptGenerationError):
        validate_script_lines([{"narrator": "Title"}], chunk, ["narrator"])
    validate_script_lines(
        [{"narrator": "Title"}, {"narrator": chunk}], chunk, ["narrator"]
    )
