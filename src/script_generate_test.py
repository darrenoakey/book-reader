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
# test exact source coverage
# fail closed on a dropped source word rather than creating an incomplete audiobook.
def test_validate_script_requires_exact_source_order() -> None:
    from src.script_generate import ScriptGenerationError, validate_script_lines

    valid = [{"narrator": "Chapter"}, {"narrator": "One two three."}]
    validate_script_lines(valid, "One two three.", ["narrator"])
    with pytest.raises(ScriptGenerationError, match="coverage"):
        validate_script_lines([{"narrator": "Chapter"}, {"narrator": "One three."}], "One two three.", ["narrator"])


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
    assert base == script_fingerprint("text", "T", ["narrator"])
    assert base != script_fingerprint("text", "T", ["narrator", "a"], True)


# ##################################################################
# test roster expansion preserves script cache key
# later cast growth cannot regenerate a chapter and change cached piece indices.
def test_fingerprint_ignores_growing_roster() -> None:
    initial = script_fingerprint("same prose", "Part", ["narrator", "ren"])
    expanded = script_fingerprint("same prose", "Part", ["narrator", "ren", "new_character"])
    assert initial == expanded


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
    validate_script_lines([{"narrator": "Title"}, {"narrator": chunk}], chunk, ["narrator"])


# ##################################################################
# test the spoken chapter heading is not duplicated
# A chapter's own text normally opens with its title, so prepending the title
# unconditionally made the narrator read the heading twice (and the assembled
# audiobook then announced it a third time). The heading is only added when the
# source does not already start with it, and coverage validation must still pass
# in both shapes.
def test_source_opens_with_title_detects_the_real_heading() -> None:
    from src.script_generate import source_opens_with_title

    assert source_opens_with_title("Chapter 1\n\nDrummer, beat, and piper, blow", "Chapter 1")
    assert source_opens_with_title("  chapter 1: the ledger\n\nBody text.", "Chapter 1")
    assert not source_opens_with_title("Drummer, beat, and piper, blow", "Chapter 1")
    # a title that merely shares a prefix with the prose is not the heading
    assert not source_opens_with_title("Chapterhouse rules applied.", "Chapter 1")


def test_heading_is_prepended_only_when_the_source_lacks_it() -> None:
    import asyncio

    import src.script_generate as sg

    original = sg.classify_spans
    try:
        # classify_spans covers the WHOLE chapter text, heading included, which is
        # exactly why prepending the title produced a second "Chapter 1".
        async def covers_heading(text, speaker_ids, *args, **kwargs):
            return [{"narrator": "Chapter 1"}, {"narrator": "Drummer, beat, and piper, blow."}]

        sg.classify_spans = covers_heading
        lines = asyncio.run(
            sg.generate_chapter_script("Chapter 1\n\nDrummer, beat, and piper, blow.", "Chapter 1", ["narrator"])
        )
        assert [next(iter(line.values())) for line in lines] == ["Chapter 1", "Drummer, beat, and piper, blow."]

        # a source that does not open with the heading still gets one spoken title
        async def no_heading(text, speaker_ids, *args, **kwargs):
            return [{"narrator": "Drummer, beat, and piper, blow."}]

        sg.classify_spans = no_heading
        lines = asyncio.run(sg.generate_chapter_script("Drummer, beat, and piper, blow.", "Chapter 1", ["narrator"]))
        assert [next(iter(line.values())) for line in lines] == ["Chapter 1", "Drummer, beat, and piper, blow."]
    finally:
        sg.classify_spans = original
