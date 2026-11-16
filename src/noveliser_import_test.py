"""Real-data tests for the noveliser annotated-script importer.

The fixture is the actual finished noveliser project "The Arrangement of Us"
(metadata, characters, chapter outline, audio chapters and section checkpoints
copied byte-for-byte). Each test imports a private copy into a private output
directory, so tests are parallel-safe and never touch real projects.
"""

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from src.epub_extract import get_output_dir
from src.noveliser_import import (
    MANIFEST_NAME,
    SEPARATOR,
    NoveliserImportError,
    import_noveliser,
    parse_aliases,
    plan_import,
    verify_cast_frozen,
    verify_import,
)
from src.pipeline import run_noveliser_pipeline, run_pipeline
from src.state import is_step_complete
from src.step_runner import run_step

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).resolve().parent / "testdata" / "noveliser_arrangement"
EXPECTED_LINES = {
    "narrator": 309,
    "elena_marchetti": 342,
    "marcus_reid": 269,
    "vivian_shaw": 92,
    "priya_desai": 44,
    "camille_baptiste": 10,
    "gerald": 13,
    "diane": 11,
    "rosa": 3,
    "station_man": 1,
}


@pytest.fixture
def novel(tmp_path: Path) -> Path:
    source = tmp_path / "The_Arrangement_of_Us"
    shutil.copytree(FIXTURE, source)
    return source


def audio(source: Path, number: int) -> list[dict]:
    return json.loads((source / f"audio_chapter_{number}.json").read_text(encoding="utf-8"))


def write_audio(source: Path, number: int, lines: list[dict]) -> None:
    (source / f"audio_chapter_{number}.json").write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")


def drop_sections(source: Path, number: int) -> None:
    for path in source.glob(f"chapter_{number}_section_*.json"):
        path.unlink()


def script_rows(path: Path) -> list[dict]:
    return [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines() if raw]


# ##################################################################
# test real import preserves every supplied speaker and text
# 1094 rows, 10 identities including narrator and the four script-only minor speakers.
def test_real_arrangement_import_is_exact(novel: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    manifest = import_noveliser(novel, out)
    assert manifest["total_lines"] == 1094
    assert set(manifest["identities"]) == set(EXPECTED_LINES)
    assert manifest["attribution"] == {"engine": "noveliser-supplied", "llm": False}
    counts: dict[str, int] = {}
    for record in manifest["chapters"]:
        supplied = audio(novel, record["number"])
        rows = script_rows(out / "script" / f"{record['stem']}.jsonl")
        assert len(rows) == len(supplied) == record["line_count"]
        for index, (row, line) in enumerate(zip(rows, supplied)):
            (speaker, text), = row.items()
            assert text == (SEPARATOR if index else "") + line["text"]
            assert speaker == manifest["label_map"][line["speaker"]]
            counts[speaker] = counts.get(speaker, 0) + 1
        chapter = (out / "chapters" / f"{record['stem']}.txt").read_text(encoding="utf-8")
        assert "".join(next(iter(row.values())) for row in rows) == chapter
    assert counts == EXPECTED_LINES
    assert manifest["chapters"][0]["title"] == "Lazy Resolutions"
    assert script_rows(out / "script" / "00-intro.jsonl") == [
        {"narrator": "The Arrangement of Us by Darren Oakey, narrated by Darren's Book Reader."}
    ]
    assert is_step_complete(out, "extract") and is_step_complete(out, "characters")
    assert not is_step_complete(out, "scripts")
    assert not (out / "audio").exists()


# ##################################################################
# test casting context
# premise facts absent from biographies survive verbatim; minor speakers get evidence, not invention.
def test_cast_profiles_carry_premise_and_textual_evidence(novel: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    import_noveliser(novel, out)
    cast = json.loads((out / "characters.json").read_text(encoding="utf-8"))
    context = json.loads((out / "source_context.json").read_text(encoding="utf-8"))
    premise = json.loads((novel / "metadata.json").read_text(encoding="utf-8"))["description"]
    assert context["description"] == premise and "34-year-old Italian-American" in premise
    for info in cast.values():
        assert info["story_context"] == premise
        assert premise in info["bio"]
        assert info["source"] == "noveliser"
    elena = cast["elena_marchetti"]
    assert elena["origin"] == "characters.json" and elena["role"] == "protagonist"
    assert elena["profile"]["wound"].startswith("Her fiancé stole")
    for minor in ("gerald", "diane", "rosa", "station_man"):
        info = cast[minor]
        assert info["origin"] == "script-label"
        assert info["profile"]["demographics"] == "unknown"
        assert info["evidence"]["spoken_line_count"] == EXPECTED_LINES[minor]
        assert info["evidence"]["spoken"]
        assert "unknown" in info["bio"]
        spoken = {line["text"] for n in range(1, 5) for line in audio(novel, n) if line["speaker"] == info["name"]}
        for excerpt in info["evidence"]["spoken"]:
            assert excerpt["text"] in spoken or excerpt["text"].endswith("[excerpt truncated]")


# ##################################################################
# test resume is idempotent and tamper-evident
def test_resume_is_idempotent_and_detects_tampering(novel: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = import_noveliser(novel, out)
    script = out / "script" / f"{first['chapters'][0]['stem']}.jsonl"
    before = script.stat().st_mtime_ns
    assert import_noveliser(novel, out) == first
    assert script.stat().st_mtime_ns == before
    assert verify_import(out)["source_fingerprint"] == first["source_fingerprint"]
    rows = script_rows(script)
    assert next(iter(rows[0])) == "camille_baptiste"
    rows[0] = {"narrator": next(iter(rows[0].values()))}
    script.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(NoveliserImportError, match="modified after import"):
        verify_import(out)


# ##################################################################
# test source change fails closed until reimport archives the old project
def test_source_change_requires_reimport_and_archives(novel: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = import_noveliser(novel, out)
    (out / "voices.json").write_text(json.dumps({i: {"description": i} for i in first["identities"]}))
    (out / "voices").mkdir()
    (out / "voices" / "narrator.wav").write_bytes(b"RIFF-real-clip-bytes")
    lines = audio(novel, 4)
    lines[-1]["text"] += " Revised."
    write_audio(novel, 4, lines)
    drop_sections(novel, 4)
    with pytest.raises(NoveliserImportError, match="audio_chapter_4.json"):
        verify_import(out)
    with pytest.raises(NoveliserImportError, match="--reimport"):
        import_noveliser(novel, out)
    second = import_noveliser(novel, out, reimport=True)
    archived = Path(second["superseded"])
    assert archived.is_dir() and (archived / MANIFEST_NAME).is_file()
    assert json.loads((archived / MANIFEST_NAME).read_text())["source_fingerprint"] == first["source_fingerprint"]
    assert second["cast_sha256"] == first["cast_sha256"]
    assert (out / "voices" / "narrator.wav").read_bytes() == b"RIFF-real-clip-bytes"
    assert is_step_complete(out, "voices_desc")
    last = script_rows(out / "script" / f"{second['chapters'][3]['stem']}.jsonl")[-1]
    assert next(iter(last.values())).endswith(" Revised.")


# ##################################################################
# test ambiguous label fails closed and an explicit alias resolves it
def test_ambiguous_label_requires_explicit_alias(novel: Path, tmp_path: Path) -> None:
    lines = audio(novel, 1)
    index = next(i for i, line in enumerate(lines) if line["speaker"] == "Elena Marchetti")
    lines[index]["speaker"] = "Elena"
    write_audio(novel, 1, lines)
    drop_sections(novel, 1)
    with pytest.raises(NoveliserImportError, match="'Elena'.*ambiguous"):
        import_noveliser(novel, tmp_path / "refused")
    assert not (tmp_path / "refused" / MANIFEST_NAME).exists()
    manifest = import_noveliser(novel, tmp_path / "out", parse_aliases(["Elena=Elena Marchetti"]))
    assert manifest["label_map"]["Elena"] == "elena_marchetti"
    assert "elena" not in manifest["identities"]
    rows = script_rows(tmp_path / "out" / "script" / f"{manifest['chapters'][0]['stem']}.jsonl")
    assert next(iter(rows[index])) == "elena_marchetti"
    with pytest.raises(NoveliserImportError, match="not an exact characters.json name"):
        plan_import(novel, {"Elena": "Elena M."})


# ##################################################################
# test case variant and malformed labels fail closed
@pytest.mark.parametrize(
    ("label", "match"),
    [("elena marchetti", "inexact variant"), (" Gerald", "malformed"), ("NARRATOR", "inexact variant")],
)
def test_variant_or_malformed_label_fails_closed(novel: Path, label: str, match: str) -> None:
    lines = audio(novel, 2)
    lines[5]["speaker"] = label
    write_audio(novel, 2, lines)
    drop_sections(novel, 2)
    with pytest.raises(NoveliserImportError, match=match):
        plan_import(novel)


# ##################################################################
# test structural problems fail closed
def test_checkpoint_mismatch_missing_chapter_and_bad_text_fail_closed(novel: Path) -> None:
    lines = audio(novel, 3)
    lines[0]["text"] = lines[0]["text"] + " extra"
    write_audio(novel, 3, lines)
    with pytest.raises(NoveliserImportError, match="differs from its section checkpoints at line 0"):
        plan_import(novel)
    drop_sections(novel, 3)
    plan_import(novel)
    lines[1]["text"] = "   "
    write_audio(novel, 3, lines)
    with pytest.raises(NoveliserImportError, match="text is empty"):
        plan_import(novel)
    lines[1]["text"] = "bad\x07bell"
    write_audio(novel, 3, lines)
    with pytest.raises(NoveliserImportError, match="control characters"):
        plan_import(novel)
    (novel / "audio_chapter_3.json").unlink()
    with pytest.raises(NoveliserImportError, match=r"missing \[3\]"):
        plan_import(novel)
    metadata = json.loads((novel / "metadata.json").read_text())
    metadata["status"] = "writing"
    (novel / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(NoveliserImportError, match="not finished"):
        plan_import(novel)


# ##################################################################
# test foreign project and frozen cast are protected
def test_refuses_foreign_project_and_cast_drift(novel: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / "audio").mkdir(parents=True)
    (out / "audio" / "01-other.wav").write_bytes(b"other book")
    with pytest.raises(NoveliserImportError, match="non-noveliser project"):
        import_noveliser(novel, out)
    clean = tmp_path / "clean"
    manifest = import_noveliser(novel, clean)
    voices = {i: {"description": i} for i in manifest["identities"]}
    voices["_cast_plan"] = {"fingerprint": "casting metadata, not a speaker"}
    (clean / "voices.json").write_text(json.dumps(voices))
    verify_cast_frozen(clean, manifest)
    del voices["station_man"]
    voices["station_master"] = {"description": "invented"}
    (clean / "voices.json").write_text(json.dumps(voices))
    with pytest.raises(NoveliserImportError, match="missing \\['station_man'\\], extra \\['station_master'\\]"):
        verify_cast_frozen(clean, manifest)


# ##################################################################
# test step runner on an imported folder
# extract imports, characters/scripts verify without any LLM, other inputs are refused.
def test_step_runner_uses_import(novel: Path, capsys: pytest.CaptureFixture[str]) -> None:
    unique = novel.with_name(f"noveliser-step-{uuid.uuid4().hex[:12]}")
    novel.rename(unique)
    out = get_output_dir(unique)
    try:
        assert run_step("voices", unique) == 2
        assert "not imported yet" in capsys.readouterr().out
        assert run_step("extract", unique) == 0
        assert "1094 lines, 10 speaker identities" in capsys.readouterr().out
        assert run_step("characters", unique) == 0
        assert "station_man: Station Man" in capsys.readouterr().out
        assert run_step("scripts", unique) == 0
        assert is_step_complete(out, "scripts")
        epub = unique.with_suffix(".epub")
        epub.write_bytes(b"not used")
        assert run_step("scripts", epub) == 2
        with pytest.raises(SystemExit, match="noveliser"):
            run_pipeline(epub)
    finally:
        shutil.rmtree(out, ignore_errors=True)


# ##################################################################
# test cli import-only
# the real ./run entrypoint imports in the venv and does no TTS/image/movie work.
def test_cli_import_only(novel: Path) -> None:
    unique = novel.with_name(f"noveliser-cli-{uuid.uuid4().hex[:12]}")
    novel.rename(unique)
    (unique / "cover.jpg").write_bytes(b"\xff\xd8\xff cover bytes")
    out = get_output_dir(unique)
    try:
        result = subprocess.run(
            [str(ROOT / "run"), "noveliser", str(unique), "--import-only", "--style", "watercolour"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "1094 lines, 10 speaker identities" in result.stdout
        assert (out / "cover.jpg").read_bytes() == b"\xff\xd8\xff cover bytes"
        assert (out / "style.txt").read_text() == "watercolour"
        assert not (out / "audio").exists() and not (out / "voices.json").exists()
        assert not (out / ".pipeline.lock").exists()
        write_audio(unique, 1, audio(unique, 1)[:-1])
        drop_sections(unique, 1)
        refused = subprocess.run(
            [str(ROOT / "run"), "noveliser", str(unique), "--import-only"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert refused.returncode == 2
        assert "noveliser import refused" in refused.stderr and "--reimport" in refused.stderr
    finally:
        for path in out.parent.glob(out.name + "*"):
            shutil.rmtree(path, ignore_errors=True)


# ##################################################################
# test pipeline import-only holds and releases the single-runner lock
def test_pipeline_import_only_releases_lock(novel: Path) -> None:
    unique = novel.with_name(f"noveliser-pipe-{uuid.uuid4().hex[:12]}")
    novel.rename(unique)
    out = get_output_dir(unique)
    try:
        assert run_noveliser_pipeline(unique, import_only=True) is None
        assert (out / MANIFEST_NAME).is_file() and not (out / ".pipeline.lock").exists()
        (out / ".pipeline.lock").write_text(str(os.getpid()))
        with pytest.raises(SystemExit, match="already running"):
            run_noveliser_pipeline(unique, import_only=True)
    finally:
        shutil.rmtree(out, ignore_errors=True)
