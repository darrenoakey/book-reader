"""Real ffmpeg fixtures: the verifier must pass a good audiobook and name each defect."""

import json
import subprocess
from pathlib import Path

import pytest

from src.audiobook_verify import (
    MANIFEST_NAME,
    canonical_digest,
    expectation_from_audio,
    image_size,
    load_manifest,
    manifest_is_intact,
    probe_m4b,
    sha256_file,
    verify_audiobook,
    write_manifest,
)

TITLE = "The Salt Road"
AUTHOR = "Ada Ferrers"
RATE = 24000


# ##################################################################
# fixtures
# a real square JPEG cover and real chapter recordings, all via ffmpeg/PIL
def _run(cmd: list[str], **kwargs) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {result.stderr[-800:]}")


def _audio(dest: Path, seconds: float, frequency: int = 440) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration={seconds}",
            "-ar",
            str(RATE),
            "-ac",
            "1",
            str(dest),
        ]
    )


def _silence(dest: Path, seconds: float) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={RATE}:cl=mono",
            "-t",
            str(seconds),
            str(dest),
        ]
    )


def _cover(dest: Path, colour: tuple[int, int, int] = (30, 70, 120)) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1400, 1400), colour)
    draw = ImageDraw.Draw(img)
    draw.ellipse((200, 200, 1200, 1200), fill=(220, 150, 40))
    draw.rectangle((0, 0, 1400, 380), fill=(10, 10, 20))
    img.save(dest, format="JPEG", quality=92)


def _project(root: Path) -> Path:
    """A project with two chapter recordings, a chime, announcements and a cover."""
    audio = root / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    _audio(audio / "00-intro.wav", 2.0)
    _audio(audio / "01-chapter_one.wav", 1.5, frequency=523)
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=784:sample_rate={RATE}:duration=0.3",
            "-ar",
            str(RATE),
            str(audio / "chime.wav"),
        ]
    )
    _audio(audio / "00-intro.announce.wav", 1.0, frequency=659)
    _audio(audio / "01-chapter_one.announce.wav", 1.0, frequency=659)
    _cover(root / "audiobook_cover.jpg")
    return root


def _assemble(
    project: Path,
    m4b: Path,
    *,
    cover_mode: str = "copy",
    tags: dict | None = None,
    truncate: bool = False,
) -> Path:
    """Assemble like m4b_assemble does, with optional deliberate defects."""
    audio = project / "audio"
    scratch = project / "_scratch"
    scratch.mkdir(exist_ok=True)
    pause = scratch / "pause.wav"
    _silence(pause, 3.0)
    chime_pause = scratch / "chime_pause.wav"
    _silence(chime_pause, 0.5)
    announce_pause = scratch / "announce_pause.wav"
    _silence(announce_pause, 1.0)

    expectation = expectation_from_audio(audio, title=TITLE, author=AUTHOR, cover=project / "audiobook_cover.jpg")
    order: list[Path] = []
    starts: list[float] = []
    for index, chapter in enumerate(expectation.chapters):
        stem = sorted(f.stem for f in audio.glob("*.wav") if not f.stem.endswith(".announce") and f.stem != "chime")[
            index
        ]
        order += [
            audio / "chime.wav",
            chime_pause,
            audio / f"{stem}.announce.wav",
            announce_pause,
            audio / f"{stem}.wav",
            pause,
        ]
        starts.append(chapter.start)
    if truncate:
        order = order[:-3]  # drop the last recording, its pause and the silence before it

    listfile = scratch / "concat.txt"
    listfile.write_text("".join(f"file '{path}'\n" for path in order), encoding="utf-8")
    full = scratch / "full.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listfile),
            "-ar",
            str(RATE),
            "-ac",
            "1",
            str(full),
        ]
    )

    metadata = scratch / "metadata.txt"
    lines = [";FFMETADATA1"]
    for chapter, start in zip(expectation.chapters, starts):
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={int(start * 1000)}",
            f"END={int(chapter.end * 1000)}",
            f"title={chapter.title}",
        ]
    metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")

    resolved = {"title": TITLE, "artist": AUTHOR, "album": TITLE}
    resolved.update(tags or {})
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(full),
        "-i",
        str(project / "audiobook_cover.jpg"),
        "-i",
        str(metadata),
    ]
    cmd += [
        "-map_metadata",
        "2",
        "-map",
        "0:a",
        "-map",
        "1:v",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
    ]
    cmd += ["-c:v", "copy"] if cover_mode == "copy" else ["-c:v", "mjpeg"]
    cmd += ["-disposition:v:0", "attached_pic"]
    for key, value in resolved.items():
        cmd += ["-metadata", f"{key}={value}"]
    cmd.append(str(m4b))
    _run(cmd)
    return m4b


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict:
    """One real project assembled the production way: exact JPEG bytes."""
    root = tmp_path_factory.mktemp("audiobook") / "salt_road"
    project = _project(root)
    m4b = _assemble(project, root / "salt_road.m4b")
    expectation = expectation_from_audio(
        project / "audio",
        title=TITLE,
        author=AUTHOR,
        cover=project / "audiobook_cover.jpg",
    )
    manifest = write_manifest(root / MANIFEST_NAME, expectation, m4b=m4b)
    return {"root": root, "project": project, "m4b": m4b, "manifest": manifest}


# ##################################################################
# expectation from audio
# the manifest must mirror m4b_assemble's own pre-roll, pause and truncation math
def test_expectation_reproduces_assembly_timing(built: dict) -> None:
    expectation = expectation_from_audio(built["project"] / "audio", title=TITLE, author=AUTHOR)
    assert expectation.announcements is True
    assert expectation.chapter_titles == ("Intro", "Chapter One")
    assert [chapter.source for chapter in expectation.chapters] == [
        "00-intro.wav",
        "01-chapter_one.wav",
    ]
    intro = expectation.chapters[0]
    # the marker opens where the pre-roll begins and closes after the recording
    assert intro.start == 0.0
    assert intro.end == pytest.approx(4.8, abs=0.01)
    second = expectation.chapters[1]
    assert second.start == pytest.approx(7.8, abs=0.01)
    assert second.end == pytest.approx(12.1, abs=0.01)
    # the trailing pause is part of the rendered audio too
    assert expectation.duration_seconds == pytest.approx(15.1, abs=0.02)


def test_expectation_numbers_duplicate_titles(tmp_path: Path) -> None:
    audio = tmp_path / "dup" / "audio"
    audio.mkdir(parents=True)
    for name in ("00-intro.wav", "01-dedication.wav", "02-dedication.wav"):
        _audio(audio / name, 1.0)
    expectation = expectation_from_audio(audio, title=TITLE, author=AUTHOR)
    assert expectation.chapter_titles == ("Intro", "Dedication 1", "Dedication 2")


def test_expectation_refuses_empty_audio_dir(tmp_path: Path) -> None:
    (tmp_path / "empty" / "audio").mkdir(parents=True)
    with pytest.raises(ValueError, match="no chapter audio files"):
        expectation_from_audio(tmp_path / "empty" / "audio", title=TITLE, author=AUTHOR)


def test_expectation_refuses_missing_audio_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="audio directory not found"):
        expectation_from_audio(tmp_path / "absent", title=TITLE, author=AUTHOR)


# ##################################################################
# manifest round trip
# the manifest is tamper-evident and refuses an edited fingerprint
def test_manifest_round_trip(built: dict) -> None:
    expectation, manifest = load_manifest(built["manifest"])
    assert manifest["version"] == 1
    assert manifest["m4b_sha256"] == sha256_file(built["m4b"])
    assert manifest["cover_sha256"] == sha256_file(built["project"] / "audiobook_cover.jpg")
    assert expectation.chapter_titles == ("Intro", "Chapter One")
    body = {key: value for key, value in manifest.items() if key != "fingerprint"}
    assert manifest["fingerprint"] == canonical_digest(body)


def test_manifest_edit_is_refused(built: dict, tmp_path: Path) -> None:
    manifest = json.loads(built["manifest"].read_text())
    manifest["duration_seconds"] += 60
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="failed its own fingerprint"):
        load_manifest(edited)


def test_manifest_version_is_enforced(built: dict, tmp_path: Path) -> None:
    manifest = json.loads(built["manifest"].read_text())
    manifest["version"] = 2
    body = {key: value for key, value in manifest.items() if key != "fingerprint"}
    manifest["fingerprint"] = canonical_digest(body)
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="version 2"):
        load_manifest(path)


def test_manifest_intactness_is_the_assembly_reuse_guard(built: dict) -> None:
    """``m4b_assemble`` reuses a previous render only when this says the record is unedited."""
    manifest = json.loads(built["manifest"].read_text())
    assert manifest_is_intact(manifest)
    tampered = {**manifest, "duration_seconds": manifest["duration_seconds"] + 60}
    assert not manifest_is_intact(tampered)
    assert not manifest_is_intact({**manifest, "fingerprint": ""})
    assert not manifest_is_intact({**manifest, "fingerprint": None})
    assert not manifest_is_intact({key: value for key, value in manifest.items() if key != "fingerprint"})
    assert not manifest_is_intact({**manifest, "version": 2})
    # it never raises: a manifest that is not even an object is merely not intact
    assert not manifest_is_intact([])
    assert not manifest_is_intact("audiobook_assembly.json")
    assert not manifest_is_intact(None)
    # and a body the digest cannot serialise is a plain False rather than a crash
    assert not manifest_is_intact({**manifest, "chapters": object()})


# ##################################################################
# the good audiobook
# a production-shaped file must pass every check with real measured detail
def test_good_audiobook_passes(built: dict) -> None:
    report = verify_audiobook(built["m4b"], built["manifest"], title=TITLE, author=AUTHOR)
    assert report.ok, report.problems
    assert all(check["ok"] for check in report.checks.values())
    assert report.checks["cover_source_parity"]["detail"].startswith("embedded cover is byte-identical")
    assert report.checks["chapter_order_parity"]["ok"] is True
    assert report.raise_for_status() is None


def test_good_audiobook_fingerprint_is_stable(built: dict) -> None:
    first = verify_audiobook(built["m4b"], built["manifest"], title=TITLE, author=AUTHOR)
    second = verify_audiobook(built["m4b"], built["manifest"], title=TITLE, author=AUTHOR)
    assert first.fingerprint == second.fingerprint
    assert first.to_json() == second.to_json()


def test_report_is_json_serialisable(built: dict) -> None:
    report = verify_audiobook(built["m4b"], built["manifest"], title=TITLE, author=AUTHOR)
    restored = json.loads(report.to_json())
    assert restored["ok"] is True and restored["problems"] == []


# ##################################################################
# defects
# each broken delivery must fail, and name the defect it found
def test_reencoded_cover_is_rejected(built: dict, tmp_path: Path) -> None:
    """The pre-fix defect: mjpeg re-encodes, so the embedded bytes are not the art."""
    m4b = _assemble(built["project"], tmp_path / "reencoded.m4b", cover_mode="reencode")
    expectation = expectation_from_audio(
        built["project"] / "audio",
        title=TITLE,
        author=AUTHOR,
        cover=built["project"] / "audiobook_cover.jpg",
    )
    manifest = write_manifest(tmp_path / "reencoded.json", expectation, m4b=m4b)
    report = verify_audiobook(m4b, manifest, title=TITLE, author=AUTHOR)
    assert not report.ok
    assert "cover_source_parity" in {problem.split(":")[0] for problem in report.problems}
    with pytest.raises(ValueError, match="cover_source_parity"):
        report.raise_for_status()


def test_wrong_cover_picture_is_rejected(built: dict, tmp_path: Path) -> None:
    project = tmp_path / "other_art"
    project.mkdir()
    (project / "audio").symlink_to(built["project"] / "audio")
    _cover(project / "audiobook_cover.jpg", colour=(200, 20, 20))
    m4b = _assemble(project, tmp_path / "wrong_cover.m4b")
    expectation = expectation_from_audio(
        project / "audio",
        title=TITLE,
        author=AUTHOR,
        cover=built["project"] / "audiobook_cover.jpg",
    )
    manifest = write_manifest(tmp_path / "wrong_cover.json", expectation, m4b=m4b)
    report = verify_audiobook(
        m4b,
        manifest,
        title=TITLE,
        author=AUTHOR,
        cover=built["project"] / "audiobook_cover.jpg",
    )
    assert not report.ok
    assert "cover_source_parity" in {problem.split(":")[0] for problem in report.problems}


def test_truncated_audio_is_rejected(built: dict, tmp_path: Path) -> None:
    m4b = _assemble(built["project"], tmp_path / "truncated.m4b", truncate=True)
    expectation = expectation_from_audio(
        built["project"] / "audio",
        title=TITLE,
        author=AUTHOR,
        cover=built["project"] / "audiobook_cover.jpg",
    )
    manifest = write_manifest(tmp_path / "truncated.json", expectation, m4b=m4b)
    report = verify_audiobook(m4b, manifest, title=TITLE, author=AUTHOR)
    assert not report.ok
    names = {problem.split(":")[0] for problem in report.problems}
    assert "duration_parity" in names


def test_copied_title_tag_is_rejected(built: dict, tmp_path: Path) -> None:
    m4b = _assemble(
        built["project"],
        tmp_path / "copied.m4b",
        tags={"title": "Another Book", "album": "Another Book"},
    )
    expectation = expectation_from_audio(
        built["project"] / "audio",
        title=TITLE,
        author=AUTHOR,
        cover=built["project"] / "audiobook_cover.jpg",
    )
    manifest = write_manifest(tmp_path / "copied.json", expectation, m4b=m4b)
    report = verify_audiobook(m4b, manifest, title=TITLE, author=AUTHOR)
    assert not report.ok
    names = {problem.split(":")[0] for problem in report.problems}
    assert names & {"title_tag", "album_tag"}


def test_wrong_author_is_rejected(built: dict, tmp_path: Path) -> None:
    m4b = _assemble(built["project"], tmp_path / "author.m4b", tags={"artist": "Someone Else"})
    expectation = expectation_from_audio(
        built["project"] / "audio",
        title=TITLE,
        author=AUTHOR,
        cover=built["project"] / "audiobook_cover.jpg",
    )
    manifest = write_manifest(tmp_path / "author.json", expectation, m4b=m4b)
    report = verify_audiobook(m4b, manifest, title=TITLE, author=AUTHOR)
    assert not report.ok
    assert "artist_tag" in {problem.split(":")[0] for problem in report.problems}


def test_job_identity_is_not_taken_from_the_manifest(built: dict) -> None:
    """The manifest cannot name the book: the job's title and author decide."""
    report = verify_audiobook(
        built["m4b"],
        built["manifest"],
        title="A Different Book",
        author="Another Writer",
    )
    assert not report.ok
    names = {problem.split(":")[0] for problem in report.problems}
    assert {"title_tag", "album_tag", "artist_tag"} <= names


def test_replaced_m4b_bytes_are_rejected(built: dict, tmp_path: Path) -> None:
    """A manifest binds the delivered bytes, so re-rendered audio must be re-recorded."""
    other = _assemble(built["project"], tmp_path / "other.m4b", tags={"album": f"{TITLE} (re-render)"})
    original = built["manifest"].read_text()
    try:
        assert sha256_file(other) != sha256_file(built["m4b"])
        report = verify_audiobook(other, built["manifest"], title=TITLE, author=AUTHOR)
        assert not report.ok
        assert "m4b_bytes_bound" in {problem.split(":")[0] for problem in report.problems}
    finally:
        assert built["manifest"].read_text() == original


def test_missing_m4b_is_rejected(built: dict, tmp_path: Path) -> None:
    report = verify_audiobook(tmp_path / "absent.m4b", built["manifest"], title=TITLE, author=AUTHOR)
    assert not report.ok
    assert "m4b_readable" in {problem.split(":")[0] for problem in report.problems}


def test_unavailable_cover_is_rejected(built: dict, tmp_path: Path) -> None:
    report = verify_audiobook(
        built["m4b"],
        built["manifest"],
        title=TITLE,
        author=AUTHOR,
        cover=tmp_path / "gone.jpg",
    )
    assert not report.ok
    assert "cover_source_parity" in {problem.split(":")[0] for problem in report.problems}


# ##################################################################
# primitives
# the small real-evidence helpers behave on real files
def test_probe_reads_real_container(built: dict) -> None:
    probed = probe_m4b(built["m4b"])
    assert probed["exit"] == 0
    assert probed["info"]["format"]["tags"]["album"] == TITLE
    assert len(probed["info"]["chapters"]) == 2


def test_probe_reports_unreadable_file(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.m4b"
    bogus.write_bytes(b"not audio")
    probed = probe_m4b(bogus)
    assert probed["exit"] != 0 and probed["info"] == {}


def test_image_size_reads_real_jpeg(built: dict) -> None:
    assert image_size(built["project"] / "audiobook_cover.jpg") == (1400, 1400)


def test_image_size_rejects_non_image(tmp_path: Path) -> None:
    bogus = tmp_path / "not.jpg"
    bogus.write_bytes(b"nope")
    assert image_size(bogus) is None
