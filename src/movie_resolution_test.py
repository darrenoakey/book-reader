# Real ffmpeg resolution and CLI coverage for movie rendering.

import json
import subprocess
import tempfile
import uuid
from pathlib import Path

from PIL import Image

from src.movie_assemble import (
    DEFAULT_RESOLUTION,
    assemble_movie,
    movie_has_resolution,
    probe_resolution,
    render_segment,
)


# ##################################################################
# make tone wav
# creates a real audio source at the production sample rate for a minimal movie project
def _make_tone(path: Path) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-ar", "24000", "-ac", "1", str(path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


# ##################################################################
# prepare project
# constructs a self-contained one-scene project using real audio and image assets
def _prepare_project(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    scenes_dir = output_dir / "scenes"
    audio_dir.mkdir()
    scenes_dir.mkdir()
    _make_tone(audio_dir / "01-chapter.wav")
    (audio_dir / "01-chapter.timeline.json").write_text(json.dumps({"chapter": "01-chapter", "lines": []}))
    Image.new("RGB", (1920, 1088), (50, 80, 120)).save(scenes_dir / "0000.png")
    (output_dir / "storyboard.json").write_text(json.dumps({"scenes": [
        {"index": 0, "start": 0.0, "end": 1.0, "characters": []},
    ]}))


# ##################################################################
# test assembly resolution switch real
# renders the same source assets at both supported resolutions without sharing segment or prescale caches
def test_assembly_resolution_switch_real() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory)
        _prepare_project(output_dir)
        first_movie = assemble_movie(output_dir, "Resolution Test")
        assert probe_resolution(first_movie) == (1280, 720)
        assert movie_has_resolution(output_dir, DEFAULT_RESOLUTION)
        assert (output_dir / "movie" / "720p" / "segments").exists()
        assert (output_dir / "movie" / "720p" / "segments" / ".prescaled").exists()

        same_destination = output_dir / "same-destination.mp4"
        render_segment(output_dir / "scenes" / "0000.png", 30, "zoom-in", same_destination)
        assert probe_resolution(same_destination) == (1280, 720)
        render_segment(output_dir / "scenes" / "0000.png", 30, "zoom-in", same_destination, resolution=1080)
        assert probe_resolution(same_destination) == (1920, 1080)

        second_movie = assemble_movie(output_dir, "Resolution Test", resolution=1080)
        assert probe_resolution(second_movie) == (1920, 1080)
        assert movie_has_resolution(output_dir, 1080)
        assert not movie_has_resolution(output_dir, DEFAULT_RESOLUTION)
        assert (output_dir / "movie" / "1080p" / "segments").exists()
        assert (output_dir / "movie" / "1080p" / "segments" / ".prescaled").exists()


# ##################################################################
# test cli resolution validation
# verifies both public commands accept their 720p default and reject unsupported choices before pipeline work starts
def test_cli_resolution_validation() -> None:
    root = Path(__file__).resolve().parents[1]
    for command in (("create", "missing.txt"), ("step", "movie", "missing.txt")):
        default_result = subprocess.run(
            [str(root / "run"), *command], cwd=root, capture_output=True, text=True, check=False
        )
        assert default_result.returncode == 1
        assert "input file not found" in default_result.stdout.lower() or "epub file not found" in default_result.stdout.lower()
        result = subprocess.run(
            [str(root / "run"), *command, "--resolution", "900"], cwd=root, capture_output=True, text=True, check=False
        )
        assert result.returncode == 2
        assert "invalid choice" in result.stderr

# ##################################################################
# test pipeline rerenders resolution real
# a complete movie state only skips when its published dimensions match the requested pipeline resolution
def test_pipeline_rerenders_when_resolution_changes_real() -> None:
    from src.epub_extract import get_output_dir
    from src.state import mark_step_complete

    with tempfile.TemporaryDirectory() as temporary_directory:
        input_path = Path(temporary_directory) / f"resolution-pipeline-{uuid.uuid4().hex}.txt"
        input_path.write_text("Pipeline input", encoding="utf-8")
        output_dir = get_output_dir(input_path)
        created_output = False
        try:
            if output_dir.exists():
                raise RuntimeError(f"test output directory unexpectedly exists: {output_dir}")
            output_dir.mkdir(parents=True)
            created_output = True
            _prepare_project(output_dir)
            chapters_dir = output_dir / "chapters"
            chapters_dir.mkdir()
            (chapters_dir / "00-intro.txt").write_text("Resolution Pipeline by Test Author, narrated by Test", encoding="utf-8")
            for step in (
                "extract", "characters", "voices_desc", "voices_clone", "scripts", "audio", "m4b", "storyboard",
                "titlepage", "refimages", "sceneimages", "movie",
            ):
                mark_step_complete(output_dir, step)
            assemble_movie(output_dir, "Resolution Pipeline")
            assert movie_has_resolution(output_dir, 720)
            root = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                [str(root / "run"), "create", str(input_path), "--resolution", "1080"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stderr
            assert movie_has_resolution(output_dir, 1080)
        finally:
            if created_output:
                import shutil

                shutil.rmtree(output_dir)
