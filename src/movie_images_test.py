"""Tests for movie_images: real Pillow contact-sheet composition and scene
reference selection. The qwen-image round trip itself is exercised once for
real against the arbiter server (the repo rule: tests are real, no mocks) —
one small t2i job, which also proves the owner-sanctioned qwen-image route
still works end to end.
"""

import tempfile
import time
from pathlib import Path

from PIL import Image

from src.movie_images import build_contact_sheet, qwen_image_to_file


# ##################################################################
# make portrait
# a real flat-color PNG standing in as a character portrait
def _make_portrait(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (256, 256), color).save(path)
    return path


# ##################################################################
# test contact sheet
# two portraits compose into one labelled strip of the right dimensions
def test_contact_sheet() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        a = _make_portrait(tmp / "keevan.png", (200, 80, 60))
        b = _make_portrait(tmp / "beterli.png", (60, 80, 200))
        sheet = build_contact_sheet([("keevan", a), ("beterli", b)], tmp / "sheet.png", tile=256)
        img = Image.open(sheet)
        # reference-sheet layout: margins, title band, label band
        assert img.size == (2 * 28 + 2 * 256 + 20, 72 + 256 + 56 + 2 * 28)
        # left tile is keevan's color, right tile beterli's (tile origins at
        # (28, 72) and (28+256+20, 72); sample each tile's centre)
        assert img.getpixel((28 + 128, 72 + 128)) == (200, 80, 60)
        assert img.getpixel((28 + 256 + 20 + 128, 72 + 128)) == (60, 80, 200)


# ##################################################################
# test style and cast condition sheet
# combines the supplied visual reference and cast likeness into the one Qwen condition image without treating either as scene content.
def test_style_and_cast_condition_sheet() -> None:
    from src.movie_images import _scene_condition

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        refs = root / "refs"
        refs.mkdir()
        _make_portrait(root / "style-reference.png", (70, 100, 140))
        _make_portrait(refs / "ira.png", (210, 120, 80))
        scenes = root / "scenes"
        scenes.mkdir()
        _make_portrait(scenes / "0000.png", (30, 40, 50))
        (root / "image_reference_mode.txt").write_text("style-and-cast-only\n", encoding="utf-8")
        condition, note = _scene_condition(root, ["ira"], 1, 0)
        assert condition is not None
        assert condition.name == "0001.png"
        assert "STYLE REFERENCE" in note
        assert "not a scene" in note
        assert "PREVIOUS SCENE" not in note
        assert "ira" in note


# ##################################################################
# test panel borders
# a synthetic two-panel image flags; a smooth scene-like gradient does not
def test_panel_borders() -> None:
    from src.movie_images import panel_borders

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        panel = Image.new("RGB", (1920, 1088), (30, 40, 60))
        right = Image.new("RGB", (960, 1088), (210, 190, 150))
        panel.paste(right, (960, 0))
        panel_path = tmp / "panel.png"
        panel.save(panel_path)
        assert len(panel_borders(panel_path)) >= 1

        smooth = Image.new("RGB", (1920, 1088))
        px = smooth.load()
        for y in range(1088):
            for x in range(1920):
                px[x, y] = (120 + x % 40, 140, 180 - y % 30)
        smooth_path = tmp / "smooth.png"
        smooth.save(smooth_path)
        assert panel_borders(smooth_path) == []


# ##################################################################
# test qwen image real
# one real small t2i job through the arbiter qwen-image adapter; proves
# submit/poll/result-bytes all work and returns a genuine PNG
def test_qwen_image_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "still.png"
        qwen_image_to_file(
            "A small bronze dragon hatchling on golden sand, painterly digital illustration.",
            dest,
            512,
            512,
            seed=99,
            steps=20,
            why="book-reader test",
        )
        assert dest.stat().st_size >= 10000
        with Image.open(dest) as img:
            assert img.format == "PNG"
            assert img.size[0] >= 480  # snapped to /16 but in the right ballpark
        replay = Path(tmpdir) / "still-replay.png"
        started = time.monotonic()
        qwen_image_to_file(
            "A small bronze dragon hatchling on golden sand, painterly digital illustration.",
            replay,
            512,
            512,
            seed=99,
            steps=20,
            why="book-reader test",
        )
        assert time.monotonic() - started < 2
        assert replay.read_bytes() == dest.read_bytes()
