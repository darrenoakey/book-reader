"""Tests for movie_images: real Pillow contact-sheet composition and scene
reference selection. The qwen-image round trip itself is exercised once for
real against the arbiter server (the repo rule: tests are real, no mocks) —
one small t2i job, which also proves the owner-sanctioned qwen-image route
still works end to end.
"""

import tempfile
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
        assert img.size == (512, 256 + 56)
        # left tile is keevan's color, right tile beterli's
        assert img.getpixel((128, 128)) == (200, 80, 60)
        assert img.getpixel((384, 128)) == (60, 80, 200)


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
        img = Image.open(dest)
        assert img.format == "PNG"
        assert img.size[0] >= 480  # snapped to /16 but in the right ballpark
