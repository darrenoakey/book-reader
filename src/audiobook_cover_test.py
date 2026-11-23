"""Tests for audiobook_cover: the generated cover must be an EDIT of the book's
own source cover, composited locally, and reused only while its binding holds.

The local contract (squared source condition, palette-relation validation,
atomic writes, typography, fingerprint reuse) runs on real PNG/JPEG bytes. The
generation test makes ONE real qwen-image edit job through the arbiter; the
content-addressed image cache (movie_images) replays that real result on every
later run, so the suite stays fast without ever faking the service.
"""

import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from src import audiobook_cover as cover

REAL_ART_SIZE = 672  # smallest /16 square the art validator accepts; keeps the real job quick



def _ramp(path: Path, size: tuple[int, int], flip: bool = False) -> Path:
    """A real, readable image with a smooth diagonal palette gradient."""
    width, height = size
    ys, xs = np.mgrid[0:height, 0:width]
    value = ((xs + ys) * 255 // (width + height)).astype(np.uint16)
    if flip:
        value = 255 - value
    rgb = np.stack([value * 5 // 10, value * 6 // 10, value], axis=-1).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path


def _flat(path: Path, size: tuple[int, int], colour: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


def test_source_cover_path_precedence() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        assert cover.source_cover_path(out) is None
        _flat(out / "cover.png", (300, 400), (40, 60, 90))
        assert cover.source_cover_path(out) == out / "cover.png"
        _flat(out / "cover.jpeg", (300, 400), (40, 60, 90))
        assert cover.source_cover_path(out) == out / "cover.jpeg"
        (out / "cover.jpg").write_bytes(b"")
        assert cover.source_cover_path(out) == out / "cover.jpeg"


# ##################################################################
# test readable image
# squareness, decodability, and minimum size all gate readability
def test_is_readable_image() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        square = _ramp(tmp / "square.png", (400, 400))
        portrait = _ramp(tmp / "portrait.png", (300, 600))
        tiny = _ramp(tmp / "tiny.png", (64, 64))
        junk = tmp / "junk.png"
        junk.write_bytes(b"not an image at all")
        assert cover.is_readable_image(square, square=True)
        assert not cover.is_readable_image(portrait, square=True)
        assert cover.is_readable_image(portrait)
        assert not cover.is_readable_image(tiny)
        assert not cover.is_readable_image(junk)
        assert not cover.is_readable_image(tmp / "missing.png")


# ##################################################################
# test prepare source art
# a portrait EPUB cover becomes a square condition image that keeps the whole
# source art (edges of the untouched art blend into the blurred extension)
def test_prepare_source_art_squares_whole_source() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        source = tmp / "cover.jpeg"
        Image.new("RGB", (600, 900), (200, 40, 40)).save(source)
        prepared = cover.prepare_source_art(source, tmp / "square.png", size=400)
        assert prepared.is_file()
        assert cover.is_readable_image(prepared, minimum=400, square=True)
        with Image.open(prepared) as img:
            assert img.size == (400, 400)
            # full-height source survives: a middle column is still solid art
            assert img.getpixel((200, 200)) == (200, 40, 40)
            # left of the fitted art is the darkened blurred extension
            assert sum(img.getpixel((2, 200))) < sum((200, 40, 40))
        with pytest.raises(ValueError):
            cover.prepare_source_art(_ramp(tmp / "small.png", (120, 160)), tmp / "out.png", size=400)


# ##################################################################
# poster
# a synthetic book cover: sky, banded ground, a figure — enough structure for
# a composition comparison to be meaningful
def _poster(path: Path, layout: str = "figure", size: tuple[int, int] = (400, 600)) -> Path:
    width, height = size
    img = Image.new("RGB", size, (18, 24, 52))
    draw = ImageDraw.Draw(img)
    if layout == "figure":
        draw.polygon([(0, 120), (200, 40), (width, 140), (width, 0), (0, 0)], fill=(40, 96, 130))
        draw.rectangle((0, int(height * 0.7), width, height), fill=(196, 168, 96))
        draw.ellipse((80, 120, 320, 360), fill=(226, 96, 48))
    else:
        draw.rectangle((0, 0, width, int(height * 0.3)), fill=(196, 168, 96))
        draw.ellipse((110, 300, 300, 520), fill=(40, 120, 70))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ##################################################################
# test grid distance and art keeps source
# a repainted, blurred, relit or heavily edited version of the source passes;
# a different cover design, a tiny image, a non-square edit and a
# byte-identical copy are all rejected
def test_art_keeps_source_accepts_edit_rejects_unrelated() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        source = cover.prepare_source_art(_poster(tmp / "cover.jpg"), tmp / "square.png", size=700)
        assert cover.grid_distance(source, source) == 0.0
        with Image.open(source) as opened:
            base = opened.convert("RGB")

        base.filter(ImageFilter.GaussianBlur(5)).save(tmp / "blurred.png")
        assert cover.grid_distance(tmp / "blurred.png", source) < 5
        assert cover.art_keeps_source(tmp / "blurred.png", source)

        ImageEnhance.Color(ImageEnhance.Brightness(base).enhance(1.12)).enhance(1.25).save(tmp / "relit.png")
        assert cover.art_keeps_source(tmp / "relit.png", source)

        repainted = base.copy()
        rd = ImageDraw.Draw(repainted)
        rd.rectangle((40, 400, 660, 640), fill=(210, 120, 60))
        rd.ellipse((270, 200, 470, 400), fill=(230, 200, 120))
        repainted.save(tmp / "repainted.png")
        assert cover.grid_distance(tmp / "repainted.png", source) <= cover.MAX_GRID_DISTANCE
        assert cover.art_keeps_source(tmp / "repainted.png", source)

        other = cover.prepare_source_art(_poster(tmp / "other.jpg", "banded"), tmp / "other.png", size=700)
        assert cover.grid_distance(other, source) > cover.MAX_GRID_DISTANCE
        assert not cover.art_keeps_source(other, source)

        flat = _flat(tmp / "flat.png", (700, 700), (0, 250, 0))
        assert not cover.art_keeps_source(flat, source)
        assert not cover.art_keeps_source(_ramp(tmp / "small_art.png", (256, 256)), source)
        assert not cover.art_keeps_source(_ramp(tmp / "wide_art.png", (600, 400)), source)

        copy = tmp / "copy.png"
        copy.write_bytes(source.read_bytes())
        assert not cover.art_keeps_source(copy, source)


# ##################################################################
# test compose cover
# title and author are drawn locally onto a real 1400x1400 JPEG, atomically
def test_compose_cover_writes_square_without_leftovers() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        art = _ramp(tmp / "art.png", (512, 512))
        dest = cover.compose_cover(art, tmp / "cover.jpg", "The Smallest Dragonboy", "Anne McCaffrey")
        assert dest.is_file()
        assert cover.is_readable_image(dest, minimum=cover.COVER_SIZE // 2, square=True)
        with Image.open(dest) as img:
            assert img.size == (cover.COVER_SIZE, cover.COVER_SIZE)
            # the title band shadows the art, so it differs from a plain resize
            plain = Image.open(art).convert("RGB").resize((cover.COVER_SIZE, cover.COVER_SIZE), Image.LANCZOS)
            assert img.getpixel((700, 140)) != plain.getpixel((700, 140))
        assert not (tmp / ".cover.jpg.partial").exists()


# ##################################################################
# test atomic writes
# a failed save leaves the previous artifact intact and no partial behind
def test_atomic_save_failure_keeps_previous() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        dest = tmp / "cover.jpg"
        dest.write_bytes(b"previous")
        # RGBA cannot be encoded as JPEG: the save fails at encode time
        with pytest.raises(OSError):
            cover.atomic_save(Image.new("RGBA", (8, 8), (1, 2, 3, 4)), dest, format="JPEG")
        assert dest.read_bytes() == b"previous"
        assert not (tmp / ".cover.jpg.partial").exists()
        cover.atomic_save(Image.new("RGB", (8, 8)), dest, format="JPEG")
        assert dest.read_bytes() != b"previous"


# ##################################################################
# test fingerprint and cover is current
# a bound cover is current only while the fingerprint still recomputes
def test_cover_is_current_binding() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        assert not cover.cover_is_current(out, "T", "A")
        source = _ramp(out / "cover.jpeg", (400, 600))
        prepared = cover.prepare_source_art(source, out / cover.SOURCE_SQUARE_NAME, size=512)
        art = _ramp(out / cover.ART_NAME, (512, 512))
        generated = cover.compose_cover(art, out / cover.COVER_NAME, "T", "A")
        meta = cover.cover_binding(
            title="T", author="A", source=source, prepared=prepared, art=art, cover=generated, seed=7, prompt="p"
        )
        binding = out / cover.BINDING_NAME
        binding.write_text(json.dumps(meta), encoding="utf-8")
        assert cover.cover_is_current(out, "T", "A")
        assert not cover.cover_is_current(out, "T", "Another Author")
        assert not cover.cover_is_current(out, "Other Title", "A")

        binding.write_text("{not json", encoding="utf-8")
        assert not cover.cover_is_current(out, "T", "A")

        meta["seed"] = 8
        binding.write_text(json.dumps(meta), encoding="utf-8")
        assert not cover.cover_is_current(out, "T", "A")

        meta["seed"] = 7
        binding.write_text(json.dumps(meta), encoding="utf-8")
        assert cover.cover_is_current(out, "T", "A")
        preserved_source = source.read_bytes()
        source.write_bytes(b"different source")
        assert not cover.cover_is_current(out, "T", "A")
        source.write_bytes(preserved_source)
        cover.compose_cover(_ramp(out / "other.png", (512, 512), flip=True), out / cover.COVER_NAME, "T", "A")
        assert not cover.cover_is_current(out, "T", "A")


# ##################################################################
# test generate cover edit (real qwen-image edit job, replayed from cache)
# the art is a real edit conditioned on the squared source cover; the bound
# cover replays without regenerating, and the source cover is never touched
def test_generate_audiobook_cover_real_edit() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        source = _poster(out / "cover.jpeg", "figure", (400, 600))
        source_bytes = source.read_bytes()
        first = cover.generate_audiobook_cover(out, "The Harbour Song", "Test Author", art_size=REAL_ART_SIZE)
        assert first == out / cover.COVER_NAME
        assert source.read_bytes() == source_bytes  # the EPUB cover survives untouched
        assert cover.art_keeps_source(out / cover.ART_NAME, out / cover.SOURCE_SQUARE_NAME)
        with Image.open(first) as img:
            assert img.size == (cover.COVER_SIZE, cover.COVER_SIZE)
        meta = json.loads((out / cover.BINDING_NAME).read_text(encoding="utf-8"))
        assert meta["source"] == "cover.jpeg"
        assert meta["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
        assert "The Harbour Song" in meta["prompt"]
        assert cover.cover_is_current(out, "The Harbour Song", "Test Author")

        from src.m4b_assemble import find_cover_image

        assert find_cover_image(out) == first  # the generated cover, not the source, is embedded

        stamp = first.stat().st_mtime_ns
        started = time.monotonic()
        assert cover.generate_audiobook_cover(out, "The Harbour Song", "Test Author", art_size=REAL_ART_SIZE) == first
        assert first.stat().st_mtime_ns == stamp
        assert time.monotonic() - started < 2


# ##################################################################
# test reuse bound art
# a retitled project re-composites from its existing derived art (no image
# job); a missing source cover fails closed before any image job
def test_generate_reuses_derived_art_and_requires_source() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        with pytest.raises(FileNotFoundError, match="no source cover"):
            cover.generate_audiobook_cover(out, "Title", "Author")
        _ramp(out / "cover.jpg", (400, 600))
        prepared = cover.prepare_source_art(out / "cover.jpg", out / cover.SOURCE_SQUARE_NAME)
        # art previously derived from this source (a real local repaint of it)
        with Image.open(prepared) as opened:
            opened.filter(ImageFilter.GaussianBlur(3)).save(out / cover.ART_NAME, format="PNG")
        art_bytes = (out / cover.ART_NAME).read_bytes()
        dest = cover.generate_audiobook_cover(out, "Title Redux", "Author")
        assert dest == out / cover.COVER_NAME and dest.is_file()
        assert (out / cover.ART_NAME).read_bytes() == art_bytes
        assert (out / "cover.jpg").is_file()  # a .jpg source is not overwritten by the generated cover
        assert json.loads((out / cover.BINDING_NAME).read_text(encoding="utf-8"))["title"] == "Title Redux"
