"""Generated audiobook cover: a qwen-image EDIT of the book's own source cover.

The audiobook cover must be an edit of the source material, not an unrelated
text-to-image picture. The EPUB's cover / title page (``cover.jpeg|jpg|png``,
written by ``epub_extract.extract_cover_image``) is therefore the CONDITION
image for a real qwen-image edit job: the model repaints this book's art at
higher fidelity, and the result must still carry the source's palette — a
generated image that ignored the reference is rejected instead of shipped.

Everything else is local and deterministic:

  * the source cover is first squared off onto a blurred extension of itself,
    so the WHOLE source art survives (no cropping away the artwork);
  * the model's art is validated (readable, square, palette still related to
    the source) before anything is composited;
  * title and author are drawn locally (correct spelling — never asked of the
    model), plus a small headphones badge marking the audio edition;
  * ``audiobook_cover.jpg`` (1400x1400, what ``m4b_assemble`` embeds first), ``cover_art.png``
    and ``cover_source_square.png`` are written atomically, and ``audiobook_cover.json``
    binds them by sha256 plus a fingerprint, so an unchanged project replays
    its cover instead of re-generating it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from src.movie_images import qwen_image_to_file
from src.title_page import (
    _GEORGIA_ITALIC,
    INK,
    INK_DIM,
    _fit_font,
    _font,
)

ART_SIZE = 1328  # Qwen-Image native 1:1 geometry
COVER_SIZE = 1400
COVER_NAME = "audiobook_cover.jpg"  # never one of the source names, so the EPUB cover survives
BINDING_NAME = "audiobook_cover.json"
ART_NAME = "cover_art.png"
SOURCE_SQUARE_NAME = "cover_source_square.png"
# Same precedence as m4b_assemble.find_cover_image, so the EPUB cover that the
# extract step wrote is the one we treat as the source.
SOURCE_EXTS = ("jpeg", "jpg", "png")

MIN_SOURCE_SIDE = 160  # smaller than this is a spacer/icon, not cover art
MIN_ART_SIDE = ART_SIZE // 2
MIN_IMAGE_BYTES = 10000

# How far the model's art may move from the squared source before it stops
# being an EDIT of this book's cover. Both images reduce to a 16x16 grid of mean
# RGB (mean channel difference, 0..255): insensitive to repainting, blur,
# relighting and grain, but it does catch a different picture. Measured margins:
# blur 0.3, relight 9.9, even a large repaint over the frame 28.6, versus a
# different cover design 60.9, two unrelated real book covers 32-71, and flat or
# random noise 90-121.
GRID = 16
GRID_MARGIN = 0.02  # ignore a hair of the edge: covers bleed to their borders
GRID_THUMB = 128
MAX_GRID_DISTANCE = 40.0

NEGATIVE = (
    "text, letters, words, typography, logo, title, caption, watermark, signature, "
    "border, picture frame, price sticker, barcode, scan, photocopy, screenshot, "
    "one-to-one copy of the reference"
)


# ##################################################################
# atomic writes
# a reader (or a crash) never observes a half-written cover artifact
def _partial_path(dest: Path) -> Path:
    return dest.with_name(f".{dest.name}.partial")


def atomic_save(img: Image.Image, dest: Path, **kwargs) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = _partial_path(dest)
    try:
        img.save(part, **kwargs)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return dest


def atomic_write_text(dest: Path, text: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = _partial_path(dest)
    try:
        part.write_text(text, encoding="utf-8")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return dest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ##################################################################
# source cover
# the EPUB's own cover file inside the project, or None when extraction
# produced no cover at all
def source_cover_path(output_dir: Path) -> Path | None:
    for ext in SOURCE_EXTS:
        path = output_dir / f"cover.{ext}"
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


# ##################################################################
# readable image
# a file we can actually decode, at least `minimum` px, optionally square
def is_readable_image(path: Path, *, minimum: int = MIN_SOURCE_SIDE, square: bool = False) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as img:
            img.load()
            width, height = img.size
    except (OSError, ValueError, Image.DecompressionBombError):
        return False
    if width < minimum or height < minimum:
        return False
    return not square or abs(width - height) <= max(8, width // 100)


# ##################################################################
# prepare source art
# the whole source cover fitted inside a square, sitting on a blurred and
# darkened extension of itself — a square condition image for the edit job
def prepare_source_art(source: Path, dest: Path, size: int = ART_SIZE) -> Path:
    with Image.open(source) as opened:
        img = ImageOps.exif_transpose(opened).convert("RGB")
    if min(img.size) < MIN_SOURCE_SIDE:
        raise ValueError(f"source cover {source.name} is too small to edit: {img.size}")
    background = ImageOps.fit(img, (size, size), Image.LANCZOS).filter(ImageFilter.GaussianBlur(size / 24))
    background = Image.eval(background, lambda v: int(v * 0.55))
    art = img.copy()
    art.thumbnail((size, size), Image.LANCZOS)
    background.paste(art, ((size - art.width) // 2, (size - art.height) // 2))
    return atomic_save(background, dest, format="PNG")


# ##################################################################
# grid signature
# coarse 16x16 mean-RGB grid of the frame interior: the composition and colour
# layout that an edit of this cover keeps and an unrelated picture does not
def grid_signature(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        img = opened.convert("RGB")
        width, height = img.size
        inner = img.crop(
            (
                int(width * GRID_MARGIN),
                int(height * GRID_MARGIN),
                int(width * (1 - GRID_MARGIN)),
                int(height * (1 - GRID_MARGIN)),
            )
        ).resize((GRID_THUMB, GRID_THUMB), Image.LANCZOS)
    step = GRID_THUMB // GRID
    arr = np.asarray(inner, dtype=np.float32)
    return arr.reshape(GRID, step, GRID, step, 3).mean(axis=(1, 3))


# ##################################################################
# grid distance
# mean absolute per-cell channel difference between two images, 0..255
def grid_distance(art: Path, source: Path) -> float:
    return float(np.abs(grid_signature(art) - grid_signature(source)).mean())


# ##################################################################
# art keeps source
# the generated art is a readable square that is really an EDIT of the source
# cover: not a byte-identical copy, and never a different picture
def art_keeps_source(art: Path, source: Path) -> bool:
    if not is_readable_image(art, minimum=MIN_ART_SIDE, square=True):
        return False
    if not is_readable_image(source, minimum=MIN_SOURCE_SIDE):
        return False
    if art.stat().st_size >= MIN_IMAGE_BYTES and art.read_bytes() == source.read_bytes():
        return False
    return grid_distance(art, source) <= MAX_GRID_DISTANCE


# ##################################################################
# cover prompt
# the EDIT instruction: repaint this book's own art, in this book's palette
def cover_prompt(title: str, author: str) -> str:
    return (
        "Edit the supplied reference image into the cover artwork of the audiobook edition of "
        f"{title} by {author}. "
        "Keep the reference's subject, figures, composition, palette and artistic medium: repaint "
        "the same scene at higher fidelity with richer light, depth and detail. "
        "Fill the frame edge to edge as one continuous square painted cover illustration, "
        "with calm, uncluttered space in the upper third and lower fifth for lettering. "
        "Remove any lettering, logos, borders, price panels or publisher marks from the art. "
        "This is a fresh painted square cover, never a scan, photocopy or one-to-one copy of the reference."
    )


# ##################################################################
# fingerprint
# one hash over every input that decides the cover's bytes, so a stale cover
# (different source, art, title or author) is regenerated instead of trusted
def fingerprint(
    *,
    title: str,
    author: str,
    source_sha256: str | None,
    art_sha256: str | None,
    cover_sha256: str,
    seed: int | None,
    prompt: str | None,
) -> str:
    payload = {
        "title": title,
        "author": author,
        "source_sha256": source_sha256,
        "art_sha256": art_sha256,
        "cover_sha256": cover_sha256,
        "seed": seed,
        "prompt": prompt,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# ##################################################################
# cover is current
# the bound cover exists, is a readable square, and still hashes to the
# fingerprint recorded when it was made for this title and author
def cover_is_current(output_dir: Path, title: str, author: str) -> bool:
    cover, binding_path = output_dir / COVER_NAME, output_dir / BINDING_NAME
    if not cover.is_file() or not binding_path.is_file():
        return False
    try:
        meta = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(meta, dict) or meta.get("title") != title or meta.get("author") != author:
        return False
    if not is_readable_image(cover, minimum=COVER_SIZE // 2, square=True):
        return False
    for member, hash_field in (("source", "source_sha256"), ("source_square", "source_square_sha256"), ("art", "art_sha256")):
        name = meta.get(member)
        if not isinstance(name, str) or Path(name).name != name:
            return False
        asset = output_dir / name
        if not asset.is_file() or _sha256(asset) != meta.get(hash_field):
            return False
    return meta.get("fingerprint") == fingerprint(
        title=title,
        author=author,
        source_sha256=meta.get("source_sha256"),
        art_sha256=meta.get("art_sha256"),
        cover_sha256=_sha256(cover),
        seed=meta.get("seed"),
        prompt=meta.get("prompt"),
    )


# ##################################################################
# compose cover
# title band at the top, author at the foot, headphones badge bottom-right
def compose_cover(art: Path, dest: Path, title: str, author: str) -> Path:
    size = COVER_SIZE
    img = Image.open(art).convert("RGB").resize((size, size), Image.LANCZOS)
    scrim = Image.new("L", (size, size), 0)
    sd = ImageDraw.Draw(scrim)
    sd.rectangle((0, 0, size, int(size * 0.30)), fill=170)
    sd.rectangle((0, int(size * 0.84), size, size), fill=170)
    scrim = scrim.filter(ImageFilter.GaussianBlur(70))
    img = Image.composite(Image.new("RGB", img.size, (8, 8, 14)), img, scrim)
    draw = ImageDraw.Draw(img)
    title_font = _fit_font(draw, title, int(size * 0.84), 150)
    author_font = _fit_font(draw, author, int(size * 0.62), 64)
    author_font = _font(_GEORGIA_ITALIC, author_font.size)

    def centered(text: str, font, fill, top: float) -> None:
        width = draw.textlength(text, font=font)
        draw.text(((size - width) / 2, top), text, font=font, fill=fill)

    centered(title, title_font, INK, size * 0.08)
    centered(author, author_font, INK_DIM, size * 0.885)
    draw_headphones_badge(draw, size)
    return atomic_save(img, dest, format="JPEG", quality=92)


# ##################################################################
# headphones badge
# a small round badge with a line-drawn pair of headphones
def draw_headphones_badge(draw: ImageDraw.ImageDraw, size: int) -> None:
    r = int(size * 0.045)
    cx, cy = size - int(size * 0.075), size - int(size * 0.075)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(20, 20, 28), outline=INK_DIM, width=max(2, r // 14))
    band = int(r * 0.55)
    width = max(3, r // 7)
    draw.arc((cx - band, cy - band, cx + band, cy + band), start=180, end=360, fill=INK, width=width)
    cup_w, cup_h = int(r * 0.26), int(r * 0.48)
    for side in (-1, 1):
        x = cx + side * band
        draw.rounded_rectangle(
            (x - cup_w // 2, cy - cup_h // 6, x + cup_w // 2, cy + cup_h * 5 // 6),
            radius=max(2, cup_w // 3),
            fill=INK,
        )


# ##################################################################
# binding
# the record that lets an unchanged project replay its cover verbatim
def cover_binding(
    *,
    title: str,
    author: str,
    source: Path,
    prepared: Path,
    art: Path,
    cover: Path,
    seed: int,
    prompt: str,
) -> dict:
    return {
        "title": title,
        "author": author,
        "source": source.name,
        "source_sha256": _sha256(source),
        "source_square": prepared.name,
        "source_square_sha256": _sha256(prepared),
        "art": art.name,
        "art_sha256": _sha256(art),
        "seed": seed,
        "prompt": prompt,
        "cover": cover.name,
        "fingerprint": fingerprint(
            title=title,
            author=author,
            source_sha256=_sha256(source),
            art_sha256=_sha256(art),
            cover_sha256=_sha256(cover),
            seed=seed,
            prompt=prompt,
        ),
    }


# ##################################################################
# cover seed
# stable per book, so the same source cover is repainted the same way
def cover_seed(title: str, author: str, source_sha256: str) -> int:
    digest = hashlib.sha256(f"{title}|{author}|{source_sha256}".encode()).hexdigest()
    return int(digest[:8], 16) % 100000


# ##################################################################
# generate audiobook cover
# idempotent: a bound, current cover is returned untouched; a previously
# generated art that still derives from the source cover is reused as-is
def generate_audiobook_cover(
    output_dir: Path,
    title: str,
    author: str,
    source_cover: Path | None = None,
    *,
    art_size: int = ART_SIZE,
) -> Path:
    dest = output_dir / COVER_NAME
    if cover_is_current(output_dir, title, author):
        return dest
    source = source_cover if source_cover is not None else source_cover_path(output_dir)
    if source is None:
        raise FileNotFoundError(
            f"no source cover in {output_dir} — the extract step must write the EPUB cover "
            f"({'/'.join('cover.' + ext for ext in SOURCE_EXTS)}) before an audiobook cover can be generated"
        )
    if not is_readable_image(source, minimum=MIN_SOURCE_SIDE):
        raise ValueError(f"source cover {source} is not a readable image at least {MIN_SOURCE_SIDE}px")

    source_sha = _sha256(source)
    prepared = prepare_source_art(source, output_dir / SOURCE_SQUARE_NAME, art_size)
    art = output_dir / ART_NAME
    seed = cover_seed(title, author, source_sha)
    prompt = cover_prompt(title, author)
    if art.is_file() and art_keeps_source(art, prepared):
        print(f"  cover art: reusing {art.name} (still derived from {source.name})")
    else:
        art.unlink(missing_ok=True)
        qwen_image_to_file(
            prompt,
            art,
            art_size,
            art_size,
            seed=seed,
            ref_image=prepared,
            why=f"audiobook cover edit of {source.name} for {title}",
            negative_prompt=NEGATIVE,
        )
        if not art_keeps_source(art, prepared):
            raise RuntimeError(
                f"qwen-image returned art for {title} that no longer derives from {source.name}: "
                "refusing to ship an unrelated generated cover"
            )

    compose_cover(art, dest, title, author)
    binding = cover_binding(
        title=title, author=author, source=source, prepared=prepared, art=art, cover=dest, seed=seed, prompt=prompt
    )
    atomic_write_text(output_dir / BINDING_NAME, json.dumps(binding, indent=2) + "\n")
    return dest


__all__ = [
    "ART_SIZE",
    "COVER_SIZE",
    "MAX_GRID_DISTANCE",
    "NEGATIVE",
    "art_keeps_source",
    "atomic_save",
    "atomic_write_text",
    "compose_cover",
    "cover_binding",
    "cover_is_current",
    "cover_prompt",
    "cover_seed",
    "draw_headphones_badge",
    "fingerprint",
    "generate_audiobook_cover",
    "grid_distance",
    "grid_signature",
    "is_readable_image",
    "prepare_source_art",
    "source_cover_path",
]
