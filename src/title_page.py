"""Title page: a generated opening card for the audiobook movie.

Shown during the opening seconds while the book is being introduced. Built in
three parts:

  1. TAGLINE — one short line capturing the point of the story, written by the
     LLM from the chapter text and cached in ``title_page.json`` (never
     regenerated once written — the card is stable across re-renders).
  2. ART — an evocative 16:9 key-art image from qwen-image (arbiter), prompted
     from the title + tagline in the storyboard's style, with calm central
     space for the overlay. Text is NOT asked of the model (spelling risk);
     it is composited locally instead.
  3. COMPOSITE — title, author, and tagline drawn with PIL over a soft scrim,
     guaranteeing legible, correctly-spelled text.

Output: ``title_page.png`` at scene geometry (1920x1088). movie_assemble
prepends it as the opening segment when the file exists.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src import llm
from src.movie_images import RESTRAINT, SCENE_HEIGHT, SCENE_WIDTH, qwen_image_to_file

# How long the title card holds on screen (carved out of scene 0's slot, so
# total runtime and A/V sync are untouched).
TITLE_SECONDS = 12.0

_GEORGIA = "/System/Library/Fonts/Supplemental/Georgia.ttf"
_GEORGIA_BOLD = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
_GEORGIA_ITALIC = "/System/Library/Fonts/Supplemental/Georgia Italic.ttf"

INK = (246, 241, 228)  # warm off-white
INK_DIM = (216, 208, 190)


# ##################################################################
# font
# Georgia at `size`, falling back to the PIL bitmap default off-Mac
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


# ##################################################################
# fit font
# largest Georgia Bold size (<= start) whose text fits within max_width
def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start: int) -> ImageFont.FreeTypeFont:
    size = start
    while size > 20:
        font = _font(_GEORGIA_BOLD, size)
        if draw.textlength(text, font=font) <= max_width:
            return font
        size -= 4
    return _font(_GEORGIA_BOLD, 20)


# ##################################################################
# story excerpt
# first chars of the concatenated chapter texts, for tagline generation
def _story_excerpt(output_dir: Path, chars: int = 4000) -> str:
    chapters_dir = output_dir / "chapters"
    parts = []
    for path in sorted(chapters_dir.glob("*.txt")):
        parts.append(path.read_text(encoding="utf-8"))
        if sum(len(p) for p in parts) >= chars:
            break
    return "\n\n".join(parts)[:chars]


# ##################################################################
# load or make tagline
# cached one-line "point of the story" for the card; LLM-written once
def load_or_make_tagline(output_dir: Path, title: str, author: str) -> str:
    meta_path = output_dir / "title_page.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("title") == title and meta.get("tagline"):
            return meta["tagline"]
    tagline = llm.ask_sync(
        (
            f"Title: {title}\nAuthor: {author}\n\n"
            f"Story excerpt:\n{_story_excerpt(output_dir)}\n\n"
            "Write a single short tagline (max 14 words) capturing the heart of this "
            "story — its emotional core or central struggle. Plain sentence case, no "
            "quotation marks, no restating the title, no trailing period. Reply with "
            "the tagline alone."
        ),
        system="You write one-line cover taglines for audiobook title cards.",
        temperature=0.4,
        max_tokens=60,
    )
    tagline = tagline.strip().strip('"').rstrip(".")
    meta_path.write_text(
        json.dumps({"title": title, "author": author, "tagline": tagline}, indent=2) + "\n",
        encoding="utf-8",
    )
    return tagline


# ##################################################################
# composite text
# draw title / author / tagline over the key art with a soft central scrim
def composite_text(art: Path, dest: Path, title: str, author: str, tagline: str) -> Path:
    img = Image.open(art).convert("RGB").resize((SCENE_WIDTH, SCENE_HEIGHT), Image.LANCZOS)

    # Soft dark scrim: a blurred black plate behind the text block so the
    # lettering reads over any art, without a hard-edged box.
    scrim = Image.new("L", (SCENE_WIDTH, SCENE_HEIGHT), 0)
    sd = ImageDraw.Draw(scrim)
    band_h = SCENE_HEIGHT // 2
    top = (SCENE_HEIGHT - band_h) // 2
    sd.rectangle((0, top, SCENE_WIDTH, top + band_h), fill=140)
    scrim = scrim.filter(ImageFilter.GaussianBlur(120))
    img = Image.composite(Image.new("RGB", img.size, (8, 8, 14)), img, scrim)

    draw = ImageDraw.Draw(img)
    cx = SCENE_WIDTH / 2
    max_w = int(SCENE_WIDTH * 0.78)

    title_font = _fit_font(draw, title, max_w, 120)
    byline = f"by {author}"
    byline_font = _font(_GEORGIA_ITALIC, 46)
    tag_font = _font(_GEORGIA_ITALIC, 42)
    wrapped = textwrap.wrap(tagline, width=52) or [tagline]

    def h(text: str, font) -> int:
        box = draw.textbbox((0, 0), text, font=font)
        return box[3] - box[1]

    gap_title_by, gap_by_tag, gap_lines = 26, 54, 14
    total_h = h(title, title_font) + gap_title_by + h(byline, byline_font) + gap_by_tag
    total_h += sum(h(line, tag_font) for line in wrapped) + gap_lines * (len(wrapped) - 1)
    y = (SCENE_HEIGHT - total_h) / 2

    def centered(text: str, font, fill, y_pos: float) -> float:
        w = draw.textlength(text, font=font)
        draw.text((cx - w / 2, y_pos), text, font=font, fill=fill)
        return y_pos + h(text, font)

    y = centered(title, title_font, INK, y) + gap_title_by
    y = centered(byline, byline_font, INK_DIM, y) + gap_by_tag
    for line in wrapped:
        y = centered(line, tag_font, INK_DIM, y) + gap_lines

    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="PNG")
    return dest


# ##################################################################
# generate title page
# full step: tagline (cached) + qwen-image key art + composited text
def generate_title_page(output_dir: Path, title: str, author: str) -> Path:
    dest = output_dir / "title_page.png"
    if dest.exists() and dest.stat().st_size >= 1000:
        return dest

    tagline = load_or_make_tagline(output_dir, title, author)
    print(f"  tagline: {tagline}")

    storyboard_path = output_dir / "storyboard.json"
    style = "Cinematic storybook illustration"
    negative = ""
    if storyboard_path.exists():
        storyboard = json.loads(storyboard_path.read_text(encoding="utf-8"))
        style = storyboard.get("style", style)
        forbidden = (storyboard.get("world_bible") or {}).get("forbidden", [])
        negative = ", ".join(forbidden)

    art = output_dir / "title_page_art.png"
    # The title itself is deliberately NOT in the prompt: qwen-image renders
    # quoted titles as logos even when told not to, and the composite draws
    # the real title locally. Typography is pushed into the NEGATIVE channel
    # (positive-prompt negations attract the concept instead of forbidding it).
    prompt = (
        "Cinematic title-card key art for an animated story. "
        f"{tagline}. "
        "One evocative emblematic scene capturing the story's heart — rich atmospheric "
        "composition, open calm sky or soft-focus space in the central area for a title overlay. "
        f"Style: {style}.{RESTRAINT}"
    )
    negative = ", ".join(filter(None, [negative, "text, letters, words, typography, logo, title, caption, watermark, signature"]))
    qwen_image_to_file(
        prompt,
        art,
        SCENE_WIDTH,
        SCENE_HEIGHT,
        seed=7777,
        why=f"title page art for {title}",
        negative_prompt=negative or None,
    )
    return composite_text(art, dest, title, author, tagline)


__all__ = ["TITLE_SECONDS", "composite_text", "generate_title_page", "load_or_make_tagline"]
