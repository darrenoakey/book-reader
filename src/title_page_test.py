"""Tests for title_page: real PIL compositing and a real ffmpeg assembly with
the title card prepended, verifying the card steals frames from scene 0 while
total runtime and A/V sync stay exact.
"""

import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from src.movie_assemble import FPS, assemble_movie, probe_duration, probe_frames
from src.title_page import TITLE_SECONDS, composite_text, load_or_make_tagline


# ##################################################################
# make still
# a real gradient PNG (gradients survive the compositing pixel check,
# flat colors do not prove the scrim drew)
def _make_gradient(path: Path) -> None:
    img = Image.new("RGB", (1920, 1088))
    px = img.load()
    for y in range(0, 1088, 4):
        for x in range(0, 1920, 4):
            px[x, y] = (x * 255 // 1920, y * 255 // 1088, 128)
    img.save(path)


# ##################################################################
# make tone wav
# real sine wav of exactly `seconds` at the pipeline rate
def _make_tone(path: Path, seconds: float) -> None:
    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-ar", "24000", "-ac", "1", str(path)],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr


# ##################################################################
# prepare project
# a minimal two-chapter project dir with scenes, audio, and a storyboard
def _prepare_project(out: Path) -> float:
    audio_dir = out / "audio"
    scenes_dir = out / "scenes"
    audio_dir.mkdir()
    scenes_dir.mkdir()
    _make_tone(audio_dir / "01-a.wav", 2.0)
    _make_tone(audio_dir / "02-b.wav", 2.0)
    for name in ("01-a", "02-b"):
        (audio_dir / f"{name}.timeline.json").write_text(json.dumps({
            "chapter": name,
            "lines": [{"index": 0, "speaker": "narrator", "text": "hi", "start": 0.0, "end": 2.0}],
        }))
    Image.new("RGB", (1920, 1088), (40, 60, 90)).save(scenes_dir / "0000.png")
    Image.new("RGB", (1920, 1088), (90, 60, 40)).save(scenes_dir / "0001.png")
    (out / "storyboard.json").write_text(json.dumps({
        "scenes": [
            {"index": 0, "start": 0.0, "end": 2.0, "characters": []},
            {"index": 1, "start": 2.0, "end": 4.0, "characters": []},
        ]
    }))
    return 4.0


# ##################################################################
# test composite text real
# compositing draws the scrim + lettering onto the art for real
def test_composite_text_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        art = tmp / "art.png"
        _make_gradient(art)
        dest = composite_text(art, tmp / "page.png", "The Smallest Dragonboy", "Anne McCaffrey", "A small boy dares to hope")
        assert dest.stat().st_size > 1000
        img = Image.open(dest)
        assert img.size == (1920, 1088)
        # centre pixels must differ from the raw art (scrim + title ink)
        src = Image.open(art).convert("RGB").resize((1920, 1088))
        assert img.getpixel((960, 544)) != src.getpixel((960, 544))


# ##################################################################
# test tagline cached
# a written title_page.json is reused verbatim — no LLM call, stable card
def test_tagline_cached() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        (out / "title_page.json").write_text(json.dumps({
            "title": "The Smallest Dragonboy", "author": "Anne McCaffrey",
            "tagline": "A small boy dares to hope",
        }))
        assert load_or_make_tagline(out, "The Smallest Dragonboy", "Anne McCaffrey") == "A small boy dares to hope"


# ##################################################################
# test assemble with title real
# title card prepended: exact TITLE_SECONDS of card, scene 0 shrinks, total
# frames and duration still match the audio exactly
def test_assemble_movie_with_title_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        total_seconds = _prepare_project(out)
        # scene 0 is only 2s, so the card caps at (scene0 - 1s) = 1s here
        Image.new("RGB", (1920, 1088), (120, 30, 30)).save(out / "title_page.png")
        movie = assemble_movie(out, "Test Book")
        expected_frames = round(total_seconds * FPS)
        assert probe_frames(movie) == expected_frames
        assert abs(probe_duration(movie) - total_seconds) < 0.1
        segments_dir = out / "movie" / "segments"
        title_seg = next(segments_dir.glob("title.*.mp4"))
        assert probe_frames(title_seg) == FPS  # capped: 2s scene − 1s reserve
        # scene 0 keeps exactly its remaining second
        assert probe_frames(next(segments_dir.glob("0000.*.mp4"))) == FPS


# ##################################################################
# test title duration constant
# the card holds a real, watchable interval by default
def test_title_seconds_sane() -> None:
    assert 5.0 <= TITLE_SECONDS <= 30.0
