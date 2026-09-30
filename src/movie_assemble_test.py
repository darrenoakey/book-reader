"""Tests for movie_assemble: real ffmpeg Ken Burns segment rendering and a
real two-scene end-to-end assembly (tone wavs + Pillow images) verifying
frame-exact A/V sync — the property the whole design exists for.
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image

from src.audio_synth import wav_duration
from src.movie_assemble import (
    FPS,
    MOVES,
    STATICS,
    assemble_movie,
    probe_duration,
    probe_frames,
    render_segment,
    scene_move,
    zoompan_filter,
)


# ##################################################################
# make tone wav
# real sine wav of exactly `seconds` at the pipeline rate
def _make_tone(path: Path, seconds: float) -> None:
    r = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ar",
            "24000",
            "-ac",
            "1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr


# ##################################################################
# make still
# a real PNG still for segment rendering
def _make_still(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (1920, 1088), color).save(path)


# ##################################################################
# test zoompan filter
# every move (moving or static) produces a complete, well-formed filter
# string carrying exactly one linear start→end ramp
def test_zoompan_filter_shapes() -> None:
    for move in list(MOVES) + list(STATICS):
        f = zoompan_filter(move, 90)
        assert "zoompan=" in f and "d=90" in f
        assert "s=854x480" in f and f"fps={FPS}" in f
        assert "format=yuv420p" in f
        assert "scale=" not in f  # pre-scaled input, never an in-graph upscale


# ##################################################################
# test move grammar
# moving moves change the view in ONE direction; statics hold perfectly
# still; every 5th scene rests; a title card shifts scene 0 off zoom-in
def test_move_grammar() -> None:
    for start, end in MOVES.values():
        assert start != end
    for start, end in STATICS.values():
        assert start == end
    assert STATICS  # the grammar MUST contain rest frames
    assert scene_move(4) in STATICS and scene_move(9) in STATICS
    assert scene_move(0) == "zoom-in"
    assert scene_move(0, title=True) != "zoom-in"
    for i in range(20):
        assert scene_move(i) in MOVES or scene_move(i) in STATICS


# ##################################################################
# test render segment real
# a real ffmpeg Ken Burns render produces exactly the requested frame count
def test_render_segment_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        still = tmp / "still.png"
        _make_still(still, (40, 60, 90))
        seg = render_segment(still, 45, "zoom-in", tmp / "seg.mp4")
        assert seg.stat().st_size > 1000
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                str(seg),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert out == "45"
        assert abs(probe_duration(seg) - 45 / FPS) < 0.1
        started = time.monotonic()
        again = render_segment(still, 45, "zoom-in", tmp / "seg-replay.mp4")
        assert time.monotonic() - started < 2
        assert probe_frames(again) == 45


# ##################################################################
# test assemble movie real
# two scenes + two chapter wavs → one mp4 whose duration matches the audio
# (the drift assertion inside assemble_movie must hold on real output)
def test_assemble_movie_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        audio_dir = out / "audio"
        scenes_dir = out / "scenes"
        audio_dir.mkdir()
        scenes_dir.mkdir()
        _make_tone(audio_dir / "01-a.wav", 2.0)
        _make_tone(audio_dir / "02-b.wav", 2.0)
        (audio_dir / "01-a.timeline.json").write_text(
            json.dumps(
                {
                    "chapter": "01-a",
                    "lines": [{"index": 0, "speaker": "narrator", "text": "hi", "start": 0.0, "end": 2.0}],
                }
            )
        )
        (audio_dir / "02-b.timeline.json").write_text(
            json.dumps(
                {
                    "chapter": "02-b",
                    "lines": [{"index": 0, "speaker": "narrator", "text": "there", "start": 0.0, "end": 2.0}],
                }
            )
        )
        _make_still(scenes_dir / "0000.png", (120, 40, 40))
        _make_still(scenes_dir / "0001.png", (40, 120, 40))
        (out / "storyboard.json").write_text(
            json.dumps(
                {
                    "style": "test",
                    "appearances": {},
                    "scenes": [
                        {"index": 0, "start": 0.0, "end": 2.0, "characters": [], "prompt": "a", "text_excerpt": ""},
                        {"index": 1, "start": 2.0, "end": 4.0, "characters": [], "prompt": "b", "text_excerpt": ""},
                    ],
                }
            )
        )
        movie = assemble_movie(out, "Test Film")
        assert movie.exists()
        assert abs(probe_duration(movie) - 4.0) < 0.5
        # audio stream present and AAC
        out_codec = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                str(movie),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert out_codec == "aac"
        # sanity: our source wavs really were 2s each
        assert abs(wav_duration(audio_dir / "01-a.wav") - 2.0) < 0.05
