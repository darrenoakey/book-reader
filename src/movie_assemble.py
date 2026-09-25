"""Movie assembly: turn the storyboard scene stills + chapter audio into a
Ken Burns "audiobook movie" — every still slowly pans or zooms so the picture
is never static, cut exactly on storyboard scene boundaries, muxed with the
full narration track.

Frame-exactness is the core design constraint (A/V sync over a long film):

  * the storyboard's scene times are seconds against the CONCATENATION of the
    chapter wavs (see movie_storyboard.load_global_lines), so the movie's
    audio track is exactly that concatenation, in the same order;
  * each scene renders at FPS with ``frames_i = round(end*FPS) -
    round(start*FPS)`` frames — cumulative rounding can never drift;
  * the final scene absorbs any remainder so total frames ==
    round(audio_seconds*FPS) exactly;
  * no ``-shortest`` anywhere (it silently drops tail frames); the mux caps
    nothing — video and audio are independently exact.

Segments render once each (idempotent skip) so a rerender after replacing a
scene image only re-renders that segment and re-concats.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from src.audio_synth import concat_wavs, wav_duration

FPS = 30
WIDTH = 1920
HEIGHT = 1080

# Ken Burns moves cycled deterministically by scene index — every scene moves,
# no two neighbours move the same way.
MOVES = ("zoom-in", "zoom-out", "pan-right", "pan-left", "pan-down", "pan-up")

# Stills arrive at 1920x1088; upscale 1.5x so pans/zooms crop from real
# pixels instead of smearing (zoompan samples from the scaled input).
_UPSCALE = f"scale={WIDTH * 3 // 2}:{HEIGHT * 3 // 2}:flags=lanczos"


# ##################################################################
# run checked
# ffmpeg/ffprobe wrapper that raises with stderr on failure
def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed: {result.stderr[-2000:]}")
    return result


# ##################################################################
# probe duration
# container duration in seconds via ffprobe
def probe_duration(path: Path) -> float:
    out = _run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ]
    ).stdout
    return float(out.strip())


# ##################################################################
# probe frames
# exact decoded frame count via ffprobe (metadata frame counts lie)
def probe_frames(path: Path) -> int:
    out = _run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
        ]
    ).stdout
    return int(out.strip())


# ##################################################################
# zoompan filter
# one Ken Burns move as a complete -vf chain: upscale, animate, deliver
# WIDTHxHEIGHT yuv420p at FPS with exactly `frames` frames per input frame
def zoompan_filter(move: str, frames: int) -> str:
    if frames < 1:
        raise ValueError("frames must be >= 1")
    if move not in MOVES:
        raise ValueError(f"unknown move {move!r}")
    t = f"on/{frames - 1}" if frames > 1 else "1"  # 0..1 progress over the segment
    if move == "zoom-in":
        z, x, y = f"1+0.12*{t}", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    elif move == "zoom-out":
        z, x, y = f"1.12-0.12*{t}", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    elif move == "pan-right":
        z, x, y = "1.14", f"(iw-iw/zoom)*{t}", "(ih-ih/zoom)/2"
    elif move == "pan-left":
        z, x, y = "1.14", f"(iw-iw/zoom)*(1-{t})", "(ih-ih/zoom)/2"
    elif move == "pan-down":
        z, x, y = "1.14", "(iw-iw/zoom)/2", f"(ih-ih/zoom)*{t}"
    else:  # pan-up
        z, x, y = "1.14", "(iw-iw/zoom)/2", f"(ih-ih/zoom)*(1-{t})"
    return (
        f"{_UPSCALE},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
        "format=yuv420p"
    )


# ##################################################################
# render segment
# render one still into an h264 segment of EXACTLY `frames` frames
def render_segment(image: Path, frames: int, move: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg", "-y", "-loop", "1", "-i", str(image),
            "-vf", zoompan_filter(move, frames),
            "-frames:v", str(frames),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-r", str(FPS), "-an", str(dest),
        ]
    )
    actual = probe_frames(dest)
    if actual != frames:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"segment {dest.name} has {actual} frames, expected {frames}")
    return dest


# ##################################################################
# chapter audio order
# the chapter wavs in exactly the order the storyboard offset them
def _chapter_wavs(output_dir: Path) -> list[Path]:
    audio_dir = output_dir / "audio"
    timelines = sorted(audio_dir.glob("*.timeline.json"))
    if not timelines:
        raise ValueError("no audio timelines — run the audio step first")
    wavs = []
    for tl in timelines:
        wav = audio_dir / f"{tl.stem.replace('.timeline', '')}.wav"
        if not wav.exists():
            raise ValueError(f"timeline {tl.name} has no wav {wav.name}")
        wavs.append(wav)
    return wavs


# ##################################################################
# assemble movie
# full step: storyboard scenes + scene stills + chapter audio → movie/movie.mp4
def assemble_movie(output_dir: Path, title: str) -> Path:
    storyboard_path = output_dir / "storyboard.json"
    if not storyboard_path.exists():
        raise ValueError("storyboard.json not found — run the storyboard step first")
    scenes = json.loads(storyboard_path.read_text(encoding="utf-8"))["scenes"]
    if not scenes:
        raise ValueError("storyboard has no scenes")

    movie_dir = output_dir / "movie"
    segments_dir = movie_dir / "segments"
    movie_dir.mkdir(parents=True, exist_ok=True)

    # Audio: the exact concatenation the storyboard timed against.
    audio_full = movie_dir / "audio_full.wav"
    if not audio_full.exists():
        concat_wavs(_chapter_wavs(output_dir), audio_full)
    audio_seconds = wav_duration(audio_full)
    total_frames = round(audio_seconds * FPS)

    # Frame budget per scene: round each boundary, never the durations.
    bounds = [round(float(s["start"]) * FPS) for s in scenes] + [total_frames]
    segments: list[Path] = []
    for i, scene in enumerate(scenes):
        frames = bounds[i + 1] - bounds[i]
        if frames < 1:
            frames = FPS  # degenerate zero-width scene — give it one second
        image = output_dir / "scenes" / f"{int(scene['index']):04d}.png"
        if not image.exists():
            raise ValueError(f"scene image missing: {image} — run the sceneimages step")
        move = MOVES[i % len(MOVES)]
        segments.append(render_segment(image, frames, move, segments_dir / f"{i:04d}.mp4"))
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(scenes)} segments rendered")

    # Concat segments (all identical codec/geometry → stream copy, no re-encode).
    video_only = movie_dir / "video_only.mp4"
    if not video_only.exists():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for seg in segments:
                f.write(f"file '{seg}'\n")
            list_file = Path(f.name)
        try:
            _run(
                [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-c", "copy", str(video_only),
                ]
            )
        finally:
            list_file.unlink(missing_ok=True)
        actual = probe_frames(video_only)
        if actual != total_frames:
            video_only.unlink(missing_ok=True)
            raise RuntimeError(f"concatenated video has {actual} frames, expected {total_frames}")

    # Mux with the narration track. Both sides are independently exact; the
    # container duration may differ by <1 frame of AAC priming, nothing drifts.
    movie_path = movie_dir / "movie.mp4"
    _run(
        [
            "ffmpeg", "-y", "-i", str(video_only), "-i", str(audio_full),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-metadata", f"title={title}",
            str(movie_path),
        ]
    )
    drift = abs(probe_duration(movie_path) - audio_seconds)
    if drift > 0.5:
        raise RuntimeError(f"movie/audio drift {drift:.3f}s exceeds 0.5s tolerance")
    print(f"  movie: {audio_seconds / 60:.1f} min, {total_frames} frames, drift {drift:.3f}s")
    return movie_path


__all__ = ["FPS", "assemble_movie", "probe_duration", "render_segment", "zoompan_filter"]
