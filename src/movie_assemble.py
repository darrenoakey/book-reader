"""Movie assembly: storyboard scenes + scene images + chapter audio → movie.mp4.

Every storyboard scene becomes one video segment with a slow Ken Burns
move (zoom in/out, pan left/right/up/down — deterministic rotation by scene
index) so the picture never sits still. Segments are frame-exact
(frames = round(duration * FPS)) and hard-cut together, so audio stays in
perfect sync with the line-level timeline the storyboard was built from;
crossfades are deliberately NOT used because each xfade consumes duration
from both sides and drifts the picture off the narration over a long book.

Layout:
  scenes/NNNN.png            from the sceneimages step
  audio/*.wav                chapter audio (announce/chime wavs excluded)
  movie/segments/NNNN.mp4    rendered Ken Burns segments (resumable cache)
  movie/movie.mp4            final film (h264 + aac)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.audio_synth import concat_wavs, wav_duration

FPS = 30
WIDTH = 1920
HEIGHT = 1080
UPSCALE_W = 3840  # zoompan input headroom — pans stay sharp, jitter subpixel
UPSCALE_H = 2160

# Ken Burns move kinds, cycled deterministically by scene index.
MOVES = ["zoom-in", "zoom-out", "pan-right", "pan-left", "pan-down", "pan-up"]


# ##################################################################
# zoompan filter
# build the ffmpeg zoompan expression for one segment; `on` is the output
# frame counter (0..frames-1). Zoom range is subtle (1.00-1.12) — a gentle
# drift, not a ride.
def zoompan_filter(move: str, frames: int) -> str:
    n = max(frames - 1, 1)
    cx = "iw/2-(iw/zoom/2)"
    cy = "ih/2-(ih/zoom/2)"
    if move == "zoom-in":
        z = f"1+0.12*on/{n}"
        x, y = cx, cy
    elif move == "zoom-out":
        z = f"1.12-0.12*on/{n}"
        x, y = cx, cy
    elif move == "pan-right":
        z = "1.12"
        x = f"(iw-iw/zoom)*on/{n}"
        y = cy
    elif move == "pan-left":
        z = "1.12"
        x = f"(iw-iw/zoom)*(1-on/{n})"
        y = cy
    elif move == "pan-down":
        z = "1.12"
        x = cx
        y = f"(ih-ih/zoom)*on/{n}"
    else:  # pan-up
        z = "1.12"
        x = cx
        y = f"(ih-ih/zoom)*(1-on/{n})"
    return (
        f"scale={UPSCALE_W}:{UPSCALE_H},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
        "format=yuv420p"
    )


# ##################################################################
# render segment
# one scene image → one silent mp4 segment of exactly `frames` frames
def render_segment(image: Path, frames: int, move: str, output: Path) -> Path:
    if output.exists() and output.stat().st_size >= 1000:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(image),
        "-vf", zoompan_filter(move, frames),
        "-frames:v", str(frames),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-an",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg segment failed for {image.name}: {result.stderr[-2000:]}")
    return output


# ##################################################################
# chapter audio wavs
# the ordered chapter wavs that back the storyboard's global timeline —
# announce/chime clips are NOT part of the spoken timeline
def chapter_audio_wavs(output_dir: Path) -> list[Path]:
    audio_dir = output_dir / "audio"
    timelines = sorted(audio_dir.glob("*.timeline.json"))
    wavs = []
    for tl in timelines:
        wav = audio_dir / f"{tl.stem.replace('.timeline', '')}.wav"
        if wav.exists():
            wavs.append(wav)
    if not wavs:
        raise ValueError("no chapter audio — run the audio step first")
    return wavs


# ##################################################################
# probe duration
# container duration in seconds via ffprobe
def probe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out.strip())


# ##################################################################
# assemble movie
# full step: render every scene segment, concat, lay the narration under it
def assemble_movie(output_dir: Path, title: str) -> Path:
    storyboard = json.loads((output_dir / "storyboard.json").read_text(encoding="utf-8"))
    scenes = storyboard["scenes"]
    scenes_dir = output_dir / "scenes"
    movie_dir = output_dir / "movie"
    segments_dir = movie_dir / "segments"
    movie_path = movie_dir / "movie.mp4"
    if movie_path.exists() and movie_path.stat().st_size >= 100000:
        return movie_path

    # Full narration track: exactly the wavs the storyboard timed against.
    audio_wav = movie_dir / "narration.wav"
    chapter_wavs = chapter_audio_wavs(output_dir)
    if not audio_wav.exists():
        audio_wav.parent.mkdir(parents=True, exist_ok=True)
        concat_wavs(chapter_wavs, audio_wav)
    audio_seconds = wav_duration(audio_wav)

    # Render segments frame-exact; the last scene is padded/truncated so the
    # video length lands exactly on the narration length.
    segments: list[Path] = []
    total_frames = round(audio_seconds * FPS)
    used_frames = 0
    for i, scene in enumerate(scenes):
        image = scenes_dir / f"{scene['index']:04d}.png"
        if not image.exists():
            raise ValueError(f"scene image missing: {image} — run the sceneimages step")
        if i == len(scenes) - 1:
            frames = total_frames - used_frames
        else:
            frames = max(round((scene["end"] - scene["start"]) * FPS), 1)
        frames = max(frames, FPS)  # never a sub-second segment
        used_frames += frames
        move = MOVES[scene["index"] % len(MOVES)]
        segments.append(render_segment(image, frames, move, segments_dir / f"{scene['index']:04d}.mp4"))
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(scenes)} segments rendered")

    # Concat the segments (identical codec params → stream copy) and mux audio.
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for seg in segments:
            f.write(f"file '{seg}'\n")
        list_file = Path(f.name)
    try:
        video_only = movie_dir / "video.mp4"
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(video_only)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr[-2000:]}")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_only),
            "-i", str(audio_wav),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-metadata", f"title={title}",
            str(movie_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed: {result.stderr[-2000:]}")
    finally:
        list_file.unlink(missing_ok=True)

    video_seconds = probe_duration(movie_path)
    drift = abs(video_seconds - audio_seconds)
    if drift > 0.5:
        raise RuntimeError(f"movie A/V drift {drift:.2f}s exceeds 0.5s (audio {audio_seconds:.2f}, video {video_seconds:.2f})")
    print(f"  movie: {video_seconds / 60:.1f} min, {len(scenes)} scenes, drift {drift:.3f}s")
    return movie_path
