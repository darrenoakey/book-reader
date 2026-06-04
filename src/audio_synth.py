import json
import subprocess
import tempfile
from pathlib import Path

from src import tts_engine

SAMPLE_RATE = 24000


# ##################################################################
# concat wavs
# concatenate per-line wavs into a single chapter wav at SAMPLE_RATE mono
def concat_wavs(line_paths: list[Path], output_path: Path) -> None:
    if not line_paths:
        raise ValueError("No line files to concatenate")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in line_paths:
            f.write(f"file '{p}'\n")
        list_file = Path(f.name)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")
    finally:
        list_file.unlink(missing_ok=True)


# ##################################################################
# synthesize chapter
# synthesize each script line via arbiter tts-clone with character ref WAVs
def split_long_text(text: str, max_words: int = 35) -> list[str]:
    """Split a long line into sentences each <= max_words. qwen3-tts has
    max_new_tokens=2048 and a 600s inference timeout. With 16 concurrent jobs,
    one slow job kills all 16, so we keep each call short (~35 words / ~200
    chars) to keep generation under ~30s."""
    if len(text.split()) <= max_words:
        return [text]
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for s in sentences:
        sw = len(s.split())
        if cur_words + sw > max_words and cur:
            chunks.append(" ".join(cur))
            cur = [s]
            cur_words = sw
        else:
            cur.append(s)
            cur_words += sw
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def synthesize_chapter(script_path: Path, audio_dir: Path,
                       voices_dir: Path, speaker_set: set[str]) -> Path:
    chapter_name = script_path.stem
    output_path = audio_dir / f"{chapter_name}.wav"
    if output_path.exists():
        return output_path
    work_dir = audio_dir / f".lines_{chapter_name}"
    work_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    line_paths: list[Path] = []
    sub_idx = 0
    with open(script_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            entry = json.loads(raw)
            speaker = list(entry.keys())[0]
            text = entry[speaker]
            if speaker not in speaker_set:
                if "narrator" not in speaker_set:
                    raise ValueError(f"speaker {speaker!r} not in voices and no narrator fallback")
                speaker = "narrator"
            for piece in split_long_text(text):
                line_path = work_dir / f"{sub_idx:05d}.wav"
                sub_idx += 1
                line_paths.append(line_path)
                if not line_path.exists():
                    jobs.append({
                        "speaker": speaker,
                        "text": piece,
                        "output_path": line_path,
                    })
    if jobs:
        tts_engine.synthesize_jobs(jobs, audio_dir.parent)
    concat_wavs(line_paths, output_path)
    return output_path


# ##################################################################
# plan chapter
# build the list of line paths and per-line jobs for one chapter; nothing
# submitted here, so the caller can group jobs across chapters by speaker
def plan_chapter(script_path: Path, audio_dir: Path, voices_dir: Path,
                 speaker_set: set[str]) -> tuple[Path, list[Path], list[dict]]:
    chapter_name = script_path.stem
    chapter_wav = audio_dir / f"{chapter_name}.wav"
    work_dir = audio_dir / f".lines_{chapter_name}"
    work_dir.mkdir(parents=True, exist_ok=True)
    line_paths: list[Path] = []
    jobs: list[dict] = []
    sub_idx = 0
    with open(script_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            entry = json.loads(raw)
            speaker = list(entry.keys())[0]
            text = entry[speaker]
            if speaker not in speaker_set:
                if "narrator" not in speaker_set:
                    raise ValueError(f"speaker {speaker!r} not in voices and no narrator fallback")
                speaker = "narrator"
            for piece in split_long_text(text):
                line_path = work_dir / f"{sub_idx:05d}.wav"
                sub_idx += 1
                line_paths.append(line_path)
                if not (line_path.exists() and line_path.stat().st_size >= 100):
                    jobs.append({
                        "speaker": speaker,
                        "text": piece,
                        "output_path": line_path,
                    })
    return chapter_wav, line_paths, jobs


# ##################################################################
# synthesize all chapters
# plan all jobs, group by speaker so each ref_wav is hot in the model the
# whole time it is processing that speaker's lines, run in those batches,
# then assemble each chapter wav in original order
def synthesize_all_chapters(output_dir: Path, max_chapters: int = 0) -> list[Path]:
    script_dir = output_dir / "script"
    audio_dir = output_dir / "audio"
    voices_dir = output_dir / "voices"
    if not script_dir.exists():
        raise ValueError("script directory not found")
    if not tts_engine.voices_ready(output_dir):
        raise ValueError("character voices not prepared — run the voices step first")
    speaker_set = tts_engine.speaker_set(output_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    script_files = sorted(script_dir.glob("*.jsonl"))
    if max_chapters > 0:
        script_files = script_files[:max_chapters]

    # Phase 1: plan every chapter without submitting anything.
    plans: list[tuple[Path, Path, list[Path]]] = []  # (script, chapter_wav, line_paths)
    all_jobs: list[dict] = []
    for script_path in script_files:
        chapter_wav, line_paths, jobs = plan_chapter(
            script_path, audio_dir, voices_dir, speaker_set,
        )
        plans.append((script_path, chapter_wav, line_paths))
        all_jobs.extend(jobs)

    # Phase 2: hand every pending line job to the selected TTS engine.
    if all_jobs:
        tts_engine.synthesize_jobs(all_jobs, output_dir)

    # Phase 3: concatenate each chapter from its (now complete) line wavs.
    created: list[Path] = []
    for script_path, chapter_wav, line_paths in plans:
        if not chapter_wav.exists():
            concat_wavs(line_paths, chapter_wav)
        created.append(chapter_wav)
    return created
