import json
import os
import subprocess
import tempfile
from pathlib import Path

from src import tts_engine
from src.data_recovery import (
    DataIssue,
    OperationalError,
    RecoveryLedger,
    bounded,
    is_data_error,
)

SAMPLE_RATE = 24000


# ##################################################################
# concat wavs
# concatenate per-line wavs into a single chapter wav at SAMPLE_RATE mono
def concat_wavs(line_paths: list[Path], output_path: Path) -> None:
    if not line_paths:
        raise ValueError("No line files to concatenate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in line_paths:
            f.write(f"file '{p}'\n")
        list_file = Path(f.name)
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")
    finally:
        list_file.unlink(missing_ok=True)


# ##################################################################
# wav duration
# duration in seconds of a PCM wav file
def wav_duration(path: Path) -> float:
    import wave

    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ##################################################################
# write timeline
# per-chapter manifest mapping every spoken line to its absolute start/end
# second within the concatenated chapter wav — the movie storyboard aligns
# scene boundaries to these line times.
def write_timeline(chapter_wav: Path, line_meta: list[dict]) -> Path:
    timeline_path = chapter_wav.with_suffix(".timeline.json")
    entries = []
    cursor = 0.0
    for meta in line_meta:
        dur = wav_duration(meta["path"])
        entries.append(
            {
                "index": meta["index"],
                "speaker": meta["speaker"],
                "text": meta["text"],
                "start": round(cursor, 4),
                "end": round(cursor + dur, 4),
            }
        )
        cursor += dur
    timeline_path.write_text(json.dumps({"chapter": chapter_wav.stem, "lines": entries}, indent=2), encoding="utf-8")
    return timeline_path


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

    sentences = re.split(r"(?<=[.!?])\s+", text)
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


def synthesize_chapter(
    script_path: Path, audio_dir: Path, voices_dir: Path, speaker_set: set[str], ledger: RecoveryLedger | None = None
) -> Path:
    chapter_wav = audio_dir / f"{script_path.stem}.wav"
    # Idempotent: a finished chapter wav is authoritative — never resynthesize
    # or overwrite it. Its timeline was written when it was created.
    if chapter_wav.exists():
        return chapter_wav
    chapter_wav, line_paths, jobs, line_meta = plan_chapter(script_path, audio_dir, voices_dir, speaker_set, ledger)
    if jobs:
        tts_engine.synthesize_jobs(jobs, audio_dir.parent)
    concat_wavs(line_paths, chapter_wav)
    write_timeline(chapter_wav, line_meta)
    return chapter_wav


# ##################################################################
# plan chapter
# build the list of line paths and per-line jobs for one chapter; nothing
# submitted here, so the caller can group jobs across chapters by speaker
def plan_chapter(
    script_path: Path, audio_dir: Path, voices_dir: Path, speaker_set: set[str], ledger: RecoveryLedger | None = None
) -> tuple[Path, list[Path], list[dict], list[dict]]:
    chapter_name = script_path.stem
    chapter_wav = audio_dir / f"{chapter_name}.wav"
    frozen_meta_path = script_path.with_name(script_path.name + ".hour.meta.json")
    try:
        frozen_meta = json.loads(frozen_meta_path.read_text(encoding="utf-8")) if frozen_meta_path.is_file() else {}
    except (OSError, ValueError) as error:
        raise OperationalError(
            "frozen_script_meta_unreadable", f"frozen script meta is unreadable: {frozen_meta_path}"
        ) from error
    remapped_lines = set(frozen_meta.get("remapped_line_indexes", []))
    is_remapped_frozen = bool(remapped_lines and frozen_meta.get("legacy_script"))
    work_dir = audio_dir / (f".lines_{chapter_name}.frozen" if is_remapped_frozen else f".lines_{chapter_name}")
    legacy_work_dir = audio_dir / f".lines_{chapter_name}"
    work_dir.mkdir(parents=True, exist_ok=True)
    line_paths: list[Path] = []
    line_meta: list[dict] = []
    jobs: list[dict] = []
    sub_idx = 0
    raw_index = 0
    ledger = ledger or RecoveryLedger(audio_dir.parent)

    def quarantine(line_number: int, code: str, message: str, raw: str) -> None:
        ledger.record(
            "audio",
            f"{script_path.name}:{line_number}",
            code,
            message,
            severity="quarantine",
            evidence={"script": script_path.name, "line": line_number, "raw": bounded(raw)},
            checkpoint={"chapter": chapter_name, "valid_lines_before": raw_index},
        )

    with open(script_path, "r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except ValueError:
                quarantine(line_number, "script_line_unparseable", "script line is not valid JSON; skipped", raw)
                continue
            if not isinstance(entry, dict) or len(entry) != 1:
                quarantine(
                    line_number,
                    "script_line_malformed",
                    "script line is not a single speaker->text object; skipped",
                    raw,
                )
                continue
            speaker = next(iter(entry.keys()))
            text = entry[speaker]
            if not isinstance(text, str) or not text.strip():
                quarantine(
                    line_number, "script_line_text_unusable", "script line text is empty or not text; skipped", raw
                )
                continue
            if speaker not in speaker_set:
                # Pending: never rewritten to narrator (that would silently skip an actor); other lines still synthesize.
                quarantine(
                    line_number,
                    "script_speaker_unknown",
                    f"speaker {speaker!r} not in voices; line left pending, not voiced",
                    raw,
                )
                continue
            for piece in split_long_text(text):
                line_path = work_dir / f"{sub_idx:05d}.wav"
                legacy_path = legacy_work_dir / f"{sub_idx:05d}.wav"
                if (
                    is_remapped_frozen
                    and raw_index not in remapped_lines
                    and legacy_path.is_file()
                    and not line_path.exists()
                ):
                    os.link(legacy_path, line_path)
                line_paths.append(line_path)
                line_meta.append({"index": sub_idx, "speaker": speaker, "text": piece, "path": line_path})
                sub_idx += 1
                if not (line_path.exists() and line_path.stat().st_size >= 100):
                    jobs.append(
                        {
                            "speaker": speaker,
                            "text": piece,
                            "output_path": line_path,
                        }
                    )
            raw_index += 1
    if not line_paths:
        raise DataIssue(
            "script_no_usable_lines", f"script has no usable lines: {script_path.name}", {"script": script_path.name}
        )
    return chapter_wav, line_paths, jobs, line_meta


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
        raise OperationalError("script_dir_missing", "script directory not found")
    if not tts_engine.voices_ready(output_dir):
        raise OperationalError("voices_not_ready", "character voices not prepared — run the voices step first")
    speaker_set = tts_engine.speaker_set(output_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    ledger = RecoveryLedger(output_dir)
    script_files = sorted(script_dir.glob("*.jsonl"))
    if max_chapters > 0:
        script_files = script_files[:max_chapters]

    # Phase 1: plan every chapter without submitting anything. Chapters whose
    # wav AND timeline already exist are fully done — skip them entirely.
    plans: list[tuple[Path, Path, list[Path], list[dict]]] = []  # (script, chapter_wav, line_paths, line_meta)
    all_jobs: list[dict] = []
    created: list[Path] = []
    for script_path in script_files:
        chapter_wav = audio_dir / f"{script_path.stem}.wav"
        if chapter_wav.exists() and chapter_wav.with_suffix(".timeline.json").exists():
            created.append(chapter_wav)
            continue
        try:
            chapter_wav, line_paths, jobs, line_meta = plan_chapter(
                script_path, audio_dir, voices_dir, speaker_set, ledger
            )
        except Exception as error:
            if not is_data_error(error):
                raise
            ledger.record_error("audio", script_path.name, error, checkpoint={"chapter": script_path.stem})
            continue
        plans.append((script_path, chapter_wav, line_paths, line_meta))
        all_jobs.extend(jobs)

    # Phase 2: hand every pending line job to the selected TTS engine.
    if all_jobs:
        tts_engine.synthesize_jobs(all_jobs, output_dir)

    # Phase 3: concatenate each chapter from its (now complete) line wavs and
    # write the per-line timing manifest the movie storyboard aligns to.
    for script_path, chapter_wav, line_paths, line_meta in plans:
        if not chapter_wav.exists():
            concat_wavs(line_paths, chapter_wav)
        if not chapter_wav.with_suffix(".timeline.json").exists():
            write_timeline(chapter_wav, line_meta)
        created.append(chapter_wav)
    return created
