#!/usr/bin/env python3
"""Jeff Book — a dedicated novelty-audiobook generator.

Generates an EPUB that is nothing but sentences made of the word "Jeff" with
wildly varied capitalisation, punctuation and length, then reads the whole
thing in ONE kokoro voice and assembles an M4B.

This deliberately bypasses the normal book-reader pipeline (no character
analysis, no voice mapping, no script generation, no LLM at all) — it is a
single voice reading pure narration. It reuses the project's kokoro batch
synthesis (src.arbiter_tts.tts_kokoro_many) and the ffmpeg helpers in
src.m4b_assemble.

Usage:
    ./run jeff --pages 200
    python jeff_book.py --pages 200 --voice am_michael --speed 1.0
"""
from __future__ import annotations

import argparse
import random
import subprocess
import uuid
from pathlib import Path

from ebooklib import epub

from src import m4b_assemble as m4b
from src.arbiter_tts import tts_kokoro_many

WORDS_PER_PAGE = 300
SENTENCES_PER_PARA = (3, 8)
PARAS_PER_CHAPTER = (8, 16)

# Terminal punctuation, weighted (period most common, plenty of variety).
_ENDINGS = ([".", 30], ["!", 14], ["?", 12], ["...", 8], ["?!", 5],
            ["!!", 4], ["—", 3], ["!?", 3], [".", 20], ["?!?", 2], ["…", 4])
# Word forms for "Jeff".
_FORMS = (["Jeff", 70], ["jeff", 18], ["JEFF", 6], ["Jeff", 6])


def _weighted(rng: random.Random, pairs) -> str:
    items = [p[0] for p in pairs]
    weights = [p[1] for p in pairs]
    return rng.choices(items, weights=weights)[0]


# ##################################################################
# make sentence
# build one "Jeff" sentence with varied length, casing and punctuation
def make_sentence(rng: random.Random) -> str:
    n = rng.choices([1, 2, 3, 4, 5, 6, 7, 8, 10, 12],
                    weights=[4, 9, 11, 11, 9, 6, 4, 3, 2, 1])[0]
    words = [_weighted(rng, _FORMS) for _ in range(n)]
    # Capitalise the first word so it reads like a sentence start.
    if words[0] in ("jeff",):
        words[0] = "Jeff"
    parts: list[str] = []
    for i, w in enumerate(words):
        last = i == len(words) - 1
        parts.append(w)
        if not last:
            r = rng.random()
            if r < 0.14:
                parts[-1] += ","          # comma beat
            elif r < 0.20:
                parts[-1] += " —"         # em-dash aside
            elif r < 0.24:
                parts[-1] += "..."        # trailing ellipsis mid-sentence
    body = " ".join(parts)
    return body + _weighted(rng, _ENDINGS)


# ##################################################################
# generate book
# produce chapters → list of (title, [paragraph strings])
def generate_book(pages: int, seed: int) -> list[tuple[str, list[str]]]:
    rng = random.Random(seed)
    target_words = pages * WORDS_PER_PAGE
    chapters: list[tuple[str, list[str]]] = []
    words_so_far = 0
    chapter_idx = 0
    # ~20 pages/chapter.
    words_per_chapter = max(WORDS_PER_PAGE, target_words // max(1, pages // 20 or 1))
    while words_so_far < target_words:
        chapter_idx += 1
        paras: list[str] = []
        chapter_words = 0
        n_paras = rng.randint(*PARAS_PER_CHAPTER)
        for _ in range(n_paras):
            if words_so_far >= target_words:
                break
            sentences = [make_sentence(rng)
                         for _ in range(rng.randint(*SENTENCES_PER_PARA))]
            para = " ".join(sentences)
            paras.append(para)
            w = sum(len(s.split()) for s in sentences)
            chapter_words += w
            words_so_far += w
            if chapter_words >= words_per_chapter:
                break
        if paras:
            chapters.append((f"Chapter {chapter_idx}", paras))
    return chapters


# ##################################################################
# write epub
# assemble the chapters into an EPUB (mirrors noveliser's epub_generator)
def write_epub(title: str, author: str, chapters: list[tuple[str, list[str]]],
               out_path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier(str(uuid.uuid4()))
    book.set_title(title)
    book.set_language("en")
    book.add_author(author)

    css = ("body{font-family:Georgia,serif;line-height:1.6;margin:2em}"
           "h1{text-align:center;margin:2em 0 1em;font-size:1.8em}"
           "p{text-indent:1.5em;margin:.3em 0}p:first-of-type{text-indent:0}")
    style = epub.EpubItem(uid="style", file_name="style/default.css",
                          media_type="text/css", content=css.encode())
    book.add_item(style)

    spine = ["nav"]
    toc = []
    for i, (ch_title, paras) in enumerate(chapters, start=1):
        html = f"<h1>{ch_title}</h1>\n"
        for para in paras:
            esc = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html += f"<p>{esc}</p>\n"
        item = epub.EpubHtml(title=ch_title, file_name=f"chapter_{i}.xhtml", lang="en")
        item.content = html
        item.add_item(style)
        book.add_item(item)
        spine.append(item)
        toc.append(item)

    book.toc = toc
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(out_path), book, {})
    return out_path


# ##################################################################
# split sentences
# split a paragraph back into individual sentences for per-line synthesis
def split_sentences(text: str) -> list[str]:
    import re
    # Keep terminal punctuation with the sentence; split on whitespace after
    # one or more end marks. Treat em-dash sentences (end "—") too.
    parts = re.split(r'(?<=[.!?…—])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


# ##################################################################
# synthesize chapter
# render every sentence of a chapter in one kokoro voice → chapter wav
def synthesize_chapter(paras: list[str], voice: str, speed: float,
                       work_dir: Path, chapter_wav: Path) -> Path:
    if chapter_wav.exists() and chapter_wav.stat().st_size >= 100:
        return chapter_wav
    work_dir.mkdir(parents=True, exist_ok=True)
    sentences: list[str] = []
    for para in paras:
        sentences.extend(split_sentences(para))
    jobs = []
    line_paths = []
    for idx, sent in enumerate(sentences):
        p = work_dir / f"{idx:05d}.wav"
        line_paths.append(p)
        jobs.append({"text": sent, "voice": voice, "speed": speed, "output_path": p})
    tts_kokoro_many(jobs)
    m4b.concatenate_audio_files(line_paths, chapter_wav)
    return chapter_wav


# ##################################################################
# assemble m4b
# concat chapter wavs with short gaps + chapter markers → m4b
def assemble_m4b(chapter_wavs: list[tuple[str, Path]], title: str, author: str,
                 out_dir: Path) -> Path:
    import tempfile
    m4b_path = out_dir / f"{title}.m4b"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        gap = work / "gap.wav"
        m4b.generate_silence(1.0, gap)
        interleaved: list[Path] = []
        chapter_info = []
        t = 0.0
        for ch_title, wav in chapter_wavs:
            start = t
            dur = m4b.get_audio_duration(wav)
            interleaved.append(wav)
            t += dur
            chapter_info.append({"title": ch_title, "start": start, "end": t})
            interleaved.append(gap)
            t += 1.0
        full = work / "full.wav"
        m4b.concatenate_audio_files(interleaved, full)
        meta = work / "meta.txt"
        m4b.write_chapter_metadata(chapter_info, meta)
        cmd = ["ffmpeg", "-y", "-i", str(full), "-i", str(meta),
               "-map_metadata", "1",
               "-metadata", f"title={title}", "-metadata", f"artist={author}",
               "-metadata", f"album={title}",
               "-c:a", "aac", "-b:a", "64k", str(m4b_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg m4b failed: {r.stderr}")
    return m4b_path


# ##################################################################
# main
def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a Jeff audiobook (epub + m4b)")
    ap.add_argument("--pages", type=int, default=200)
    ap.add_argument("--voice", default="am_michael")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--title", default="Infinite Jeff")
    ap.add_argument("--author", default="Jeff")
    ap.add_argument("--out", default="output/Jeff Book")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()  # absolute — ffmpeg concat needs it
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    print(f"[1/4] Generating ~{args.pages} pages of Jeff (seed={args.seed})...")
    chapters = generate_book(args.pages, args.seed)
    total_sentences = sum(len(split_sentences(" ".join(p))) for _, p in chapters)
    total_words = sum(len(" ".join(p).split()) for _, p in chapters)
    print(f"      {len(chapters)} chapters, {total_sentences} sentences, "
          f"~{total_words} words (~{total_words // WORDS_PER_PAGE} pages)")

    print("[2/4] Writing EPUB...")
    epub_path = write_epub(args.title, args.author, chapters, out_dir / f"{args.title}.epub")
    print(f"      {epub_path}")

    print(f"[3/4] Synthesising with kokoro voice '{args.voice}' (speed {args.speed})...")
    chapter_wavs: list[tuple[str, Path]] = []
    for i, (ch_title, paras) in enumerate(chapters, start=1):
        cw = audio_dir / f"{i:03d}.wav"
        work = audio_dir / f".lines_{i:03d}"
        print(f"      {ch_title} ({i}/{len(chapters)})...")
        synthesize_chapter(paras, args.voice, args.speed, work, cw)
        chapter_wavs.append((ch_title, cw))

    print("[4/4] Assembling M4B...")
    m4b_path = assemble_m4b(chapter_wavs, args.title, args.author, out_dir)
    dur = m4b.get_audio_duration(m4b_path)
    print(f"\nDone: {m4b_path}")
    print(f"  {dur / 60:.1f} min, {len(chapters)} chapters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
