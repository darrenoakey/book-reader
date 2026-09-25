# Book Reader

EPUB (or .md/.txt story) to **audiobook movie**: multi-character M4B audiobook
plus a Ken Burns video of scene illustrations. Narrator narrates; every
character speaks their own dialogue in a voice designed for them.

## Pipeline Steps

1. `extract` - EPUB to chapter text files (or text_ingest chunks .md/.txt at
   paragraph boundaries, ~900 words/chapter)
2. `characters` - per-chapter LLM analysis → characters.json
3. `voices_desc` - single-call LLM → voices.json (voice descriptions)
4. `voices` - prepare per-character voices for the selected TTS engine
   (breeze: voice-design a reference clip per character → voices/*.wav +
   breeze_voices.json; kokoro: LLM maps each description → kokoro voice →
   kokoro_voices.json; qwen: tts-design a reference WAV per character)
5. `scripts` - LLM → speaker-attributed JSONL
6. `audio` - selected TTS engine per line, ffmpeg concat → chapter WAVs +
   per-line `.timeline.json` (the movie storyboard aligns to these)
7. `m4b` - ffmpeg assembly → M4B with chapters
8. `storyboard` - ~30s scenes snapped to line boundaries + LLM image prompts →
   storyboard.json (style, appearances, scenes)
9. `refimages` - qwen-image identity portrait per visual character → refs/
10. `sceneimages` - qwen-image 1920x1088 still per scene → scenes/, conditioned
    on the scene's character portraits (contact sheet when several)
11. `movie` - Ken Burns pan/zoom segments (frame-exact vs audio) →
    movie/movie.mp4

## Inspect UI

`./run serve` / `./run deploy` (auto service `book-reader-inspect`, port 8769):
projects, step progress, character portraits + voice clips, storyboard
filmstrip, audiobook + movie playback. Stdlib HTTP server, hashed static
assets, Range-aware /media for AV seeking (src/server.py + static/).

## LLM backend (all LLM work)

All LLM calls go through `src/llm.py` → **qwen3.6:35b-a3b on the boringstack
machine** (Ollama HTTP at `http://10.0.0.42:11434` — DHCP-assigned, was .237; check asus dnsmasq leases if unreachable, `think=false`, stdlib
urllib only — no local model process, no per-call subprocess). Configurable via
`BOOK_LLM_HOST`, `BOOK_LLM_MODEL`, `BOOK_LLM_CONCURRENCY`. The old
daz-agent-sdk/Claude path (which spawned a ~400 MB Node CLI per call and
exhausted local memory) is gone.

`src/llm.py` semaphore is keyed **per running event loop** — the pipeline runs
each step under its own `asyncio.run()`, and an asyncio.Semaphore is bound to
the loop it was created in; a module-level singleton raises "bound to a
different event loop" on the second step.

## TTS engines (pluggable)

`src/tts_engine.py` dispatches on env `BOOK_TTS_ENGINE` (default `breeze`):

- **breeze** — Breeze TTS 2 via arbiter `tts-breeze`. Voices: voice-DESIGN one
  short reference clip per character (instruction = voices.json description,
  cfg_scale 4) saved to voices/<id>.wav + breeze_voices.json. Audio: every
  line CLONES that clip (ref_audio_file + exact ref_text), so timbre is stable
  across the whole book. Lines are batched 40/job (`items`) and sliced back
  apart via `item_samples` — Breeze is near-realtime in eager mode, so a batch
  job takes minutes, not the scheduler-overhead-dominated one-job-per-line.
- **kokoro** — Kokoro-82M via arbiter `tts-kokoro` (new adapter on spark).
  Hundreds of times faster than qwen3-tts. No cloning: each character is mapped
  by the LLM to a kokoro voice (gender/accent match, optional 2-voice blend,
  speed) in `src/kokoro_voices.py` → `kokoro_voices.json`. A voice spec is
  `"af_heart"` or a blend `"af_heart*0.6+am_michael*0.4"`.
- **qwen** — original path: `tts-design` a reference WAV per character, then
  `tts-clone` per line (uses ref WAVs in `voices/`).

The "perfect world" goal (qwen3-tts creates voices → kokoro clones them) needs
**kvoicewalk** to evolve a real kokoro `.pt` style tensor per voice (~30-90 min
GPU each, one-time) — documented in `arbiter/KOKORO_TTS.md` as a future hybrid
(grow the bank from 54 → ~500 voices, then map onto that). Not yet wired in.

Arbiter is reached **directly at spark** (`http://10.0.0.254:8400`, override
with `ARBITER_BASE`); kokoro jobs need no ref staging so the local
arbiter-proxy/client services (8399/8401) are not required.

## Key Files

- `output/<epub-stem>/state.jsonl` - Append-only progress tracking
- `output/<epub-stem>/characters.json` - Character bios (physical/voice focus)
- `output/<epub-stem>/voices.json` - TTS voice descriptions
- `output/<epub-stem>/breeze_voices.json` - per-character breeze ref clip+text
- `output/<epub-stem>/kokoro_voices.json` - per-character kokoro voice + speed
- `output/<epub-stem>/script/*.jsonl` - Speaker-attributed lines
- `output/<epub-stem>/audio/*.timeline.json` - per-line start/end seconds
- `output/<epub-stem>/storyboard.json` - ~30s scenes with image prompts
- `output/<epub-stem>/refs/*.png` - character identity portraits (qwen-image)
- `output/<epub-stem>/scenes/*.png` - 1920x1088 scene stills (qwen-image)
- `output/<epub-stem>/movie/movie.mp4` - the audiobook movie

## Gotchas

### Output folder naming
Uses exact EPUB stem with spaces: `Scott Lynch - Book Title/` not normalized.

### Character analysis prompt
Must request PHYSICAL descriptions for voice generation. "Crime boss" has no value. "Elderly man with gravelly voice" has value. Exclude plot roles and cross-character relationships.

CRITICAL: nationality/region and a specific age MUST be inferred from setting cues (era, vocabulary, place names, period detail). Past failure: Paul Jonas in Otherland is clearly British (WW1, Webley gun, "fraulein", trench dialogue) but the bio just said "Young man, soldier" — wrong on nationality and age. War setting alone does not imply young. Use ALL setting evidence to nail down accent and age.

### Script generation prompt
Must be extremely terse: "Output JSONL only. No explanations." Otherwise Haiku explains what it would do instead of outputting JSONL.

### Speaker ID matching
Scripts must use exact IDs from voices.json. Pass full speaker list to prompt.

### Sonnet for deduplication
Single call, not retries. Character dedup and voice generation both use Sonnet in one call each.

### Tests are real
No mocks. Tests call actual Claude APIs and TTS tools. Audio synthesis test may hang if TTS unavailable.

### Haiku JSONL format variance
Haiku sometimes outputs `{"speaker_id": "name", "text": "..."}` instead of `{"name": "text"}`. The `parse_jsonl_response` function normalizes both formats.

### TTS via arbiter (qwen3-tts)
All voice synthesis goes through arbiter `tts-design` at `http://10.0.0.254:8400`:
- `audio_synth.py` calls `tts-design` per line with the speaker's `voices.json` description as `instruct`. Lines are submitted in parallel per chapter, then concatenated via ffmpeg.
- `m4b_assemble.py` chapter announcements use `tts-design` with the narrator description.
- No reference WAVs — qwen3-tts generates the voice directly from description each call. There is no separate "voices_clone" step.
- Helpers live in `src/arbiter_tts.py`. Uses the `arbiter_client` package directly (not daz-agent-sdk).

### Arbiter force=True required
All TTS submissions pass `force=True`. Arbiter dedup returns cached jobs whose `result_path` points to `/home/darren/src/arbiter/local_output/jobs/<id>/result.wav` — a spark-local path that arbiter-client can't resolve once the job dir is gone. force=True bypasses dedup and re-runs the job, returning fresh `data` (base64) inline.

### Arbiter refs are broken for tts-clone
`POST /v1/refs?filename=` succeeds and stores under `local_output/refs/`, but the adapter resolves `ref:<id>` to `output/refs/<id>` (relative to cwd) which doesn't exist. Don't use refs; pass base64 instead.

### get_result_bytes over copy_result
Use `client.get_result_bytes(job_id)` not `copy_result()` — it checks the inline `data` field first (works even when `result_path` is unresolvable), only falling back to mount/scp lookup.

### M4B idempotency caveat
`assemble_m4b` skips if the .m4b file exists. When re-running with different `--max-chapters`, delete the existing .m4b first. Also delete `chime.wav` if changing the chime, since it's cached.

### Cover art extraction
`extract_epub` extracts cover image from EPUB to `output/<stem>/cover.jpeg`. Tries ITEM_COVER, `get_item_with_id("cover")`, `get_item_with_id("cover-image")`, then ITEM_IMAGE filename scan. M4B assembly embeds it as `attached_pic` if present.

### Front/back matter detection
Uses Claude Haiku (via claude-agent-sdk) to classify EPUB sections as content vs non-content. Scans from front until first real content, then from back until last real content. Everything in between is kept. Replaces keyword-based filtering.

### Chapter announcements
M4B assembly generates a 3-note chime + narrator TTS announcement ("{title}. {chapter name}.") before each chapter except intro. Announcement WAVs are cached in `audio/` as `XX-name.announce.wav`.

### Sample rate consistency (CRITICAL)
All audio must be 24000 Hz mono to match TTS output. The chime, silence, and concat steps all use `SAMPLE_RATE = 24000`. Mismatched sample rates cause silent/garbled audio when concatenated.

### Duplicate chapter titles
M4B assembly auto-numbers duplicate titles: "Interlude" → "Interlude 1", "Interlude 2". Single-occurrence titles unchanged.

## Commands

```bash
./run create book.epub                       # Full pipeline (EPUB or .md/.txt)
./run step <step> book.epub                  # Single step (extract…movie)
./run step audio book.epub --max-chapters 6  # First 6 chapters only
./run serve                                  # Inspect UI (port 8769)
./run deploy                                 # Register UI with auto
./run health                                 # Probe the live UI
./run test <target>                          # Run tests
```

## Movie assembly gotchas

- **Frame-exact A/V**: total video frames = round(audio_seconds * 30); the last
  scene absorbs the rounding remainder. Never use `-shortest` (it drops video
  frames); the video track is authoritative.
- **Scene offsets** are global seconds across the concatenated chapter WAVs in
  sorted-timeline order — the movie audio must be exactly that concatenation
  (no chimes/announcements; those exist only in the M4B).
- **zoompan**: stills are upscaled to 2880x1620 first so pans have travel room;
  expressions use `on/(frames-1)` so moves hit their endpoint exactly.
- **qwen-image result is file-only** (`{"file": "result.png"}`) — resolve via
  `local_result("/mnt/arbiter-store/output/jobs/<id>/result.png")`; inline
  `data`/`result_path` shapes are also handled (movie_images._fetch_image).
- **Contact sheets**: qwen-image takes ONE condition image, so multi-character
  scenes pass a labelled portrait strip (movie_images.build_contact_sheet).
