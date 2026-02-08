# Book Reader

EPUB to M4B audiobook converter with multi-character voice synthesis.

## Pipeline Steps

1. `extract` - EPUB to chapter text files
2. `characters` - Claude Haiku per-chapter analysis → characters.json
3. `voices_desc` - Claude Sonnet single-call → voices.json
4. `voices_clone` - TTS voice cloning → voice zips
5. `scripts` - Claude Haiku → speaker-attributed JSONL
6. `audio` - TTS synthesis → WAV files
7. `m4b` - ffmpeg assembly → M4B with chapters

## Key Files

- `output/<epub-stem>/state.jsonl` - Append-only progress tracking
- `output/<epub-stem>/characters.json` - Character bios (physical/voice focus)
- `output/<epub-stem>/voices.json` - TTS voice descriptions
- `output/<epub-stem>/script/*.jsonl` - Speaker-attributed lines

## Gotchas

### Output folder naming
Uses exact EPUB stem with spaces: `Scott Lynch - Book Title/` not normalized.

### Character analysis prompt
Must request PHYSICAL descriptions for voice generation. "Crime boss" has no value. "Elderly man with gravelly voice" has value. Exclude plot roles and cross-character relationships.

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

### TTS multi command
Audio synthesis uses `tts multi` per chapter (one subprocess call per chapter, not per line). The multi command handles per-line synthesis and concatenation internally via a work directory.

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
./run create book.epub                       # Full pipeline
./run step <step> book.epub                  # Single step
./run step audio book.epub --max-chapters 6  # First 6 chapters only
./run step m4b book.epub --max-chapters 6    # M4B from first 6 chapters
./run test <target>                          # Run tests
```
