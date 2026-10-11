# Scriptor attribution in production (Weakest Beast Tamer)

Switched 2026-10-11 on Darren's explicit authorization (conv-70a09ad1f7c73f5b). From now on, every not-yet-voiced
hourly script line comes from the fine-tuned scriptor model (`src/scriptor_attribution.py`). No audio was redone.

## Audio boundary

- Hours 1-44 saved (`hours.json`, audio-only). Hour 44 ends at chapter file `549-part_549` piece 1
  (cursor `next_chapter=548, next_piece=2`).
- The paused 2026-10-09 hour 45 had already cached classifier TTS for `549` pieces 2-63, `550` (62) and `551` (60).
  Those classifier scripts stay so that cached audio is reused, not regenerated.
- First line with no audio: `552-part_552` line 0 (0 wavs). The classifier script for 552 (and its meta and the
  hour-045 symlink) was moved to `output/weakest_beast_tamer/backup-scriptor-switch-20261011/` with a `MANIFEST.json`.
  Nothing was deleted. Chapter boundary, so `hour_runner` needed no mid-chapter switch.
- Hour 45 therefore voices 549 p2 to 551 from cached classifier audio, then 552 onward from scriptor scripts.

## Configuration (gitignored `local/config.toml`)

```toml
[scriptor]
enabled = true
model = "scriptor:v4"
primary_url = "http://10.0.0.42:11434"   # boringstack Ollama
primary_ping_host = "10.0.0.42"
backup_url = "http://127.0.0.1:11434"    # laptop Ollama
backup_ping_host = "127.0.0.1"
backup_identity = "Darrens-MacBook-Pro"
```

`primary_model` and `backup_model` are deliberately unset so both hosts inherit `model`. The laptop fallback
is therefore always the same scriptor, never the general `[llm]` backup (`qwen3:8b`), which earlier hours silently
fell back to whenever boringstack ICMP was down.

Scripting happens per chapter just before that chapter's audio (`script_for_chapter` inside `run_hour`), so each
hour picks up the model configured when it runs.

## Switching to a newer scriptor

1. Publish the new model under the same tag on **both** Ollama hosts (`ollama list | grep scriptor` on each).
2. Change only `model = "scriptor:vN"` in `local/config.toml`. Takes effect at the next chapter that has no script.
3. Already-scripted chapters keep their scripts (and audio). Never move a script whose chapter already has wavs
   in `audio_cache/.lines_<chapter>`.

## Provenance

Each new `script_cache/<chapter>.jsonl.hour.meta.json` records
`"attribution": {"engine": "scriptor", "model": "scriptor:v4"}` (or `{"engine": "span-classifier"}`).
Scripts written before 2026-10-11 have no `attribution` key. Hour logs (`local/audio-hour-N.log`) also show
`"model": "scriptor:v4"` on each attribution call.

## Operations

The audio-hours supervisor (`sh local/audio-hours-supervisor.sh N`) runs for hours (one hour of audio takes about
four hours). Always start it detached (`nohup ... &`) and check on it later. Never hold an agent turn waiting on it.
