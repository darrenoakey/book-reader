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

## Long audiobook renders must be detached

A full audiobook render (extract -> scripts -> audio -> cover -> M4B) runs for hours and
MUST be started detached (`/tmp/abd/render.sh PROJECT EPUB LOG`, or
`./run audiobook "Title" --author A` which spawns a detached worker). Never hold an agent
turn watching it: start it, then `sleep` and re-inspect the project log on waking.
The scriptor host must have `scriptor:v5` published before step 5 or the run dies with
`permanent HTTP 404`; restore it by rsyncing the manifest plus its blobs from
`~/.ollama/models` (local offload store) to Boringstack's `~/.ollama/models`, then verify
`curl :11434/api/generate -d '{"model":"scriptor:v5",...}'`.

Measured breeze throughput (2026-11-24): Arbiter's `tts-breeze` model runs on ONE
`max_concurrent=1` slot and averages ~400s execution plus ~305s queue wait per
40-line job, i.e. ~10s per line, so a 3000-line book needs ~9h of GPU time plus
queueing. This is a server-side model limit: client batch size and window
changes buy nothing, and submitting extra concurrent jobs only lengthens the
queue. Budget renders in hours and sleep on them; never wait inside a turn.
