![](banner.jpg)

```markdown
# Book Reader

Convert EPUB books into M4B audiobooks with distinct voices for each character using AI-powered voice synthesis.

## Overview

Book Reader is a command-line pipeline that takes an EPUB file and produces a fully chaptered M4B audiobook. Each character in the book is assigned a unique synthesised voice. The output includes chapter markers, a cover image, and chime announcements between chapters.

## Requirements

- Python 3.10+
- `ffmpeg` available on your PATH
- A running TTS (text-to-speech) service accessible via the `tts` CLI
- An Anthropic API key set in your environment (`ANTHROPIC_API_KEY`)

## Installation

Clone the repository and install dependencies into a local virtual environment:

```bash
git clone <repo-url>
cd book-reader
./run install
```

This creates a `.venv` directory and installs all Python dependencies automatically.

## Pipeline Steps

The conversion process runs as a sequence of steps:

| Step | Name | Description |
|------|------|-------------|
| 1 | `extract` | Extracts chapter text from the EPUB file |
| 2 | `characters` | Analyses each chapter to identify characters |
| 3 | `voices` | Generates voice descriptions for each character |
| 4 | `clone` | Clones TTS voices from descriptions |
| 5 | `scripts` | Converts chapters into speaker-attributed dialogue scripts |
| 6 | `audio` | Synthesises audio for each chapter |
| 7 | `m4b` | Assembles all audio into a final M4B file with chapter markers |

## Usage

### Convert an entire book

```bash
./run create "My Book.epub"
```

### Run a single pipeline step

```bash
./run step <step-name> "My Book.epub"
```

### Convert only the first N chapters

```bash
./run step audio "My Book.epub" --max-chapters 6
./run step m4b   "My Book.epub" --max-chapters 6
```

### Run tests

```bash
./run test src/
./run test src/epub_extract_test.py::test_normalize_name
```

### Lint the source

```bash
./run lint
```

### Run the full quality gate suite

```bash
./run check
```

## Examples

**Full conversion of a single EPUB:**

```bash
./run create "Scott Lynch - The Lies of Locke Lamora.epub"
```

Output is written to `output/Scott Lynch - The Lies of Locke Lamora/`.

**Previewing the first five chapters before committing to a full run:**

```bash
./run step audio "Scott Lynch - The Lies of Locke Lamora.epub" --max-chapters 5
./run step m4b   "Scott Lynch - The Lies of Locke Lamora.epub" --max-chapters 5
```

**Re-running the M4B assembly step after changing chapter count:**

Delete the existing `.m4b` file first, then re-run:

```bash
rm "output/Scott Lynch - The Lies of Locke Lamora/"*.m4b
./run step m4b "Scott Lynch - The Lies of Locke Lamora.epub"
```

**Resuming a partially completed pipeline:**

Each step tracks its own progress. Simply run the step you want to resume from — completed work is not repeated.

```bash
./run step audio "My Book.epub"
```

## Output Structure

```
output/
└── <epub-stem>/
    ├── state.jsonl          # Append-only progress log
    ├── characters.json      # Character profiles
    ├── voices.json          # TTS voice assignments
    ├── cover.jpeg           # Cover art extracted from EPUB
    ├── chapters/            # Extracted chapter text files
    ├── script/              # Speaker-attributed JSONL scripts
    ├── audio/               # Synthesised WAV files per chapter
    └── <stem>.m4b           # Final audiobook
```

## Notes

- The M4B assembly step is skipped if the output `.m4b` already exists. Delete it manually to force a rebuild.
- All pipeline steps are idempotent — re-running a completed step is safe.
- Progress is tracked in `state.jsonl`; individual steps check this before doing work.
```