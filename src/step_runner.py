import json
import sys
from pathlib import Path

from colorama import Fore, Style, init

from src.audio_synth import synthesize_all_chapters
from src.character_analysis import analyze_characters_sync
from src.epub_extract import extract_epub, get_output_dir
from src.m4b_assemble import assemble_m4b
from src.script_generate import generate_scripts_sync
from src.state import mark_step_complete
from src.voice_description import generate_voices_sync

init(autoreset=True)


# ##################################################################
# get book info
# extract title and author from intro file
def get_book_info(output_dir: Path) -> tuple[str, str]:
    intro = (output_dir / "chapters" / "00-intro.txt").read_text()
    parts = intro.split(" by ")
    title = parts[0]
    author = parts[1].split(", narrated by")[0] if len(parts) > 1 else "Unknown"
    return title, author


# ##################################################################
# run step
# execute a single pipeline step
def run_step(step: str, epub_path: Path, max_chapters: int = 0) -> int:
    output_dir = get_output_dir(epub_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if step == "extract":
        title, author, written = extract_epub(epub_path, output_dir)
        print(f"Extracted {len(written)} files")
        mark_step_complete(output_dir, "extract")

    elif step == "characters":
        title, author = get_book_info(output_dir)
        analyze_characters_sync(output_dir, title, author)
        chars = json.loads((output_dir / "characters.json").read_text())
        print(f"Found {len(chars)} characters")
        for char_id, info in chars.items():
            print(f"  {char_id}: {info['name']}")
        mark_step_complete(output_dir, "characters")

    elif step == "voices":
        generate_voices_sync(output_dir)
        voices = json.loads((output_dir / "voices.json").read_text())
        print(f"\n{Fore.CYAN}Generated {len(voices)} voice descriptions:{Style.RESET_ALL}\n")
        for char_id, info in voices.items():
            print(f"{Fore.GREEN}{char_id}{Style.RESET_ALL}")
            print(f"  {info['description']}\n")
        mark_step_complete(output_dir, "voices_desc")

    elif step == "scripts":
        scripts = generate_scripts_sync(output_dir)
        print(f"Generated {len(scripts)} script files")
        mark_step_complete(output_dir, "scripts")

    elif step == "audio":
        audio = synthesize_all_chapters(output_dir, max_chapters=max_chapters)
        print(f"Synthesized {len(audio)} audio segments")
        if max_chapters == 0:
            mark_step_complete(output_dir, "audio")

    elif step == "m4b":
        title, author = get_book_info(output_dir)
        m4b = assemble_m4b(output_dir, title, author, max_chapters=max_chapters)
        print(f"Created {m4b}")
        if max_chapters == 0:
            mark_step_complete(output_dir, "m4b")

    else:
        print(f"Unknown step: {step}")
        print("Valid steps: extract, characters, voices, scripts, audio, m4b")
        return 1

    return 0


# ##################################################################
# main
# entry point for step runner
def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="step_runner")
    parser.add_argument("step")
    parser.add_argument("epub_path")
    parser.add_argument("--max-chapters", type=int, default=0, help="Limit to first N chapters")
    args = parser.parse_args()
    return run_step(args.step, Path(args.epub_path), max_chapters=args.max_chapters)


if __name__ == "__main__":
    sys.exit(main())
