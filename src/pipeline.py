from pathlib import Path

from colorama import Fore, Style, init

from src.audio_synth import synthesize_all_chapters
from src.character_analysis import analyze_characters_sync
from src.epub_extract import extract_epub, get_output_dir
from src.m4b_assemble import assemble_m4b
from src.script_generate import generate_scripts_sync
from src.state import is_step_complete, mark_step_complete
from src.voice_description import generate_voices_sync

init(autoreset=True)


# ##################################################################
# print step
# print a step header
def print_step(step_num: int, name: str) -> None:
    print(f"\n{Fore.BLUE}[Step {step_num}]{Style.RESET_ALL} {Fore.WHITE}{name}{Style.RESET_ALL}")


# ##################################################################
# print done
# print step completion
def print_done(message: str) -> None:
    print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {message}")


# ##################################################################
# print skip
# print step skip message
def print_skip(message: str) -> None:
    print(f"  {Fore.YELLOW}→{Style.RESET_ALL} {message}")


# ##################################################################
# run pipeline
# execute the full book-reader pipeline
def run_pipeline(epub_path: Path) -> Path:
    output_dir = get_output_dir(epub_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{Fore.CYAN}Book Reader Pipeline{Style.RESET_ALL}")
    print(f"  Input: {epub_path}")
    print(f"  Output: {output_dir}")
    print_step(1, "Extract chapters from EPUB")
    if is_step_complete(output_dir, "extract"):
        print_skip("Already extracted")
        title_line = (output_dir / "chapters" / "00-intro.txt").read_text().split(" by ")
        title = title_line[0]
        author = title_line[1].split(", narrated by")[0] if len(title_line) > 1 else "Unknown"
    else:
        title, author, written = extract_epub(epub_path, output_dir)
        print_done(f"Extracted {len(written)} chapter files")
        mark_step_complete(output_dir, "extract")
    print_step(2, "Analyze characters with Claude Haiku")
    if is_step_complete(output_dir, "characters"):
        print_skip("Characters already analyzed")
    else:
        analyze_characters_sync(output_dir, title, author)
        print_done("Character analysis complete")
        mark_step_complete(output_dir, "characters")
    print_step(3, "Generate voice descriptions")
    if is_step_complete(output_dir, "voices_desc"):
        print_skip("Voice descriptions already generated")
    else:
        generate_voices_sync(output_dir)
        print_done("Voice descriptions generated")
        mark_step_complete(output_dir, "voices_desc")
    print_step(4, "Generate scripts with Claude Haiku")
    if is_step_complete(output_dir, "scripts"):
        print_skip("Scripts already generated")
    else:
        scripts = generate_scripts_sync(output_dir)
        print_done(f"Generated {len(scripts)} script files")
        mark_step_complete(output_dir, "scripts")
    print_step(5, "Synthesize audio")
    if is_step_complete(output_dir, "audio"):
        print_skip("Audio already synthesized")
    else:
        audio_files = synthesize_all_chapters(output_dir)
        print_done(f"Synthesized {len(audio_files)} audio segments")
        mark_step_complete(output_dir, "audio")
    print_step(6, "Assemble M4B audiobook")
    if is_step_complete(output_dir, "m4b"):
        print_skip("M4B already assembled")
        m4b_path = output_dir / f"{output_dir.name}.m4b"
    else:
        m4b_path = assemble_m4b(output_dir, title, author)
        print_done(f"Created {m4b_path.name}")
        mark_step_complete(output_dir, "m4b")
    print(f"\n{Fore.GREEN}Complete!{Style.RESET_ALL} Audiobook: {m4b_path}")
    return m4b_path
