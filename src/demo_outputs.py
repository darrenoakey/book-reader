#!/usr/bin/env python3
import json
import tempfile
from pathlib import Path

from colorama import Fore, Style, init

from src.character_analysis import analyze_characters_sync
from src.script_generate import generate_scripts_sync
from src.voice_description import generate_voices_sync

init(autoreset=True)

SAMPLE_CHAPTER = """
The morning sun cast long shadows across the cobblestones of Camorr's Temple District.
Locke Lamora walked with practiced nonchalance, his threadbare coat hiding the fortune
in jewels sewn into its lining.

"You're late," said Jean Tannen, emerging from a doorway. He was a large man, deceptively
quick for his size, with hands that could crush walnuts or pick pockets with equal ease.

Locke shrugged. "The Sanza twins needed handling. They're getting ambitious."

"Ambitious enough to cut us out?" Jean's voice held an edge of concern.

Locke thought about the twins - Calo with his quick tongue and Galdo with his quicker
blade. They were family, in a way that had nothing to do with blood. He would never
betray them, and they would never betray him. That was the way of the Gentlemen Bastards.

"Never," Locke said firmly. "We're brothers."

Jean nodded, satisfied. "Then let's go steal from a duke."

Father Chains had taught them well - how to speak like nobles, fight like demons, and
lie like the very gods themselves. The old priest was dead now, but his lessons lived
on in every con they ran.

"The Duke of Ashen Vale arrives at noon," Jean said, reviewing their plan. "His wife
has a weakness for fortune tellers."

"And I," said Locke with a theatrical bow, "am the finest fortune teller in all of Camorr."
"""


def print_section(title: str) -> None:
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{title}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        chapters_dir = output_dir / "chapters"
        chapters_dir.mkdir(parents=True)

        # Create test files
        intro = chapters_dir / "00-intro.txt"
        intro.write_text("The Lies of Locke Lamora by Scott Lynch.")

        chapter1 = chapters_dir / "01-chapter_one.txt"
        chapter1.write_text(SAMPLE_CHAPTER.strip())

        # Step 1: Character Analysis
        print_section("STEP 1: CHARACTER ANALYSIS (Claude Haiku)")
        print(f"{Fore.WHITE}Analyzing chapter to find characters with speaking parts...{Style.RESET_ALL}\n")

        analyze_characters_sync(output_dir, "The Lies of Locke Lamora", "Scott Lynch")

        chars = json.loads((output_dir / "characters.json").read_text())
        for char_id, info in chars.items():
            print(f"{Fore.GREEN}{char_id}{Style.RESET_ALL}")
            print(f"  Name: {info['name']}")
            print(f"  Bio: {info['bio'][:200]}...")
            print()

        # Step 2: Voice Descriptions
        print_section("STEP 2: VOICE DESCRIPTIONS (Claude Haiku)")
        print(f"{Fore.WHITE}Converting character biographies to voice descriptions...{Style.RESET_ALL}\n")

        generate_voices_sync(output_dir)

        voices = json.loads((output_dir / "voices.json").read_text())
        for char_id, info in voices.items():
            print(f"{Fore.GREEN}{char_id}{Style.RESET_ALL}")
            print(f"  {info['description']}")
            print()

        # Step 3: Script Generation
        print_section("STEP 3: SCRIPT GENERATION (Claude Haiku)")
        print(f"{Fore.WHITE}Breaking chapter into speaker-attributed segments...{Style.RESET_ALL}\n")

        generate_scripts_sync(output_dir)

        script_path = output_dir / "script" / "01-chapter_one.jsonl"
        lines = script_path.read_text().strip().split("\n")
        for i, line in enumerate(lines[:15]):  # Show first 15 lines
            entry = json.loads(line)
            speaker = list(entry.keys())[0]
            text = entry[speaker][:80]
            if len(entry[speaker]) > 80:
                text += "..."
            color = Fore.YELLOW if speaker == "narrator" else Fore.MAGENTA
            print(f"{color}{speaker:12}{Style.RESET_ALL} {text}")

        if len(lines) > 15:
            print(f"\n  ... and {len(lines) - 15} more lines")


if __name__ == "__main__":
    main()
