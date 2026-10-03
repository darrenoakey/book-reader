"""Tests for the demo module's pure content and output helpers."""

from src.demo_outputs import SAMPLE_CHAPTER, print_section


# ##################################################################
# test sample chapter
# the sample is real prose with quoted dialogue from several named speakers for the analysis demo
def test_sample_chapter_has_dialogue_and_characters() -> None:
    text = SAMPLE_CHAPTER.strip()
    assert len(text.split()) > 150
    assert text.count('"') >= 8 and text.count('"') % 2 == 0
    for name in ("Locke", "Jean", "Father Chains"):
        assert name in text


# ##################################################################
# test print section
# a section prints its title between two rule lines
def test_print_section(capsys) -> None:
    print_section("STEP 9: DEMO")
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 3
    assert "=" * 60 in lines[0] and "=" * 60 in lines[2]
    assert "STEP 9: DEMO" in lines[1]
