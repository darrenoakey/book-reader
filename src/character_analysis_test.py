import tempfile
from pathlib import Path

from src.character_analysis import (
    MAX_DEDUP_GROUP,
    analyze_characters_sync,
    apply_dedup_groups,
    merge_character_info,
    parse_json_response,
)


# ##################################################################
# test parse json response
# verify json extraction from plain and markdown responses
def test_parse_json_response() -> None:
    plain = '{"name": "test"}'
    assert parse_json_response(plain) == {"name": "test"}
    markdown = '```json\n{"name": "test"}\n```'
    assert parse_json_response(markdown) == {"name": "test"}


# ##################################################################
# test merge character info
# verify merging character info from multiple chapters
def test_merge_character_info() -> None:
    chapters = [
        {"characters": {"john": {"name": "John", "details": "A tall man."}}},
        {"characters": {"john": {"name": "John", "details": "He likes coffee."}}},
        {"characters": {"mary": {"name": "Mary", "details": "John's sister."}}},
    ]
    merged = merge_character_info(chapters)
    assert "john" in merged
    assert "mary" in merged
    assert "A tall man." in merged["john"]["bio"]
    assert "He likes coffee." in merged["john"]["bio"]


# ##################################################################
# test analyze characters with real haiku
# calls claude haiku to analyze a minimal chapter
def test_analyze_characters_real() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        chapters_dir = output_dir / "chapters"
        chapters_dir.mkdir(parents=True)
        intro = chapters_dir / "00-intro.txt"
        intro.write_text("Test Book by Test Author.")
        chapter1 = chapters_dir / "01-chapter_one.txt"
        chapter1.write_text(
            """
John walked into the room, his heavy boots echoing on the wooden floor. He was a tall man, well over six feet, with dark hair streaked with gray at the temples. His deep voice rumbled when he spoke. At forty-two, he carried himself with the careful movements of someone who had seen too much.

"Hello, Mary," he said, his voice low and gravelly. "It's been a long time."

Mary looked up from her chair. She was a small woman in her mid-thirties, with bright red hair and a soft, melodic voice. Where John was large and imposing, she was delicate and quick.

"John! You finally came." Her voice was warm but carried a hint of an Irish accent she'd never quite lost. "I was beginning to think you'd forgotten about your little sister."

John thought about their childhood in Dublin. Those were simpler times, before the war changed everything. He remembered how she used to sing while doing the dishes, her young voice clear as a bell.

"I missed you," he replied, his gruff tone softening. "It's been too long since we talked. Far too long."

"Come, sit," Mary said, gesturing to the chair across from her. "Tell me everything. I want to hear it all in that booming voice of yours."

John lowered his large frame into the chair, which creaked under his weight. He began to speak, his bass voice filling the small room with stories of where he had been.
        """.strip()
        )
        result_path = analyze_characters_sync(output_dir, "Test Book", "Test Author")
        assert result_path.exists()
        import json

        chars = json.loads(result_path.read_text())
        assert "narrator" in chars
        assert "name" in chars["narrator"]
        assert "bio" in chars["narrator"]
        char_ids = [k for k in chars if k != "narrator"]
        assert len(char_ids) >= 1


# ##################################################################
# dedup must never swallow distinct characters
# Dragonsinger's production shipped with 17 of 49 analysed characters because a
# single over-broad group merged every protagonist into a few ids; one surviving
# id carried a 17,007-character blob of other people's bios.
def _cast() -> dict:
    return {
        cid: {"name": cid.title(), "bio": f"{cid} speaks.", "look": ""}
        for cid in ("menolly", "piemur", "sebell", "audiva", "pona", "talmor", "jerint", "morshal", "benis")
    }


def test_over_broad_dedup_group_is_refused_and_keeps_every_character() -> None:
    cast = _cast()
    too_wide = [list(cast)]
    result = apply_dedup_groups(cast, too_wide)
    # nothing merged: every principal survives with its own bio
    assert set(result) == set(cast)
    for cid in cast:
        assert result[cid]["bio"] == f"{cid} speaks."
    assert len(too_wide[0]) > MAX_DEDUP_GROUP


def test_plausible_dedup_group_still_merges() -> None:
    cast = {
        "don_salvara": {"name": "Don Salvara", "bio": "older man", "look": ""},
        "don_lorenzo_salvara": {"name": "Don Lorenzo Salvara", "bio": "the don", "look": ""},
        "menolly": {"name": "Menolly", "bio": "girl", "look": ""},
    }
    result = apply_dedup_groups(cast, [["don_salvara", "don_lorenzo_salvara"]])
    assert "don_salvara" not in result
    assert "don_lorenzo_salvara" in result
    # the unrelated character is untouched
    assert result["menolly"]["bio"] == "girl"


def test_group_bound_admits_only_small_same_person_groups() -> None:
    assert MAX_DEDUP_GROUP >= 2
    assert MAX_DEDUP_GROUP <= 6
