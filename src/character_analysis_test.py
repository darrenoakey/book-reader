import tempfile
from pathlib import Path

from src.character_analysis import analyze_characters_sync, merge_character_info, parse_json_response


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
        chapter1.write_text("""
John walked into the room, his heavy boots echoing on the wooden floor. He was a tall man, well over six feet, with dark hair streaked with gray at the temples. His deep voice rumbled when he spoke. At forty-two, he carried himself with the careful movements of someone who had seen too much.

"Hello, Mary," he said, his voice low and gravelly. "It's been a long time."

Mary looked up from her chair. She was a small woman in her mid-thirties, with bright red hair and a soft, melodic voice. Where John was large and imposing, she was delicate and quick.

"John! You finally came." Her voice was warm but carried a hint of an Irish accent she'd never quite lost. "I was beginning to think you'd forgotten about your little sister."

John thought about their childhood in Dublin. Those were simpler times, before the war changed everything. He remembered how she used to sing while doing the dishes, her young voice clear as a bell.

"I missed you," he replied, his gruff tone softening. "It's been too long since we talked. Far too long."

"Come, sit," Mary said, gesturing to the chair across from her. "Tell me everything. I want to hear it all in that booming voice of yours."

John lowered his large frame into the chair, which creaked under his weight. He began to speak, his bass voice filling the small room with stories of where he had been.
        """.strip())
        result_path = analyze_characters_sync(output_dir, "Test Book", "Test Author")
        assert result_path.exists()
        import json
        chars = json.loads(result_path.read_text())
        assert "narrator" in chars
        assert "name" in chars["narrator"]
        assert "bio" in chars["narrator"]
        char_ids = [k for k in chars.keys() if k != "narrator"]
        assert len(char_ids) >= 1
