import pytest

from src.hourly_spans import classify_spans, generate_hourly_script_sync, immutable_spans


def test_spans_preserve_source() -> None:
    text = "Klein said Look at the failure egg. He laughed."
    spans = immutable_spans(text)
    assert "".join(spans) == text and len(spans) == 2
    with pytest.raises(ValueError):
        immutable_spans("  \n ")


# ##################################################################
# scoped reference notes
# a reference bound to chapter hash, span hash and offset annotates just that span and never becomes a global alias.
def test_scoped_reference_notes_bind_exact_mention_and_reject_mismatch() -> None:
    import hashlib

    from src.hourly_spans import scoped_reference_notes

    text = "Young Ren smiled. Young stood up. Young left."
    spans = immutable_spans(text)
    assert "".join(spans) == text and len(spans) == 3
    reference = {
        "chapter_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "quote_sha256": hashlib.sha256(spans[1].encode()).hexdigest(),
        "label": "Young",
        "span_start": spans[1].index("Young"),
        "canonical": "young_ren",
    }
    speakers = {"narrator", "young_ren"}
    notes = scoped_reference_notes(text, spans, speakers, [reference])
    assert list(notes) == [1] and "young_ren" in notes[1][0] and "'Young'" in notes[1][0]
    assert scoped_reference_notes(text, spans, speakers, None) == {}
    # a different chapter content hash never applies
    assert scoped_reference_notes(text + " ", immutable_spans(text + " "), speakers, [reference]) == {}
    with pytest.raises(ValueError, match="source span"):
        scoped_reference_notes(text, spans, speakers, [{**reference, "span_start": 3}])
    with pytest.raises(ValueError, match="valid speakers"):
        scoped_reference_notes(text, spans, {"narrator"}, [reference])


# ##################################################################
# live scriptor attribution
# the real small model splits a quote around its dialogue tag; the script reconstructs the source exactly.
def test_classify_spans_live_splits_quote_from_tag(tmp_path) -> None:
    import asyncio
    import json

    text = 'Bob looked up. "Yellow," Bob said, "is the colour of the sun."\n\n"Not at night," Alice replied.'
    lines = asyncio.run(classify_spans(text, ["narrator", "bob", "alice"], names={"bob": "Bob", "alice": "Alice"}))
    assert "".join(next(iter(line.values())) for line in lines) == text
    assert [next(iter(line)) for line in lines] == ["narrator", "bob", "narrator", "bob", "alice", "narrator"]
    assert lines[1] == {"bob": ' "Yellow,"'} and lines[4] == {"alice": '\n\n"Not at night,"'}
    chapter = tmp_path / "01-one.txt"
    chapter.write_text(text, encoding="utf-8")
    script = generate_hourly_script_sync(chapter, tmp_path / "01-one.jsonl", ["narrator", "bob", "alice"])
    rows = [json.loads(line) for line in script.read_text(encoding="utf-8").splitlines()]
    assert "".join(next(iter(row.values())) for row in rows) == text
    with pytest.raises(ValueError):
        asyncio.run(classify_spans(text, ["bob"]))
