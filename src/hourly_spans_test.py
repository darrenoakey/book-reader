import pytest

from src.hourly_spans import (
    immutable_spans,
    parse_assignments,
    parse_speakers,
    speaker_array_schema,
)


def test_spans_preserve_source_and_complete_bijection() -> None:
    text = 'Klein said Look at the failure egg. He laughed.'
    spans = immutable_spans(text)
    assert ''.join(spans) == text
    parsed = parse_assignments('{"index": 0, "speaker_id": "klein"}\n{"index": 1, "speaker_id": "narrator"}', 0, 2, {"narrator", "klein"})
    assert parsed == {0: "klein", 1: "narrator"}
    pretty_stream = '''{
  "index": 0,
  "speaker_id": "klein"
}
{
  "index": 1,
  "speaker_id": "narrator"
}'''
    assert parse_assignments(pretty_stream, 0, 2, {"narrator", "klein"}) == {0: "klein", 1: "narrator"}
    assert parse_assignments('[{"index": 0, "speaker_id": "klein"}, {"index": 1, "speaker_id": "narrator"}]', 0, 2, {"narrator", "klein"}) == {0: "klein", 1: "narrator"}
    with pytest.raises(ValueError):
        parse_assignments(pretty_stream + " trailing", 0, 2, {"narrator", "klein"})


def test_schema_speaker_array_requires_one_valid_id_per_span() -> None:
    schema = speaker_array_schema(["narrator", "klein"], 2)
    assert schema["minItems"] == schema["maxItems"] == 2
    assert schema["items"]["enum"] == ["narrator", "klein"]
    assert parse_speakers('["klein", "narrator"]', 2, {"narrator", "klein"}) == ["klein", "narrator"]
    with pytest.raises(ValueError):
        parse_speakers('["klein"]', 2, {"narrator", "klein"})
    with pytest.raises(ValueError):
        parse_speakers('["klein", "unknown"]', 2, {"narrator", "klein"})


# ##################################################################
# scoped references reach the span prompt only at their exact mention
# a reference bound to chapter hash, span hash and offset annotates just that span and never becomes a global alias.
def test_scoped_reference_notes_bind_exact_mention_and_reject_mismatch() -> None:
    import hashlib

    from src.hourly_spans import classification_prompt, scoped_reference_notes

    text = "Young Ren smiled. Young stood up. Young left."
    spans = immutable_spans(text)
    assert ''.join(spans) == text and len(spans) == 3
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
    prompt = classification_prompt(["narrator", "young_ren"], "", notes[1], "1: " + spans[1])
    assert "span 1" in prompt and "ONLY to that exact mention" in prompt
    assert "young_ren->" not in prompt and "Approved aliases that must use their canonical speaker ID: (none)" in prompt
