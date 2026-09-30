import pytest

from src.hourly_spans import immutable_spans, parse_assignments


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
