import pytest

from src.arbiter_tts import _client, _submit, sanitize_why


# ##################################################################
# sanitize provenance
# remove C0 controls and normalize whitespace while preserving a bounded human-readable job reason.
def test_sanitize_why_controls_and_bound() -> None:
    assert sanitize_why("hour\nscene\x00source\ttext") == "hour scene source text"
    assert sanitize_why("é" * 300).encode("utf-8").decode("utf-8") == sanitize_why("é" * 300)
    assert len(sanitize_why("é" * 300).encode("utf-8")) <= 256


# ##################################################################
# submit sanitized qwen provenance real
# exercise real arbiter HTTP validation with controls in provenance, while server dedup makes this a tiny non-rendering request.
def test_qwen_provenance_real() -> None:
    raw = "hour 1 scene\nsource\x00excerpt\tfor provenance"
    job_id = _submit(
        _client(120),
        "qwen-image",
        {
            "prompt": "A tiny quiet watercolor pebble on a plain cream background.",
            "width": 512,
            "height": 512,
            "steps": 20,
            "seed": 124994,
        },
        why=raw,
    )
    source = _client(120).status(job_id).get("source")
    assert source == {"who": "book-reader", "why": sanitize_why(raw)}


# ##################################################################
# reject client error real
# a server-side client-error response must propagate once instead of entering the long transient retry loop.
def test_http_4xx_fails_fast() -> None:
    with pytest.raises(RuntimeError, match="HTTP 4"):
        _submit(_client(120), "not-a-real-arbiter-job", {}, why="invalid\nrequest")
