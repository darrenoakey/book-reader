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


# ##################################################################
# classify empty staged input
# only the captured empty-staged-input HTTP 400 is transient; every other denial fails closed.
def test_only_empty_staged_input_rejection_is_retryable() -> None:
    from src.arbiter_tts import is_empty_staged_input_rejection

    captured = RuntimeError(
        "submit tts-breeze rejected with HTTP 400: job rejected: 2 input paths unreadable "
        "/mnt/arbiter-store/inbox/c686d7adc26a_professor_lynn.wav bad input file (empty, 0 bytes)"
    )
    assert is_empty_staged_input_rejection(captured)
    assert not is_empty_staged_input_rejection(RuntimeError("submit tts-breeze rejected with HTTP 400: bad seed"))
    assert not is_empty_staged_input_rejection(RuntimeError("submit tts-breeze rejected with HTTP 403: unreadable empty"))


# ##################################################################
# breeze one line real
# real staging and live Breeze synthesis of one line from an unchanged real reference voice.
def test_breeze_many_one_line_real(tmp_path) -> None:
    import subprocess
    from pathlib import Path

    from src.arbiter_tts import tts_breeze_many

    # gitignored real outputs live only in the canonical checkout, not in greenline worktrees.
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True, text=True, check=True, cwd=Path(__file__).parent,
    ).stdout.strip()
    output_dir = Path(common).parent / "output" / "weakest_beast_tamer"
    out = tmp_path / "line.wav"
    tts_breeze_many([{"text": "A short test line.", "speaker": "narrator", "output_path": out}], output_dir)
    assert out.stat().st_size > 1000
