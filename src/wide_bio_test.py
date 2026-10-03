"""Real-file tests for the wide-context biography contract (tiny real byte-level BPE vocabulary, no model, no network)."""

import json
from pathlib import Path

import pytest

from src.wide_bio import (
    CATEGORIES,
    ContractError,
    build_counter,
    build_plan,
    load_proof_config,
    main,
    project_chapters,
    run_extraction,
    validate_calibration,
    validate_response,
)
from src.wide_bio_tokenizer import (
    TokenizerRefusal,
    build_capture,
    byte_alphabet,
    capture_digest,
    load_exact_tokenizer,
    pre_tokenize,
)

SOURCE = (
    "Intro line.\n\n# Chapter One\n\nRen was tall, with green eyes. She smiled.\n\n"
    "Luna Starwaver, a wizard of forty, laughed.\n\n# Chapter Two\n\nOrphan heading text.\n"
)
CHAPTERS = {
    "00-intro.txt": "Intro line.",
    "01-one.txt": "Ren was tall, with green eyes. She smiled.\n\nLuna Starwaver, a wizard of forty, laughed.\n\n",
    "02-two.txt": "Different words entirely.\n",
}


def capture_file(
    root: Path, model: str = "proof-model:1", pre: str = "qwen35"
) -> tuple[Path, str]:
    chars, _ = byte_alphabet()
    tokens = list(chars.values())
    merges = []
    for left, right in (("R", "e"), ("Re", "n"), ("Ġ", "t"), ("Ġt", "a")):
        merges.append(f"{left} {right}")
        tokens.append(left + right)
    tokens.append("<|endoftext|>")
    types = [1] * (len(tokens) - 1) + [3]
    show = {
        "model_info": {
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": pre,
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.token_type": types,
            "tokenizer.ggml.merges": merges,
            "x.context_length": 262144,
        }
    }
    path = root / "capture.json"
    path.write_text(
        json.dumps(build_capture(model, "http://proof.invalid:11434", show)),
        encoding="utf-8",
    )
    return path, capture_digest(path)


def make_config(root: Path, digest: str, **over) -> Path:
    llm = {
        "primary_url": "http://proof.invalid:11434",
        "primary_model": "proof-model:1",
        "primary_num_ctx": 262144,
        **over.pop("llm", {}),
    }
    wide = {
        "tokenizer_capture": "capture.json",
        "tokenizer_sha256": digest,
        "output_tokens": 2048,
        "reserve_tokens": 1024,
        "tolerance_percent": 2,
        "chunk_chars": 120,
        **over,
    }
    lines = (
        ["[llm]"]
        + [f"{key} = {json.dumps(value)}" for key, value in llm.items()]
        + ["[wide_bio]"]
        + [f"{key} = {json.dumps(value)}" for key, value in wide.items()]
    )
    path = root / f"proof-{abs(hash(json.dumps([llm, wide], sort_keys=True)))}.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def world(tmp_path: Path):
    _path, digest = capture_file(tmp_path)
    config_path = make_config(tmp_path, digest)
    project = tmp_path / "project"
    (project / "chapters").mkdir(parents=True)
    for name, text in CHAPTERS.items():
        (project / "chapters" / name).write_text(text, encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text(SOURCE, encoding="utf-8")
    return tmp_path, config_path, digest, project, source


def test_config_pins_primary_wide_context_only(world) -> None:
    root, config_path, digest, _, _ = world
    config = load_proof_config(config_path)
    assert (
        config.backend.num_ctx == 262144 and config.input_budget == 262144 - 2048 - 1024
    )
    for bad, code in (
        (
            make_config(root, digest, llm={"primary_num_ctx": 32768}),
            "config_wrong_context",
        ),
        (
            make_config(root, digest, llm={"primary_url": "http://127.0.0.1:11434"}),
            "config_is_fallback",
        ),
        (
            make_config(root, digest, llm={"primary_model": "qwen3:8b"}),
            "config_is_fallback",
        ),
        (make_config(root, digest, llm={"api_key": "x"}), "config_has_secret"),
    ):
        with pytest.raises(ContractError) as error:
            load_proof_config(bad)
        assert error.value.code == code
    with pytest.raises(ContractError) as error:
        load_proof_config(root / "missing.toml")
    assert error.value.code == "config_missing"


def test_canonical_config_is_refused() -> None:
    from src.wide_bio import CANONICAL_CONFIG

    if CANONICAL_CONFIG.is_file():
        with pytest.raises(ContractError) as error:
            load_proof_config(CANONICAL_CONFIG)
        assert error.value.code == "config_is_canonical"


def test_tokenizer_fails_closed(world) -> None:
    root, _, digest, _, _ = world
    path = root / "capture.json"
    assert load_exact_tokenizer(path, digest, "proof-model:1").count("Ren was tall") > 0
    for pin, model, code in (
        ("0" * 64, "proof-model:1", "capture_hash_mismatch"),
        (digest, "other:1", "capture_model_mismatch"),
    ):
        with pytest.raises(TokenizerRefusal) as error:
            load_exact_tokenizer(path, pin, model)
        assert error.value.code == code
    (root / "o").mkdir()
    other, other_digest = capture_file(root / "o", pre="llama-bpe")
    with pytest.raises(TokenizerRefusal) as error:
        load_exact_tokenizer(other, other_digest, "proof-model:1")
    assert error.value.code == "pre_tokenizer_unproven"
    tokenizer = load_exact_tokenizer(path, digest, "proof-model:1")
    for text, code in (
        ("a<|endoftext|>b", "text_contains_special_token"),
        ("e\u0301", "text_not_nfc"),
    ):
        with pytest.raises(TokenizerRefusal) as error:
            tokenizer.count(text)
        assert error.value.code == code
    with pytest.raises(TokenizerRefusal):
        load_exact_tokenizer(root / "nope.json", digest, "proof-model:1")


def test_pre_tokenize_tiles_text_and_merges_apply(world) -> None:
    text = "Ren's  tall,\n\n  house 42!\r\nx"
    assert "".join(pre_tokenize(text)) == text
    root, _, digest, _, _ = world
    tokenizer = load_exact_tokenizer(root / "capture.json", digest, "proof-model:1")
    assert (
        tokenizer.count("Ren") == 1
        and tokenizer.decode(tokenizer.encode("Ren tall")) == "Ren tall"
    )


def test_plan_covers_source_exactly_once_with_witnesses(world) -> None:
    _, config_path, _, project, source = world
    config = load_proof_config(config_path)
    plan = build_plan(
        source.read_text(encoding="utf-8"),
        project_chapters(project),
        config,
        build_counter(config),
    )
    coverage = plan.artifact["coverage"]
    assert (
        coverage["exactly_once"]
        and coverage["source_chars"] == len(SOURCE)
        and coverage["unmapped_paragraphs"] >= 1
    )
    assert "".join(item.chunk.paragraphs[0].text[:0] for item in plan.chunks) == ""
    assert plan.chunks[0].chunk.start == 0 and plan.chunks[-1].chunk.end == len(SOURCE)
    mapped = [p for item in plan.chunks for p in item.chunk.paragraphs if p.witness]
    assert any(p.witness["chapter"] == "01-one.txt" for p in mapped)
    assert plan.artifact["calibration_required"]


def test_plan_refuses_unmapped_source_and_budget(world) -> None:
    root, config_path, digest, project, source = world
    config = load_proof_config(config_path)
    with pytest.raises(ContractError) as error:
        build_plan(
            "Nothing alike at all.\n",
            project_chapters(project),
            config,
            build_counter(config),
        )
    assert error.value.code == "source_unmapped"
    tiny = config
    with pytest.raises(ContractError) as error:
        build_plan(
            source.read_text(encoding="utf-8"),
            project_chapters(project),
            tiny,
            lambda text: 10**9,
        )
    assert error.value.code == "input_budget_exceeded"
    small = load_proof_config(make_config(root, digest, chunk_chars=5))
    with pytest.raises(ContractError) as error:
        build_plan(
            source.read_text(encoding="utf-8"),
            project_chapters(project),
            small,
            build_counter(small),
        )
    assert error.value.code == "paragraph_oversize"


def one_chunk(world):
    _, config_path, _, project, source = world
    config = load_proof_config(config_path)
    plan = build_plan(
        source.read_text(encoding="utf-8"),
        project_chapters(project),
        config,
        build_counter(config),
    )
    return config, plan, project_chapters(project)


def test_validate_response_checks_quotes_and_witnesses(world) -> None:
    _, plan, _ = one_chunk(world)[0], one_chunk(world)[1], None
    chunk = next(
        item.chunk
        for item in plan.chunks
        if any(p.text.startswith("Ren was") for p in item.chunk.paragraphs)
    )
    ren = next(p for p in chunk.paragraphs if p.text.startswith("Ren was"))
    good = {
        "subject": "Ren",
        "category": "appearance",
        "value": "tall",
        "quote": "Ren was tall",
        "paragraph_id": ren.id,
    }
    bad = [
        {**good, "quote": "Ren was short"},
        {**good, "paragraph_id": "999999"},
        {**good, "category": "mood"},
        {"subject": "Ren"},
    ]
    result = validate_response(json.dumps({"facts": [good, good, *bad]}), chunk)
    assert (
        len(result["claims"]) == 1
        and result["claims"][0]["witness"]["chapter"] == "01-one.txt"
    )
    assert [item["reason"] for item in result["rejected"]] == [
        "duplicate_fact",
        "quote_not_in_paragraph",
        "unknown_paragraph_id",
        "unknown_category",
        "wrong_fields",
    ]
    assert validate_response("nope", chunk)["status"] == "invalid_json"
    assert validate_response('{"facts":{},"x":1}', chunk)["status"] == "invalid_shape"
    orphan = next(
        p for item in plan.chunks for p in item.chunk.paragraphs if not p.witness
    )
    holder = next(item.chunk for item in plan.chunks if orphan in item.chunk.paragraphs)
    pending = validate_response(
        json.dumps(
            {"facts": [{**good, "quote": orphan.text[:5], "paragraph_id": orphan.id}]}
        ),
        holder,
    )
    assert (
        not pending["claims"]
        and pending["pending"][0]["pending_reason"] == "no_chapter_witness"
    )
    assert set(CATEGORIES) >= {"voice", "kinship"}


def calibration(config, digest: str, drift: float = 0.0) -> dict:
    return {
        "model": config.backend.model,
        "tokenizer_capture_sha256": digest,
        "samples": [
            {"local_tokens": 20000, "prompt_eval_count": int(20000 * (1 + drift / 100))}
        ]
        * 3,
    }


def test_calibration_gate(world) -> None:
    config, _, _ = one_chunk(world)
    assert (
        validate_calibration(calibration(config, config.tokenizer_sha256), config)[
            "samples"
        ]
        == 3
    )
    for record, code in (
        (calibration(config, config.tokenizer_sha256, 5), "calibration_drift"),
        (
            {**calibration(config, config.tokenizer_sha256), "model": "x"},
            "calibration_mismatch",
        ),
        (
            {**calibration(config, config.tokenizer_sha256), "samples": []},
            "calibration_insufficient",
        ),
    ):
        with pytest.raises(ContractError) as error:
            validate_calibration(record, config)
        assert error.value.code == code


def test_run_saves_evidence_never_reasks_and_reports_fraction(world) -> None:
    root, *_ = world
    config, plan, chapters = one_chunk(world)
    out = root / "out"
    calls = []

    def transport(url: str, payload: bytes, timeout: float) -> dict:
        calls.append(timeout)
        return {
            "model": config.backend.model,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 3,
            "content": json.dumps({"facts": []}),
        }

    summary = run_extraction(
        chapters,
        config,
        plan,
        out,
        transport,
        calibration(config, config.tokenizer_sha256),
    )
    assert (
        summary["state"] == "ready"
        and summary["source_fraction"] == 1.0
        and len(calls) == len(plan.chunks)
    )
    assert max(calls) <= 240
    assert (out / "raw" / "k0000.response.txt").is_file() and (
        out / "claims.jsonl"
    ).is_file()
    assert (
        json.loads((out / "pending_reconciliation.json").read_text())["status"]
        == "pending"
    )
    again = run_extraction(
        chapters,
        config,
        plan,
        out,
        transport,
        calibration(config, config.tokenizer_sha256),
    )
    assert len(calls) == len(plan.chunks) and again["state"] == "ready"
    offline = run_extraction(chapters, config, plan, root / "empty", None)
    assert (
        offline["state"] == "not_ready"
        and offline["source_fraction"] == 0.0
        and offline["chunk_status"] == {"missing": len(plan.chunks)}
    )


def test_run_rejects_fallback_truncation_deadline_and_project_output(world) -> None:
    root, *_ = world
    config, plan, chapters = one_chunk(world)
    cal = calibration(config, config.tokenizer_sha256)
    with pytest.raises(ContractError) as error:
        run_extraction(
            chapters,
            config,
            plan,
            root / "fb",
            lambda u, p, t: {"model": "qwen3:8b", "content": "{}"},
            cal,
        )
    assert error.value.code == "fallback_route"
    cut = run_extraction(
        chapters,
        config,
        plan,
        root / "cut",
        lambda u, p, t: {
            "model": config.backend.model,
            "done_reason": "length",
            "content": '{"facts":[]}',
        },
        cal,
    )
    assert (
        cut["state"] == "not_ready"
        and cut["source_fraction"] == 0.0
        and "truncated" in cut["chunk_status"]
    )
    ticks = iter(range(0, 10_000, 60))
    late = run_extraction(
        chapters,
        config,
        plan,
        root / "late",
        lambda u, p, t: {
            "model": config.backend.model,
            "done_reason": "stop",
            "content": '{"facts":[]}',
        },
        cal,
        clock=lambda: float(next(ticks)),
    )
    assert late["state"] == "not_ready" and 0 < late["source_fraction"] < 1
    with pytest.raises(ContractError) as error:
        run_extraction(chapters, config, plan, chapters[0].parent / "out", None)
    assert error.value.code == "output_inside_project"
    with pytest.raises(ContractError) as error:
        run_extraction(chapters, config, plan, root / "nocal", lambda u, p, t: {}, None)
    assert error.value.code == "calibration_mismatch"


def test_cli_plan_validate_and_run_gate(world, capsys: pytest.CaptureFixture) -> None:
    root, config_path, _, project, source = world
    base = [
        str(project),
        "--source",
        str(source),
        "--config",
        str(config_path),
        "--out",
        str(root / "cli"),
    ]
    assert main(["plan", *base]) == 0
    assert json.loads(capsys.readouterr().out)["coverage"]["exactly_once"]
    assert main(["validate", *base]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "not_ready"
    cal = root / "cal.json"
    cal.write_text("{}", encoding="utf-8")
    assert main(["run", *base, "--calibration", str(cal)]) == 2
    assert "execute_required" in capsys.readouterr().err
