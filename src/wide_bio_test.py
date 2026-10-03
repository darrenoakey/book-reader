"""Real-file tests for the wide-context biography contract (tiny real byte-level BPE vocabulary, no model, no network)."""

import json
from pathlib import Path

import pytest

from src.wide_bio import (
    CATEGORIES,
    MAX_FIXED_OVERHEAD_TOKENS,
    SCHEMA,
    Chunk,
    ContractError,
    build_counter,
    build_plan,
    canonical_json,
    chunk_schema,
    load_proof_config,
    main,
    padded,
    project_chapters,
    request_record,
    run_extraction,
    sha256_text,
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


def overhead_record(config, pairs) -> dict:
    return {
        "model": config.backend.model,
        "tokenizer_capture_sha256": config.tokenizer_sha256,
        "samples": [
            {"local_tokens": local, "prompt_eval_count": server}
            for local, server in pairs
        ],
    }


REAL_PAIRS = ((236, 252), (3030, 3046), (10122, 10138))


def test_calibration_accepts_constant_template_overhead(world) -> None:
    config, _, _ = one_chunk(world)
    measured = validate_calibration(overhead_record(config, REAL_PAIRS), config)
    assert measured == {"samples": 3, "fixed_overhead_tokens": 16}
    assert (
        validate_calibration(calibration(config, config.tokenizer_sha256), config)[
            "fixed_overhead_tokens"
        ]
        == 0
    )


def test_calibration_rejects_variable_negative_and_excessive_overhead(world) -> None:
    config, _, _ = one_chunk(world)
    cap = MAX_FIXED_OVERHEAD_TOKENS
    cases = (
        # proportional drift: 0.5% of each prompt is not a constant
        (((236, 237), (3030, 3045), (10122, 10173)), "calibration_drift"),
        # one sample off by a single token
        (((236, 252), (3030, 3046), (10122, 10139)), "calibration_drift"),
        (((236, 252), (3030, 3029), (10122, 10138)), "calibration_drift"),
        (((236, 235), (3030, 3029), (10122, 10121)), "calibration_overhead_negative"),
        (
            ((236, 236 + cap + 1), (3030, 3030 + cap + 1), (10122, 10122 + cap + 1)),
            "calibration_drift",
        ),
        (((20000, 20016),) * 3, "calibration_insufficient"),
    )
    for pairs, code in cases:
        with pytest.raises(ContractError) as error:
            validate_calibration(overhead_record(config, pairs), config)
        assert error.value.code == code, pairs
    edge = tuple((n, n + cap) for n in (236, 3030, 10122))
    assert (
        validate_calibration(overhead_record(config, edge), config)[
            "fixed_overhead_tokens"
        ]
        == cap
    )
    with pytest.raises(ContractError) as error:
        validate_calibration(
            overhead_record(config, ((236, 252), (3030, 3046), (5000, 5016))), config
        )
    assert error.value.code == "calibration_insufficient"


def test_plan_budgets_fixed_overhead_and_records_it(world) -> None:
    _, config_path, _, project, source = world
    config = load_proof_config(config_path)
    args = (
        source.read_text(encoding="utf-8"),
        project_chapters(project),
        config,
        build_counter(config),
    )
    base = build_plan(*args)
    shifted = build_plan(*args, 16)
    assert base.artifact["fixed_overhead_tokens"] == 0
    assert shifted.artifact["fixed_overhead_tokens"] == 16
    assert base.artifact["plan_sha256"] != shifted.artifact["plan_sha256"]
    for before, after in zip(base.chunks, shifted.chunks):
        assert after.input_tokens == before.input_tokens
        assert after.padded_tokens > before.padded_tokens
    assert [c["padded_tokens"] for c in shifted.artifact["chunks"]] == [
        item.padded_tokens for item in shifted.chunks
    ]
    for bad in (-1, MAX_FIXED_OVERHEAD_TOKENS + 1, 1.5, True):
        with pytest.raises(ContractError) as error:
            build_plan(*args, bad)
        assert error.value.code == "overhead_invalid"


def test_overhead_pushes_chunk_over_input_budget(world) -> None:
    _, config_path, _, project, source = world
    config = load_proof_config(config_path)
    args = (source.read_text(encoding="utf-8"), project_chapters(project), config)
    half = max(
        n
        for n in range(config.input_budget)
        if padded(2 * n, config.tolerance_percent) <= config.input_budget
    )
    build_plan(*args, lambda text: half, 0)
    with pytest.raises(ContractError) as error:
        build_plan(*args, lambda text: half, 16)
    assert error.value.code == "input_budget_exceeded"


def test_run_refuses_plan_built_without_calibrated_overhead(world) -> None:
    root, *_ = world
    config, plan, chapters = one_chunk(world)
    record = overhead_record(config, REAL_PAIRS)
    calls = []

    def transport(url: str, payload: bytes, timeout: float) -> dict:
        calls.append(url)
        return {}

    with pytest.raises(ContractError) as error:
        run_extraction(chapters, config, plan, root / "mismatch", transport, record)
    assert error.value.code == "calibration_overhead_mismatch"
    assert calls == []


def test_cli_run_builds_plan_with_calibrated_overhead(world, capsys) -> None:
    root, config_path, _, project, source = world
    config = load_proof_config(config_path)
    cal = root / "real-cal.json"
    cal.write_text(json.dumps(overhead_record(config, REAL_PAIRS)), encoding="utf-8")
    base = [
        str(project),
        "--source",
        str(source),
        "--config",
        str(config_path),
    ]
    out = root / "cli-cal"
    assert main(["plan", *base, "--out", str(out), "--calibration", str(cal)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["fixed_overhead_tokens"] == 16
    assert json.loads((out / "plan.json").read_text())["fixed_overhead_tokens"] == 16
    bad = root / "bad-cal.json"
    bad.write_text(
        json.dumps(overhead_record(config, ((236, 237), (3030, 3045), (10122, 10173)))),
        encoding="utf-8",
    )
    runout = root / "cli-run"
    assert (
        main(
            ["run", *base, "--out", str(runout), "--calibration", str(bad), "--execute"]
        )
        == 2
    )
    assert "calibration_drift" in capsys.readouterr().err
    assert not (runout / "plan.json").exists()


def sent_paragraph_id_schema(config, item) -> dict:
    _, payload = request_record(config, item)
    return json.loads(payload)["format"]["properties"]["facts"]["items"]["properties"][
        "paragraph_id"
    ]


def test_each_request_constrains_paragraph_id_to_that_chunks_exact_ids(world) -> None:
    config, plan, _ = one_chunk(world)
    assert len(plan.chunks) > 1
    template = SCHEMA["properties"]["facts"]["items"]["properties"]["paragraph_id"]
    assert "enum" not in template
    for item, entry in zip(plan.chunks, plan.artifact["chunks"]):
        ids = [paragraph.id for paragraph in item.chunk.paragraphs]
        assert ids and all(len(i) == 6 and i.isdigit() for i in ids)
        assert sent_paragraph_id_schema(config, item) == {
            "type": "string",
            "enum": ids,
        }
        assert entry["paragraph_ids"] == ids
        assert entry["schema_sha256"] == sha256_text(
            canonical_json(chunk_schema(item.chunk))
        )
    assert len({entry["schema_sha256"] for entry in plan.artifact["chunks"]}) == len(
        plan.chunks
    )
    assert plan.artifact["schema_template_sha256"] == sha256_text(
        canonical_json(SCHEMA)
    )
    assert "schema_sha256" not in plan.artifact


def test_invented_marker_paragraph_id_is_schema_blocked_and_locally_rejected(
    world,
) -> None:
    config, plan, _ = one_chunk(world)
    item = next(
        item
        for item in plan.chunks
        if any(p.text.startswith("Ren was") for p in item.chunk.paragraphs)
    )
    ren = next(p for p in item.chunk.paragraphs if p.text.startswith("Ren was"))
    allowed = sent_paragraph_id_schema(config, item)["enum"]
    assert "P 000001" not in allowed
    good = {
        "subject": "Ren",
        "category": "appearance",
        "value": "tall",
        "quote": "Ren was tall",
        "paragraph_id": ren.id,
    }
    assert good["paragraph_id"] in allowed
    result = validate_response(
        json.dumps({"facts": [good, {**good, "paragraph_id": "P 000001"}]}), item.chunk
    )
    assert len(result["claims"]) == 1
    assert [r["reason"] for r in result["rejected"]] == ["unknown_paragraph_id"]
    assert result["rejected"][0]["fact"]["paragraph_id"] == "P 000001"
    assert result["status"] == "partial"


def test_chunk_without_shown_paragraphs_schema_allows_no_facts() -> None:
    schema = chunk_schema(Chunk("k9999", 0, 3, ()))
    assert schema["properties"]["facts"] == {"type": "array", "maxItems": 0}


# ##################################################################
# provider shapes
# a real local HTTP server answers with each provider's real response shape; the generic transport must parse both.
@pytest.fixture
def provider_server():
    import http.server
    import threading

    seen = []
    reply = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            seen.append((self.path, json.loads(self.rfile.read(length))))
            body = json.dumps(reply["body"]).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", seen, reply
    server.shutdown()
    server.server_close()


OPENAI_BODY = {
    "model": "proof-model:1",
    "choices": [
        {"message": {"role": "assistant", "content": '{"facts": []}'}, "finish_reason": "length"}
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
}
OLLAMA_BODY = {
    "model": "proof-model:1",
    "message": {"role": "assistant", "content": '{"facts": []}'},
    "done_reason": "length",
    "prompt_eval_count": 11,
    "eval_count": 7,
}
PARSED = {
    "model": "proof-model:1",
    "done_reason": "length",
    "prompt_eval_count": 11,
    "eval_count": 7,
    "content": '{"facts": []}',
}


def test_chat_transport_parses_both_provider_shapes(provider_server) -> None:
    from src.wide_bio import chat_transport

    base, _seen, reply = provider_server
    for body in (OPENAI_BODY, OLLAMA_BODY):
        reply["body"] = body
        assert chat_transport(base + "/x", b"{}", 5) == PARSED
    reply["body"] = {"unexpected": 1}
    parsed = chat_transport(base + "/x", b"{}", 5)
    assert parsed["model"] is None and parsed["content"] == ""
    reply["body"] = {"model": "m", "choices": [], "usage": None}
    assert chat_transport(base + "/x", b"{}", 5)["content"] == ""


def test_openai_proof_config_loads_and_stays_strict(world) -> None:
    root, _, digest, *_ = world
    config = load_proof_config(
        make_config(root, digest, llm={"primary_style": "openai"})
    )
    assert (
        config.backend.style == "openai"
        and config.backend.num_ctx == 262144
        and config.input_budget == 262144 - 2048 - 1024
    )
    for llm, code in (
        ({"primary_style": "vllm"}, "config_invalid"),
        ({"primary_style": "openai", "primary_num_ctx": 32768}, "config_wrong_context"),
        (
            {"primary_style": "openai", "primary_url": "http://localhost:8000"},
            "config_is_fallback",
        ),
        ({"primary_style": "openai", "primary_model": "x-8b"}, "config_is_fallback"),
    ):
        with pytest.raises(ContractError) as error:
            load_proof_config(make_config(root, digest, llm=llm))
        assert error.value.code == code


def test_openai_run_uses_chat_completions_and_rejects_fallback(
    world, provider_server
) -> None:
    from src.wide_bio import chat_transport

    root, _, digest, *_ = world
    base, seen, reply = provider_server
    openai_path = make_config(root, digest, llm={"primary_style": "openai"})
    config = load_proof_config(openai_path)
    _, plan, chapters = one_chunk(world)
    cal = calibration(config, config.tokenizer_sha256)

    def transport(url: str, payload: bytes, timeout: float) -> dict:
        assert url.endswith("/v1/chat/completions")
        return chat_transport(base + "/v1/chat/completions", payload, timeout)

    reply["body"] = OPENAI_BODY | {
        "choices": [
            {"message": {"content": '{"facts": []}'}, "finish_reason": "stop"}
        ]
    }
    out = root / "oa"
    summary = run_extraction(chapters, config, plan, out, transport, cal)
    assert summary["state"] == "ready"
    assert seen and seen[0][0] == "/v1/chat/completions"
    assert "options" not in seen[0][1] and seen[0][1]["model"] == "proof-model:1"
    meta = json.loads((out / "raw" / "k0000.meta.json").read_text())
    assert meta["prompt_eval_count"] == 11 and meta["eval_count"] == 7
    count = len(seen)
    reply["body"] = OPENAI_BODY | {"model": "qwen3:8b"}
    with pytest.raises(ContractError) as error:
        run_extraction(chapters, config, plan, root / "oafb", transport, cal)
    assert error.value.code == "fallback_route"
    assert len(seen) == count + 1
