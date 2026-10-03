"""Real-file tests for the resumable wide-bio delta runner (tiny real byte-level BPE vocabulary, real temp source, no model, no network)."""

import json
import re
import shutil
import uuid
from pathlib import Path

import pytest

from src.cast_freeze import MANIFEST_NAME, RecoveryLedger, prepare_cast
from src.epub_extract import get_output_dir
from src.wide_bio import (
    ContractError,
    build_counter,
    load_proof_config,
    project_chapters,
)
from src.wide_bio_delta import (
    DeltaSettings,
    WideBioHook,
    apply_to_cast,
    build_delta_plan,
    delta_schema,
    main,
    pinned_seed,
    reconstruct_quote,
    run_delta,
)
from src.wide_bio_test import calibration, capture_file, make_config

TEXT = (
    "Ren was tall, with green eyes.\n\n"
    "Later Ren wore a red cloak and was twelve years old.\n\n"
    "Luna Starwaver, the Master, smiled.\n\n"
)
REGISTRY = {
    "ren": {
        "name": "Ren",
        "bio": "original bio",
        "look": "original look",
        "facts": {"voice": ["calm"], "look": []},
    }
}
ALIASES = {"ren": "ren"}


def fact(name, ref, category, value, pid):
    return {
        "subject": {"name": name, "ref": ref},
        "category": category,
        "value": value,
        "paragraph_id": pid,
    }


@pytest.fixture
def world(tmp_path: Path):
    _path, digest = capture_file(tmp_path)
    config_path = make_config(tmp_path, digest)
    project = tmp_path / "project"
    (project / "chapters").mkdir(parents=True)
    (project / "chapters" / "01-one.txt").write_text(TEXT, encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text(TEXT, encoding="utf-8")
    config = load_proof_config(config_path)
    chapters = project_chapters(project)
    count = build_counter(config)
    seed = pinned_seed(tmp_path / "out", REGISTRY, ALIASES)
    plan = build_delta_plan(
        TEXT, chapters, config, count, DeltaSettings(1, 512, 1), 0, "seed"
    )
    return (
        tmp_path,
        config_path,
        config,
        chapters,
        count,
        seed,
        plan,
        project,
        source,
        digest,
    )


def scripted(config, calls):
    """One scripted reply per excerpt; the ref of the known actor is read from the real prompt it was offered in."""

    def transport(url: str, payload: bytes, timeout: float) -> dict:
        body = json.loads(payload)
        user = body["messages"][1]["content"]
        chunk = re.search(r"Excerpt (k\d+)", user).group(1)
        calls.append((chunk, user, body["format"]))
        facts = {
            "k0000": [
                fact("Ren", "ren", "look", "tall, with green eyes", "000000"),
                fact("Ren", "ren", "mood", "x", "000000"),
            ],
            "k0001": [
                fact("Ren", "ren", "look", "tall, with green eyes", "000001"),
                fact("Ren", "ren", "look", "a red cloak", "000001"),
                fact("Ren", "ren", "age", "twelve years old", "000001"),
            ],
            "k0002": [
                fact("Luna Starwaver", "novel", "role", "the Master", "000002"),
                fact("Luna Starwaver", "novel", "alias", "Master", "000002"),
                fact("Master", "novel", "role", "Master", "000002"),
                fact("Luna Starwaver", "novel", "age", "ninety winters", "000002"),
                fact("Someone", "ambiguous", "look", "smiled", "000002"),
            ],
        }[chunk]
        return {
            "model": config.backend.model,
            "done_reason": "stop",
            "prompt_eval_count": 7,
            "eval_count": 3,
            "content": json.dumps({"facts": facts}),
        }

    return transport


def test_plan_covers_every_paragraph_once_in_token_chunks(world) -> None:
    _, _, config, chapters, count, _seed, _, _, _, _ = world
    one = build_delta_plan(
        TEXT, chapters, config, count, DeltaSettings(100000, 512, 2), 0, "s"
    )
    many = build_delta_plan(
        TEXT, chapters, config, count, DeltaSettings(1, 512, 2), 0, "s"
    )
    assert len(one.chunks) == 1 and len(many.chunks) == 3
    for plan in (one, many):
        ids = [p.id for chunk in plan.chunks for p in chunk.paragraphs]
        assert ids == ["000000", "000001", "000002"]
        assert "".join(TEXT[c.start : c.end] for c in plan.chunks) == TEXT
    assert many.chunks[0].oversize and not one.chunks[0].oversize
    assert one.artifact["plan_sha256"] != many.artifact["plan_sha256"]
    with pytest.raises(ContractError):
        DeltaSettings(0)


def test_schema_is_per_chunk_and_short(world) -> None:
    schema = delta_schema(["000001"], ["ren"])
    item = schema["properties"]["facts"]["items"]["properties"]
    assert item["paragraph_id"]["enum"] == ["000001"] and item["subject"]["properties"][
        "ref"
    ]["enum"] == ["novel", "ambiguous", "ren"]
    assert "quote" not in item and set(item["category"]["enum"]) == {
        "look",
        "role",
        "gender",
        "kin",
        "beast",
        "age",
        "alias",
    }
    assert delta_schema([], [])["properties"]["facts"]["maxItems"] == 0


def test_run_is_exact_once_delta_prompted_typed_and_resumable(world) -> None:
    root, _, config, chapters, count, seed, plan, *_ = world
    out = root / "out"
    calls = []
    cal = calibration(config, config.tokenizer_sha256)
    summary = run_delta(
        chapters,
        config,
        plan,
        out,
        scripted(config, calls),
        count,
        cal,
        registry=seed["registry"],
        aliases=seed["aliases"],
    )
    assert (
        summary["state"] == "ready" and summary["calls_total"] == 3 and len(calls) == 3
    )
    assert summary["source_fraction"] == 1.0
    # every chunk shows its own source paragraph; the established section lists Ren and already cited traits
    for chunk, user, schema in calls:
        assert "[[P " in user and "Source paragraphs:" in user
    assert "[ren] Ren" in calls[1][1] and "look: tall, with green eyes" in calls[1][1]
    assert calls[1][2]["properties"]["facts"]["items"]["properties"]["paragraph_id"][
        "enum"
    ] == ["000001"]
    claims = [json.loads(x) for x in (out / "claims.jsonl").read_text().splitlines()]
    held = [json.loads(x) for x in (out / "pending.jsonl").read_text().splitlines()]
    by = {(c["subject"], c["category"], c["value"]) for c in claims}
    assert ("Ren", "look", "a red cloak") in by and (
        "Ren",
        "age",
        "twelve years old",
    ) in by
    assert ("Ren", "look", "tall, with green eyes") in by  # first appearance is new
    assert ("Luna Starwaver", "alias", "Master") in by
    red = next(c for c in claims if c["value"] == "a red cloak")
    assert red[
        "quote"
    ] == "Later Ren wore a red cloak and was twelve years old." and TEXT[
        red["source_offset"] :
    ].startswith(red["quote"])
    assert red["witness"]["chapter"] == "01-one.txt"
    reasons = sorted(item["pending_reason"] for item in held)
    assert {
        "unsupported_category",
        "role_scope_uncertain",
        "unsupported_value",
        "ambiguous_subject",
    } <= set(reasons)
    assert summary["duplicates"] == 0 and reasons.count("unsupported_value") == 2
    budget = json.loads((out / "call_budget.json").read_text())
    assert (
        budget["max_model_calls"] == 3
        and budget["calls_used"] == 3
        and "none" in budget["speed_claim"]
    )
    assert (out / "raw" / "k0000.a1.response.txt").is_file() and (
        out / "journal.jsonl"
    ).is_file()

    def never(*a):
        raise AssertionError("a saved chunk must never be re-asked")

    again = run_delta(
        chapters,
        config,
        plan,
        out,
        never,
        count,
        cal,
        registry=seed["registry"],
        aliases=seed["aliases"],
    )
    assert (
        again["state"] == "ready"
        and again["calls_this_invocation"] == 0
        and again["claims"] == summary["claims"]
    )


def test_deadline_is_durable_not_ready_then_resumes_without_reasking(world) -> None:
    root, _, config, chapters, count, seed, plan, *_ = world
    out = root / "late"
    calls = []
    cal = calibration(config, config.tokenizer_sha256)
    first = run_delta(
        chapters,
        config,
        plan,
        out,
        scripted(config, calls),
        count,
        cal,
        clock=lambda: 200.0 * len(calls),
        registry=seed["registry"],
        aliases=seed["aliases"],
    )
    assert (
        first["state"] == "not_ready"
        and first["reason"] == "deadline"
        and 0 < first["source_fraction"] < 1
    )
    saved = json.loads((out / "summary.json").read_text())
    assert saved["state"] == "not_ready" and saved["calls_total"] == len(calls) == 1
    done = run_delta(
        chapters,
        config,
        plan,
        out,
        scripted(config, calls),
        count,
        cal,
        registry=seed["registry"],
        aliases=seed["aliases"],
    )
    assert done["state"] == "ready" and len(calls) == 3 and done["calls_total"] == 3


def test_transport_error_truncation_and_fallback_never_ready(world) -> None:
    root, _, config, chapters, count, seed, plan, *_ = world
    cal = calibration(config, config.tokenizer_sha256)

    def boom(*a):
        raise OSError("down")

    err = run_delta(
        chapters,
        config,
        plan,
        root / "e",
        boom,
        count,
        cal,
        registry=seed["registry"],
        aliases=seed["aliases"],
    )
    assert (
        err["state"] == "not_ready"
        and err["reason"] == "transport_error"
        and err["calls_total"] == 1
    )
    cut = run_delta(
        chapters,
        config,
        plan,
        root / "c",
        lambda u, p, t: {
            "model": config.backend.model,
            "done_reason": "length",
            "content": '{"facts":[]}',
        },
        count,
        cal,
        registry=seed["registry"],
        aliases=seed["aliases"],
    )
    assert (
        cut["state"] == "not_ready"
        and cut["source_fraction"] == 0.0
        and "truncated" in cut["chunk_status"]
    )
    with pytest.raises(ContractError) as caught:
        run_delta(
            chapters,
            config,
            plan,
            root / "f",
            lambda u, p, t: {"model": "qwen3:8b", "content": "{}"},
            count,
            cal,
            registry=seed["registry"],
            aliases=seed["aliases"],
        )
    assert caught.value.code == "fallback_route"
    assert json.loads((root / "f" / "summary.json").read_text())["state"] == "not_ready"
    offline = run_delta(
        chapters,
        config,
        plan,
        root / "o",
        None,
        count,
        None,
        registry=seed["registry"],
        aliases=seed["aliases"],
    )
    assert (
        offline["state"] == "not_ready"
        and offline["reason"] == "offline"
        and offline["calls_total"] == 0
    )


def test_journal_torn_tail_is_quarantined_and_tampering_refused(world) -> None:
    root, _, config, chapters, count, seed, plan, *_ = world
    out = root / "j"
    calls = []
    cal = calibration(config, config.tokenizer_sha256)
    kw = {"registry": seed["registry"], "aliases": seed["aliases"]}
    run_delta(
        chapters,
        config,
        plan,
        out,
        scripted(config, calls),
        count,
        cal,
        clock=lambda: 200.0 * len(calls),
        **kw,
    )
    with (out / "journal.jsonl").open("ab") as stream:
        stream.write(b'{"seq": 1, "torn')
    done = run_delta(
        chapters, config, plan, out, scripted(config, calls), count, cal, **kw
    )
    assert done["state"] == "ready" and list(out.glob("journal.torn-*"))
    lines = (out / "journal.jsonl").read_text().splitlines()
    (out / "journal.jsonl").write_text(
        "\n".join([lines[0].replace('"attempt":1', '"attempt":1,"x":1'), *lines[1:]])
        + "\n"
    )
    with pytest.raises(ContractError) as caught:
        run_delta(
            chapters, config, plan, out, scripted(config, calls), count, cal, **kw
        )
    assert caught.value.code == "journal_corrupt"


def test_apply_to_cast_preserves_originals_and_requires_source_bridge(world) -> None:
    root, _, config, chapters, count, seed, plan, *_ = world
    out = root / "ap"
    calls = []
    run_delta(
        chapters,
        config,
        plan,
        out,
        scripted(config, calls),
        count,
        calibration(config, config.tokenizer_sha256),
        registry=seed["registry"],
        aliases=seed["aliases"],
    )
    claims = [json.loads(x) for x in (out / "claims.jsonl").read_text().splitlines()]
    registry = json.loads(json.dumps(REGISTRY))
    aliases = dict(ALIASES)
    outcome = apply_to_cast(registry, aliases, claims)
    ren = registry["ren"]
    assert (ren["name"], ren["bio"], ren["look"]) == (
        "Ren",
        "original bio",
        "original look",
    )
    assert (
        ren["facts"]["voice"] == ["calm"]
        and "a red cloak" in ren["facts"]["look"]
        and ren["facts"]["age"] == ["twelve years old"]
    )
    assert (
        registry["luna_starwaver"]["origin"] == "prepared"
        and aliases["master"] == "luna_starwaver"
    )
    assert outcome["pending"] == []
    assert (
        apply_to_cast(registry, aliases, claims)["applied"] == outcome["applied"]
        and registry["ren"]["facts"]["look"].count("a red cloak") == 1
    )
    bare = dict(
        claims[0]
        | {
            "category": "alias",
            "subject": "Ren",
            "subject_id": "ren",
            "value": "Ronny",
            "quote": "Ren is Ronny.",
        }
    )
    held = apply_to_cast(registry, aliases, [bare])
    assert [i["pending_reason"] for i in held["pending"]] == [
        "alias_unbridged"
    ] and "ronny" not in aliases


def test_reconstruct_quote_is_the_containing_sentence() -> None:
    text = "One. Ren wore a red cloak, truly. Two."
    assert reconstruct_quote(text, "a red cloak") == "Ren wore a red cloak, truly."
    assert reconstruct_quote("x" * 400 + " a red cloak", "a red cloak") == "a red cloak"


def test_hook_in_prepare_cast_applies_only_when_ready_and_never_freezes_on_not_ready(
    world,
) -> None:
    root, config_path, config, chapters, *_ = world
    out = root / "hook"
    hook = WideBioHook(
        config_path,
        out,
        DeltaSettings(1, 512, 1),
        root / "cal.json",
        True,
        scripted(config, calls := []),
    )
    (root / "cal.json").write_text(
        json.dumps(calibration(config, config.tokenizer_sha256)), encoding="utf-8"
    )
    project = root / "hproj"
    project.mkdir()
    progress = {"registry": json.loads(json.dumps(REGISTRY)), "aliases": dict(ALIASES)}
    recovery = RecoveryLedger(project)
    result = hook(project, chapters, progress, TEXT, recovery)
    assert (
        result["state"] == "applied"
        and len(calls) == 3
        and progress["wide_bio"] == result
    )
    assert (
        progress["registry"]["ren"]["bio"] == "original bio"
        and "luna_starwaver" in progress["registry"]
    )
    rows = recovery.open_pending("cast")
    assert rows and all(r["code"].startswith("wide_bio_pending_") for r in rows)
    assert (
        hook(project, chapters, progress, TEXT, recovery) == result and len(calls) == 3
    )
    offline = WideBioHook(config_path, root / "off", DeltaSettings(1, 512, 1))
    assert (
        offline(project, chapters, {"registry": {}, "aliases": {}}, TEXT, recovery)[
            "state"
        ]
        == "not_ready"
    )


def test_prepare_cast_pathway_returns_preparing_when_wide_bio_not_ready(world) -> None:
    root, config_path, *_ = world
    source = root / f"book-{uuid.uuid4().hex}.txt"
    plain = "it rained all day.\n"
    source.write_text(plain, encoding="utf-8")
    project = get_output_dir(source)
    try:
        (project / "chapters").mkdir(parents=True)
        (project / "chapters" / "00-intro.txt").write_text(
            "Book by Tester, narrated by Narrator", encoding="utf-8"
        )
        (project / "chapters" / "01-plain.txt").write_text(plain, encoding="utf-8")
        from src.cast_freeze import ANCHOR_IDS

        (project / "characters.json").write_text(
            json.dumps({a: {"name": a, "bio": "", "look": ""} for a in ANCHOR_IDS}),
            encoding="utf-8",
        )

        def never(*a, **k):
            raise AssertionError(
                "no cast model call is needed for a source without names"
            )

        hook = WideBioHook(config_path, root / "pc", DeltaSettings(1000, 512, 1))
        result = prepare_cast(source, ask=never, wide_bio=hook)
        assert (
            result["status"] == "preparing"
            and result["wide_bio"]["state"] == "not_ready"
        )
        assert not (project / MANIFEST_NAME).exists()
    finally:
        shutil.rmtree(project, ignore_errors=True)


def test_cli_plan_writes_exact_call_budget_and_run_needs_execute(
    world, capsys: pytest.CaptureFixture
) -> None:
    root, config_path, _, _, _, _, _, project, source, _ = world
    base = [
        str(project),
        "--source",
        str(source),
        "--config",
        str(config_path),
        "--out",
        str(root / "cli"),
        "--target-tokens",
        "1",
        "--max-attempts",
        "2",
    ]
    assert main(["plan", *base]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["chunks"] == 3 and printed["max_model_calls"] == 6
    budget = json.loads((root / "cli" / "call_budget.json").read_text())
    assert (
        budget["max_model_calls"] == 6
        and budget["calls_used"] == 0
        and len(budget["per_chunk"]) == 3
    )
    assert main(["validate", *base]) == 3
    assert json.loads(capsys.readouterr().out)["state"] == "not_ready"
    cal = root / "cal.json"
    cal.write_text("{}", encoding="utf-8")
    assert main(["run", *base, "--calibration", str(cal)]) == 2
    assert "execute_required" in capsys.readouterr().err
