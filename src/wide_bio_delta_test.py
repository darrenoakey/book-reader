"""Real-file tests for the resumable wide-bio delta runner (tiny real byte-level BPE vocabulary, real temp source, no model, no network)."""

import contextlib
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
    sha256_text,
)
from src.wide_bio_delta import (
    DELTA_VERSION,
    PENDING_REASONS,
    SYSTEM_PROMPT,
    VALUE_MAX,
    DeltaSettings,
    Established,
    WideBioHook,
    apply_to_cast,
    build_delta_plan,
    delta_schema,
    main,
    pinned_seed,
    prepare,
    reconstruct_quote,
    run_delta,
    validate_delta_response,
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
        "voice",
        "power",
    }
    assert "at most once" in schema["properties"]["facts"]["description"]
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


# ##################################################################
# wide_bio_required gate (prepare_cast integration): real project on disk, real hook; the legacy `ask` raises if it is ever called
def forbidden_ask(*_args, **_kwargs):
    raise AssertionError(
        "the legacy scanner must never be asked in wide_bio_required mode"
    )


def one_chunk_transport(config, facts, calls):
    def transport(url: str, payload: bytes, timeout: float) -> dict:
        calls.append(json.loads(payload)["messages"][1]["content"])
        return {
            "model": config.backend.model,
            "done_reason": "stop",
            "prompt_eval_count": 7,
            "eval_count": 3,
            "content": json.dumps({"facts": facts}),
        }

    return transport


@contextlib.contextmanager
def required_book(root: Path, text: str):
    from src.cast_freeze import ANCHOR_IDS

    source = root / f"book-{uuid.uuid4().hex}.txt"
    source.write_text(text, encoding="utf-8")
    project = get_output_dir(source)
    try:
        (project / "chapters").mkdir(parents=True)
        (project / "chapters" / "00-intro.txt").write_text(
            "Book by Tester, narrated by Narrator", encoding="utf-8"
        )
        (project / "chapters" / "01-one.txt").write_text(text, encoding="utf-8")
        profiles = {a: {"name": a, "bio": "", "look": ""} for a in ANCHOR_IDS}
        profiles["ren"] = {
            "name": "Ren",
            "bio": "original bio",
            "look": "original look",
        }
        (project / "characters.json").write_text(json.dumps(profiles), encoding="utf-8")
        yield source, project
    finally:
        shutil.rmtree(project, ignore_errors=True)


def required_run(root, config_path, config, source, facts, ask=forbidden_ask):
    (root / "cal.json").write_text(
        json.dumps(calibration(config, config.tokenizer_sha256)), encoding="utf-8"
    )
    calls: list[str] = []
    hook = WideBioHook(
        config_path,
        root / f"req-{uuid.uuid4().hex}",
        DeltaSettings(100000, 512, 1),
        root / "cal.json",
        True,
        one_chunk_transport(config, facts, calls),
    )
    result = prepare_cast(source, ask=ask, wide_bio=hook, wide_bio_required=True)
    return result, calls


def seed_pending(
    project: Path, label: str, quote: str, chapter: Path, mention_hash=None
):
    """A legacy typed-identity pending row exactly as the scanner records it (mention scope + source hashes)."""
    from src.cast_freeze import file_digest

    recovery = RecoveryLedger(project)
    recovery.record(
        "cast",
        f"chapters 1-1:{label.lower()}",
        "pending_new_identity",
        f"new_identity review of {label!r} is pending",
        severity="pending",
        evidence={
            "type": "new_identity",
            "label": label,
            "reason": "x",
            "mentions": [
                {
                    "chapter": chapter.name,
                    "chapter_sha256": sha256_text(chapter.read_text(encoding="utf-8")),
                    "quote_sha256": mention_hash or sha256_text(quote),
                    "label": label,
                    "span_start": 0,
                }
            ],
            "source": [chapter.name],
            "source_hash": {chapter.name: file_digest(chapter)},
        },
    )


def test_required_complete_applies_early_closes_exact_scope_only_and_never_asks(
    world,
) -> None:
    root, config_path, config, *_ = world
    with required_book(root, "Ren wore a red cloak.\n") as (source, project):
        chapter = project / "chapters" / "01-one.txt"
        seed_pending(project, "Ren", "Ren wore a red cloak.", chapter)
        seed_pending(
            project, "Zed", "Zed ran.", chapter
        )  # no claim proves this mention
        facts = [fact("Ren", "ren", "look", "a red cloak", "000000")]
        result, calls = required_run(root, config_path, config, source, facts)
        progress = json.loads((project / "cast_preparation_progress.json").read_text())
        assert len(calls) == 1 and not (project / MANIFEST_NAME).exists()
        assert progress["wide_bio"]["accounting"]["complete"] is True
        assert progress["wide_bio"]["closed_pending"] == [
            {"item": "chapters 1-1:ren", "code": "pending_new_identity"}
        ]
        # early application: original profile kept, fact appended, cursors advanced from the accounting
        ren = progress["registry"]["ren"]
        assert ren["bio"] == "original bio" and ren["look"] == "original look"
        assert ren["facts"]["look"] == ["a red cloak"]
        assert (
            progress["next_chapter"]
            == progress["semantic_coverage"]["next_chapter"]
            == 1
        )
        # only the exactly proven row was closed; the unproven one stays open and blocks
        open_items = [r["item"] for r in RecoveryLedger(project).open_pending("cast")]
        assert open_items == ["chapters 1-1:zed"]
        assert result["status"] == "blocked" and result["pending"] == 1


def test_required_mode_allows_openai_primary_without_a_legacy_ask(world) -> None:
    root, config_path, config, *_ = world
    with required_book(root, "Ren wore a red cloak.\n") as (source, project):
        chapter = project / "chapters" / "01-one.txt"
        seed_pending(project, "Zed", "Zed ran.", chapter)
        facts = [fact("Ren", "ren", "look", "a red cloak", "000000")]
        result, calls = required_run(root, config_path, config, source, facts, ask=None)
        assert len(calls) == 1 and result["status"] == "blocked"


def test_required_invalid_claim_is_retained_pending_with_hashes_and_not_applied(
    world,
) -> None:
    root, config_path, config, *_ = world
    with required_book(root, "Ren wore a red cloak.\n\nMara smiled.\n") as (
        source,
        project,
    ):
        facts = [
            fact("Ren", "ren", "look", "a red cloak", "000000"),
            fact("Mara", "ambiguous", "look", "smiled", "000001"),
        ]
        result, _ = required_run(root, config_path, config, source, facts)
        progress = json.loads((project / "cast_preparation_progress.json").read_text())
        rows = RecoveryLedger(project).open_pending("cast")
        assert [r["code"] for r in rows] == ["wide_bio_pending_ambiguous_subject"]
        evidence = rows[0]["evidence"]
        assert (
            evidence["plan_sha256"]
            and evidence["source_hash"]
            and evidence["fact_sha256"]
        )
        assert "mara" not in progress["registry"] and "mara" not in progress["aliases"]
        assert progress["wide_bio"]["accounting"]["pending"] == 1
        assert result["status"] == "blocked" and not (project / MANIFEST_NAME).exists()


def test_required_blocks_without_native_candidate_accounting(world) -> None:
    root, config_path, config, *_ = world
    with required_book(root, "Ren wore a red cloak.\n\nMara smiled.\n") as (
        source,
        project,
    ):
        facts = [fact("Ren", "ren", "look", "a red cloak", "000000")]
        result, _ = required_run(root, config_path, config, source, facts)
        progress = json.loads((project / "cast_preparation_progress.json").read_text())
        assert result["status"] == "blocked"
        assert result["reason"] == "wide_bio_required_candidate_accounting_incomplete"
        assert [
            u["label"] for u in result["wide_bio"]["accounting"]["unaccounted"]
        ] == ["Mara"]
        assert progress["next_chapter"] == 0 and not (project / MANIFEST_NAME).exists()


def test_required_offline_scan_is_preparing_and_never_asks(world) -> None:
    root, config_path, *_ = world
    with required_book(root, "Ren wore a red cloak.\n") as (source, project):
        hook = WideBioHook(config_path, root / "off2", DeltaSettings(100000, 512, 1))
        result = prepare_cast(
            source, ask=forbidden_ask, wide_bio=hook, wide_bio_required=True
        )
        assert (
            result["status"] == "preparing"
            and result["wide_bio"]["state"] == "not_ready"
        )
        assert not (project / MANIFEST_NAME).exists()


# ##################################################################
# adaptive output retry: a finished base-cap run is reused read-only; only a raw-proven length chunk is retried once at the larger cap
def tree(path: Path) -> dict:
    return {
        str(item.relative_to(path)): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def adaptive_transport(config, calls, retry_reply):
    """Real prompts, scripted model: k0001 hits the base cap (done_reason length, eval_count == cap); the 8192 reply is `retry_reply`."""
    inner = scripted(config, [])

    def transport(url: str, payload: bytes, timeout: float) -> dict:
        body = json.loads(payload)
        cap = body["options"]["num_predict"]
        chunk = re.search(r"Excerpt (k\d+)", body["messages"][1]["content"]).group(1)
        calls.append((chunk, cap))
        if chunk == "k0001" and cap == config.output_tokens:
            return {
                "model": config.backend.model,
                "done_reason": "length",
                "prompt_eval_count": 7,
                "eval_count": cap,
                "content": '{"facts":[{"subject":{"name":"Ren"',
            }
        if chunk == "k0001":
            return retry_reply(inner, url, payload, timeout, cap)
        return inner(url, payload, timeout)

    return transport


def stop_reply(inner, url, payload, timeout, cap):
    return inner(url, payload, timeout)


def length_reply(config):
    def reply(inner, url, payload, timeout, cap):
        return {
            "model": config.backend.model,
            "done_reason": "length",
            "prompt_eval_count": 7,
            "eval_count": cap,
            "content": '{"facts":[{"subject"',
        }

    return reply


@pytest.fixture
def parent_run(world):
    root, config_path, config, _, _, _, _, project, source, digest = world
    cal = root / "cal.json"
    cal.write_text(json.dumps(calibration(config, digest)), encoding="utf-8")
    calls: list = []
    parent = root / "parent"
    base = [
        str(project),
        "--source",
        str(source),
        "--config",
        str(config_path),
        "--target-tokens",
        "1",
        "--delta-tokens",
        "512",
        "--calibration",
        str(cal),
    ]
    code = main(
        ["run", *base, "--out", str(parent), "--execute"],
        adaptive_transport(config, calls, stop_reply),
    )
    assert code == 3 and [c[1] for c in calls] == [config.output_tokens] * 3
    return root, config, base, parent


def adaptive_args(base, parent, out, *extra):
    return ["adaptive", *base, "--from-out", str(parent), "--out", str(out), *extra]


def test_adaptive_partial_parent_reuses_saved_rows_retries_length_and_continues_missing(
    parent_run, capsys
) -> None:
    root, config, base, parent = parent_run
    partial = root / "partial-parent"
    before = tree(parent)
    shutil.copytree(parent, partial)
    records = (partial / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    # A real bounded pass has only a valid hash-chain prefix, not invented records
    # for the source chunks it has not reached yet.
    (partial / "journal.jsonl").write_text(
        "\n".join(records[:2]) + "\n", encoding="utf-8"
    )
    for suffix in (".response.txt", ".meta.json"):
        (partial / "raw" / f"k0002.a1{suffix}").unlink()
    calls: list = []
    assert (
        main(
            adaptive_args(base, partial, root / "adaptive", "--execute"),
            adaptive_transport(config, calls, stop_reply),
        )
        == 0
    )
    assert calls == [("k0001", 8192), ("k0002", config.output_tokens)]
    assert tree(parent) == before


def test_adaptive_offline_reuses_parent_without_calls_or_mutation(
    parent_run, capsys
) -> None:
    root, _config, base, parent = parent_run
    before = tree(parent)
    capsys.readouterr()
    assert main(adaptive_args(base, parent, root / "ad")) == 3
    summary = json.loads(capsys.readouterr().out)
    assert summary["chunk_status"] == {"ready": 2, "truncated": 1}
    assert summary["calls_this_invocation"] == 0 and summary["calls_total"] == 3
    assert tree(parent) == before
    ledger = json.loads((root / "ad" / "plan.json").read_text())["adaptive"]
    assert ledger["parent_chunks"]["k0001"]["server"]["done_reason"] == "length"
    assert ledger["retry_output_tokens"] == 8192


def test_adaptive_retries_only_proven_length_chunk_once_at_8192(
    parent_run, capsys
) -> None:
    root, config, base, parent = parent_run
    before = tree(parent)
    calls: list = []
    transport = adaptive_transport(config, calls, stop_reply)
    out = root / "ad"
    assert main(adaptive_args(base, parent, out, "--execute"), transport) == 0
    assert calls == [("k0001", 8192)]
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["state"] == "ready" and summary["calls_this_invocation"] == 1
    assert tree(parent) == before
    # ready chunks were reused from the parent raw files: their journal records are byte-identical
    old = (parent / "journal.jsonl").read_text().splitlines()
    new = (out / "journal.jsonl").read_text().splitlines()
    assert new[: len(old)] == old and len(new) == len(old) + 1
    assert json.loads(new[-1])["attempt"] == 2
    # a rerun neither re-asks nor changes anything
    assert main(adaptive_args(base, parent, out, "--execute"), transport) == 0
    assert calls == [("k0001", 8192)]


def test_adaptive_length_at_8192_stays_typed_pending_and_is_never_retried(
    parent_run, capsys
) -> None:
    root, config, base, parent = parent_run
    calls: list = []
    transport = adaptive_transport(config, calls, length_reply(config))
    out = root / "ad"
    assert main(adaptive_args(base, parent, out, "--execute"), transport) == 3
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["reason"] == "output_length_exhausted"
    assert summary["length_exhausted"] == ["k0001"] and summary["claims"] > 0
    assert main(adaptive_args(base, parent, out, "--execute"), transport) == 3
    assert calls == [("k0001", 8192)]


def test_adaptive_length_without_exact_cap_eval_count_is_not_retried(
    parent_run, capsys
) -> None:
    root, config, base, parent = parent_run
    forged = root / "forged"
    shutil.copytree(parent, forged)
    for meta_path in (forged / "raw").glob("k0001.a1.meta.json"):
        meta = json.loads(meta_path.read_text())
        meta["eval_count"] = config.output_tokens - 1
        meta_path.write_text(json.dumps(meta, sort_keys=True))
    capsys.readouterr()
    calls: list = []
    transport = adaptive_transport(config, calls, stop_reply)
    # the edited meta no longer reproduces the parent journal's meta hash, so it is refused outright
    assert main(adaptive_args(base, forged, root / "ad", "--execute"), transport) == 2
    assert "journal_inconsistent" in capsys.readouterr().err and calls == []


@pytest.mark.parametrize("target", ["response", "meta_request", "journal", "plan"])
def test_adaptive_refuses_tampered_or_mismatched_parent(
    parent_run, capsys, target
) -> None:
    root, config, base, parent = parent_run
    bad = root / "bad"
    shutil.copytree(parent, bad)
    if target == "response":
        path = bad / "raw" / "k0000.a1.response.txt"
        path.write_text(path.read_text() + " ")
    elif target == "meta_request":
        path = bad / "raw" / "k0002.a1.meta.json"
        meta = json.loads(path.read_text())
        meta["request_sha256"] = "0" * 64
        path.write_text(json.dumps(meta, sort_keys=True))
    elif target == "journal":
        path = bad / "journal.jsonl"
        path.write_bytes(path.read_bytes()[:-5])
    else:
        path = bad / "plan.json"
        plan = json.loads(path.read_text())
        plan["system_prompt_sha256"] = "0" * 64
        path.write_text(json.dumps(plan))
    capsys.readouterr()
    calls: list = []
    code = main(
        adaptive_args(base, bad, root / "ad", "--execute"),
        adaptive_transport(config, calls, stop_reply),
    )
    assert code == 2 and calls == []
    assert json.loads(capsys.readouterr().err)["refused"] in {
        "journal_inconsistent",
        "adaptive_parent_invalid",
        "adaptive_parent_mismatch",
    }


def test_adaptive_refuses_other_chunking_and_output_inside_parent(
    parent_run, capsys
) -> None:
    root, config, base, parent = parent_run
    other = list(base)
    other[other.index("--target-tokens") + 1] = "100000"
    capsys.readouterr()
    calls: list = []
    transport = adaptive_transport(config, calls, stop_reply)
    assert main(adaptive_args(other, parent, root / "ad", "--execute"), transport) == 2
    assert json.loads(capsys.readouterr().err)["refused"] == "adaptive_parent_mismatch"
    assert main(adaptive_args(base, parent, parent, "--execute"), transport) == 2
    assert (
        main(adaptive_args(base, parent, parent / "sub", "--execute"), transport) == 2
    )
    assert calls == []


def test_adaptive_input_budget_guard_uses_retry_cap_and_margin(parent_run) -> None:
    root, config, base, parent = parent_run
    out = root / "ad"
    assert main(adaptive_args(base, parent, out)) == 3
    plan = json.loads((out / "plan.json").read_text())
    assert plan["adaptive"]["retry_margin_tokens"] == 16
    assert (
        config.backend.num_ctx - 8192 - config.reserve_tokens
        > plan["input_budget"] - 8192 - 1
    )
    assert max(c["max_padded_prompt_tokens"] for c in plan["chunks"]) < (
        config.backend.num_ctx - 8192 - config.reserve_tokens
    )


# ##################################################################
# finalize-adaptive: no-provider finalization of a completed adaptive output into the current prepare_cast progress
def adaptive_facts(config, facts_by_chunk, length_chunks, calls, exhaust=()):
    def transport(url: str, payload: bytes, timeout: float) -> dict:
        body = json.loads(payload)
        cap = body["options"]["num_predict"]
        chunk = re.search(r"Excerpt (k\d+)", body["messages"][1]["content"]).group(1)
        calls.append((chunk, cap))
        if chunk in exhaust or (chunk in length_chunks and cap == config.output_tokens):
            return {
                "model": config.backend.model,
                "done_reason": "length",
                "prompt_eval_count": 7,
                "eval_count": cap,
                "content": '{"facts":[{"subject"',
            }
        return {
            "model": config.backend.model,
            "done_reason": "stop",
            "prompt_eval_count": 7,
            "eval_count": 3,
            "content": json.dumps({"facts": facts_by_chunk.get(chunk, [])}),
        }

    return transport


def build_adaptive(
    root, config_path, config, source, project, facts, length=(), exhaust=()
):
    """A real parent run + a real adaptive run (the only calls ever made), both through the CLI."""
    cal = root / "cal.json"
    cal.write_text(
        json.dumps(calibration(config, config.tokenizer_sha256)), encoding="utf-8"
    )
    base = [
        str(project),
        "--source",
        str(source),
        "--config",
        str(config_path),
        "--target-tokens",
        "1",
        "--delta-tokens",
        "512",
        "--calibration",
        str(cal),
    ]
    calls: list = []
    transport = adaptive_facts(config, facts, set(length), calls, set(exhaust))
    parent, out = root / "fparent", root / "fadaptive"
    # the seed a real run pins is the preparation registry at scan time
    pinned_seed(parent, json.loads(json.dumps(REGISTRY)), dict(ALIASES))
    main(["run", *base, "--out", str(parent), "--execute"], transport)
    code = main(
        ["adaptive", *base, "--from-out", str(parent), "--out", str(out), "--execute"],
        transport,
    )
    return out, code, calls


def finalize_args(project, source, config_path, out):
    return [
        "finalize-adaptive",
        str(project),
        "--source",
        str(source),
        "--config",
        str(config_path),
        "--out",
        str(out),
    ]


@pytest.fixture
def no_network(monkeypatch):
    import socket

    def refuse(*_a, **_k):
        raise AssertionError("finalize-adaptive must never open a connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_finalize_adaptive_applies_to_current_progress_closes_exact_old_pending_and_blocks(
    world, capsys, no_network
) -> None:
    root, config_path, config, *_ = world
    text = "Ren wore a red cloak.\n\nMara smiled.\n"
    with required_book(root, text) as (source, project):
        chapter = project / "chapters" / "01-one.txt"
        seed_pending(project, "Ren", "Ren wore a red cloak.", chapter)
        seed_pending(project, "Zed", "Zed ran.", chapter)
        facts = {
            "k0000": [fact("Ren", "ren", "look", "a red cloak", "000000")],
            "k0001": [fact("Mara", "ambiguous", "look", "smiled", "000001")],
        }
        out, code, calls = build_adaptive(
            root, config_path, config, source, project, facts, length=["k0000"]
        )
        assert code == 0 and ("k0000", 8192) in calls
        journal = (out / "journal.jsonl").read_bytes()
        used = len(calls)
        capsys.readouterr()
        assert main(finalize_args(project, source, config_path, out)) == 3
        result = json.loads(capsys.readouterr().out)
        assert len(calls) == used and (out / "journal.jsonl").read_bytes() == journal
        progress = json.loads((project / "cast_preparation_progress.json").read_text())
        ren = progress["registry"]["ren"]
        assert ren["bio"] == "original bio" and ren["facts"]["look"] == ["a red cloak"]
        assert "mara" not in progress["registry"]
        assert progress["wide_bio"]["closed_pending"] == [
            {"item": "chapters 1-1:ren", "code": "pending_new_identity"}
        ]
        # exact typed pending for the ambiguous subject; the unproven old row also stays open
        rows = RecoveryLedger(project).open_pending("cast")
        assert sorted(r["item"] for r in rows if r["code"].startswith("pending_")) == [
            "chapters 1-1:zed"
        ]
        typed = [r for r in rows if r["code"] == "wide_bio_pending_ambiguous_subject"]
        assert len(typed) == 1 and typed[0]["evidence"]["fact_sha256"]
        assert result["status"] == "blocked" and not (project / MANIFEST_NAME).exists()
        # idempotent over the same progress
        assert main(finalize_args(project, source, config_path, out)) == 3
        assert len(calls) == used


def test_finalize_adaptive_with_candidate_gap_is_blocked_and_never_freezes(
    world, capsys, no_network
) -> None:
    root, config_path, config, *_ = world
    with required_book(root, "Ren wore a red cloak.\n\nMara smiled.\n") as (
        source,
        project,
    ):
        facts = {"k0000": [fact("Ren", "ren", "look", "a red cloak", "000000")]}
        out, code, _ = build_adaptive(root, config_path, config, source, project, facts)
        assert code == 0
        capsys.readouterr()
        assert main(finalize_args(project, source, config_path, out)) == 3
        result = json.loads(capsys.readouterr().out)
        assert result["reason"] == "wide_bio_required_candidate_accounting_incomplete"
        assert [
            u["label"] for u in result["wide_bio"]["accounting"]["unaccounted"]
        ] == ["Mara"]
        assert not (project / MANIFEST_NAME).exists()


def test_finalize_adaptive_complete_and_clean_proceeds_to_freeze_gates(
    world, capsys, no_network
) -> None:
    root, config_path, config, *_ = world
    with required_book(root, "Ren wore a red cloak.\n") as (source, project):
        facts = {"k0000": [fact("Ren", "ren", "look", "a red cloak", "000000")]}
        out, code, _ = build_adaptive(root, config_path, config, source, project, facts)
        assert code == 0
        capsys.readouterr()
        # nothing wide-bio related blocks any more, so prepare_cast proceeds to the freeze gates; the asset
        # stage (voices.json, never present in this tiny project) is the first thing that stops it
        from src.data_recovery import OperationalError

        with pytest.raises(OperationalError, match="voice profiles"):
            main(finalize_args(project, source, config_path, out))
        progress = json.loads((project / "cast_preparation_progress.json").read_text())
        assert progress["wide_bio"]["accounting"]["complete"] is True
        assert progress["wide_bio"]["pending"] == 0
        assert RecoveryLedger(project).open_pending("cast") == []
        assert not (project / MANIFEST_NAME).exists()


def refused(capsys) -> str:
    return json.loads(capsys.readouterr().err)["refused"]


def test_finalize_adaptive_refuses_incomplete_exhausted_and_foreign_outputs(
    world, capsys, no_network
) -> None:
    root, config_path, config, *_ = world
    with required_book(root, "Ren wore a red cloak.\n\nMara smiled.\n") as (
        source,
        project,
    ):
        facts = {"k0000": [fact("Ren", "ren", "look", "a red cloak", "000000")]}
        # k0001 stays length-exhausted at 8192
        out, code, _ = build_adaptive(
            root, config_path, config, source, project, facts, exhaust=["k0001"]
        )
        assert code == 3
        args = finalize_args(project, source, config_path, out)
        capsys.readouterr()
        assert main(args) == 2 and refused(capsys) == "adaptive_output_incomplete"
        assert not (project / "cast_preparation_progress.json").exists()
        # a plain (non-adaptive) run directory is not accepted
        assert main(finalize_args(project, source, config_path, root / "fparent")) == 2
        assert refused(capsys) in {
            "adaptive_output_invalid",
            "adaptive_output_incomplete",
        }
        # a project that is not the source's output directory is refused
        assert main(finalize_args(root, source, config_path, out)) == 2
        assert refused(capsys) == "project_mismatch"


def test_finalize_adaptive_refuses_tampering_missing_exit_and_other_source(
    world, capsys, no_network
) -> None:
    root, config_path, config, *_ = world
    with required_book(root, "Ren wore a red cloak.\n\nMara smiled.\n") as (
        source,
        project,
    ):
        facts = {"k0000": [fact("Ren", "ren", "look", "a red cloak", "000000")]}
        out, code, _ = build_adaptive(
            root, config_path, config, source, project, facts, length=["k0000"]
        )
        assert code == 0
        capsys.readouterr()
        for name in ("k0000.a2.response.txt", "k0001.a1.response.txt"):
            bad = root / f"bad-{name}"
            shutil.copytree(out, bad)
            path = bad / "raw" / name
            path.write_text(path.read_text() + " ")
            assert main(finalize_args(project, source, config_path, bad)) == 2
            assert refused(capsys) == "journal_inconsistent"
        for victim, expected in (
            ("run.exit", "adaptive_output_incomplete"),
            ("seed.json", "adaptive_output_mismatch"),
        ):
            bad = root / f"bad-{victim}"
            shutil.copytree(out, bad)
            if victim == "run.exit":
                (bad / victim).unlink()
            else:
                seed = json.loads((bad / victim).read_text())
                seed["registry"]["ren"]["bio"] = "forged"
                (bad / victim).write_text(json.dumps(seed))
            assert main(finalize_args(project, source, config_path, bad)) == 2
            assert refused(capsys) == expected
        plan_bad = root / "bad-plan"
        shutil.copytree(out, plan_bad)
        plan = json.loads((plan_bad / "plan.json").read_text())
        plan["adaptive"]["retry_output_tokens"] = 4096
        (plan_bad / "plan.json").write_text(json.dumps(plan))
        assert main(finalize_args(project, source, config_path, plan_bad)) == 2
        assert refused(capsys) == "adaptive_output_mismatch"
        # a changed source no longer reproduces the plan
        source.write_text("Ren wore a red cloak.\n\nMara frowned.\n", encoding="utf-8")
        assert main(finalize_args(project, source, config_path, out)) == 2
        assert refused(capsys) in {"adaptive_output_mismatch", "journal_inconsistent"}
        assert not (project / "cast_preparation_progress.json").exists()


# compact-density contract (delta v2)
# captured-style source: dialogue-heavy paragraphs around explicit traits, as in the raw 4096/8192 runs that hit the output cap.
DENSE = (
    "Mira Vale, the harbour pilot, had copper hair and grey eyes, and she was twenty-six years old.\n\n"
    '"I told you we would sail at dawn," said Mira, "and I will not wait for the tide to turn again, whatever the captain says!"\n\n'
    "Tomas was her younger brother and the half-brother of Ren, and he could call lightning and speak with gulls.\n\n"
    "Mira slammed the door, crossed the quay and shouted for the boat to be brought round at once.\n\n"
    "Everyone on the quay knew Mira Vale as the Gull Queen, a gruff and patient woman with a voice like gravel.\n\n"
)


@pytest.fixture
def dense(tmp_path: Path):
    _path, digest = capture_file(tmp_path)
    config = load_proof_config(make_config(tmp_path, digest))
    project = tmp_path / "project"
    (project / "chapters").mkdir(parents=True)
    (project / "chapters" / "01-one.txt").write_text(DENSE, encoding="utf-8")
    chapters = project_chapters(project)
    plan = build_delta_plan(
        DENSE,
        chapters,
        config,
        build_counter(config),
        DeltaSettings(4000, 512, 1),
        0,
        "seed",
    )
    assert len(plan.chunks) == 1
    state = Established()
    state.seed(REGISTRY, ALIASES)
    candidates = {"Mira Vale": False, "Mira": False, "Tomas": False}
    return plan.chunks[0], state, candidates


def verdict(dense, rows):
    chunk, state, candidates = dense
    return validate_delta_response(
        json.dumps({"facts": rows}), chunk, state, [], candidates
    )


def reasons(result):
    return [item["pending_reason"] for item in result["pending"]]


def test_compact_discriminative_json_is_fully_accepted_without_per_category_cap(
    dense,
) -> None:
    rows = [
        fact("Mira Vale", "novel", "role", "harbour pilot", "000000"),
        fact("Mira Vale", "novel", "look", "copper hair", "000000"),
        fact("Mira Vale", "novel", "look", "grey eyes", "000000"),
        fact("Mira Vale", "novel", "age", "twenty-six years old", "000000"),
        fact("Mira Vale", "novel", "alias", "the Gull Queen", "000004"),
        fact("Mira Vale", "novel", "voice", "a voice like gravel", "000004"),
        fact("Mira Vale", "novel", "voice", "gruff and patient", "000004"),
        fact("Tomas", "novel", "kin", "younger brother", "000002"),
        fact("Tomas", "novel", "kin", "half-brother of Ren", "000002"),
        fact("Tomas", "novel", "power", "call lightning", "000002"),
        fact("Tomas", "novel", "power", "speak with gulls", "000002"),
        fact("Tomas", "novel", "power", "speak with gulls", "000002"),
        fact("Mira Vale", "novel", "alias", "Gull Queen", "000004"),
    ]
    result = verdict(dense, rows)
    assert result["pending"] == [], result["pending"]
    assert len(result["claims"]) == 11
    # the same literal trait slot is deduplicated, never a distinct trait
    assert [item["reason"] for item in result["duplicates"]] == [
        "duplicate_in_response"
    ] * 2
    assert {claim["category"] for claim in result["claims"]} >= {
        "look",
        "kin",
        "power",
        "voice",
    }
    assert sum(1 for c in result["claims"] if c["category"] == "look") == 2
    assert sum(1 for c in result["claims"] if c["category"] == "kin") == 2


def test_verbose_dialogue_and_action_rows_are_typed_pending_never_claims(dense) -> None:
    verbose = [
        fact(
            "Mira",
            "novel",
            "role",
            '"I told you we would sail at dawn," said Mira',
            "000001",
        ),
        fact("Mira", "novel", "voice", "whatever the captain says!", "000001"),
        fact(
            "Mira",
            "novel",
            "look",
            "I will not wait for the tide to turn again",
            "000001",
        ),
        fact(
            "Mira",
            "novel",
            "role",
            "slammed the door, crossed the quay and shouted",
            "000003",
        ),
        fact("Mira", "novel", "look", "Mira slammed the door", "000003"),
        fact("Mira", "novel", "power", "shouted for the boat", "000003"),
        fact("Mira", "novel", "kin", "Mira", "000003"),
        fact("Mira", "novel", "role", "was the harbour pilot", "000000"),
        fact("Mira", "novel", "age", "she was twenty-six years old", "000000"),
        fact(
            "Mira",
            "novel",
            "look",
            "Mira Vale, the harbour pilot, had copper hair and grey eyes, and she was",
            "000000",
        ),
        fact(
            "Mira Vale",
            "novel",
            "alias",
            "Everyone on the quay knew Mira Vale as the Gull Queen",
            "000004",
        ),
    ]
    result = verdict(dense, verbose)
    assert result["claims"] == []
    assert set(reasons(result)) <= set(PENDING_REASONS)
    assert len(result["pending"]) == len(verbose)
    assert {
        "dialogue_value",
        "clause_value",
        "value_not_compact",
        "category_incompatible_value",
    } <= set(reasons(result))
    # the valid compact phrase for the same subject/category is still accepted next to the rejected verbose rows
    mixed = verdict(
        dense,
        [
            verbose[0],
            fact("Mira Vale", "novel", "role", "harbour pilot", "000000"),
            verbose[3],
            fact("Mira Vale", "novel", "alias", "the Gull Queen", "000004"),
        ],
    )
    assert [c["value"] for c in mixed["claims"]] == ["harbour pilot", "the Gull Queen"]
    assert reasons(mixed) == ["dialogue_value", "value_not_compact"]


def test_unknown_subject_stays_ambiguous_pending_and_is_never_forced_to_an_actor(
    dense,
) -> None:
    result = verdict(dense, [fact("he", "ambiguous", "look", "grey eyes", "000000")])
    assert result["claims"] == [] and reasons(result) == ["ambiguous_subject"]


def test_plan_fingerprint_changes_and_old_runs_stay_incompatible(world) -> None:
    root, _, config, chapters, count, _seed, plan, *_ = world
    assert DELTA_VERSION == 3 and plan.artifact["delta_version"] == 3
    assert VALUE_MAX <= 100 and "SHORTEST" in SYSTEM_PROMPT
    assert "earliest paragraph id" in SYSTEM_PROMPT and "MINIFIED JSON" in SYSTEM_PROMPT
    assert plan.artifact["compact_contract_sha256"]
    out = root / "old-run"
    out.mkdir()
    legacy = {
        key: value
        for key, value in plan.artifact.items()
        if key != "compact_contract_sha256"
    }
    legacy["delta_version"] = 1
    legacy["plan_sha256"] = "0" * 64
    (out / "plan.json").write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ContractError, match="different plan"):
        run_delta(
            chapters,
            config,
            plan,
            out,
            None,
            count,
            None,
            cast=None,
            registry=REGISTRY,
            aliases=ALIASES,
        )


# ##################################################################
# caretaker sampling proof: sampled plan fingerprint + one-chunk run
# TensorFold's OpenAI endpoint takes top-level `temperature` and `seed` (verified in its request_options); the whitespace grammar bound is a server constant with no request field, so nothing is sent for it.
@pytest.fixture
def sampled(world):
    root, _, _, _, _, _, _, project, source, digest = world
    config_path = make_config(root, digest, llm={"primary_style": "openai"})
    config = load_proof_config(config_path)
    return root, config_path, config, project, source


def sampling_args(sampled, out, *extra):
    _root, config_path, _, project, source = sampled
    return [
        str(project),
        "--source",
        str(source),
        "--config",
        str(config_path),
        "--out",
        str(out),
        "--target-tokens",
        "1",
        "--sampling-temperature",
        "0.15",
        "--sampling-seed",
        "1729",
        *extra,
    ]


def source_tree(path: Path) -> dict:
    return {
        str(item.relative_to(path)): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_sampling_changes_fingerprint_and_is_sent_only_where_openai_supports_it(
    world, sampled
) -> None:
    _, _, _, chapters, count, _seed, plan, *_ = world
    _, _, config, *_ = sampled
    sampled_plan = build_delta_plan(
        TEXT, chapters, config, count, DeltaSettings(1, 512, 1, 0.15, 1729), 0, "seed"
    )
    plain_openai = build_delta_plan(
        TEXT, chapters, config, count, DeltaSettings(1, 512, 1), 0, "seed"
    )
    assert sampled_plan.artifact["sampling"] == {"temperature": 0.15, "seed": 1729}
    assert "sampling" not in plain_openai.artifact
    assert plain_openai.artifact["plan_sha256"] == plan.artifact["plan_sha256"]
    assert (
        sampled_plan.artifact["plan_sha256"] != plain_openai.artifact["plan_sha256"]
    )
    state = Established()
    sent = prepare(config, sampled_plan, sampled_plan.chunks[0], state, count)
    body = json.loads(sent.payload)
    assert body["temperature"] == 0.15 and body["seed"] == 1729
    assert sent.url.endswith("/v1/chat/completions")
    assert not {"format", "options", "top_p", "top_k"} & body.keys()
    unsampled = json.loads(
        prepare(config, plain_openai, plain_openai.chunks[0], state, count).payload
    )
    assert unsampled["temperature"] == 0.0 and "seed" not in unsampled
    # an ollama-style primary has no verified seed field: refused, never silently dropped
    _, _, ollama_config, *_ = world
    with pytest.raises(ContractError, match="openai-style"):
        build_delta_plan(
            TEXT,
            chapters,
            ollama_config,
            count,
            DeltaSettings(1, 512, 1, 0.15, 1729),
            0,
            "seed",
        )
    for bad in ((0.15, None), (None, 1729), (0.0, 1), (3.0, 1), (0.15, True)):
        with pytest.raises(ContractError):
            DeltaSettings(1, 512, 1, *bad)


def test_only_chunk_sends_exactly_the_selected_chunk_with_temperature_and_seed(
    sampled, capsys: pytest.CaptureFixture
) -> None:
    root, _, config, *_ = sampled
    out = root / "proof"
    sent = []

    def transport(url: str, payload: bytes, timeout: float) -> dict:
        sent.append((url, json.loads(payload)))
        return {
            "model": config.backend.model,
            "done_reason": "stop",
            "prompt_eval_count": 7,
            "eval_count": 3,
            "content": json.dumps(
                {"facts": [fact("Ren", "ren", "look", "a red cloak", "000001")]}
            ),
        }

    cal = root / "cal.json"
    cal.write_text(json.dumps(calibration(config, config.tokenizer_sha256)))
    argv = [
        "run",
        *sampling_args(sampled, out, "--calibration", str(cal), "--execute"),
        "--only-chunk",
        "k0001",
    ]
    assert main(argv, transport) == 3
    summary = json.loads(capsys.readouterr().out)
    assert len(sent) == 1
    _url, body = sent[0]
    assert body["temperature"] == 0.15 and body["seed"] == 1729
    assert "Excerpt k0001" in body["messages"][1]["content"]
    assert "Excerpt k0000" not in body["messages"][1]["content"]
    assert summary["state"] == "not_ready" and summary["reason"] == "only_chunk_proof"
    assert summary["only_chunk"] == "k0001" and summary["calls_total"] == 1
    assert summary["chunk_status"] == {"missing": 2, "ready": 1}
    assert summary["chunks"] == 3 and summary["source_fraction"] < 1.0
    plan = json.loads((out / "plan.json").read_text())
    assert len(plan["chunks"]) == 3 and plan["sampling"]["seed"] == 1729
    assert plan["plan_sha256"] == summary["plan_sha256"]
    results = json.loads((out / "chunk_results.json").read_text())
    assert [item["status"] for item in results] == ["missing", "ready", "missing"]


def test_only_chunk_refuses_invalid_id_nonfresh_out_and_sends_nothing(
    sampled, capsys: pytest.CaptureFixture
) -> None:
    root, _, config, project, _source = sampled
    sent = []

    def transport(url: str, payload: bytes, timeout: float) -> dict:
        sent.append(payload)
        raise AssertionError("no provider call may happen on a refusal")

    cal = root / "cal.json"
    cal.write_text(json.dumps(calibration(config, config.tokenizer_sha256)))

    def go(out, chunk):
        return main(
            [
                "run",
                *sampling_args(sampled, out, "--calibration", str(cal), "--execute"),
                "--only-chunk",
                chunk,
            ],
            transport,
        )

    for bad in ("k0009", "0001", "", "K0001"):
        assert go(root / f"bad-{bad or 'empty'}", bad) == 2
        assert "only_chunk_invalid" in capsys.readouterr().err
    assert not sent
    # a directory that already holds a journal is not new
    used = root / "used"
    used.mkdir()
    (used / "journal.jsonl").write_text("")
    before = source_tree(used)
    assert go(used, "k0001") == 2
    assert "only_chunk_output_not_new" in capsys.readouterr().err
    after = source_tree(used)
    assert after["journal.jsonl"] == before["journal.jsonl"] == b""
    assert set(after) <= {"journal.jsonl", "seed.json", "run.pid", "run.exit"}
    # output inside the book project is refused as before
    assert go(project / "inside", "k0001") == 2
    assert "output_inside_project" in capsys.readouterr().err
    assert not sent


def test_only_chunk_leaves_full_plan_and_source_integrity_unchanged(
    sampled, capsys: pytest.CaptureFixture
) -> None:
    root, _, config, project, source = sampled
    source_before, project_before = source.read_bytes(), source_tree(project)
    full = root / "full-plan"
    assert main(["plan", *sampling_args(sampled, full)]) == 0
    full_plan = json.loads(capsys.readouterr().out)
    out = root / "one"
    cal = root / "cal.json"
    cal.write_text(json.dumps(calibration(config, config.tokenizer_sha256)))

    def transport(url: str, payload: bytes, timeout: float) -> dict:
        return {
            "model": config.backend.model,
            "done_reason": "stop",
            "prompt_eval_count": 7,
            "eval_count": 3,
            "content": json.dumps({"facts": []}),
        }

    assert (
        main(
            [
                "run",
                *sampling_args(sampled, out, "--calibration", str(cal), "--execute"),
                "--only-chunk",
                "k0002",
            ],
            transport,
        )
        == 3
    )
    capsys.readouterr()
    one_plan = json.loads((out / "plan.json").read_text())
    full_plan_file = json.loads((full / "plan.json").read_text())
    assert one_plan == full_plan_file and one_plan["plan_sha256"] == full_plan["plan_sha256"]
    assert source.read_bytes() == source_before == TEXT.encode()
    assert source_tree(project) == project_before
