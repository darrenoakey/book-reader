import itertools
import random
import re
from pathlib import Path

from src.scriptor_attribution import (
    Character,
    characters_for,
    load_scriptor_config,
    parse_lines,
    partition_chapter,
    partition_passage,
    passage_bounds,
    select_characters,
)

PASSAGE = (
    '"Hello," said Mary.\n\nJohn nodded. "Good to see you." He sat down.\n\n'
    "\u201cWhere were you?\u201d she asked. \u201cWe waited\u2014all night!\u201d"
)
GOLD = [
    ("mary", "Hello,"),
    ("narrator", "said Mary."),
    ("narrator", "John nodded."),
    ("john", "Good to see you."),
    ("narrator", "He sat down."),
    ("mary", "Where were you?"),
    ("narrator", "she asked."),
    ("mary", "We waited\u2014all night!"),
]


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


# ##################################################################
# exact partition
# quotes travel with speech, whitespace leads the following line, and the slices rebuild the source.
def test_partition_puts_quotes_with_speech_and_rebuilds_source() -> None:
    got = partition_passage(PASSAGE, GOLD)
    assert "".join(t for _, t in got) == PASSAGE
    assert [s for s, _ in got] == [s for s, _ in GOLD]
    assert got[:4] == [
        ("mary", '"Hello,"'),
        ("narrator", " said Mary."),
        ("narrator", "\n\nJohn nodded."),
        ("john", ' "Good to see you."'),
    ]
    assert got[5] == ("mary", "\n\n\u201cWhere were you?\u201d")


def test_partition_survives_damaged_or_garbage_model_output() -> None:
    rng = random.Random(7)
    for _ in range(200):
        damaged = []
        for speaker, text in GOLD:
            words = text.split()
            if len(words) > 1 and rng.random() < 0.4:
                del words[rng.randrange(len(words))]
            if rng.random() < 0.3:
                words.insert(rng.randrange(len(words) + 1), "inserted")
            damaged.append((speaker, " ".join(words)))
        for pred in (damaged, [], [("bob", "nothing matches")], list(reversed(GOLD))):
            got = partition_passage(PASSAGE, pred)
            assert "".join(t for _, t in got) == PASSAGE and all(t for _, t in got)
    speech = [t for s, t in partition_passage(PASSAGE, GOLD) if s != "narrator"]
    assert _words(" ".join(speech)) == _words(" ".join(t for s, t in GOLD if s != "narrator"))


def test_parse_lines_maps_unlisted_speakers_and_skips_junk() -> None:
    text = '{"bob": "Hi"}\nnot json\n{"zed": "Yo"}\n{"a": 1}\n{"narrator": "he said."}'
    assert parse_lines(text, {"narrator", "bob", "unknown"}) == [("bob", "Hi"), ("unknown", "Yo"), ("narrator", "he said.")]


# ##################################################################
# passages
def test_passage_bounds_tile_text_and_respect_limit() -> None:
    rng = random.Random(5)
    words = ["alpha", "beta.", "gamma!", "delta", "eps?", "zeta,"]
    for _ in range(60):
        paras = [" ".join(rng.choice(words) for _ in range(rng.randint(1, 900))) for _ in range(rng.randint(1, 8))]
        text = rng.choice(["", "\n"]) + rng.choice(["\n\n", "\n", "\n \n"]).join(paras) + rng.choice(["", "\n"])
        bounds = passage_bounds(text, limit=1200)
        assert bounds[0][0] == 0 and bounds[-1][1] == len(text)
        assert all(a[1] == b[0] for a, b in itertools.pairwise(bounds))
        assert all(len(text[s:e].strip()) <= 1200 for s, e in bounds)


# ##################################################################
# speaker list
def test_select_characters_prefers_named_and_recent_and_caps() -> None:
    cast = [Character(f"c{i}", f"Name{i}", "m") for i in range(60)] + [Character("ren", "Ren", "m", ("Young Ren",))]
    history = [("c5", "hi"), ("narrator", "Name7 looked up."), ("c9", "yo")]
    listed = [c.id for c in select_characters(cast, "Ren smiled at Name42.", history)]
    assert set(listed[:2]) == {"ren", "c42"} and {"c5", "c9", "c7"} <= set(listed) and len(listed) <= 20
    assert "c41" not in listed


def test_characters_for_uses_names_and_alias_ids() -> None:
    chars = characters_for(
        ["narrator", "professor_lynn", "tiger_boy"], {"tiger_boy": "Tiger Boy"}, {"gene": "tiger_boy", "lynn": "professor_lynn"}
    )
    assert chars == [Character("professor_lynn", "Professor Lynn", "", ("Lynn",)), Character("tiger_boy", "Tiger Boy", "", ("Gene",))]


def test_partition_chapter_rebuilds_with_any_model_output() -> None:
    text = "\n\n".join([PASSAGE] * 40) + "\n\n* * *\n\n" + PASSAGE
    replies = iter(['{"mary": "x"}', "garbage", '{"narrator": "said"}\n{"zed": "John"}'] * 50)
    seen: list[list[dict]] = []

    def chat(messages: list[dict]) -> str:
        seen.append(messages)
        return next(replies)

    got = partition_chapter(text, [Character("mary", "Mary", "f"), Character("john", "John", "m")], chat)
    assert "".join(t for _, t in got) == text
    assert len(seen) >= 2 and "Previous lines:\n(start of text)" in seen[0][1]["content"]
    assert "(start of text)" not in seen[1][1]["content"]


def test_config_defaults_and_overrides(tmp_path: Path) -> None:
    default = load_scriptor_config(tmp_path / "missing.toml")
    assert default.primary.url == "http://10.0.0.42:11434" and default.primary.style == "ollama"
    assert default.backup.ping_host == "127.0.0.1" and default.primary.model == default.backup.model
    config = tmp_path / "config.toml"
    config.write_text('[scriptor]\nmodel = "scriptor:v9"\nbackup_identity = "Mac"\n', encoding="utf-8")
    loaded = load_scriptor_config(config)
    assert loaded.primary.model == loaded.backup.model == "scriptor:v9" and loaded.backup.expected_identity == "Mac"
