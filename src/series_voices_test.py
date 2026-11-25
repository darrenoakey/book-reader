import json
import tempfile
import wave
from pathlib import Path

import pytest

from src.series_voices import (
    REGISTRY_FILE,
    copy_parent_reference,
    established_identities,
    inherited_descriptions,
    latest_series_version,
    load_series_registry,
    normalize_name,
    publish_series,
    registry_root,
    require_series_cast,
    resolve_series_identities,
    series_parent,
    set_series_parent,
    verify_required_characters,
)


# ##################################################################
# make production
# a synthetic parent/child production: real (tiny, silent) wav clips, voices
# and breeze manifests — no model calls anywhere in this suite
def _make_production(root: Path, cast: dict, clips: dict[str, bytes] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    voices = {cid: {"description": desc} for cid, desc in cast.items()}
    (root / "voices.json").write_text(json.dumps(voices, indent=2), encoding="utf-8")
    (root / "voices").mkdir(exist_ok=True)
    manifest = {}
    for cid, desc in cast.items():
        wav = root / "voices" / f"{cid}.wav"
        _write_wav(wav)
        manifest[cid] = {
            "ref_wav": f"voices/{cid}.wav",
            "ref_text": "Listen to me. I have waited my whole life for this moment, and I am not giving up now.",
            "description": desc,
        }
    for cid, payload in (clips or {}).items():
        (root / "voices" / f"{cid}.wav").write_bytes(payload)
        manifest[cid]["ref_wav"] = f"voices/{cid}.wav"
    (root / "breeze_voices.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "characters.json").write_text(
        json.dumps({cid: {"name": name} for cid, name in cast_names(cast).items()}, indent=2),
        encoding="utf-8",
    )
    return root


def cast_names(cast: dict) -> dict:
    return {cid: f"{cid.replace('_', ' ').title()}" for cid in cast}


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 3200)


PARENT_CAST = {
    "narrator": "A warm measured voice.",
    "menolly": "Female, teen, light clear musical voice.",
    "petiron": "Male, older, weathered baritone.",
}


# ##################################################################
# test normalize name
# identity keys are casefolded with whitespace collapsed
# ##################################################################
# pre-synthesis series cast gate
# The series contract must fail in seconds on a fresh cast, not after a render.
# Dragonsinger rendered in full before failing because Menolly had no voice.
def test_require_series_cast_passes_when_required_present() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "book"
        project.mkdir()
        cast = {"menolly": {"name": "Menolly"}, "narrator": {"name": "Narrator"}}
        verified = require_series_cast(project, cast, ["Menolly"])
        assert verified == ["menolly"]


def test_require_series_cast_fails_closed_when_required_missing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "book"
        project.mkdir()
        cast = {"narrator": {"name": "Narrator"}, "silvina": {"name": "Silvina"}}
        with pytest.raises(ValueError) as caught:
            require_series_cast(project, cast, ["Menolly"])
        message = str(caught.value)
        assert "menolly" in message
        # the failure names the character, so the operator knows what to fix
        assert "series contract" in message


def test_require_series_cast_names_every_missing_character() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "book"
        project.mkdir()
        cast = {"narrator": {"name": "Narrator"}}
        with pytest.raises(ValueError) as caught:
            require_series_cast(project, cast, ["Menolly", "Piemur"])
        message = str(caught.value)
        assert "menolly" in message or "piemur" in message


def test_require_series_cast_is_a_no_op_without_a_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "book"
        project.mkdir()
        assert require_series_cast(project, {"narrator": {"name": "Narrator"}}, None) == []


def test_require_series_cast_ignores_id_spelling_and_uses_display_name() -> None:
    # the contract names people ("Menolly"); a book may key her differently as
    # long as the display name matches, which is how the publish-time check works
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "book"
        project.mkdir()
        cast = {"menolly_of_half_circle": {"name": "Menolly"}, "narrator": {"name": "Narrator"}}
        assert require_series_cast(project, cast, ["Menolly"]) == ["menolly_of_half_circle"]


def test_normalize_name() -> None:
    assert normalize_name("  Menolly   Harper ") == "menolly harper"
    assert normalize_name("MENOLLY") == "menolly"


# ##################################################################
# test set and read series parent
# the parent binding is recorded idempotently and read back resolved
def test_set_and_read_series_parent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parent = _make_production(tmp / "parent", PARENT_CAST)
        child = tmp / "child"
        path = set_series_parent(child, parent)
        assert path == child / "series.json"
        assert series_parent(child) == parent.resolve()
        # idempotent: same parent again succeeds and keeps one file
        assert set_series_parent(child, parent) == path


# ##################################################################
# test repointing fails closed
# a different parent after the first is set is a hard error
def test_repoint_parent_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parent = _make_production(tmp / "parent", PARENT_CAST)
        other = _make_production(tmp / "other", PARENT_CAST)
        child = tmp / "child"
        set_series_parent(child, parent)
        with pytest.raises(ValueError, match="already inherits voices"):
            set_series_parent(child, other)


# ##################################################################
# test parent without cast fails closed
# both set and read fail when the parent production has no voice manifests
def test_parent_without_cast_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "hollow").mkdir()
        child = tmp / "child"
        with pytest.raises(ValueError, match="produce the earlier book"):
            set_series_parent(child, tmp / "hollow")
        (child / "series.json").parent.mkdir(parents=True, exist_ok=True)
        (child / "series.json").write_text(json.dumps({"parent": str(tmp / "hollow")}), encoding="utf-8")
        with pytest.raises(ValueError, match="has no voices.json"):
            series_parent(child)


# ##################################################################
# test established identities
# the parent's cast is offered as id (name) rows, narrator excluded
def test_established_identities_lists_parent_cast() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parent = _make_production(tmp / "parent", PARENT_CAST)
        child = tmp / "child"
        set_series_parent(child, parent)
        text = established_identities(child)
        assert "menolly (Menolly)" in text
        assert "petiron (Petiron)" in text
        assert "narrator" not in text
        # no series binding means nothing established
        assert established_identities(tmp / "standalone") == ""


# ##################################################################
# test inherited descriptions
# shared ids take the parent description when it satisfies the plan profile
def test_inherited_descriptions_matches_plan() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parent = _make_production(tmp / "parent", PARENT_CAST)
        child = tmp / "child"
        set_series_parent(child, parent)
        characters = {"narrator": {}, "menolly": {}, "brand": {}}
        plan = {"characters": {"menolly": {"gender": "female", "age_band": "teen"}}}
        inherited = inherited_descriptions(child, characters, plan)
        # menolly's parent description matches its plan profile exactly and is
        # reused; the narrator is the reading voice, so it is never inherited
        # from the parent's performance — the narrator policy supplies it
        assert inherited == {"menolly": {"description": PARENT_CAST["menolly"]}}
        report = json.loads((child / "series_voices_report.json").read_text(encoding="utf-8"))
        assert report["inherited"] == ["menolly"]
        assert report["parent"] == str(parent.resolve())


# ##################################################################
# test conflicting profile is designed fresh
# a parent description contradicting the new plan is rejected and recorded,
# never forced onto the new book
def test_inherited_descriptions_reject_conflict() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parent = _make_production(tmp / "parent", PARENT_CAST)
        child = tmp / "child"
        set_series_parent(child, parent)
        characters = {"menolly": {}}
        plan = {"characters": {"menolly": {"gender": "male", "age_band": "teen"}}}
        inherited = inherited_descriptions(child, characters, plan)
        assert inherited == {}
        report = json.loads((child / "series_voices_report.json").read_text(encoding="utf-8"))
        assert report["designed_fresh_conflicts"]["menolly"]


# ##################################################################
# test byte identical reference copy
# an exact id + description match copies the parent clip and verifies sha256
def test_copy_parent_reference_is_byte_identical() -> None:
    import hashlib

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parent = _make_production(tmp / "parent", PARENT_CAST)
        child = tmp / "child"
        set_series_parent(child, parent)
        entry = copy_parent_reference(child, "menolly", PARENT_CAST["menolly"])
        assert entry is not None
        target = child / entry["ref_wav"]
        source = parent / "voices" / "menolly.wav"
        assert target.read_bytes() == source.read_bytes()
        assert entry["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert entry["description"] == PARENT_CAST["menolly"]
        assert entry["inherited_from"] == str(parent.resolve())
        # a different description means design fresh
        assert copy_parent_reference(child, "menolly", "some other description") is None
        assert copy_parent_reference(child, "brand", PARENT_CAST["menolly"]) is None


# ##################################################################
# test missing parent clip fails closed
# a manifest entry whose wav vanished is an error, never a silent redesign
def test_copy_parent_reference_missing_clip_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        parent = _make_production(tmp / "parent", PARENT_CAST)
        (parent / "voices" / "menolly.wav").unlink()
        child = tmp / "child"
        set_series_parent(child, parent)
        with pytest.raises(ValueError, match="missing"):
            copy_parent_reference(child, "menolly", PARENT_CAST["menolly"])


# ##################################################################
# test publish and read back registry
# publishing appends version 1 under the delivery root and fails closed
# when a required character voice is missing
def test_publish_series_creates_versioned_registry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        production = _make_production(tmp / "prod", PARENT_CAST)
        entry = publish_series(
            tmp,
            "Harper Hall of Pern",
            production,
            character_names={"narrator": "Narrator", "menolly": "Menolly", "petiron": "Petiron"},
            required_characters=["Menolly"],
        )
        assert entry["version"] == 1
        assert entry["required_characters"] == ["menolly", "narrator"]
        registry = load_series_registry(tmp)
        assert registry["format"] == "series-registry-v1"
        assert registry["series"]["Harper Hall of Pern"][0]["production"] == str(production.resolve())
        assert registry_root(tmp) == tmp / "series-registry"
        # second publish appends, never overwrites
        publish_series(tmp, "Harper Hall of Pern", production, {})
        version, latest = latest_series_version(tmp, "Harper Hall of Pern")
        assert version == 2
        assert latest["production"] == str(production.resolve())
        # unknown series fails closed
        with pytest.raises(ValueError, match="not in the registry"):
            latest_series_version(tmp, "Other Series")


# ##################################################################
# test publish requires narrator and required characters
# a production without a narrator voice, or missing a pinned series
# character, fails closed before writing anything
def test_publish_requires_narrator_and_pinned_characters() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        no_narrator = {k: v for k, v in PARENT_CAST.items() if k != "narrator"}
        production = _make_production(tmp / "prod", no_narrator)
        with pytest.raises(ValueError, match="narrator voice"):
            publish_series(tmp, "Series", production, {}, required_characters=["Menolly"])
        assert not (registry_root(tmp) / REGISTRY_FILE).exists()
        full = _make_production(tmp / "full", PARENT_CAST)
        with pytest.raises(ValueError, match="required series voices: .*lessa"):
            publish_series(tmp, "Series", full, {}, required_characters=["Lessa"])
        # still nothing written by the failed attempts
        assert not (registry_root(tmp) / REGISTRY_FILE).exists()
        publish_series(tmp, "Series", full, {})
        assert latest_series_version(tmp, "Series")[0] == 1


# ##################################################################
# test registry format guard
# a foreign or corrupt registry fails closed instead of degrading
def test_load_registry_format_guard() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with pytest.raises(ValueError, match="publish the series parent first"):
            load_series_registry(tmp)
        registry_root(tmp).mkdir()
        (registry_root(tmp) / REGISTRY_FILE).write_text(json.dumps({"format": "other-v9"}), encoding="utf-8")
        with pytest.raises(ValueError, match="series-registry-v1"):
            load_series_registry(tmp)


# ##################################################################
# test resolve identities by name and alias
# exact normalized names map local ids to series ids; a registered alias
# maps to the same canonical character; unknown names stay absent
def test_resolve_series_identities_names_and_aliases() -> None:
    series_characters = {
        "menolly": {"name": "Menolly", "aliases": ["Menolly Harper"]},
        "petiron": {"name": "Petiron"},
    }
    local = {"menolly": "Menolly", "menolly_harper": "menolly harper", "brand": "Brand", "PETIRON": "Petiron"}
    resolved = resolve_series_identities(series_characters, local)
    assert resolved == {"menolly": "menolly", "menolly_harper": "menolly", "PETIRON": "petiron"}


# ##################################################################
# test identity conflict fails closed
# one name pointing at two different registered characters is never resolved
def test_resolve_series_identities_conflict_fails_closed() -> None:
    series_characters = {
        "menolly": {"name": "Menolly", "aliases": ["Fireling"]},
        "sella": {"name": "Sella", "aliases": ["fireling"]},
    }
    with pytest.raises(ValueError, match="collides with different series characters"):
        resolve_series_identities(series_characters, {"m": "Fireling"})


# ##################################################################
# test required character verification
# narrator is always required; pinned names must exist in the book's cast
def test_verify_required_characters() -> None:
    entry = {"required_characters": ["narrator", "menolly"]}
    local = {"narrator": "Narrator", "menolly_local": "Menolly"}
    assert verify_required_characters(entry, local, None) == ["menolly_local", "narrator"]
    # extra runtime requirement is honoured when the cast carries it
    wider = {"narrator": "Narrator", "menolly_local": "Menolly", "petiron_local": "Petiron"}
    assert verify_required_characters(entry, wider, ["Petiron"]) == ["menolly_local", "narrator", "petiron_local"]
    with pytest.raises(ValueError, match="requires it"):
        verify_required_characters(entry, {"narrator": "Narrator"}, None)
