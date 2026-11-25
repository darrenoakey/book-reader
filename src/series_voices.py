"""Series-stable voices: a later book in a series reuses an earlier book's cast.

A project may name ONE earlier production as its series parent in
``series.json``. Every character id the two books share keeps the parent's
exact voice description and its exact Breeze reference clip (byte-identical
copy), so the same person sounds the same across books. Characters new to the
later book are designed fresh. A parent description that contradicts the new
book's cast plan (e.g. a different gender for a reused id) is never forced on
the new book; that character is designed fresh and the decision is recorded.

The parent binding is immutable: re-pointing a project at a different parent
after it has been set fails closed, because its voices/audio already depend on
the first parent's clips.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import shutil
from pathlib import Path

from src.voice_acoustic import description_violations

SERIES_FILE = "series.json"
REGISTRY_FILE = "series_registry.json"
REGISTRY_FORMAT = "series-registry-v1"

# ##################################################################
# registry root
# the persistent registry directory under the configured delivery root;
# created on first write, never touching the productions it indexes
def registry_root(delivery_root: Path) -> Path:
    return Path(delivery_root) / "series-registry"


# ##################################################################
# normalize name
# identity keys are casefolded, whitespace-collapsed display names
def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).casefold()


# ##################################################################
# publish series
# record a finished production as the latest version of a series in the
# persistent registry under the delivery root (versioned, non-destructive:
# each publish appends a version, earlier versions stay readable forever).
# required_characters: display names the series contract demands for reuse
# (e.g. narrator plus series protagonists). The narrator is always required.
def publish_series(
    delivery_root: Path,
    series: str,
    production: Path,
    character_names: dict[str, str],
    narrator_name: str = "Narrator",
    required_characters: list[str] | None = None,
) -> dict:
    production = production.resolve()
    _require_parent_cast(production)
    voices = _load(production / "voices.json")
    if "narrator" not in voices:
        raise ValueError(f"production {production.name} has no narrator voice; a series parent always publishes one")
    required = ["narrator", *(normalize_name(n) for n in (required_characters or []))]
    missing = [char_id for char_id in dict.fromkeys(required) if not voices.get(char_id, {}).get("description")]
    if missing:
        raise ValueError(f"production {production.name} is missing required series voices: {', '.join(missing)}")
    root = registry_root(delivery_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / REGISTRY_FILE
    registry = _load(path) if path.is_file() else {"format": REGISTRY_FORMAT, "series": {}}
    if registry.get("format") != REGISTRY_FORMAT:
        raise ValueError(f"{path} is not a {REGISTRY_FORMAT} registry")
    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    entry = {
        "series": series,
        "production": str(production),
        "published_at": now,
        "narrator": {"description": voices["narrator"]["description"]},
        "characters": {
            char_id: {
                "name": (character_names.get(char_id) or char_id),
                "description": voices[char_id]["description"],
            }
            for char_id in sorted(voices)
            if isinstance(voices[char_id], dict) and voices[char_id].get("description")
        },
        "required_characters": sorted(set(required)),
    }
    versions = registry["series"].setdefault(series, [])
    entry["version"] = len(versions) + 1
    versions.append(entry)
    _write_json_atomic(path, registry)
    return entry


# ##################################################################
# write registry atomically
# a crash mid-write can never leave a torn registry behind
def _write_json_atomic(path: Path, data: dict) -> None:
    partial = path.with_suffix(f"{path.suffix}.partial")
    partial.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


# ##################################################################
# load series registry
# the registry document (fails closed when absent or foreign-format)
def load_series_registry(delivery_root: Path) -> dict:
    path = registry_root(delivery_root) / REGISTRY_FILE
    if not path.is_file():
        raise ValueError(f"no series registry under {registry_root(delivery_root)}; publish the series parent first")
    registry = _load(path)
    if registry.get("format") != REGISTRY_FORMAT:
        raise ValueError(f"{path} is not a {REGISTRY_FORMAT} registry")
    return registry


# ##################################################################
# latest series version
# the newest registry entry for a series, with its version number (1-based)
def latest_series_version(delivery_root: Path, series: str) -> tuple[int, dict]:
    versions = load_series_registry(delivery_root)["series"].get(series)
    if not versions:
        raise ValueError(f"series {series!r} is not in the registry under {registry_root(delivery_root)}")
    return len(versions), versions[-1]


# ##################################################################
# resolve series identity
# map a new book's character id to the series character id it must reuse.
# Exact normalized names win; recurring aliases map to the same canonical
# character. A name that collides with a DIFFERENT registered character is a
# conflict and fails closed (never silently re-voiced); unknown names return
# None so the book designs them fresh. Returns {local_id: series_char_id}.
def resolve_series_identities(series_characters: dict[str, dict], local_characters: dict[str, str]) -> dict[str, str]:
    aliases = _alias_index(series_characters)
    resolved: dict[str, str] = {}
    for local_id, name in local_characters.items():
        canonical = aliases.get(normalize_name(name))
        if canonical is None:
            continue
        if canonical == "_conflict":
            raise ValueError(
                f"character name {name!r} collides with different series characters; "
                "rename the character or widen the registry aliases before publishing"
            )
        resolved[local_id] = canonical
    return resolved


def _alias_index(series_characters: dict[str, dict]) -> dict[str, str]:
    index: dict[str, str] = {}
    for char_id, info in series_characters.items():
        names = [info.get("name", ""), *(info.get("aliases") or [])]
        for name in names:
            if not name:
                continue
            key = normalize_name(name)
            existing = index.get(key)
            if existing and existing != char_id:
                index[key] = "_conflict"
            else:
                index[key] = char_id
    return index


# ##################################################################
# required series characters
# verify a production actually carries every character the series contract
# requires (the narrator by construction, plus each configured name).
# Returns the verified char ids; a missing or unnamed character fails closed.
def verify_required_characters(entry: dict, local_characters: dict[str, str], required: list[str] | None) -> list[str]:
    required_ids = set(entry.get("required_characters") or ["narrator"])
    if required:
        required_ids |= {normalize_name(n) for n in required}
    by_name = {}
    for local_id, name in local_characters.items():
        by_name.setdefault(normalize_name(name), local_id)
    verified = []
    for key in sorted(required_ids):
        if key == "narrator":
            verified.append("narrator")
            continue
        local_id = by_name.get(key)
        if not local_id:
            raise ValueError(f"series character {key!r} is missing from this book's cast; the series contract requires it")
        verified.append(local_id)
    return verified


# ##################################################################
# set series parent
# record the earlier production this project inherits voices from (idempotent)
def set_series_parent(project: Path, parent: Path) -> Path:
    parent = parent.resolve()
    _require_parent_cast(parent)
    path = project / SERIES_FILE
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8")).get("parent")
        if existing != str(parent):
            raise ValueError(f"{project.name} already inherits voices from {existing}; refusing to re-point it at {parent}")
        return path
    project.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"parent": str(parent)}, indent=2) + "\n", encoding="utf-8")
    return path


# ##################################################################
# series parent
# the recorded parent production, or None; a recorded-but-missing parent fails closed
def series_parent(project: Path) -> Path | None:
    path = project / SERIES_FILE
    if not path.is_file():
        return None
    parent = Path(json.loads(path.read_text(encoding="utf-8"))["parent"])
    _require_parent_cast(parent)
    return parent


# ##################################################################
# established identities
# the parent's character ids and names, offered to this book's character
# analysis so the same person keeps the same id (and therefore voice)
def established_identities(project: Path) -> str:
    parent = series_parent(project)
    if parent is None:
        return ""
    characters = _load(parent / "characters.json") if (parent / "characters.json").is_file() else {}
    rows = [f"{cid} ({info.get('name', cid)})" for cid, info in sorted(characters.items()) if isinstance(info, dict) and cid != "narrator"]
    return ", ".join(rows)


def _require_parent_cast(parent: Path) -> None:
    for name in ("voices.json", "breeze_voices.json"):
        if not (parent / name).is_file():
            raise ValueError(f"series parent {parent} has no {name}; produce the earlier book's voices first")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ##################################################################
# inherited descriptions
# shared character ids take the parent's exact description when it satisfies
# this book's cast-plan profile; returns {char_id: {"description": ...}}
def inherited_descriptions(project: Path, characters: dict, plan: dict | None) -> dict:
    parent = series_parent(project)
    if parent is None:
        return {}
    parent_voices = _load(parent / "voices.json")
    profiles = (plan or {}).get("characters", {})
    inherited: dict = {}
    rejected: dict = {}
    for char_id in sorted(characters):
        entry = parent_voices.get(char_id)
        if not isinstance(entry, dict) or not entry.get("description"):
            continue
        if char_id == "narrator":
            # the narrator is the reading voice, configured rather than inherited
            # from the earlier book's performance; resolve_for_project already
            # consulted this parent, so the inherited clip would be stale art
            continue
        profile = profiles.get(char_id)
        problems = description_violations(entry["description"], profile) if profile else []
        if problems:
            rejected[char_id] = problems
            continue
        inherited[char_id] = {"description": entry["description"]}
    report = {"parent": str(parent), "inherited": sorted(inherited), "designed_fresh_conflicts": rejected}
    (project / "series_voices_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return inherited


# ##################################################################
# copy parent reference
# when the parent designed a Breeze clip for this exact id + description,
# materialize a byte-identical copy instead of designing a new voice.
# Returns the manifest entry to record, or None when a fresh design is needed.
def copy_parent_reference(project: Path, char_id: str, description: str) -> dict | None:
    parent = series_parent(project)
    if parent is None:
        return None
    entry = _load(parent / "breeze_voices.json").get(char_id)
    if not entry or entry.get("description") != description:
        return None
    source = parent / entry["ref_wav"]
    if not source.is_file() or source.stat().st_size < 100:
        raise ValueError(f"series parent clip for {char_id} is missing at {source}")
    target = project / "voices" / f"{char_id}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    shutil.copyfile(source, partial)
    if _sha256(partial) != _sha256(source):
        partial.unlink(missing_ok=True)
        raise ValueError(f"copied series clip for {char_id} does not match its source bytes")
    partial.replace(target)
    return {
        "ref_wav": str(target.relative_to(project)),
        "ref_text": entry["ref_text"],
        "description": description,
        "inherited_from": str(parent),
        "sha256": _sha256(target),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "REGISTRY_FILE",
    "SERIES_FILE",
    "copy_parent_reference",
    "established_identities",
    "inherited_descriptions",
    "latest_series_version",
    "load_series_registry",
    "normalize_name",
    "publish_series",
    "registry_root",
    "resolve_series_identities",
    "series_parent",
    "set_series_parent",
    "verify_required_characters",
]


# ##################################################################
# pre-synthesis series cast gate
# The series contract is currently enforced when a production PUBLISHES, which
# is after hours of synthesis: Dragonsinger spent a full render before failing
# on a missing required voice (Menolly). This runs the SAME existing validator
# (`verify_required_characters`) on the freshly analysed cast, scoped to the
# required returning series characters, so a missing one fails in seconds with
# the identities named instead of after a render.
def require_series_cast(project: Path, characters: dict, required: list[str] | None) -> list[str]:
    """Fail closed before synthesis when a required series character is absent.

    Returns the verified local ids (narrator plus each required name). A book
    without a series contract has nothing to verify and returns an empty list.
    """
    parent = series_parent(project)
    if parent is None and not required:
        return []
    required_names = list(required or [])
    if parent is not None:
        parent_voices = _load(parent / "voices.json")
        # the parent's own required set plus every id it actually voiced is the
        # contract a sequel inherits; only names are asserted here
        required_names = required_names or [
            char_id for char_id in parent_voices if char_id == "narrator"
        ]
    if not required_names:
        return []
    names = {
        str(info.get("name", char_id)): char_id
        for char_id, info in characters.items()
        if isinstance(info, dict)
    }
    entry = {"required_characters": [normalize_name(name) for name in required_names]}
    return verify_required_characters(entry, {cid: name for name, cid in names.items()}, required_names)
