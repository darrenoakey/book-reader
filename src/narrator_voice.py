"""The durable narrator default and its explicit per-book/series override.

Darren's standing preference: future audiobooks are narrated by a deep male
voice unless a book or series explicitly asks for something else. Character
voices and series continuity are untouched — this governs the ``narrator``
identity only.

Two layers, both durable in git-ignored config rather than an environment
variable:

* ``[audiobook] narrator_voice`` in ``local/config.toml`` — the machine default
  applied to every future request. Unset means the built-in deep male default.
* an explicit per-job override recorded on the request (``--narrator-voice``)
  and, for a whole series, on the series registry entry the parent publishes.

Resolution is fail-closed: an unrecognized value raises instead of silently
narrating in the wrong voice.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

DEFAULT_NARRATOR_VOICE = "deep male"
DEEP_MALE = "deep male"
FEMALE = "female"
NEUTRAL = "neutral"

KNOWN_NARRATOR_VOICES = (DEEP_MALE, FEMALE, NEUTRAL)

# The acoustic instruction handed to the voice-design model. Only the leading
# gender/timbre clause changes between choices; the rest keeps narration clear
# and level so the default is an actual narration voice, not a performance.
_DESCRIPTION_TEMPLATES = {
    DEEP_MALE: (
        "A deep male audiobook narrator's voice in his forties: low-pitched, rich and resonant, "
        "with a warm chest-tone. Steady, unhurried pacing and clear articulation, calm and "
        "authoritative, carrying long passages of prose without strain. He keeps character "
        "dialogue clearly distinct from the narration."
    ),
    FEMALE: (
        "A female audiobook narrator's voice in her forties: clear and warm, mid-to-low register, "
        "with an even, resonant tone. Steady, unhurried pacing and precise articulation, calm and "
        "authoritative, carrying long passages of prose without strain. She keeps character "
        "dialogue clearly distinct from the narration."
    ),
    NEUTRAL: (
        "A neutral audiobook narrator's voice: clear, warm and evenly pitched, with a smooth "
        "mid register. Steady, unhurried pacing and precise articulation, calm and authoritative, "
        "carrying long passages of prose without strain. The narrator keeps character dialogue "
        "clearly distinct from the narration."
    ),
}

# Words a narrator description must contain for the deep-male contract to hold;
# checked against the materialised description so a drifting model response
# cannot quietly deliver a different narrator than the one configured.
_DEEP_MALE_CUES = ("male", "deep", "low")

CONFIG_SECTION = "audiobook"
CONFIG_KEY = "narrator_voice"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "local" / "config.toml"


class NarratorVoiceError(ValueError):
    """An unsupported narrator voice was configured or requested."""


# ##################################################################
# normalize
# Accept the documented names, their obvious spellings, and a free-form
# instruction ("a deep, gravelly male voice") that still names a known choice.
def normalize(value: str | None) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if not text:
        raise NarratorVoiceError("an empty narrator voice is not a valid choice")
    if text in KNOWN_NARRATOR_VOICES:
        return text
    words = set(re.findall(r"[a-z]+", text))
    if "male" in words and ("deep" in words or "low" in words or "baritone" in words or "bass" in words):
        return DEEP_MALE
    if "female" in words:
        return FEMALE
    if "neutral" in words or "androgynous" in words:
        return NEUTRAL
    raise NarratorVoiceError(
        f"unsupported narrator voice {value!r}; use one of {', '.join(KNOWN_NARRATOR_VOICES)}, "
        "or a phrase naming one of them (for example 'deep male')"
    )


# ##################################################################
# configured default
# The machine-wide preference from [audiobook] narrator_voice; the built-in
# deep male default applies when the key is absent.
def configured_default(path: Path | None = None) -> str:
    path = Path(path) if path is not None else CONFIG_PATH
    if not path.is_file():
        return DEFAULT_NARRATOR_VOICE
    section = tomllib.loads(path.read_text(encoding="utf-8")).get(CONFIG_SECTION, {})
    value = section.get(CONFIG_KEY)
    if value is None:
        return DEFAULT_NARRATOR_VOICE
    return normalize(str(value))


# ##################################################################
# resolve
# An explicit request beats the series preference, which beats the machine
# default. Nothing is ever silently ignored: each supplied layer is validated.
def resolve(job_override: str | None = None, series_override: str | None = None, config_path: Path | None = None) -> str:
    if job_override is not None and str(job_override).strip():
        return normalize(job_override)
    if series_override is not None and str(series_override).strip():
        return normalize(series_override)
    return configured_default(config_path)


# ##################################################################
# per-project override
# A book or series that explicitly overrides the default records the choice in
# PROJECT_FILE beside the production, so the override is durable for every
# later step and re-run of that project rather than living in one process.
PROJECT_FILE = "narrator_voice.txt"


def project_override(project: Path) -> str | None:
    path = Path(project) / PROJECT_FILE
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def series_override(project: Path) -> str | None:
    """The series parent's recorded narrator choice, if this book belongs to one.

    A published parent is the voice contract its sequels inherit, so a series
    that chose a narrator keeps it rather than flipping to the machine default.
    """
    from src.series_voices import series_parent

    parent = series_parent(Path(project))
    return project_override(parent) if parent is not None else None


def resolve_for_project(project: Path, config_path: Path | None = None) -> str:
    """The narrator for one production: book override, then series, then default."""
    project = Path(project)
    return resolve(
        job_override=project_override(project),
        series_override=series_override(project),
        config_path=config_path,
    )


def description_for_project(project: Path, config_path: Path | None = None) -> str:
    return description(resolve_for_project(project, config_path))


# ##################################################################
# description
# The exact instruction the voice-design model receives for the narrator.
def description(choice: str | None = None) -> str:
    resolved = normalize(choice) if choice is not None else DEFAULT_NARRATOR_VOICE
    return _DESCRIPTION_TEMPLATES[resolved]


# ##################################################################
# deep male contract check
# True when a materialised narrator description is a deep male narration voice.
def satisfies(choice: str, description_text: str) -> bool:
    resolved = normalize(choice)
    text = self_description(description_text).lower()
    if resolved != DEEP_MALE:
        return True
    return all(re.search(rf"\b{re.escape(cue)}\b", text) for cue in _DEEP_MALE_CUES)


def self_description(text: str) -> str:
    # local import keeps this module free of a cycle with voice_acoustic
    from src.voice_acoustic import self_description as strip_contrasts

    return strip_contrasts(text)


__all__ = [
    "CONFIG_KEY",
    "CONFIG_PATH",
    "CONFIG_SECTION",
    "DEEP_MALE",
    "DEFAULT_NARRATOR_VOICE",
    "FEMALE",
    "KNOWN_NARRATOR_VOICES",
    "NEUTRAL",
    "PROJECT_FILE",
    "NarratorVoiceError",
    "configured_default",
    "description",
    "description_for_project",
    "normalize",
    "project_override",
    "resolve",
    "resolve_for_project",
    "satisfies",
    "series_override",
]
