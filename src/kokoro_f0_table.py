"""Measured fundamental-frequency medians for the 28 English Kokoro bank voices.

Provenance: derived 2026-11-16 from the Timbre listening atlas
(~/src/timbre/static/data/atlas.json, generated 2026-10-28; upstream Kokoro-82M
English bank, narration passage, f0_median Hz) so voice selection is
deterministic everywhere — no runtime dependency on the sibling checkout.
Lower f0 = deeper/larger perceived voice; higher f0 = lighter/brighter.
"""

F0_MEDIANS: dict[str, float] = {
    "af_alloy": 143.7,
    "af_aoede": 176.5,
    "af_bella": 206.9,
    "af_heart": 203.4,
    "af_jessica": 206.9,
    "af_kore": 148.1,
    "af_nicole": 165.0,
    "af_nova": 157.9,
    "af_river": 172.7,
    "af_sarah": 198.3,
    "af_sky": 154.8,
    "am_adam": 125.0,
    "am_echo": 113.5,
    "am_eric": 157.9,
    "am_fenrir": 134.1,
    "am_liam": 128.3,
    "am_michael": 121.2,
    "am_onyx": 87.6,
    "am_puck": 108.6,
    "am_santa": 161.1,
    "bf_alice": 213.3,
    "bf_emma": 183.2,
    "bf_isabella": 203.4,
    "bf_lily": 195.1,
    "bm_daniel": 125.0,
    "bm_fable": 119.4,
    "bm_george": 140.4,
    "bm_lewis": 105.7,
}

ATLAS_PROVENANCE = {
    "source": "timbre listening atlas (kokoro-82M English bank)",
    "atlas_generated": "2026-10-28",
    "derived": "2026-11-16",
    "measurement": "f0_median Hz, narration passage",
}
