# Shared movie render resolution settings and geometry.

DEFAULT_RESOLUTION = 480
SUPPORTED_RESOLUTIONS = frozenset({480, 720, 1080})

# H.264 requires even dimensions. 854×480 is the conventional 16:9 480p
# frame (the mathematical width rounds from 853.33).
DIMENSIONS = {480: (854, 480), 720: (1280, 720), 1080: (1920, 1080)}


# ##################################################################
# movie dimensions
# maps supported vertical resolutions to their 16:9 frame dimensions
def movie_dimensions(resolution: int) -> tuple[int, int]:
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {sorted(SUPPORTED_RESOLUTIONS)}, got {resolution}")
    return DIMENSIONS[resolution]
