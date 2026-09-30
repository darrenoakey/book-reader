# Shared movie render resolution settings and geometry.

DEFAULT_RESOLUTION = 720
SUPPORTED_RESOLUTIONS = frozenset({720, 1080})


# ##################################################################
# movie dimensions
# maps supported vertical resolutions to their 16:9 frame dimensions
def movie_dimensions(resolution: int) -> tuple[int, int]:
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {sorted(SUPPORTED_RESOLUTIONS)}, got {resolution}")
    return resolution * 16 // 9, resolution
