"""BVH interpolation helpers."""

from broom.bvh.interpolation.interpolation import Interpolation
from broom.bvh.interpolation.schemas import (
    InterpolationResult,
)
from broom.bvh.interpolation.utils import (
    blend_euler_degrees,
    blend_frame_ranges,
    count_trailing_static_frames,
    interpolate_documents,
    interpolate_motion_values,
    trim_trailing_static_frames,
)

count_trailing_identical_frames = count_trailing_static_frames

__all__ = (
    "Interpolation",
    "InterpolationResult",
    "blend_euler_degrees",
    "blend_frame_ranges",
    "count_trailing_identical_frames",
    "count_trailing_static_frames",
    "interpolate_documents",
    "interpolate_motion_values",
    "trim_trailing_static_frames",
)
