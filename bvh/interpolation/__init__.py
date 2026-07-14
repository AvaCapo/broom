"""BVH interpolation helpers."""

from bvh.interpolation.interpolation import Interpolation
from bvh.interpolation.schemas import (
    InterpolationResult,
)
from bvh.interpolation.utils import (
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
