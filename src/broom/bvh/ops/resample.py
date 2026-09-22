"""BVH frame-rate resampling operations."""

from __future__ import annotations

import numpy as np

from broom.bvh.interpolation.utils import (
    blend_euler_degrees,
    rotation_channel_groups,
)
from broom.bvh.ops.motion import (
    _require_frame_time,
    with_motion_values,
)
from broom.bvh.schemas import BVHDocument


def resample_fps(document: BVHDocument, target_fps: float) -> BVHDocument:
    """Return a copy of the document resampled to ``target_fps``."""

    target_fps = float(target_fps)
    if target_fps <= 0.0:
        raise ValueError("Target FPS must be positive.")

    source_frame_time = _require_frame_time(document)
    target_frame_time = 1.0 / target_fps
    source_values = document.motion_values

    if document.frame_count <= 1:
        return with_motion_values(
            document=document,
            motion_values=source_values.copy(),
            frame_time=target_frame_time,
        )

    duration = (document.frame_count - 1) * source_frame_time
    target_frame_count = max(1, int(round(duration * target_fps)) + 1)

    source_times = np.arange(document.frame_count, dtype=np.float64)
    source_times *= source_frame_time
    target_times = np.linspace(
        0.0,
        duration,
        target_frame_count,
        dtype=np.float64,
    )

    resampled = np.empty(
        (target_frame_count, document.total_channels),
        dtype=np.float64,
    )
    _resample_motion_channels(
        source_values=source_values,
        source_times=source_times,
        target_times=target_times,
        resampled=resampled,
    )
    _resample_rotation_channels(
        document=document,
        source_times=source_times,
        target_times=target_times,
        resampled=resampled,
    )
    return with_motion_values(
        document=document,
        motion_values=resampled,
        frame_time=target_frame_time,
    )


def _resample_motion_channels(
    source_values: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
    resampled: np.ndarray,
) -> None:
    """Linearly resample all channels as the baseline pass."""

    for channel_index in range(source_values.shape[1]):
        resampled[:, channel_index] = np.interp(
            target_times,
            source_times,
            source_values[:, channel_index],
        )


def _resample_rotation_channels(
    document: BVHDocument,
    source_times: np.ndarray,
    target_times: np.ndarray,
    resampled: np.ndarray,
) -> None:
    """Replace linearly interpolated rotation channels with SLERP results."""

    source_values = document.motion_values
    for indices, order in rotation_channel_groups(document):
        for target_index, target_time in enumerate(target_times):
            right = int(np.searchsorted(source_times, target_time, side="right"))
            if right <= 0:
                resampled[target_index, indices] = source_values[0, indices]
                continue
            if right >= len(source_times):
                resampled[target_index, indices] = source_values[-1, indices]
                continue

            left = right - 1
            span = source_times[right] - source_times[left]
            alpha = (
                0.0
                if span <= 0.0
                else (target_time - source_times[left]) / span
            )
            resampled[target_index, indices] = _blend_rotation_degrees(
                first_angles=source_values[left, indices],
                second_angles=source_values[right, indices],
                order=order,
                alpha=float(alpha),
            )


def _blend_rotation_degrees(
    first_angles: np.ndarray,
    second_angles: np.ndarray,
    order: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend Euler angles with SLERP and fall back to shortest-angle lerp."""

    try:
        return blend_euler_degrees(
            first_angles=first_angles,
            second_angles=second_angles,
            order=order,
            alpha=alpha,
        )
    except ValueError:
        delta = (second_angles - first_angles + 180.0) % 360.0 - 180.0
        return (first_angles + delta * alpha + 180.0) % 360.0 - 180.0
