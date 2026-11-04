"""Frame-rate resampling operations."""

from __future__ import annotations

from typing import Literal

import numpy as np

from broom import Hierarchy, Motion
from broom.rotations.euler import blend_euler_degrees


def resample_fps(
    motion: Motion,
    target_fps: float,
    *,
    endpoint_policy: Literal["preserve", "truncate"] = "preserve",
) -> Motion:
    """Return a Motion sampled at an exact ``target_fps``.

    ``endpoint_policy='preserve'`` retains the first and last source poses
    and chooses the nearest frame count for ``target_fps``. The returned
    Motion still records exactly ``1 / target_fps`` as its frame time, so its
    duration can differ slightly from the source duration through intentional
    uniform time scaling. ``endpoint_policy='truncate'`` samples only source
    times on the exact target-FPS grid and omits a final partial interval.

    Non-rotation channels are interpolated linearly. One-axis rotation
    channels are unwrapped over source frames before interpolation, so each
    adjacent rotation follows the shortest path. Two-axis rotation groups use
    linear channel interpolation. Three-axis groups use declared-order SLERP.
    """
    if not isinstance(motion, Motion):
        raise TypeError("motion must be a Motion.")

    target_fps = float(target_fps)
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("Target FPS must be a positive finite number.")
    if endpoint_policy not in {"preserve", "truncate"}:
        raise ValueError("endpoint_policy must be 'preserve' or 'truncate'.")

    target_frame_time = 1.0 / target_fps
    source_values = motion.values

    if motion.frame_count <= 1:
        return Motion(motion.hierarchy, source_values, target_frame_time)

    duration = (motion.frame_count - 1) * motion.frame_time
    if endpoint_policy == "preserve":
        target_interval_count = max(
            1,
            int(np.floor(duration * target_fps + 0.5)),
        )
    else:
        target_interval_count = int(np.floor(duration * target_fps))
    target_frame_count = target_interval_count + 1

    source_times = np.arange(motion.frame_count, dtype=np.float64)
    source_times *= motion.frame_time
    if endpoint_policy == "preserve":
        sample_times = np.linspace(
            0.0,
            duration,
            target_frame_count,
            dtype=np.float64,
        )
        sample_times[0] = source_times[0]
        sample_times[-1] = source_times[-1]
    else:
        sample_times = (
            np.arange(target_frame_count, dtype=np.float64) * target_frame_time
        )

    resampled = np.empty(
        (target_frame_count, motion.hierarchy.total_channels),
        dtype=np.float64,
    )
    _resample_motion_channels(
        source_values=source_values,
        source_times=source_times,
        sample_times=sample_times,
        resampled=resampled,
    )
    _resample_rotation_channels(
        hierarchy=motion.hierarchy,
        source_values=source_values,
        source_times=source_times,
        sample_times=sample_times,
        resampled=resampled,
    )
    return Motion(motion.hierarchy, resampled, target_frame_time)


def _resample_motion_channels(
    source_values: np.ndarray,
    source_times: np.ndarray,
    sample_times: np.ndarray,
    resampled: np.ndarray,
) -> None:
    """Linearly resample all channels as the baseline pass."""

    for channel_index in range(source_values.shape[1]):
        resampled[:, channel_index] = np.interp(
            sample_times,
            source_times,
            source_values[:, channel_index],
        )


def _resample_rotation_channels(
    hierarchy: Hierarchy,
    source_values: np.ndarray,
    source_times: np.ndarray,
    sample_times: np.ndarray,
    resampled: np.ndarray,
) -> None:
    """Apply the interpolation policy for each joint rotation group.

    A one-axis group is unwrapped before linear interpolation, treating
    adjacent source rotations as shortest-path changes. A two-axis group
    retains the baseline linear channel interpolation. A three-axis group is
    replaced with declared-order SLERP.
    """

    for joint in hierarchy.joints:
        indices = list(hierarchy.rotation_channel_indices(joint.name))
        if len(indices) == 1:
            channel_index = indices[0]
            unwrapped = np.rad2deg(
                np.unwrap(np.deg2rad(source_values[:, channel_index]))
            )
            resampled[:, channel_index] = np.interp(
                sample_times,
                source_times,
                unwrapped,
            )
            continue
        if len(indices) != 3:
            continue
        order = np.asarray(
            [
                channel[0]
                for channel in joint.channels
                if channel.endswith("rotation")
            ]
        )
        for target_index, sample_time in enumerate(sample_times):
            right = int(np.searchsorted(source_times, sample_time, side="right"))
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
                else (sample_time - source_times[left]) / span
            )
            resampled[target_index, indices] = blend_euler_degrees(
                first_angles=source_values[left, indices],
                second_angles=source_values[right, indices],
                order=order,
                alpha=float(alpha),
            )
