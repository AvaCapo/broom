"""Frame-rate resampling operations."""

from __future__ import annotations

import numpy as np

from broom import Hierarchy, Motion
from broom.bvh.interpolation.utils import blend_euler_degrees # TODO: fix blend_euler_degrees import


def resample_fps(motion: Motion, target_fps: float) -> Motion:
    """Return a copy of the Motion resampled to ``target_fps``."""
    if not isinstance(motion, Motion):
        raise TypeError("motion must be a Motion.")

    target_fps = float(target_fps)
    if target_fps <= 0.0:
        raise ValueError("Target FPS must be positive.")

    target_frame_time = 1.0 / target_fps
    source_values = motion.values

    if motion.frame_count <= 1:
        return Motion(motion.hierarchy, source_values, target_frame_time)

    duration = (motion.frame_count - 1) * motion.frame_time
    target_frame_count = max(1, int(round(duration * target_fps)) + 1)

    source_times = np.arange(motion.frame_count, dtype=np.float64)
    source_times *= motion.frame_time
    target_times = np.linspace(
        0.0,
        duration,
        target_frame_count,
        dtype=np.float64,
    )

    resampled = np.empty(
        (target_frame_count, motion.hierarchy.total_channels),
        dtype=np.float64,
    )
    _resample_motion_channels(
        source_values=source_values,
        source_times=source_times,
        target_times=target_times,
        resampled=resampled,
    )
    _resample_rotation_channels(
        hierarchy=motion.hierarchy,
        source_values=source_values,
        source_times=source_times,
        target_times=target_times,
        resampled=resampled,
    )
    return Motion(motion.hierarchy, resampled, target_frame_time)


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
    hierarchy: Hierarchy,
    source_values: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
    resampled: np.ndarray,
) -> None:
    """Replace linearly interpolated rotation channels with SLERP results."""

    for joint in hierarchy.joints:
        indices = list(hierarchy.rotation_channel_indices(joint.name))
        if len(indices) != 3:
            continue
        order = np.asarray(
            [
                channel[0].lower()
                for channel in joint.channels
                if channel.endswith("rotation")
            ]
        )
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


# TODO: Correct the existing numerical behavior before treating resampling as final.
# - ``linspace`` can disagree with the recorded ``1 / target_fps`` time grid.
# - Euler orders are converted to lowercase and differ from FK's intrinsic order.
# - ValueError silently falls back from SLERP to channel-space interpolation.
# - One- and two-axis rotation groups use baseline linear interpolation.
# - Wrapped Euler output loses winding information.
