"""Numpy helpers specific to PCA in-place BVH transforms."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from broom.math_helpers import moving_average


def validate_axis_set(
    axes: Sequence[str],
    expected_axes: Sequence[str],
    context: str,
) -> None:
    """Validate that the provided axes match the expected axis set."""

    axes = tuple(axes)
    expected_axes = tuple(expected_axes)
    if len(axes) != len(expected_axes) or set(axes) != set(expected_axes):
        raise ValueError(
            f"{context} expects axes: {', '.join(expected_axes)}."
        )


def remove_smoothed_pca_trend(
    root_points: np.ndarray,
    sensor_points: np.ndarray,
    direction: np.ndarray,
    smooth_window: int,
    zero_origin: bool,
) -> np.ndarray:
    """Remove the smoothed PCA trend from the root points to preserve
        the original trajectory shape.

    :param root_points: An array of shape (N, 2) representing the
        root X/Z points.
    :param sensor_points: An array of shape (N, 2) representing
        the sensor X/Z points used for PCA.
    :param direction: An array of shape (2,) representing the principal
        direction of the sensor points.
    :param smooth_window: Smoothing window size for the PCA trend.
        Even values are rounded up to the next odd value.
    :param zero_origin: If True, the output points will be shifted
        to start at the origin (0, 0
    """
    forward = (sensor_points - sensor_points[0]) @ direction
    trend = moving_average(forward, smooth_window)
    correction = (trend - trend[0])[:, None] * direction[None, :]
    transformed = root_points - correction

    if zero_origin:
        return transformed - transformed[0]
    return transformed + (root_points[0] - transformed[0])
