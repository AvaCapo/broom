"""Euler-angle interpolation helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def blend_euler_degrees(
    first_angles: np.ndarray,
    second_angles: np.ndarray,
    order: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend Euler angles in degrees with SciPy quaternion SLERP."""
    order_string = "".join(str(axis) for axis in order)
    key_rotations = Rotation.from_euler(
        order_string,
        np.vstack([first_angles, second_angles]),
        degrees=True,
    )
    blended = Slerp([0.0, 1.0], key_rotations)([float(alpha)])
    return wrap_degrees(blended.as_euler(order_string, degrees=True)[0])


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    """Wrap angles to the [-180, 180) interval."""
    return (values + 180.0) % 360.0 - 180.0
