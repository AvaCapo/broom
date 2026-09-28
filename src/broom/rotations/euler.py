"""Euler-angle interpolation helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# TODO: Define an explicit temporal Euler-branch policy for multi-axis
# interpolation. Canonical Euler output preserves orientation but cannot
# preserve winding across a sequence of samples by itself.
def blend_euler_degrees(
    first_angles: np.ndarray,
    second_angles: np.ndarray,
    order: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend declared-order Euler angles in degrees with SciPy SLERP.

    ``order`` uses uppercase intrinsic axis names, matching Broom FK and the
    declared order of BVH rotation channels.
    """
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


def axis_rotation_matrices(axis: str, radians: np.ndarray) -> np.ndarray:
    """Return rotation matrices for an axis and per-frame angles."""

    matrices = np.zeros((len(radians), 3, 3), dtype=np.float64)
    cos_v = np.cos(radians)
    sin_v = np.sin(radians)

    if axis == "X":
        matrices[:, 0, 0] = 1.0
        matrices[:, 1, 1] = cos_v
        matrices[:, 1, 2] = -sin_v
        matrices[:, 2, 1] = sin_v
        matrices[:, 2, 2] = cos_v
    elif axis == "Y":
        matrices[:, 0, 0] = cos_v
        matrices[:, 0, 2] = sin_v
        matrices[:, 1, 1] = 1.0
        matrices[:, 2, 0] = -sin_v
        matrices[:, 2, 2] = cos_v
    elif axis == "Z":
        matrices[:, 0, 0] = cos_v
        matrices[:, 0, 1] = -sin_v
        matrices[:, 1, 0] = sin_v
        matrices[:, 1, 1] = cos_v
        matrices[:, 2, 2] = 1.0
    else:
        raise ValueError(f"Unsupported rotation axis '{axis}'.")

    return matrices
