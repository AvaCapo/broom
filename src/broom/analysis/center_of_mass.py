"""Center-of-mass, joint-center, and stability utilities."""

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from ._validation import resolve_time_range, validate_joints_world


_DIMS_IDX = {"x": 0, "y": 1, "z": 2}


def validate_plane(plane: Sequence[str]) -> Tuple[str, str]:
    if len(plane) != 2:
        raise ValueError("plane must contain exactly 2 axes, e.g. ('x', 'z').")
    a0, a1 = str(plane[0]).lower(), str(plane[1]).lower()
    if a0 not in _DIMS_IDX or a1 not in _DIMS_IDX:
        raise ValueError("plane axes must be chosen from {'x', 'y', 'z'}.")
    if a0 == a1:
        raise ValueError("plane axes must be different.")
    return a0, a1


def validate_joint_indices(joint_indices: Sequence[int], joint_count: int) -> np.ndarray:
    if len(joint_indices) == 0:
        raise ValueError("joint_indices must contain at least one joint.")
    idx = np.asarray(joint_indices, dtype=int)
    if np.any(idx < 0) or np.any(idx >= joint_count):
        raise IndexError("joint_indices contain out-of-range values.")
    return idx


def validate_weights(weights: Sequence[float], expected_len: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != expected_len:
        raise ValueError("weights must be a 1D array with the same length as joint_indices.")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite.")
    if np.allclose(np.sum(w), 0.0):
        raise ValueError("weights must not sum to zero.")
    return w


def get_weighted_joint_center(
    joints_world: np.ndarray,
    joint_indices: Sequence[int],
    *,
    weights: Optional[Sequence[float]] = None,
    t_range: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Return weighted centroid trajectory [T_sel, 3] for selected joints.

    If weights is None, all selected joints are weighted equally.
    """
    t, j = validate_joints_world(joints_world)
    idx = validate_joint_indices(joint_indices, j)
    sl = resolve_time_range(t, t_range)

    traj = joints_world[sl][:, idx, :].astype(np.float64, copy=False)
    if weights is None:
        w = np.ones((len(idx),), dtype=np.float64)
    else:
        w = validate_weights(weights, len(idx))

    w = w / np.sum(w)
    return np.einsum("tjd,j->td", traj, w)


def get_joint_planar_trajectory(
    joints_world: np.ndarray,
    joint_idx: int,
    *,
    plane: Sequence[str] = ("x", "z"),
    t_range: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Return planar trajectory [T_sel, 2] for one joint.
    """
    t, j = validate_joints_world(joints_world)
    if not (0 <= int(joint_idx) < j):
        raise IndexError("joint_idx is out of range.")

    a0, a1 = validate_plane(plane)
    sl = resolve_time_range(t, t_range)
    dims = (_DIMS_IDX[a0], _DIMS_IDX[a1])
    return joints_world[sl, int(joint_idx), :][:, dims].astype(np.float64, copy=False)


def get_weighted_joint_center_trajectory(
    joints_world: np.ndarray,
    joint_indices: Sequence[int],
    *,
    weights: Optional[Sequence[float]] = None,
    plane: Sequence[str] = ("x", "z"),
    t_range: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Return weighted centroid trajectory [T_sel, 2] for selected joints.

    If weights is None, all selected joints are weighted equally.
    """
    a0, a1 = validate_plane(plane)
    dims = (_DIMS_IDX[a0], _DIMS_IDX[a1])
    center = get_weighted_joint_center(
        joints_world,
        joint_indices,
        weights=weights,
        t_range=t_range,
    )
    return center[:, dims]


def compute_center_velocity(
    center: np.ndarray,
    *,
    fps: float,
) -> np.ndarray:
    """
    Compute center velocity with central differences.
    """
    arr = np.asarray(center, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError("center must have shape [T, 2] or [T, 3].")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number.")

    dt = 1.0 / float(fps)
    return np.gradient(arr, dt, axis=0)


def compute_xcom(
    center: np.ndarray,
    *,
    fps: float,
    com_height: Optional[float | np.ndarray] = None,
    gravity: float = 9.81,
) -> np.ndarray:
    """
    Compute extrapolated center of mass (XCoM).

    Formula:
      XCoM = CoM + V_CoM * sqrt(l / g)

    If `center` is [T, 3] and `com_height` is None, the vertical coordinate
    `center[:, 1]` is used as `l`. If `center` is planar [T, 2], `com_height`
    must be provided.
    """
    arr = np.asarray(center, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError("center must have shape [T, 2] or [T, 3].")
    if not np.isfinite(gravity) or gravity <= 0:
        raise ValueError("gravity must be a positive finite number.")

    velocity = compute_center_velocity(arr, fps=fps)

    if com_height is None:
        if arr.shape[1] != 3:
            raise ValueError("com_height must be provided when center has shape [T, 2].")
        height = arr[:, 1]
    else:
        height = np.asarray(com_height, dtype=np.float64)
        if height.ndim == 0:
            height = np.full((arr.shape[0],), float(height), dtype=np.float64)
        if height.ndim != 1 or height.shape[0] != arr.shape[0]:
            raise ValueError("com_height must be a scalar or an array with shape [T].")

    height = np.maximum(height, 1.0e-6)
    pendulum_time = np.sqrt(height / float(gravity))[:, None]
    return arr + velocity * pendulum_time


def compute_com_support_offset(
    com_trajectory: np.ndarray,
    support_bounds: np.ndarray | Mapping[str, np.ndarray],
    *,
    plane: Sequence[str] = ("x", "z"),
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """
    Measure how far projected CoM is from the center of support bounds.

    Returns offsets relative to the support-rectangle center and Euclidean
    distance on the chosen support plane.
    """
    com = np.asarray(com_trajectory, dtype=np.float64)
    if com.ndim != 2 or com.shape[1] not in (2, 3):
        raise ValueError("com_trajectory must have shape [T, 2] or [T, 3].")

    axis0_name, axis1_name = validate_plane(plane)
    dims = (_DIMS_IDX[axis0_name], _DIMS_IDX[axis1_name])
    com_proj = com if com.shape[1] == 2 else com[:, dims]

    if isinstance(support_bounds, Mapping):
        axis0_min = np.asarray(support_bounds["axis0_min"], dtype=np.float64)
        axis0_max = np.asarray(support_bounds["axis0_max"], dtype=np.float64)
        axis1_min = np.asarray(support_bounds["axis1_min"], dtype=np.float64)
        axis1_max = np.asarray(support_bounds["axis1_max"], dtype=np.float64)
    else:
        bounds = np.asarray(support_bounds, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 4:
            raise ValueError("support_bounds must have shape [T, 4].")
        axis0_min, axis0_max, axis1_min, axis1_max = bounds.T

    if axis0_min.shape[0] != com_proj.shape[0]:
        raise ValueError("support_bounds and com_trajectory must have the same number of frames.")

    center = np.column_stack(
        [
            0.5 * (axis0_min + axis0_max),
            0.5 * (axis1_min + axis1_max),
        ]
    )
    offset = com_proj - center
    distance = np.linalg.norm(offset, axis=1)

    half_extent = np.column_stack(
        [
            0.5 * (axis0_max - axis0_min),
            0.5 * (axis1_max - axis1_min),
        ]
    )
    half_diagonal = np.linalg.norm(half_extent, axis=1)
    half_diagonal = np.where(half_diagonal > 1.0e-8, half_diagonal, np.nan)

    inside_mask = (
        (com_proj[:, 0] >= axis0_min)
        & (com_proj[:, 0] <= axis0_max)
        & (com_proj[:, 1] >= axis1_min)
        & (com_proj[:, 1] <= axis1_max)
    )

    result = {
        "com_projection": com_proj,
        "support_center": center,
        "offset_vector": offset,
        "distance_to_center": distance,
        "inside_support": inside_mask,
    }
    if normalize:
        result["normalized_distance_to_center"] = distance / half_diagonal

    return result


def compute_com_outside_support_distance(
    com_trajectory: np.ndarray,
    support_bounds: np.ndarray | Mapping[str, np.ndarray],
    *,
    plane: Sequence[str] = ("x", "z"),
    normalize: bool = False,
) -> dict[str, np.ndarray]:
    """
    Measure how far projected CoM lies outside rectangular support bounds.

    The metric is:
    - 0 when CoM is inside the support rectangle or on its boundary
    - > 0 when CoM is outside
    """
    com = np.asarray(com_trajectory, dtype=np.float64)
    if com.ndim != 2 or com.shape[1] not in (2, 3):
        raise ValueError("com_trajectory must have shape [T, 2] or [T, 3].")

    axis0_name, axis1_name = validate_plane(plane)
    dims = (_DIMS_IDX[axis0_name], _DIMS_IDX[axis1_name])
    com_proj = com if com.shape[1] == 2 else com[:, dims]

    if isinstance(support_bounds, Mapping):
        axis0_min = np.asarray(support_bounds["axis0_min"], dtype=np.float64)
        axis0_max = np.asarray(support_bounds["axis0_max"], dtype=np.float64)
        axis1_min = np.asarray(support_bounds["axis1_min"], dtype=np.float64)
        axis1_max = np.asarray(support_bounds["axis1_max"], dtype=np.float64)
    else:
        bounds = np.asarray(support_bounds, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 4:
            raise ValueError("support_bounds must have shape [T, 4].")
        axis0_min, axis0_max, axis1_min, axis1_max = bounds.T

    if axis0_min.shape[0] != com_proj.shape[0]:
        raise ValueError("support_bounds and com_trajectory must have the same number of frames.")

    axis0_excess = np.maximum(axis0_min - com_proj[:, 0], com_proj[:, 0] - axis0_max)
    axis1_excess = np.maximum(axis1_min - com_proj[:, 1], com_proj[:, 1] - axis1_max)
    axis0_excess = np.maximum(axis0_excess, 0.0)
    axis1_excess = np.maximum(axis1_excess, 0.0)

    outside_vector = np.column_stack([axis0_excess, axis1_excess])
    outside_distance = np.linalg.norm(outside_vector, axis=1)
    inside_mask = (axis0_excess == 0.0) & (axis1_excess == 0.0)

    result = {
        "com_projection": com_proj,
        "outside_vector": outside_vector,
        "outside_distance": outside_distance,
        "inside_support": inside_mask,
        "outside_support": ~inside_mask,
    }

    if normalize:
        half_extent = np.column_stack(
            [
                0.5 * (axis0_max - axis0_min),
                0.5 * (axis1_max - axis1_min),
            ]
        )
        half_diagonal = np.linalg.norm(half_extent, axis=1)
        half_diagonal = np.where(half_diagonal > 1.0e-8, half_diagonal, np.nan)
        result["normalized_outside_distance"] = outside_distance / half_diagonal

    return result


def compute_margin_of_stability(
    center_trajectory: np.ndarray,
    support_bounds: np.ndarray | Mapping[str, np.ndarray],
    *,
    plane: Sequence[str] = ("x", "z"),
) -> dict[str, np.ndarray]:
    """
    Compute signed margin of stability (MoS) for rectangular support bounds.

    In the classical formulation this function should be called with `XCoM`.
    The implementation itself is generic and accepts any projected center-like
    trajectory with shape [T, 2] or [T, 3].

    Returns signed margins:
    - positive inside support
    - zero on the boundary
    - negative outside support
    """
    center = np.asarray(center_trajectory, dtype=np.float64)
    if center.ndim != 2 or center.shape[1] not in (2, 3):
        raise ValueError("center_trajectory must have shape [T, 2] or [T, 3].")

    axis0_name, axis1_name = validate_plane(plane)
    dims = (_DIMS_IDX[axis0_name], _DIMS_IDX[axis1_name])
    center_proj = center if center.shape[1] == 2 else center[:, dims]

    if isinstance(support_bounds, Mapping):
        axis0_min = np.asarray(support_bounds["axis0_min"], dtype=np.float64)
        axis0_max = np.asarray(support_bounds["axis0_max"], dtype=np.float64)
        axis1_min = np.asarray(support_bounds["axis1_min"], dtype=np.float64)
        axis1_max = np.asarray(support_bounds["axis1_max"], dtype=np.float64)
    else:
        bounds = np.asarray(support_bounds, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 4:
            raise ValueError("support_bounds must have shape [T, 4].")
        axis0_min, axis0_max, axis1_min, axis1_max = bounds.T

    if axis0_min.shape[0] != center_proj.shape[0]:
        raise ValueError("support_bounds and center_trajectory must have the same number of frames.")

    margin_axis0_lower = center_proj[:, 0] - axis0_min
    margin_axis0_upper = axis0_max - center_proj[:, 0]
    margin_axis1_lower = center_proj[:, 1] - axis1_min
    margin_axis1_upper = axis1_max - center_proj[:, 1]

    mos_axis0 = np.minimum(margin_axis0_lower, margin_axis0_upper)
    mos_axis1 = np.minimum(margin_axis1_lower, margin_axis1_upper)
    mos_min = np.minimum(mos_axis0, mos_axis1)

    inside_support = (
        (center_proj[:, 0] >= axis0_min)
        & (center_proj[:, 0] <= axis0_max)
        & (center_proj[:, 1] >= axis1_min)
        & (center_proj[:, 1] <= axis1_max)
    )

    return {
        "center_projection": center_proj,
        "mos_axis0": mos_axis0,
        "mos_axis1": mos_axis1,
        "mos_min": mos_min,
        "inside_support": inside_support,
    }
