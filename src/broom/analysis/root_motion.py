from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .center_of_mass import (
    _DIMS_IDX,
    get_joint_planar_trajectory,
    get_weighted_joint_center_trajectory,
    validate_plane as _validate_plane,
)
from ._validation import resolve_time_range, validate_joints_world


@dataclass
class PlanarTrendResult:
    source_name: str
    joint_idx: Optional[int]
    plane: Tuple[str, str]
    degree: int
    frame_indices: np.ndarray
    time_normalized: np.ndarray
    trajectory: np.ndarray
    trend: np.ndarray
    oscillation: np.ndarray
    coefficients: np.ndarray


def _normalized_time(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros((n,), dtype=np.float64)
    return np.linspace(-1.0, 1.0, n, dtype=np.float64)


def fit_planar_polynomial_trend(
    trajectory: np.ndarray,
    *,
    degree: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit polynomial trend independently for both planar coordinates.

    Returns:
      trend: [T, 2]
      coefficients: [2, degree+1]
      time_normalized: [T]
    """
    if degree < 0:
        raise ValueError("degree must be >= 0.")
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("trajectory must have shape [T, 2].")

    n = trajectory.shape[0]
    degree_eff = min(int(degree), max(0, n - 1))
    tau = _normalized_time(n)

    coeffs = np.empty((2, degree_eff + 1), dtype=np.float64)
    trend = np.empty_like(trajectory, dtype=np.float64)

    for dim in range(2):
        coeff = np.polyfit(tau, trajectory[:, dim], degree_eff)
        coeffs[dim, :] = coeff
        trend[:, dim] = np.polyval(coeff, tau)

    return trend, coeffs, tau


def decompose_planar_trajectory(
    trajectory: np.ndarray,
    *,
    degree: int = 1,
    plane: Sequence[str] = ("x", "z"),
    frame_indices: Optional[np.ndarray] = None,
    source_name: str = "trajectory",
    joint_idx: Optional[int] = None,
) -> PlanarTrendResult:
    """
    Decompose any planar trajectory [T, 2] into polynomial trend and oscillation.
    """
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("trajectory must have shape [T, 2].")

    trend, coeffs, tau = fit_planar_polynomial_trend(trajectory, degree=degree)
    if frame_indices is None:
        frames = np.arange(trajectory.shape[0], dtype=np.int64)
    else:
        frames = np.asarray(frame_indices, dtype=np.int64)
        if frames.shape[0] != trajectory.shape[0]:
            raise ValueError("frame_indices length must match trajectory length.")

    trajectory_f64 = np.asarray(trajectory, dtype=np.float64)
    return PlanarTrendResult(
        source_name=str(source_name),
        joint_idx=None if joint_idx is None else int(joint_idx),
        plane=_validate_plane(plane),
        degree=min(int(degree), max(0, trajectory.shape[0] - 1)),
        frame_indices=frames,
        time_normalized=tau,
        trajectory=trajectory_f64,
        trend=trend,
        oscillation=trajectory_f64 - trend,
        coefficients=coeffs,
    )


def decompose_joint_planar_motion(
    joints_world: np.ndarray,
    joint_idx: int,
    *,
    degree: int = 1,
    plane: Sequence[str] = ("x", "z"),
    t_range: Optional[Tuple[int, int]] = None,
) -> PlanarTrendResult:
    """
    Decompose planar joint motion into polynomial trend and residual oscillation.
    """
    t, _ = validate_joints_world(joints_world)
    sl = resolve_time_range(t, t_range)
    trajectory = get_joint_planar_trajectory(
        joints_world,
        joint_idx,
        plane=plane,
        t_range=t_range,
    )
    frames = np.arange(t, dtype=np.int64)[sl]

    return decompose_planar_trajectory(
        trajectory,
        degree=degree,
        plane=plane,
        frame_indices=frames,
        source_name="joint",
        joint_idx=int(joint_idx),
    )


def decompose_weighted_joint_center_motion(
    joints_world: np.ndarray,
    joint_indices: Sequence[int],
    *,
    weights: Optional[Sequence[float]] = None,
    degree: int = 1,
    plane: Sequence[str] = ("x", "z"),
    t_range: Optional[Tuple[int, int]] = None,
) -> PlanarTrendResult:
    """
    Decompose weighted joint-center motion into polynomial trend and oscillation.

    Use weights=None for an unweighted centroid.
    """
    t, _ = validate_joints_world(joints_world)
    sl = resolve_time_range(t, t_range)
    trajectory = get_weighted_joint_center_trajectory(
        joints_world,
        joint_indices,
        weights=weights,
        plane=plane,
        t_range=t_range,
    )
    frames = np.arange(t, dtype=np.int64)[sl]

    return decompose_planar_trajectory(
        trajectory,
        degree=degree,
        plane=plane,
        frame_indices=frames,
        source_name="weighted_joint_center",
    )


def remove_joint_planar_trend_from_clip(
    joints_world: np.ndarray,
    joint_idx: int,
    *,
    degree: int = 1,
    plane: Sequence[str] = ("x", "z"),
    t_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Estimate planar trend from one joint and subtract it from all joints.
    """
    t, _ = validate_joints_world(joints_world)
    sl = resolve_time_range(t, t_range)
    a0, a1 = _validate_plane(plane)
    dims = (_DIMS_IDX[a0], _DIMS_IDX[a1])

    result = decompose_joint_planar_motion(
        joints_world,
        joint_idx,
        degree=degree,
        plane=(a0, a1),
        t_range=t_range,
    )

    joints_world_inplace = joints_world.astype(np.float64, copy=True)
    joints_world_inplace[sl, :, dims[0]] -= result.trend[:, 0][:, None]
    joints_world_inplace[sl, :, dims[1]] -= result.trend[:, 1][:, None]

    return {
        "joints_world_inplace": joints_world_inplace,
        "source_name": result.source_name,
        "trend": result.trend,
        "oscillation": result.oscillation,
        "trajectory": result.trajectory,
        "coefficients": result.coefficients,
        "frame_indices": result.frame_indices,
        "time_normalized": result.time_normalized,
    }


def remove_weighted_joint_center_trend_from_clip(
    joints_world: np.ndarray,
    joint_indices: Sequence[int],
    *,
    weights: Optional[Sequence[float]] = None,
    degree: int = 1,
    plane: Sequence[str] = ("x", "z"),
    t_range: Optional[Tuple[int, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Estimate planar trend from a weighted joint centroid and subtract it from all joints.

    Use weights=None for a plain centroid.
    """
    t, _ = validate_joints_world(joints_world)
    sl = resolve_time_range(t, t_range)
    a0, a1 = _validate_plane(plane)
    dims = (_DIMS_IDX[a0], _DIMS_IDX[a1])

    result = decompose_weighted_joint_center_motion(
        joints_world,
        joint_indices,
        weights=weights,
        degree=degree,
        plane=(a0, a1),
        t_range=t_range,
    )

    joints_world_inplace = joints_world.astype(np.float64, copy=True)
    joints_world_inplace[sl, :, dims[0]] -= result.trend[:, 0][:, None]
    joints_world_inplace[sl, :, dims[1]] -= result.trend[:, 1][:, None]

    return {
        "joints_world_inplace": joints_world_inplace,
        "source_name": result.source_name,
        "trend": result.trend,
        "oscillation": result.oscillation,
        "trajectory": result.trajectory,
        "coefficients": result.coefficients,
        "frame_indices": result.frame_indices,
        "time_normalized": result.time_normalized,
    }



@dataclass
class ReconstructionErrorResult:
    mean_error: float
    max_error: float
    rms_error: float
    per_frame_mean: np.ndarray
    per_joint_mean: np.ndarray
    per_point_error: np.ndarray


def reconstruction_error(
    joints_world_reference: np.ndarray,
    joints_world_reconstructed: np.ndarray,
    *,
    joint_indices: Optional[Sequence[int]] = None,
) -> ReconstructionErrorResult:
    """
    Compute reconstruction error between reference and reconstructed joint trajectories.

    Returns Euclidean position errors in world space.
    """
    t_ref, j_ref = validate_joints_world(joints_world_reference)
    t_rec, j_rec = validate_joints_world(joints_world_reconstructed)
    if (t_ref, j_ref) != (t_rec, j_rec):
        raise ValueError("reference and reconstructed arrays must have the same shape [T, J, 3].")

    ref = joints_world_reference.astype(np.float64, copy=False)
    rec = joints_world_reconstructed.astype(np.float64, copy=False)

    if joint_indices is not None:
        idx = np.asarray(joint_indices, dtype=int)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("joint_indices must be a non-empty 1D sequence.")
        if np.any(idx < 0) or np.any(idx >= j_ref):
            raise IndexError("joint_indices contain out-of-range values.")
        ref = ref[:, idx, :]
        rec = rec[:, idx, :]

    per_point_error = np.linalg.norm(ref - rec, axis=-1)  # [T, J_sel]
    per_frame_mean = np.mean(per_point_error, axis=1)
    per_joint_mean = np.mean(per_point_error, axis=0)

    return ReconstructionErrorResult(
        mean_error=float(np.mean(per_point_error)),
        max_error=float(np.max(per_point_error)),
        rms_error=float(np.sqrt(np.mean(np.square(per_point_error)))),
        per_frame_mean=per_frame_mean,
        per_joint_mean=per_joint_mean,
        per_point_error=per_point_error,
    )


def reconstruction_error_summary(
    joints_world_reference: np.ndarray,
    joints_world_reconstructed: np.ndarray,
    *,
    joint_indices: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """
    Lightweight dict summary for quick comparisons in notebooks.
    """
    result = reconstruction_error(
        joints_world_reference,
        joints_world_reconstructed,
        joint_indices=joint_indices,
    )
    return {
        "mean_error": result.mean_error,
        "max_error": result.max_error,
        "rms_error": result.rms_error,
    }
