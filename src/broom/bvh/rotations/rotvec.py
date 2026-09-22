"""Analytic differential of rotation-vector (SO(3)) coordinates."""

import numpy as np


def right_jacobian(rotvec: np.ndarray) -> np.ndarray:
    """Return Jr(phi), shape (..., 3, 3), for rotation vectors in radians.

    Exp(phi + delta) = Exp(phi) Exp(Jr(phi) delta) + O(||delta||**2).
    Taylor coefficients avoid cancellation and division by zero near identity.
    """
    phi = np.asarray(rotvec, dtype=np.float64)
    if phi.ndim == 0 or phi.shape[-1] != 3 or not np.all(np.isfinite(phi)):
        raise ValueError("rotvec must be finite with shape (..., 3).")
    theta2 = np.sum(phi * phi, axis=-1)
    small = theta2 < 1e-6
    safe = np.where(small, 1.0, theta2)
    theta = np.sqrt(safe)
    a = np.where(small, .5 - theta2 / 24 + theta2**2 / 720,
                 (1 - np.cos(theta)) / safe)
    b = np.where(small, 1 / 6 - theta2 / 120 + theta2**2 / 5040,
                 (theta - np.sin(theta)) / (safe * theta))
    skew = np.cross(np.eye(3), phi[..., None, :])
    return np.eye(3) - a[..., None, None] * skew + b[..., None, None] * (skew @ skew)
