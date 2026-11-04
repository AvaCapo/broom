"""Math helpers shared across BVH processing modules."""

from __future__ import annotations

import numpy as np


def principal_direction(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Calculate the principal direction of the given 2D points using PCA.

    :param points: An array of shape (N, 2) representing the 2D points.
    """
    if points.shape[0] < 2:
        raise ValueError(
            "At least two points are required"
            "to calculate principal direction."
        )

    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / points.shape[0]
    trace = float(np.trace(covariance))
    if trace <= 1.0e-12:
        return endpoint_direction(points), 0.0

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest_index = int(np.argmax(eigenvalues))
    direction = normalize_2d(eigenvectors[:, largest_index])

    endpoint = points[-1] - points[0]
    if float(direction @ endpoint) < 0.0:
        direction = -direction

    explained_ratio = float(eigenvalues[largest_index] / trace)
    return direction, explained_ratio


def endpoint_direction(points: np.ndarray) -> np.ndarray:
    """Calculate the direction from the first point to the last point.

    :param points: An array of shape (N, 2) representing the 2D points.
    """
    direction = points[-1] - points[0]
    return normalize_2d(direction)

#TODO NEED to check those functions, maybe unify them 

def normalize_2d(vector: np.ndarray) -> np.ndarray:
    """Normalize a 2D vector to have unit length.

    :param vector: An array of shape (2,) representing the 2D vector.
    """
    return normalize_vectors(vector, fallback=np.array([1.0, 0.0]))



def normalize_vectors(
    v: np.ndarray,
    eps: float = 1e-12,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """
    Normalize vector(s) along the last axis.

    Works with:
    - shape (D,)
    - shape (N, D)
    - shape (..., D)

    Near-zero vectors are replaced with fallback.
    """
    v = np.asarray(v, dtype=np.float64)

    if fallback is None:
        fallback = np.zeros(v.shape[-1], dtype=np.float64)
        fallback[0] = 1.0
    else:
        fallback = np.asarray(fallback, dtype=np.float64)

    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    mask = norm <= eps

    normalized = v / np.where(mask, 1.0, norm)

    return np.where(mask, fallback, normalized)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a moving average filter to the given 1D values.

    :param values: An array of shape (N,) representing the input values.
    :param window: Smoothing window size. Even values are rounded up to
        the next odd value so the filter stays centered on the current frame.
    """
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window == 1 or len(values) <= 1:
        return values.copy()

    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def smoothstep(value: float) -> float:
    """Clamp the input to [0, 1] and apply smoothstep interpolation.

    :param value: The input value to smooth, typically in the range [0, 1].
    """
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)
