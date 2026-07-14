"""Numerical calculus helpers for time-series analysis."""

from typing import Optional

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

try:
    from scipy.integrate import cumulative_simpson
except Exception:  # pragma: no cover - optional SciPy feature
    cumulative_simpson = None


def diff_five_point(x: np.ndarray, dt: float) -> np.ndarray:
    """5-point finite difference for the first derivative along axis 0."""

    t = x.shape[0]
    if t <= 1:
        return np.zeros_like(x)
    if t < 5:
        return np.gradient(x, dt, axis=0, edge_order=min(2, t - 1))

    dx = np.zeros_like(x)
    dx[0] = (
        -25.0 * x[0]
        + 48.0 * x[1]
        - 36.0 * x[2]
        + 16.0 * x[3]
        - 3.0 * x[4]
    ) / (12.0 * dt)
    dx[1] = (
        -3.0 * x[0]
        - 10.0 * x[1]
        + 18.0 * x[2]
        - 6.0 * x[3]
        + 1.0 * x[4]
    ) / (12.0 * dt)
    dx[2:-2] = (
        -x[4:]
        + 8.0 * x[3:-1]
        - 8.0 * x[1:-3]
        + x[0:-4]
    ) / (12.0 * dt)
    dx[-2] = (
        3.0 * x[-1]
        + 10.0 * x[-2]
        - 18.0 * x[-3]
        + 6.0 * x[-4]
        - 1.0 * x[-5]
    ) / (12.0 * dt)
    dx[-1] = (
        25.0 * x[-1]
        - 48.0 * x[-2]
        + 36.0 * x[-3]
        - 16.0 * x[-4]
        + 3.0 * x[-5]
    ) / (12.0 * dt)
    return dx


def _gradient_n(
    x: np.ndarray,
    dt: float,
    order: int,
    *,
    edge_order: int = 2,
) -> np.ndarray:
    out = x
    for _ in range(order):
        t = out.shape[0]
        edge_order_eff = min(max(1, int(edge_order)), 2, max(1, t - 1))
        out = np.gradient(out, dt, axis=0, edge_order=edge_order_eff)
    return out


def _five_point_n(x: np.ndarray, dt: float, order: int) -> np.ndarray:
    out = x
    for _ in range(order):
        out = diff_five_point(out, dt)
    return out


def _savgol_n(
    x: np.ndarray,
    dt: float,
    order: int,
    *,
    window: int = 9,
    polyorder: int = 3,
    mode: str = "interp",
) -> np.ndarray:
    t = x.shape[0]
    if t <= 1:
        return np.zeros_like(x)

    window = int(window)
    polyorder = int(polyorder)
    if window % 2 != 1 or window < 3:
        raise ValueError("window must be an odd integer >= 3.")
    if polyorder < 1 or polyorder >= window:
        raise ValueError("polyorder must be >= 1 and < window.")
    if order > polyorder:
        raise ValueError("order must be <= polyorder for Savitzky-Golay.")
    if t < window:
        return _gradient_n(x, dt, order, edge_order=2)

    return savgol_filter(
        x,
        window_length=window,
        polyorder=polyorder,
        deriv=order,
        delta=dt,
        axis=0,
        mode=str(mode),
    ).astype(x.dtype, copy=False)


def derivative(
    x: np.ndarray,
    dt: float,
    *,
    order: int = 1,
    method: str = "gradient",
    edge_order: int = 2,
    window: int = 9,
    polyorder: int = 3,
    mode: str = "interp",
) -> np.ndarray:
    """Compute a time derivative of order ``order`` along axis 0."""

    if order < 0:
        raise ValueError("order must be >= 0.")
    if order == 0:
        return np.asarray(x)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite number.")

    method_normalized = str(method).lower()
    if method_normalized == "gradient":
        return _gradient_n(x, dt, order, edge_order=edge_order)
    if method_normalized in ("five_point", "five-point", "fivepoint"):
        return _five_point_n(x, dt, order)
    if method_normalized == "savgol":
        return _savgol_n(
            x,
            dt,
            order,
            window=window,
            polyorder=polyorder,
            mode=mode,
        )
    raise ValueError(
        "method must be 'gradient', 'five_point', or 'savgol'."
    )


def integrate_trapezoid(
    y: np.ndarray,
    dt: float,
    *,
    y0: Optional[np.ndarray] = None,
    axis: int = 0,
) -> np.ndarray:
    """Cumulative trapezoidal integration along the selected axis."""

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite number.")
    if y.size == 0:
        raise ValueError("Input cannot be empty.")

    y64 = y.astype(np.float64, copy=False)
    sample0 = np.take(y64, 0, axis=axis)
    y0_array = (
        np.zeros_like(sample0, dtype=np.float64)
        if y0 is None
        else np.asarray(y0, dtype=np.float64)
    )
    out = cumulative_trapezoid(y64, dx=dt, axis=axis, initial=0.0)
    out = out + np.expand_dims(y0_array, axis=axis)
    return out.astype(y.dtype, copy=False)


def integrate_rect(
    y: np.ndarray,
    dt: float,
    *,
    y0: Optional[np.ndarray] = None,
    axis: int = 0,
    variant: str = "left",
) -> np.ndarray:
    """Cumulative rectangular integration along the selected axis."""

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite number.")
    if y.size == 0:
        raise ValueError("Input cannot be empty.")

    y64 = y.astype(np.float64, copy=False)
    sample0 = np.take(y64, 0, axis=axis)
    y0_array = (
        np.zeros_like(sample0, dtype=np.float64)
        if y0 is None
        else np.asarray(y0, dtype=np.float64)
    )

    n = y64.shape[axis]
    out = np.empty_like(y64)
    idx0 = [slice(None)] * y64.ndim
    idx0[axis] = 0
    out[tuple(idx0)] = y0_array
    if n == 1:
        return out.astype(y.dtype, copy=False)

    variant_normalized = str(variant).lower()
    if variant_normalized not in ("left", "right"):
        raise ValueError("variant must be 'left' or 'right'.")

    sl_out = [slice(None)] * y64.ndim
    sl_out[axis] = slice(1, None)
    sl_src = [slice(None)] * y64.ndim
    sl_src[axis] = (
        slice(0, -1)
        if variant_normalized == "left"
        else slice(1, None)
    )

    increments = y64[tuple(sl_src)] * dt
    csum = np.cumsum(increments, axis=axis)
    out[tuple(sl_out)] = np.expand_dims(y0_array, axis=axis) + csum
    return out.astype(y.dtype, copy=False)


def integrate_simpson(
    y: np.ndarray,
    dt: float,
    *,
    y0: Optional[np.ndarray] = None,
    axis: int = 0,
) -> np.ndarray:
    """Cumulative Simpson integration along the selected axis."""

    if cumulative_simpson is None:
        raise RuntimeError(
            "cumulative_simpson is not available in this SciPy version."
        )
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive finite number.")
    if y.size == 0:
        raise ValueError("Input cannot be empty.")

    y64 = y.astype(np.float64, copy=False)
    sample0 = np.take(y64, 0, axis=axis)
    y0_array = (
        np.zeros_like(sample0, dtype=np.float64)
        if y0 is None
        else np.asarray(y0, dtype=np.float64)
    )
    out = cumulative_simpson(y64, dx=dt, axis=axis, initial=0.0)
    out = out + np.expand_dims(y0_array, axis=axis)
    return out.astype(y.dtype, copy=False)


def drift_correct_to_reference(
    x: np.ndarray,
    x_ref: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply a linear endpoint drift correction relative to ``x_ref``."""

    if x.shape != x_ref.shape:
        raise ValueError("x and x_ref must have the same shape.")
    if not np.isfinite(strength):
        raise ValueError("strength must be finite.")

    t = x.shape[0]
    if t <= 1:
        return x.copy()

    x64 = x.astype(np.float64, copy=False)
    x_ref64 = x_ref.astype(np.float64, copy=False)
    e0 = x_ref64[0] - x64[0]
    e1 = x_ref64[-1] - x64[-1]
    weights = np.linspace(0.0, 1.0, t, dtype=np.float64).reshape(
        (t,) + (1,) * (x.ndim - 1)
    )
    trend = e0 + (e1 - e0) * weights
    corrected = x64 + float(strength) * trend
    return corrected.astype(x.dtype, copy=False)


def integrate(
    y: np.ndarray,
    dt: float,
    *,
    method: str = "trapezoid",
    axis: int = 0,
    y0: Optional[np.ndarray] = None,
    variant: str = "left",
) -> np.ndarray:
    """Unified cumulative integration API."""

    method_normalized = str(method).lower()
    if method_normalized in ("trapezoid", "trapz", "trap"):
        return integrate_trapezoid(y, dt, y0=y0, axis=axis)
    if method_normalized in ("rect", "rectangle"):
        return integrate_rect(y, dt, y0=y0, axis=axis, variant=variant)
    if method_normalized in ("simpson", "cumulative_simpson"):
        return integrate_simpson(y, dt, y0=y0, axis=axis)
    raise ValueError("method must be 'trapezoid', 'rect', or 'simpson'.")
