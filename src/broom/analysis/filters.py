"""Filtering helpers for motion-analysis signals."""

import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt
except Exception:  # pragma: no cover - SciPy availability depends on env
    butter = None
    sosfiltfilt = None


def lowpass_butter(
    x: np.ndarray,
    fps: float,
    cutoff_hz: float = 6.0,
    order: int = 2,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter along axis 0."""

    if butter is None or sosfiltfilt is None:
        raise RuntimeError("SciPy is required for lowpass_butter.")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number.")
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be a positive finite number.")

    nyquist = 0.5 * float(fps)
    normalized_cutoff = float(cutoff_hz) / nyquist
    if normalized_cutoff >= 1.0:
        raise ValueError("cutoff_hz must be less than the Nyquist frequency.")

    sos = butter(int(order), normalized_cutoff, btype="lowpass", output="sos")
    return sosfiltfilt(sos, x, axis=0)
