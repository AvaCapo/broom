from typing import Optional, Tuple
import numpy as np

def validate_joints_world(joints_world: np.ndarray) -> Tuple[int, int]:
    if not isinstance(joints_world, np.ndarray):
        raise TypeError("joints_world must be a numpy array.")
    if joints_world.ndim != 3 or joints_world.shape[-1] != 3:
        raise ValueError("joints_world must have shape [T, J, 3].")
    if joints_world.dtype.kind not in ("f", "i", "u"):
        raise TypeError("joints_world must be numeric.")
    t, j, _ = joints_world.shape
    if t < 1 or j < 1:
        raise ValueError("joints_world must have T>=1 and J>=1.")
    return t, j

def validate_fps(fps: float) -> float:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number.")
    return float(fps)

def resolve_time_range(t: int, t_range: Optional[Tuple[int, int]]) -> slice:
    if t_range is None:
        return slice(0, t)
    if len(t_range) != 2:
        raise ValueError("t_range must be a (start, end) tuple.")
    start, end = t_range
    start = 0 if start is None else int(start)
    end = t if end is None else int(end)
    start = max(0, start)
    end = min(t, end)
    if end <= start:
        raise ValueError("t_range end must be greater than start.")
    return slice(start, end)
