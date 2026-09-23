"""Editable scalar animation samples."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from broom.hierarchy import Hierarchy


def _owned_values(values: Any, hierarchy: Hierarchy) -> np.ndarray:
    """Return an owned scalar table matching a hierarchy."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != hierarchy.total_channels:
        raise ValueError(
            f"Motion values must have shape (frames, {hierarchy.total_channels})."
        )
    if array.shape[0] == 0:
        raise ValueError("Motion values must contain at least one frame.")
    if not np.isfinite(array).all():
        raise ValueError("Motion values must contain only finite values.")
    return np.array(array, dtype=np.float64, copy=True, order="C")


@dataclass(frozen=True, slots=True)
class Motion:
    """Store editable scalar animation samples for one hierarchy.

    Parameters
    ----------
    hierarchy:
        Skeleton topology that defines the scalar channel layout.
    values:
        Two-dimensional samples with shape ``(F, C)``, where ``F >= 1`` is the
        number of frames and ``C`` is ``hierarchy.total_channels``. Columns
        follow the hierarchy's joint order and each joint's declared channel
        order. Values are converted to an owned, C-contiguous ``float64``
        NumPy array.
    frame_time:
        Positive finite duration of one frame in seconds.

    Raises
    ------
    TypeError
        If ``hierarchy`` is not a :class:`Hierarchy`.
    ValueError
        If ``values`` is not a two-dimensional non-empty table with the
        required width, or if ``frame_time`` is not positive and finite.

    Notes
    -----
    The constructor copies ``values``. The stored array and rows returned by
    :meth:`frame` are writable; :meth:`with_values` creates a separate Motion
    with the same hierarchy and frame time.
    """

    hierarchy: Hierarchy
    values: Any
    frame_time: float

    def __post_init__(self) -> None:
        if not isinstance(self.hierarchy, Hierarchy):
            raise TypeError("Motion hierarchy must be a Hierarchy.")
        if self.hierarchy.total_channels == 0:
            raise ValueError("Motion hierarchy must declare at least one channel.")
        try:
            frame_time = float(self.frame_time)
        except (TypeError, ValueError) as error:
            raise ValueError("frame_time must be a positive finite number.") from error
        if (
            isinstance(self.frame_time, bool)
            or not np.isfinite(frame_time)
            or frame_time <= 0
        ):
            raise ValueError("frame_time must be a positive finite number.")
        object.__setattr__(self, "frame_time", frame_time)
        object.__setattr__(self, "values", _owned_values(self.values, self.hierarchy))

    @property
    def frame_count(self) -> int:
        """Return the number of stored frames."""
        return int(self.values.shape[0])

    @property
    def duration(self) -> float:
        """Return the duration between the first and last frame."""
        return max(self.frame_count - 1, 0) * self.frame_time

    def frame(self, index: int) -> np.ndarray:
        """Return a writable view of one frame.

        Assigning through the returned array updates :attr:`values` of this
        Motion instance.
        """
        return self.values[index]

    def with_values(self, values: Any) -> Motion:
        """Return an independent Motion with replacement samples."""
        return Motion(self.hierarchy, values, self.frame_time)
