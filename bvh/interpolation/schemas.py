import numpy as np
from dataclasses import dataclass

from bvh.schemas import BVHDocument




@dataclass(frozen=True)
class InterpolationResult:
    """Result of BVH interpolation."""

    document: BVHDocument
    motion_values: np.ndarray
