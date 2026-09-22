from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from broom.bvh.in_place.config import PCASource


@dataclass(frozen=True)
class InPlacePCAResult:
    """Result metadata for a PCA-based in-place conversion."""

    input_path: Path
    output_path: Path
    root_name: str
    source: PCASource
    frames: int
    dominant_direction: tuple[float, float]
    explained_ratio: float
    selected_joints: tuple[str, ...]
    foot_lock: bool
