"""Data structures for BVH in-place transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

PCA_SOURCE_ROOT = "root"
PCA_SOURCE_BODY = "body"
PCASource = Literal["root", "body"]

DEFAULT_ROOT_AXES = ("Xposition", "Zposition")
DEFAULT_BODY_JOINT_WEIGHTS = (
    ("Hips", 2.0),
    ("Spine", 1.2),
    ("Spine1", 1.2),
    ("Spine2", 1.2),
    ("Neck", 0.8),
    ("Head", 0.8),
    ("LeftUpLeg", 1.0),
    ("RightUpLeg", 1.0),
    ("LeftLeg", 1.0),
    ("RightLeg", 1.0),
    ("LeftFoot", 1.3),
    ("RightFoot", 1.3),
    ("LeftToe", 1.3),
    ("RightToe", 1.3),
    ("LeftShoulder", 0.5),
    ("RightShoulder", 0.5),
)


@dataclass(frozen=True)
class InPlacePCAConfig:
    """Configuration for PCA-based in-place conversion."""

    root_name: str | None = "Hips"
    source: PCASource = PCA_SOURCE_BODY
    axes: tuple[str, str] = DEFAULT_ROOT_AXES
    smooth_window: int = 5
    zero_origin: bool = False
    precision: int = 6
    foot_blend_frames: int = 8
    foot_height_threshold: float | None = None
    foot_velocity_threshold: float | None = None
    body_joint_weights: Sequence[tuple[str, float]] = (
        DEFAULT_BODY_JOINT_WEIGHTS
    )
    use_all_joints: bool = False
