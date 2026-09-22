import numpy as np
from dataclasses import dataclass
from broom.bvh.schemas import BVHDocument


@dataclass(frozen=True)
class JointMatch:
    """Resolved joint match with confidence metadata."""

    source: str
    target: str
    score: float
    reason: str


@dataclass(frozen=True)
class MappingResult:
    """Full output of automatic skeleton mapping."""

    mapping: dict[str, str]
    matches: tuple[JointMatch, ...]
    unmapped_sources: tuple[str, ...]
    unmapped_targets: tuple[str, ...]


@dataclass(frozen=True)
class RetargetResult:
    """Result of retargeting source motion onto a target BVH skeleton."""

    document: BVHDocument
    motion_values: np.ndarray
    joint_map: dict[str, str]
    unmapped_source_joints: tuple[str, ...]
    unmapped_target_joints: tuple[str, ...]
    root_scale: float


@dataclass(frozen=True)
class RotationChannels:
    """Absolute channel indices and Euler order for one joint."""

    indices: tuple[int, int, int]
    order: str
