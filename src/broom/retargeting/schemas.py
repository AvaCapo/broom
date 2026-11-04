from dataclasses import dataclass

from broom import Motion


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
    """Result of automatic motion retargeting onto a target hierarchy."""

    motion: Motion
    joint_map: dict[str, str]
    unmapped_source_joints: tuple[str, ...]
    unmapped_target_joints: tuple[str, ...]
    scale: float
