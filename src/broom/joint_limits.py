"""Joint limit helpers for BVH metadata."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_JOINT_LIMITS_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "mujoco_humanoid_joint_limits.json"
)


@dataclass(frozen=True)
class JointLimit:
    """Resolved limits for one BVH joint."""

    dof: int
    min_values: tuple[float, ...]
    max_values: tuple[float, ...]


def load_joint_limits(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a joint-limit JSON file and index entries by normalized aliases."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    indexed: dict[str, dict[str, Any]] = {}
    for name, entry in data.get("joints", {}).items():
        aliases = (name, *entry.get("aliases", ()))
        for alias in aliases:
            indexed[_joint_key(alias)] = entry
    return indexed


def resolve_joint_limit(
    name: str,
    channels: Sequence[str],
    joint_limits: Mapping[str, Mapping[str, Any]] | None,
) -> JointLimit:
    """Return channel-ordered rotation limits for a BVH joint."""

    if not joint_limits:
        return JointLimit(dof=0, min_values=(), max_values=())

    entry = joint_limits.get(_joint_key(name))
    if entry is None:
        return JointLimit(dof=0, min_values=(), max_values=())

    source_channels = tuple(str(channel) for channel in entry["channels"])
    source_min = tuple(float(value) for value in entry["min_values"])
    source_max = tuple(float(value) for value in entry["max_values"])
    if (
        len(source_channels) != len(source_min)
        or len(source_channels) != len(source_max)
    ):
        raise ValueError(
            f"Joint limit entry for {name!r} has inconsistent channel/value lengths."
        )

    minima_by_channel = dict(zip(source_channels, source_min, strict=True))
    maxima_by_channel = dict(zip(source_channels, source_max, strict=True))
    rotation_channels = tuple(
        channel for channel in channels if channel.endswith("rotation")
    )
    min_values = tuple(
        minima_by_channel[channel]
        for channel in rotation_channels
        if channel in minima_by_channel
    )
    max_values = tuple(
        maxima_by_channel[channel]
        for channel in rotation_channels
        if channel in maxima_by_channel
    )

    return JointLimit(
        dof=int(entry["dof"]),
        min_values=min_values,
        max_values=max_values,
    )


def _joint_key(name: str) -> str:
    """Normalize BVH and namespaced Mixamo-style joint names."""

    name = name.rsplit(":", maxsplit=1)[-1]
    return re.sub(r"[^a-z0-9]+", "", name.lower())
