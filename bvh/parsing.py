"""Parsing helpers for BVH hierarchy and motion data."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from bvh.joint_limits import (
    DEFAULT_JOINT_LIMITS_PATH,
    load_joint_limits,
    resolve_joint_limit,
)
from bvh.schemas import BVHJoint


def find_motion_index(
    lines: Sequence[str], motion_name: str = "MOTION"
) -> int:
    """Find the line index that starts the MOTION section."""

    for index, line in enumerate(lines):
        if line.strip().upper() == motion_name.upper():
            return index
    raise ValueError(f"BVH file does not contain a {motion_name} section.")


def find_motion_data_start(lines: Sequence[str], motion_index: int) -> int:
    """Find the line index where motion frame rows start."""

    for index in range(motion_index + 1, len(lines)):
        if re.match(
            r"^\s*Frame\s+Time\s*:", lines[index], flags=re.IGNORECASE
        ):
            return index + 1
    raise ValueError("BVH MOTION section does not contain a Frame Time line.")


def extract_frame_count(
    motion_lines: Sequence[str],
) -> int | None:
    """Extract the declared frame count from MOTION header lines."""

    for line in motion_lines:
        match = re.match(r"^\s*Frames\s*:\s*(\d+)\s*$", line)
        if match:
            return int(match.group(1))
    return None


def extract_frame_time(
    motion_lines: Sequence[str],
) -> float | None:
    """Extract the frame time from MOTION header lines."""

    for line in motion_lines:
        match = re.match(
            r"^\s*Frame\s+Time\s*:\s*([\-\d\.eE]+)\s*$", line
        )
        if match:
            return float(match.group(1))
    return None


def parse_hierarchy(
    hierarchy_lines: Sequence[str],
    joint_limits_path: str | Path | None = DEFAULT_JOINT_LIMITS_PATH,
) -> tuple[list[BVHJoint], int]:
    """Parse BVH hierarchy lines into joints and total channel count."""

    joint_limits = (
        load_joint_limits(joint_limits_path)
        if joint_limits_path is not None
        else None
    )
    joints: list[BVHJoint] = []
    stack: list[int | None] = []
    pending_joint: int | None = None
    pending_end_site = False
    channel_cursor = 0

    for line in hierarchy_lines:
        stripped = line.strip()
        if not stripped or stripped == "HIERARCHY":
            continue

        root_match = re.match(r"^ROOT\s+(.+?)\s*$", stripped)
        joint_match = re.match(r"^JOINT\s+(.+?)\s*$", stripped)
        if root_match or joint_match:
            parent = active_joint(stack)
            name = (root_match or joint_match).group(1)
            pending_joint = len(joints)
            joints.append(
                BVHJoint(
                    name=name,
                    parent=parent,
                    offset=np.zeros(3, dtype=np.float64),
                    channels=(),
                    channel_start=channel_cursor,
                )
            )
            continue

        if stripped == "End Site":
            pending_end_site = True
            continue

        if stripped == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
            elif pending_end_site:
                stack.append(None)
                pending_end_site = False
            continue

        if stripped == "}":
            if stack:
                stack.pop()
            continue

        if stack and stack[-1] is None:
            continue

        active = active_joint(stack)
        if active == -1:
            continue

        offset_match = re.match(
            r"^OFFSET\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)",
            stripped,
        )
        if offset_match:
            joints[active] = replace_joint(
                joints[active],
                offset=np.asarray(offset_match.groups(), dtype=np.float64),
            )
            continue

        channel_match = re.match(r"^CHANNELS\s+(\d+)\s+(.+?)\s*$", stripped)
        if channel_match:
            channel_count = int(channel_match.group(1))
            channels = tuple(channel_match.group(2).split())
            if len(channels) != channel_count:
                raise ValueError(
                    f"Joint '{joints[active].name}' declares {channel_count} "
                    f"channels, but {len(channels)} names were found."
                )
            joints[active] = replace_joint(
                joints[active],
                channels=channels,
                channel_start=channel_cursor,
                joint_limits=joint_limits,
            )
            channel_cursor += channel_count

    return joints, channel_cursor


def active_joint(stack: Sequence[int | None]) -> int:
    """Return the currently active joint index from the parser stack."""

    for joint_index in reversed(stack):
        if joint_index is not None:
            return joint_index
    return -1


def replace_joint(
    joint: BVHJoint,
    offset: np.ndarray | None = None,
    channels: tuple[str, ...] | None = None,
    channel_start: int | None = None,
    joint_limits: Mapping[str, Mapping[str, Any]] | None = None,
) -> BVHJoint:
    """Return a copy of a joint with selected fields replaced."""

    resolved_channels = joint.channels if channels is None else channels
    limit = resolve_joint_limit(joint.name, resolved_channels, joint_limits)
    return BVHJoint(
        name=joint.name,
        parent=joint.parent,
        offset=joint.offset if offset is None else offset,
        channels=resolved_channels,
        channel_start=(
            joint.channel_start if channel_start is None else channel_start
        ),
        dof=limit.dof if joint_limits is not None else joint.dof,
        min_values=(
            limit.min_values if joint_limits is not None else joint.min_values
        ),
        max_values=(
            limit.max_values if joint_limits is not None else joint.max_values
        ),
    )


def extract_root_channels(
    hierarchy_lines: Sequence[str],
    expected_root_name: str | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Extract the root joint name and root channel names."""

    root_name = None

    for line in hierarchy_lines:
        root_match = re.match(r"^\s*ROOT\s+(.+?)\s*$", line)
        if root_match:
            root_name = root_match.group(1)
            if (
                expected_root_name is not None
                and root_name != expected_root_name
            ):
                raise ValueError(
                    f"Expected root joint '{expected_root_name}', "
                    f"got '{root_name}'."
                )
            continue

        if root_name is None:
            continue

        channel_match = re.match(r"^\s*CHANNELS\s+(\d+)\s+(.+?)\s*$", line)
        if channel_match:
            channel_count = int(channel_match.group(1))
            channels = tuple(channel_match.group(2).split())
            if len(channels) != channel_count:
                raise ValueError(
                    f"Root CHANNELS declares {channel_count} channels, "
                    f"but {len(channels)} names were found."
                )
            return root_name, channels

    raise ValueError("BVH hierarchy does not contain root CHANNELS.")


def read_motion_rows(
    lines: Sequence[str],
    total_channels: int,
) -> tuple[list[tuple[tuple[str, ...], str]], np.ndarray]:
    """Read motion data rows and numeric values."""

    rows = []
    values = []

    for line in lines:
        if not line.strip():
            continue

        row, newline = split_newline(line)
        tokens = tuple(row.split())
        if len(tokens) != total_channels:
            raise ValueError(
                f"Motion row has {len(tokens)} values, but hierarchy declares "
                f"{total_channels} channels."
            )
        numeric = np.asarray(tokens, dtype=np.float64)
        if np.isinf(numeric).any():
            raise ValueError("Motion rows cannot contain infinite values.")
        rows.append((tokens, newline))
        values.append(numeric)

    if not values:
        return rows, np.empty((0, total_channels), dtype=np.float64)
    return rows, np.vstack(values)


def split_newline(line: str) -> tuple[str, str]:
    """Split a line into content and newline components."""

    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""
