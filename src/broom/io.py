"""I/O helpers for loading and writing BVH documents."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from broom.hierarchy import Hierarchy, Joint
from broom.motion import Motion


def _find_motion_index(lines: Sequence[str], motion_name: str) -> int:
    """Return the line index of the MOTION section."""
    for index, line in enumerate(lines):
        if line.strip().upper() == motion_name.upper():
            return index
    raise ValueError(f"BVH file does not contain a {motion_name} section.")


def _find_motion_data_start(lines: Sequence[str], motion_index: int) -> int:
    """Return the first motion-data line index."""
    for index in range(motion_index + 1, len(lines)):
        if re.match(r"^\s*Frame\s+Time\s*:", lines[index], flags=re.IGNORECASE):
            return index + 1
    raise ValueError("BVH MOTION section does not contain a Frame Time line.")


def _extract_frame_count(lines: Sequence[str]) -> int | None:
    """Return the declared BVH frame count."""
    for line in lines:
        match = re.match(r"^\s*Frames\s*:\s*(\d+)\s*$", line)
        if match:
            return int(match.group(1))
    return None


def _extract_frame_time(lines: Sequence[str]) -> float | None:
    """Return the declared BVH frame duration."""
    for line in lines:
        match = re.match(r"^\s*Frame\s+Time\s*:\s*([-\d.eE]+)\s*$", line)
        if match:
            return float(match.group(1))
    return None


def _parse_hierarchy(lines: Sequence[str]) -> Hierarchy:
    """Parse BVH hierarchy text into a Hierarchy."""
    joints: list[Joint] = []
    stack: list[int | None] = []
    end_site_parents: list[int] = []
    pending_joint: int | None = None
    pending_end_site_parent: int | None = None
    offset_joints: set[int] = set()
    channel_joints: set[int] = set()
    end_site_offsets: set[int] = set()

    def normalize_name(name: str) -> str:
        normalized = re.sub(r"\s+", "_", name)
        if normalized != name:
            print(
                f"Renamed BVH joint {name!r} to {normalized!r} to improve "
                "compatibility with common BVH readers."
            )
        return normalized

    def active_joint() -> int:
        for index in reversed(stack):
            if index is not None:
                return index
        return -1

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == "HIERARCHY":
            continue
        match = re.match(r"^(ROOT|JOINT)\s+(.+?)\s*$", stripped)
        if match:
            kind, name = match.groups()
            if (kind == "ROOT" and joints) or (kind == "JOINT" and not joints):
                raise ValueError(
                    "BVH hierarchy must start with one ROOT followed by JOINT entries."
                )
            parent = active_joint()
            name = normalize_name(name)
            if any(joint.name == name for joint in joints):
                raise ValueError(
                    f"BVH joint name {name!r} is not unique after normalization."
                )
            pending_joint = len(joints)
            joints.append(Joint(name, parent, (0.0, 0.0, 0.0)))
            continue
        if stripped == "End Site":
            pending_end_site_parent = active_joint()
            if pending_end_site_parent == -1:
                raise ValueError("BVH End Site must belong to a joint.")
            continue
        if stripped == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
            elif pending_end_site_parent is not None:
                stack.append(None)
                end_site_parents.append(pending_end_site_parent)
                pending_end_site_parent = None
            else:
                raise ValueError("Unexpected opening brace in BVH hierarchy.")
            continue
        if stripped == "}":
            if not stack:
                raise ValueError("Unexpected closing brace in BVH hierarchy.")
            closed = stack.pop()
            if closed is None:
                parent = end_site_parents.pop()
                if parent not in end_site_offsets:
                    raise ValueError(
                        f"End Site for joint {joints[parent].name!r} has no OFFSET."
                    )
            continue
        active = active_joint()
        if active == -1:
            continue
        offset_match = re.match(
            r"^OFFSET\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s*$",
            stripped,
        )
        if offset_match:
            offset = tuple(float(value) for value in offset_match.groups())
            if stack and stack[-1] is None:
                parent = end_site_parents[-1]
                if joints[parent].end_site_offset is not None:
                    raise ValueError("A BVH joint may declare only one End Site.")
                joints[parent] = replace(joints[parent], end_site_offset=offset)
                end_site_offsets.add(parent)
            else:
                if active in offset_joints:
                    raise ValueError(
                        f"Joint {joints[active].name!r} declares OFFSET more than once."
                    )
                joints[active] = replace(joints[active], offset=offset)
                offset_joints.add(active)
            continue
        if stripped.startswith("OFFSET"):
            raise ValueError(
                "BVH OFFSET must contain exactly three numeric components."
            )
        channel_match = re.match(r"^CHANNELS\s+(\d+)\s+(.+?)\s*$", stripped)
        if channel_match:
            if active in channel_joints:
                raise ValueError(
                    f"Joint {joints[active].name!r} declares CHANNELS more than once."
                )
            count = int(channel_match.group(1))
            channels = tuple(channel_match.group(2).split())
            if len(channels) != count:
                raise ValueError(
                    f"Joint '{joints[active].name}' declares {count} channels, "
                    f"but {len(channels)} names were found."
                )
            joints[active] = replace(joints[active], channels=channels)
            channel_joints.add(active)
    if stack or pending_joint is not None or pending_end_site_parent is not None:
        raise ValueError("BVH hierarchy has unclosed or incomplete blocks.")
    if not joints:
        raise ValueError("BVH hierarchy does not contain joints.")
    missing_offsets = [
        joint.name for index, joint in enumerate(joints) if index not in offset_joints
    ]
    if missing_offsets:
        raise ValueError(f"BVH joints without OFFSET: {', '.join(missing_offsets)}.")
    hierarchy = Hierarchy(tuple(joints))
    # TODO: Verify whether a zero-length End Site is the right fallback for every BVH consumer.
    completed_joints = []
    for index, joint in enumerate(hierarchy.joints):
        if not hierarchy.children(index) and joint.end_site_offset is None:
            print(
                f"Added zero-offset End Site for leaf joint {joint.name!r}: "
                "the input BVH did not declare one."
            )
            joint = replace(joint, end_site_offset=(0.0, 0.0, 0.0))
        completed_joints.append(joint)
    return Hierarchy(tuple(completed_joints))


def _read_motion_values(lines: Sequence[str], total_channels: int) -> np.ndarray:
    """Parse numeric BVH motion rows."""
    values = []
    for line in lines:
        if not line.strip():
            continue
        tokens = tuple(line.split())
        if len(tokens) != total_channels:
            raise ValueError(
                f"Motion row has {len(tokens)} values, but hierarchy declares "
                f"{total_channels} channels."
            )
        numeric = np.asarray(tokens, dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(
                f"Motion row {len(values)} contains NaN or infinite values."
            )
        values.append(numeric)
    return (
        np.empty((0, total_channels), dtype=np.float64)
        if not values
        else np.vstack(values)
    )


def _read_motion_from_lines(
    lines: Sequence[str], *, root_name: str | None, motion_name: str
) -> Motion:
    """Read a Motion from already decoded BVH lines."""
    lines = list(lines)
    motion_index = _find_motion_index(lines, motion_name)
    hierarchy = _parse_hierarchy(lines[:motion_index])
    if root_name is not None and hierarchy.root_name != root_name:
        raise ValueError(
            f"Expected root joint '{root_name}', got '{hierarchy.root_name}'."
        )
    data_start_index = _find_motion_data_start(lines, motion_index)
    values = _read_motion_values(lines[data_start_index:], hierarchy.total_channels)
    if values.shape[0] == 0:
        raise ValueError("No motion frames found in BVH document.")
    header = lines[motion_index:data_start_index]
    declared_frames = _extract_frame_count(header)
    if declared_frames is not None and declared_frames != values.shape[0]:
        raise ValueError(
            f"BVH declares {declared_frames} frames but contains {values.shape[0]}."
        )
    return Motion(hierarchy, values, _extract_frame_time(header))


def load_bvh(
    path: str | Path,
    root_name: str | None = None,
    motion_name: str = "MOTION",
) -> Motion:
    """Read a BVH file into a Motion."""
    return _read_motion_from_lines(
        Path(path).read_text(encoding="utf-8-sig").splitlines(keepends=True),
        root_name=root_name,
        motion_name=motion_name,
    )


def load_bvh_from_text(
    text: str, root_name: str | None = None, motion_name: str = "MOTION"
) -> Motion:
    """Read BVH text into a Motion."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return _read_motion_from_lines(
        text.splitlines(keepends=True), root_name=root_name, motion_name=motion_name
    )


def load_bvh_from_bytes(
    data: bytes | bytearray | memoryview,
    root_name: str | None = None,
    motion_name: str = "MOTION",
    encoding: str = "utf-8-sig",
) -> Motion:
    """Read BVH bytes into a Motion."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes, bytearray, or memoryview.")
    try:
        text = bytes(data).decode(encoding)
    except UnicodeDecodeError as error:
        raise ValueError(f"BVH data is not valid {encoding} text.") from error
    return load_bvh_from_text(text, root_name=root_name, motion_name=motion_name)


def _serialize_joint(
    hierarchy: Hierarchy, index: int, indent: str, precision: int
) -> list[str]:
    """Serialize one joint and its descendants."""
    joint = hierarchy.joints[index]
    label = "ROOT" if joint.parent == -1 else "JOINT"
    lines = [f"{indent}{label} {joint.name}\n", f"{indent}{{\n"]
    lines.append(
        f"{indent}    OFFSET {' '.join(f'{value:.{precision}f}' for value in joint.offset)}\n"
    )
    if joint.channels:
        lines.append(
            f"{indent}    CHANNELS {len(joint.channels)} {' '.join(joint.channels)}\n"
        )
    children = hierarchy.children(index)
    for child_index in children:
        lines.extend(
            _serialize_joint(hierarchy, child_index, indent + "    ", precision)
        )
    if joint.end_site_offset is not None or not children:
        # TODO: Verify whether a zero-length End Site is the right fallback for every BVH consumer.
        if joint.end_site_offset is None:
            print(
                f"Added zero-offset End Site for leaf joint {joint.name!r}: "
                "the Hierarchy did not define one."
            )
        end_site_offset = joint.end_site_offset or (0.0, 0.0, 0.0)
        offset = " ".join(f"{value:.{precision}f}" for value in end_site_offset)
        lines.extend(
            (
                f"{indent}    End Site\n",
                f"{indent}    {{\n",
                f"{indent}        OFFSET {offset}\n",
                f"{indent}    }}\n",
            )
        )
    lines.append(f"{indent}}}\n")
    return lines


def write_bvh(motion: Motion, output_path: str | Path, precision: int = 8) -> None:
    """Write a Motion as a BVH file."""
    if not isinstance(motion, Motion):
        raise TypeError("motion must be a Motion.")
    if isinstance(precision, bool) or not isinstance(precision, int) or precision < 0:
        raise ValueError("precision must be a non-negative integer.")
    if motion.values.ndim != 2 or motion.values.shape != (
        motion.frame_count,
        motion.hierarchy.total_channels,
    ):
        raise ValueError("Motion values no longer match the Hierarchy layout.")
    if not np.isfinite(motion.values).all():
        raise ValueError("BVH output requires finite motion values.")
    if any(
        joint.local_orientation != (1.0, 0.0, 0.0, 0.0)
        for joint in motion.hierarchy.joints
    ):
        # TODO: Implement a verified local-orientation basis conversion for BVH output.
        raise NotImplementedError("BVH output does not support local_orientation yet.")
    lines = [
        "HIERARCHY\n",
        *_serialize_joint(motion.hierarchy, motion.hierarchy.root, "", precision),
    ]
    lines.extend(
        (
            "MOTION\n",
            f"Frames: {motion.frame_count}\n",
            f"Frame Time: {motion.frame_time:.{precision}f}\n",
        )
    )
    for frame in motion.values:
        lines.append(" ".join(f"{value:.{precision}f}" for value in frame) + "\n")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(lines), encoding="utf-8")
