"""Load and write BVH motions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from broom import Hierarchy, Joint, Motion


def _parse_finite_float(token: str, *, context: str) -> float:
    """Parse one finite BVH number, replacing NaN with zero."""
    try:
        value = float(token)
    except ValueError as error:
        raise ValueError(f"{context} must be a finite number, got {token!r}.") from error
    if np.isnan(value):
        return 0.0
    if not np.isfinite(value):
        raise ValueError(f"{context} must be finite, got {token!r}.")
    return value


def _parse_offset(line: str, *, context: str) -> tuple[float, float, float]:
    """Parse an OFFSET declaration with exactly three values."""
    tokens = line.split()
    if len(tokens) != 4 or tokens[0] != "OFFSET":
        raise ValueError("BVH OFFSET must contain exactly three numeric components.")
    return tuple(
        _parse_finite_float(token, context=f"{context} OFFSET") for token in tokens[1:]
    )  # type: ignore[return-value]


def _normalise_joint_name(name: str) -> str:
    """Normalise whitespace in a BVH joint name with an explicit notice."""
    normalised = re.sub(r"\s+", "_", name)
    if normalised != name:
        print(
            f"Renamed BVH joint {name!r} to {normalised!r} to improve "
            "compatibility with BVH readers."
        )
    return normalised


def _parse_hierarchy(lines: Sequence[str]) -> Hierarchy:
    """Parse a strict BVH hierarchy into a Hierarchy."""
    joints: list[Joint] = []
    stack: list[int | None] = []
    end_site_parents: list[int] = []
    pending_joint: int | None = None
    pending_end_site_parent: int | None = None
    offset_joints: set[int] = set()
    channel_joints: set[int] = set()
    end_site_offsets: set[int] = set()
    hierarchy_seen = False

    def active_joint() -> int:
        if not stack or stack[-1] is None:
            return -1
        return stack[-1]

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if not hierarchy_seen:
            if stripped != "HIERARCHY":
                raise ValueError("BVH hierarchy must begin with HIERARCHY.")
            hierarchy_seen = True
            continue
        if stripped == "HIERARCHY":
            raise ValueError("BVH hierarchy may declare HIERARCHY only once.")
        if pending_joint is not None or pending_end_site_parent is not None:
            if stripped != "{":
                raise ValueError("BVH joint and End Site declarations must be followed by {.")
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
            else:
                stack.append(None)
                end_site_parents.append(pending_end_site_parent)  # type: ignore[arg-type]
                pending_end_site_parent = None
            continue
        if stripped == "{":
            raise ValueError("Unexpected opening brace in BVH hierarchy.")
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

        declaration = stripped.split(maxsplit=1)
        keyword = declaration[0]
        active = active_joint()
        if keyword in ("ROOT", "JOINT"):
            if len(declaration) != 2 or not declaration[1].strip():
                raise ValueError(f"BVH {keyword} declaration requires a joint name.")
            if keyword == "ROOT":
                if joints or stack:
                    raise ValueError("BVH hierarchy must declare exactly one root first.")
                parent = -1
            else:
                if not joints or active == -1:
                    raise ValueError("BVH JOINT must be declared inside a joint block.")
                parent = active
            name = _normalise_joint_name(declaration[1])
            if any(joint.name == name for joint in joints):
                raise ValueError(
                    f"BVH joint name {name!r} is not unique after normalization."
                )
            pending_joint = len(joints)
            joints.append(Joint(name, parent, (0.0, 0.0, 0.0)))
            continue
        if stripped == "End Site":
            if active == -1:
                raise ValueError("BVH End Site must belong to a joint.")
            pending_end_site_parent = active
            continue
        if keyword == "OFFSET":
            if not stack:
                raise ValueError("BVH OFFSET must belong to a joint or End Site.")
            offset = _parse_offset(stripped, context="BVH")
            if stack[-1] is None:
                parent = end_site_parents[-1]
                if parent in end_site_offsets:
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
        if keyword == "CHANNELS":
            if active == -1 or stack[-1] is None:
                raise ValueError("BVH CHANNELS must belong to a joint.")
            if active not in offset_joints:
                raise ValueError("BVH CHANNELS must follow its joint OFFSET.")
            if active in channel_joints:
                raise ValueError(
                    f"Joint {joints[active].name!r} declares CHANNELS more than once."
                )
            tokens = stripped.split()
            if len(tokens) < 2 or not tokens[1].isdigit():
                raise ValueError("BVH CHANNELS requires a non-negative integer count.")
            count = int(tokens[1])
            channels = tuple(tokens[2:])
            if len(channels) != count:
                raise ValueError(
                    f"Joint {joints[active].name!r} declares {count} channels, "
                    f"but {len(channels)} names were found."
                )
            joints[active] = replace(joints[active], channels=channels)
            channel_joints.add(active)
            continue
        raise ValueError(f"Unknown or misplaced BVH hierarchy declaration: {stripped!r}.")

    if not hierarchy_seen:
        raise ValueError("BVH hierarchy must begin with HIERARCHY.")
    if stack or pending_joint is not None or pending_end_site_parent is not None:
        raise ValueError("BVH hierarchy has unclosed or incomplete blocks.")
    if not joints:
        raise ValueError("BVH hierarchy does not contain joints.")
    missing_offsets = [
        joint.name for index, joint in enumerate(joints) if index not in offset_joints
    ]
    if missing_offsets:
        raise ValueError(f"BVH joints without OFFSET: {', '.join(missing_offsets)}.")
    missing_channels = [
        joint.name for index, joint in enumerate(joints) if index not in channel_joints
    ]
    if missing_channels:
        raise ValueError(
            f"BVH joints without CHANNELS: {', '.join(missing_channels)}."
        )
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


def _find_motion_index(lines: Sequence[str], motion_name: str) -> int:
    """Return the unique MOTION section line index."""
    matches = [
        index
        for index, line in enumerate(lines)
        if line.strip().upper() == motion_name.upper()
    ]
    if len(matches) != 1:
        raise ValueError(f"BVH file must contain exactly one {motion_name} section.")
    return matches[0]


def _parse_motion_header(lines: Sequence[str], motion_index: int) -> tuple[int, float, int]:
    """Parse required Frames and Frame Time declarations."""
    nonempty = [
        (index, line.strip())
        for index, line in enumerate(lines[motion_index + 1 :], start=motion_index + 1)
        if line.strip()
    ]
    if len(nonempty) < 2:
        raise ValueError("BVH MOTION section requires Frames and Frame Time headers.")
    frame_index, frames_line = nonempty[0]
    time_index, time_line = nonempty[1]
    frame_match = re.fullmatch(r"Frames\s*:\s*(\d+)", frames_line)
    if frame_match is None:
        raise ValueError("BVH MOTION section must begin with a Frames header.")
    frame_count = int(frame_match.group(1))
    time_match = re.fullmatch(r"Frame\s+Time\s*:\s*(\S+)", time_line, flags=re.IGNORECASE)
    if time_match is None:
        raise ValueError("BVH MOTION section must contain one Frame Time header.")
    frame_time = _parse_finite_float(time_match.group(1), context="BVH Frame Time")
    if frame_time <= 0:
        raise ValueError("BVH Frame Time must be positive.")
    return frame_count, frame_time, time_index + 1


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
        numeric = np.asarray(
            [_parse_finite_float(token, context="BVH motion value") for token in tokens],
            dtype=np.float64,
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
            f"Expected root joint {root_name!r}, got {hierarchy.root_name!r}."
        )
    declared_frames, frame_time, data_start_index = _parse_motion_header(
        lines, motion_index
    )
    values = _read_motion_values(lines[data_start_index:], hierarchy.total_channels)
    if values.shape[0] == 0:
        raise ValueError("No motion frames found in BVH document.")
    if declared_frames != values.shape[0]:
        raise ValueError(
            f"BVH declares {declared_frames} frames but contains {values.shape[0]}."
        )
    return Motion(hierarchy, values, frame_time)


def load_bvh(
    path: str | Path,
    root_name: str | None = None,
    motion_name: str = "MOTION",
) -> Motion:
    """Read a BVH file into a Motion, replacing NaN numeric values with zero."""
    return _read_motion_from_lines(
        Path(path).read_text(encoding="utf-8-sig").splitlines(keepends=True),
        root_name=root_name,
        motion_name=motion_name,
    )


def load_bvh_from_text(
    text: str, root_name: str | None = None, motion_name: str = "MOTION"
) -> Motion:
    """Read BVH text into a Motion, replacing NaN numeric values with zero."""
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
    """Read BVH bytes into a Motion, replacing NaN numeric values with zero."""
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
    """Serialize one joint and its descendants in DFS order."""
    joint = hierarchy.joints[index]
    label = "ROOT" if joint.parent == -1 else "JOINT"
    lines = [f"{indent}{label} {joint.name}\n", f"{indent}{{\n"]
    offset = " ".join(f"{value:.{precision}f}" for value in joint.offset)
    lines.append(f"{indent}    OFFSET {offset}\n")
    channels = " ".join(joint.channels)
    lines.append(
        f"{indent}    CHANNELS {len(joint.channels)}"
        f"{f' {channels}' if channels else ''}\n"
    )
    for child_index in hierarchy.children(index):
        lines.extend(_serialize_joint(hierarchy, child_index, indent + "    ", precision))
    if joint.end_site_offset is not None or not hierarchy.children(index):
        # TODO: Verify whether a zero-length End Site is the right fallback for every BVH consumer.
        if joint.end_site_offset is None:
            print(
                f"Added zero-offset End Site for leaf joint {joint.name!r}: "
                "the Hierarchy did not define one."
            )
        end_site_offset = joint.end_site_offset or (0.0, 0.0, 0.0)
        end_offset = " ".join(
            f"{value:.{precision}f}" for value in end_site_offset
        )
        lines.extend(
            (
                f"{indent}    End Site\n",
                f"{indent}    {{\n",
                f"{indent}        OFFSET {end_offset}\n",
                f"{indent}    }}\n",
            )
        )
    lines.append(f"{indent}}}\n")
    return lines


def _dfs_joint_indices(hierarchy: Hierarchy, index: int) -> tuple[int, ...]:
    """Return hierarchy joint indices in BVH serialization order."""
    indices = [index]
    for child_index in hierarchy.children(index):
        indices.extend(_dfs_joint_indices(hierarchy, child_index))
    return tuple(indices)


def _export_values_in_dfs_order(motion: Motion) -> np.ndarray:
    """Return temporary motion rows aligned with the serialized hierarchy."""
    column_indices = []
    for index in _dfs_joint_indices(motion.hierarchy, motion.hierarchy.root):
        start = motion.hierarchy.channel_start(index)
        column_indices.extend(range(start, start + motion.hierarchy.channel_count(index)))
    return motion.values[:, column_indices]


def _format_frame_time(value: float, precision: int) -> str:
    """Format a positive frame duration without scientific notation."""
    formatted = f"{value:.{precision}f}"
    if float(formatted) > 0:
        return formatted
    return np.format_float_positional(value, unique=True, trim="-")


def write_bvh(
    motion: Motion,
    output_path: str | Path,
    precision: int = 8,
    frame_time_precision: int = 8,
) -> None:
    """Write a Motion as canonical DFS-ordered BVH.

    BVH hierarchy records are emitted in depth-first order. If the in-memory
    Hierarchy uses another valid parent-before-child order, this function
    permutes complete joint channel blocks only in temporary export rows. The
    source Motion and its values are not modified.
    """
    if not isinstance(motion, Motion):
        raise TypeError("motion must be a Motion.")
    for name, value in (
        ("precision", precision),
        ("frame_time_precision", frame_time_precision),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
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
    export_values = _export_values_in_dfs_order(motion)
    lines = [
        "HIERARCHY\n",
        *_serialize_joint(motion.hierarchy, motion.hierarchy.root, "", precision),
        "MOTION\n",
        f"Frames: {motion.frame_count}\n",
        f"Frame Time: {_format_frame_time(motion.frame_time, frame_time_precision)}\n",
    ]
    for frame in export_values:
        lines.append(" ".join(f"{value:.{precision}f}" for value in frame) + "\n")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(lines), encoding="utf-8")
