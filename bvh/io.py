"""I/O helpers for loading and writing BVH documents."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from bvh.channels import root_channel_indices
from bvh.joint_limits import DEFAULT_JOINT_LIMITS_PATH
from bvh.parsing import (
    extract_frame_count,
    extract_frame_time,
    extract_root_channels,
    find_motion_data_start,
    find_motion_index,
    parse_hierarchy,
    read_motion_rows,
)
from bvh.schemas import BVHDocument


def load_bvh_document(
    path: str | Path, root_name: str | None = None,
    motion_name: str = "MOTION",
    joint_limits_path: str | Path | None = DEFAULT_JOINT_LIMITS_PATH,
) -> BVHDocument:
    """Load a BVH document from the specified file path."""

    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    motion_index = find_motion_index(lines, motion_name=motion_name)
    hierarchy_lines = lines[:motion_index]
    motion_header_lines = lines[motion_index:]
    joints, total_channels = parse_hierarchy(
        hierarchy_lines,
        joint_limits_path=joint_limits_path,
    )
    if not joints:
        raise ValueError("BVH hierarchy does not contain joints.")

    actual_root_name, root_channels = extract_root_channels(
        hierarchy_lines,
        expected_root_name=root_name,
    )
    data_start_index = find_motion_data_start(lines, motion_index)
    motion_rows, motion_values = read_motion_rows(
        lines[data_start_index:],
        total_channels=total_channels,
    )
    if motion_values.shape[0] == 0:
        raise ValueError("No motion frames found in BVH file.")

    return BVHDocument(
        path=source,
        prefix_lines=tuple(lines[:data_start_index]),
        motion_rows=tuple(motion_rows),
        motion_values=motion_values,
        joints=tuple(joints),
        total_channels=total_channels,
        root_name=actual_root_name,
        root_channels=tuple(root_channels),
        frame_time=extract_frame_time(motion_header_lines),
        declared_frames=extract_frame_count(motion_header_lines),
    )


def write_bvh_with_channel_values(
    document: BVHDocument,
    output_path: str | Path,
    channel_indices: Sequence[int],
    values: np.ndarray,
    precision: int,
) -> None:
    """Write a BVH copy with selected motion channels replaced."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape != (document.frame_count, len(channel_indices)):
        raise ValueError(
            "Replacement values must have shape "
            f"({document.frame_count}, {len(channel_indices)}), "
            f"got {values.shape}."
        )

    motion_values = document.motion_values.copy()
    motion_values[:, list(channel_indices)] = values
    write_bvh_with_motion_values(
        document=document,
        output_path=output_path,
        motion_values=motion_values,
        precision=precision,
    )


def write_bvh_with_motion_values(
    document: BVHDocument,
    output_path: str | Path,
    motion_values: np.ndarray,
    precision: int,
) -> None:
    """Write a BVH document with complete replacement motion values."""

    values = validate_motion_values(
        document=document,
        motion_values=motion_values,
    )
    destination = Path(output_path)
    output_lines = list(
        update_prefix_frame_count(
            prefix_lines=document.prefix_lines,
            frame_count=values.shape[0],
        )
    )

    for frame in values:
        output_lines.append(
            " ".join(f"{value:.{precision}f}" for value in frame) + "\n"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(output_lines), encoding="utf-8")


def write_bvh_with_root_channels(
    document: BVHDocument,
    output_path: str | Path,
    root_channel_values: np.ndarray,
    axes: Sequence[str],
    precision: int,
) -> None:
    """Write a BVH copy with selected root channels replaced."""

    channel_indices = root_channel_indices(document.root_channels, axes)
    write_bvh_with_channel_values(
        document=document,
        output_path=output_path,
        channel_indices=channel_indices,
        values=root_channel_values,
        precision=precision,
    )


def validate_motion_values(
    document: BVHDocument,
    motion_values: np.ndarray,
) -> np.ndarray:
    """Validate replacement motion values and return a float64 copy."""

    values = np.asarray(motion_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Motion values must be a 2D array.")
    if values.shape[1] != document.total_channels:
        raise ValueError(
            "Motion channel count does not match the BVH hierarchy: "
            f"expected {document.total_channels}, got {values.shape[1]}."
        )
    if np.isinf(values).any():
        raise ValueError("Motion values cannot contain infinite values.")
    return values.copy()


def update_prefix_frame_count(
    prefix_lines: Sequence[str],
    frame_count: int,
) -> tuple[str, ...]:
    """Return BVH prefix lines with an updated Frames header."""

    updated = []
    replaced = False
    for line in prefix_lines:
        if line.strip().lower().startswith("frames:"):
            newline = line[len(line.rstrip("\r\n")):]
            updated.append(f"Frames: {frame_count}{newline}")
            replaced = True
            continue
        updated.append(line)

    if not replaced:
        raise ValueError("BVH prefix does not contain a Frames header.")
    return tuple(updated)
