"""Basic BVH motion editing operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional
import numpy as np

from broom.bvh.interpolation.utils import (
    root_position_indices,
    validate_compatible_documents,
)
from broom.bvh.io import validate_motion_values
from broom.bvh.schemas import BVHDocument


def fill_motion(
    document: BVHDocument,
    frame_count: int,
    pose: np.ndarray | Sequence[float] | None = None,
    frame_time: float | None = None,
) -> BVHDocument:
    """Create a document whose motion is filled with one repeated pose.

    If ``pose`` is omitted, the motion is filled with zeros.
    """

    frame_count = int(frame_count)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")

    if pose is None:
        motion_values = np.zeros(
            (frame_count, document.total_channels),
            dtype=np.float64,
        )
    else:
        pose_values = _single_pose_values(document=document, pose=pose)
        motion_values = np.repeat(pose_values[None, :], frame_count, axis=0)

    return with_motion_values(
        document=document,
        motion_values=motion_values,
        frame_time=frame_time,
    )


def trim_frames(
    document: BVHDocument,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> BVHDocument:
    """Return a copy of the document containing a frame slice.

    ``end_frame`` is exclusive, matching normal Python slicing.
    """

    start, end = _normalize_frame_slice(
        frame_count=document.frame_count,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    return with_motion_values(
        document=document,
        motion_values=document.motion_values[start:end],
    )


def slice_by_time(
    document: BVHDocument,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> BVHDocument:
    """Return a copy of the document sliced by seconds.

    end_seconds is exclusive. The start time is floored to the nearest
    source frame and the end time is ceiled so the requested time range is not
    accidentally shortened.
    """

    frame_time = _require_frame_time(document)
    start_frame = (
        None
        if start_seconds is None
        else int(np.floor(float(start_seconds) / frame_time))
    )
    end_frame = (
        None if end_seconds is None else int(np.ceil(float(end_seconds) / frame_time))
    )
    return trim_frames(
        document=document,
        start_frame=start_frame,
        end_frame=end_frame,
    )


def reverse(
    document: BVHDocument,
    keep_root_start: bool = False,
) -> BVHDocument:
    """Return a copy of the document with frames in reverse order."""

    motion_values = document.motion_values[::-1].copy()
    if keep_root_start and motion_values.shape[0] > 0:
        indices = root_position_indices(document)
        if indices:
            offset = document.motion_values[0, indices] - motion_values[0, indices]
            motion_values[:, indices] += offset
    return with_motion_values(document=document, motion_values=motion_values)


def zero_origin(
    document: BVHDocument,
    axes: Sequence[str] | None = None,
) -> BVHDocument:
    """Return a bvh animation where position channels start at zero.

    :param document: BVH document to modify.
    :param axes: Optional list of root position channel names to zero.
    """

    indices = _root_position_indices_for_axes(document=document, axes=axes)
    motion_values = document.motion_values.copy()
    if indices and motion_values.shape[0] > 0:
        motion_values[:, indices] -= motion_values[0, indices]
    return with_motion_values(document=document, motion_values=motion_values)


def _root_position_indices_for_axes(
    document: BVHDocument,
    axes: Sequence[str] | None,
) -> list[int]:
    
    """Return absolute root position channel indices for selected axes.
    
    :param document: BVH document to query for root channels.
    :param axes: Optional list of root position channel names to select.
    """

    root_joint = next(
        (joint for joint in document.joints if joint.parent == -1),
        None,
    )
    if root_joint is None:
        return []

    if axes is None:
        return [
            root_joint.channel_start + offset
            for offset, channel in enumerate(root_joint.channels)
            if channel.endswith("position")
        ]

    indices = []
    for axis in axes:
        if axis not in root_joint.channels:
            raise ValueError(
                f"Root channel '{axis}' was not found. "
                f"Available root channels: {', '.join(root_joint.channels)}"
            )
        indices.append(
            root_joint.channel_start + root_joint.channels.index(axis)
            )
    return indices


def _normalize_frame_slice(
    frame_count: int,
    start_frame: Optional[int] = 0,
    end_frame: Optional[int] = None,
) -> tuple[int, int]:
    """Normalize optional frame slice bounds and reject empty slices."""

    end = frame_count if end_frame is None else int(end_frame)

    start = min(max(start_frame, 0), frame_count)
    end = min(max(end_frame, start_frame), frame_count)
    if end < start:
        raise ValueError(f"Frame slice end must be >= start, got {start}..{end}.")
    if end == start:
        raise ValueError("Frame slice cannot be empty.")
    return start, end


def _require_frame_time(document: BVHDocument) -> float:
    """Return positive frame time or raise a clear error."""

    if document.frame_time is None or document.frame_time <= 0.0:
        raise ValueError("BVH document does not have a positive frame_time.")
    return float(document.frame_time)


def with_motion_values(
    document: BVHDocument,
    motion_values: np.ndarray,
    frame_time: float | None = None,
) -> BVHDocument:
    """Create a document copy with replaced motion values and metadata."""

    values = validate_motion_values(document=document, motion_values=motion_values)
    new_frame_time = document.frame_time if frame_time is None else float(frame_time)
    return BVHDocument(
        path=document.path,
        prefix_lines=_update_prefix_metadata(
            prefix_lines=document.prefix_lines,
            frame_count=values.shape[0],
            frame_time=new_frame_time,
        ),
        motion_rows=tuple(),
        motion_values=values,
        joints=document.joints,
        total_channels=document.total_channels,
        root_name=document.root_name,
        root_channels=document.root_channels,
        frame_time=new_frame_time,
        declared_frames=values.shape[0],
    )


def _single_pose_values(
    document: BVHDocument,
    pose: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Normalize one motion pose to a flat channel vector."""

    pose_values = np.asarray(pose, dtype=np.float64)
    if pose_values.ndim == 2:
        if pose_values.shape[0] != 1:
            raise ValueError(
                "pose must be a single frame with shape (channels,) or "
                "(1, channels)."
            )
        pose_values = pose_values[0]
    elif pose_values.ndim != 1:
        raise ValueError(
            "pose must be a single frame with shape (channels,) or "
            "(1, channels)."
        )

    if pose_values.shape[0] != document.total_channels:
        raise ValueError(
            "pose channel count must match document.total_channels, got "
            f"{pose_values.shape[0]} and {document.total_channels}."
        )

    return np.nan_to_num(pose_values, nan=0.0)


def _update_prefix_metadata(
    prefix_lines: Sequence[str],
    frame_count: int,
    frame_time: float | None,
) -> tuple[str, ...]:
    """Return prefix lines with updated frame count and frame time."""

    updated = []
    replaced_frames = False
    replaced_frame_time = False

    for line in prefix_lines:
        stripped = line.strip().lower()
        newline = line[len(line.rstrip("\r\n")) :]
        if stripped.startswith("frames:"):
            updated.append(f"Frames: {frame_count}{newline}")
            replaced_frames = True
            continue
        if frame_time is not None and stripped.startswith("frame time:"):
            updated.append(f"Frame Time: {frame_time:.8f}{newline}")
            replaced_frame_time = True
            continue
        updated.append(line)

    if not replaced_frames:
        raise ValueError("BVH prefix does not contain a Frames header.")
    if frame_time is not None and not replaced_frame_time:
        raise ValueError("BVH prefix does not contain a Frame Time header.")
    return tuple(updated)
