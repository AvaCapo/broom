"""Skeleton hierarchy helpers derived from BVH rest-pose offsets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from broom.bvh.io import validate_motion_values
from broom.bvh.schemas import BVHDocument, BVHJoint

DEFAULT_HIPS_NAMES = (
    "hips",
    "pelvis",
    "mixamorig:hips",
    "root",
)


def skeleton_tree(
    skeleton: BVHDocument | Sequence[BVHJoint],
    *,
    show_channels: bool = False,
    show_offsets: bool = False,
    max_depth: int | None = None,
) -> str:
    """Return the skeleton hierarchy formatted as an ASCII tree."""

    joints = (
        skeleton.joints
        if isinstance(skeleton, BVHDocument)
        else tuple(skeleton)
    )
    if not joints:
        return ""

    children: list[list[int]] = [[] for _ in joints]
    roots: list[int] = []
    for index, joint in enumerate(joints):
        if joint.parent == -1:
            roots.append(index)
        else:
            children[joint.parent].append(index)

    lines: list[str] = []
    stack: list[tuple[int, str, bool, int]] = [
        (root_index, "", root_offset == len(roots) - 1, 0)
        for root_offset, root_index in reversed(list(enumerate(roots)))
    ]
    while stack:
        joint_index, prefix, is_last, depth = stack.pop()
        branch = "`-- " if is_last else "|-- "
        joint = joints[joint_index]
        line = prefix + branch + joint.name

        details: list[str] = []
        if show_channels and joint.channels:
            details.append(f"channels={list(joint.channels)}")
        if show_offsets:
            offset = np.asarray(joint.offset, dtype=np.float64)
            details.append(
                "offset=["
                f"{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}"
                "]"
            )
        if details:
            line += "  (" + ", ".join(details) + ")"
        lines.append(line)

        if max_depth is not None and depth >= max_depth:
            continue

        child_prefix = prefix + ("    " if is_last else "|   ")
        joint_children = children[joint_index]
        for child_offset, child_index in reversed(list(enumerate(joint_children))):
            stack.append(
                (
                    child_index,
                    child_prefix,
                    child_offset == len(joint_children) - 1,
                    depth + 1,
                )
            )
    return "\n".join(lines)


def compute_rest_joint_positions(joints: Sequence[BVHJoint]) -> np.ndarray:
    """Return global rest-pose joint positions reconstructed from offsets."""

    positions = np.zeros((len(joints), 3), dtype=np.float64)
    for index, joint in enumerate(joints):
        if joint.parent == -1:
            positions[index] = joint.offset
        else:
            positions[index] = positions[joint.parent] + joint.offset
    return positions


def scale_skeleton(
    document: BVHDocument,
    scale: float,
    *,
    scale_position_channels: bool = True,
) -> BVHDocument:
    """Return a document with skeleton offsets scaled by ``scale``.

    By default all position channels are scaled too, which keeps the motion
    consistent with the resized skeleton.
    """

    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    if np.isclose(scale, 1.0):
        return document

    joints = tuple(
        replace(
            joint,
            offset=np.asarray(joint.offset, dtype=np.float64) * scale,
        )
        for joint in document.joints
    )
    motion_values = document.motion_values.copy()
    if scale_position_channels and motion_values.size > 0:
        indices = _position_channel_indices(document)
        if indices:
            motion_values[:, indices] *= scale

    values = validate_motion_values(document=document, motion_values=motion_values)
    return BVHDocument(
        path=document.path,
        prefix_lines=document.prefix_lines,
        motion_rows=tuple(),
        motion_values=values,
        joints=joints,
        total_channels=document.total_channels,
        root_name=document.root_name,
        root_channels=document.root_channels,
        frame_time=document.frame_time,
        declared_frames=values.shape[0],
    )


def estimate_skeleton_height(joints: Sequence[BVHJoint]) -> float:
    """Estimate rest-pose skeleton height as Y-axis span."""

    if not joints:
        return 0.0

    positions = compute_rest_joint_positions(joints)
    return max(float(np.ptp(positions[:, 1])), 1.0e-6)


def estimate_hips_height(
    joints: Sequence[BVHJoint],
    hips_name: str | None = None,
) -> float:
    """Estimate hips height above the lowest rest-pose point along Y."""

    if not joints:
        return 0.0

    positions = compute_rest_joint_positions(joints)
    hips_index = find_hips_joint_index(joints, hips_name=hips_name)
    floor_y = float(positions[:, 1].min())
    hips_y = float(positions[hips_index, 1])
    return max(hips_y - floor_y, 1.0e-6)


def find_hips_joint_index(
    joints: Sequence[BVHJoint],
    hips_name: str | None = None,
) -> int:
    """Return the hierarchy index of the hips/pelvis joint."""

    if not joints:
        raise ValueError("Cannot find hips joint in an empty skeleton.")
    
    # search by provided hips joint name
    if hips_name is not None:
        hips_name_normalized = hips_name.casefold()
        for index, joint in enumerate(joints):
            if joint.name.casefold() == hips_name_normalized:
                return index
        raise ValueError(f"Hips joint '{hips_name}' was not found.")
    
    # exact search default known hip joint names
    joint_names = [joint.name.casefold() for joint in joints]
    for candidate in DEFAULT_HIPS_NAMES:
        candidate_normalized = candidate.casefold()
        for index, joint_name in enumerate(joint_names):
            if joint_name == candidate_normalized:
                return index
    
    # fallback to non-exact search
    for candidate in ("hip", "pelvis"):
        candidate_normalized = candidate.casefold()
        for index, joint_name in enumerate(joint_names):
            if candidate_normalized in joint_name:
                return index

    return 0


def _position_channel_indices(document: BVHDocument) -> list[int]:
    indices: list[int] = []
    for joint in document.joints:
        for offset, channel in enumerate(joint.channels):
            if channel.endswith("position"):
                indices.append(joint.channel_start + offset)
    return indices
