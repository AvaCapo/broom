"""Foot-floor analysis utilities for BVH joint trajectories.

Adapted from our internal main-project BVH branch foot-contact utilities.
Direct branch URL is not available in this workspace; this module is the
local adaptation for the `joints_world[T, J, 3]` representation.
"""

# TODO: Consolidate this module with broom.bvh.foot_lock before extending either
# contact algorithm. They currently overlap and must not diverge.

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from ._validation import validate_joints_world


JointRef = int | str


def contact_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Find continuous segments of True values in a boolean mask.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 1:
        raise ValueError("mask must be a 1D boolean array.")

    segments: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(mask_bool):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            segments.append((start, index - 1))
            start = None

    if start is not None:
        segments.append((start, len(mask_bool) - 1))

    return segments


def resolve_joint_groups(
    joint_groups: Mapping[str, Sequence[JointRef]],
    *,
    joint_names: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Convert joint-group definitions to integer indices.

    `joint_groups` may contain joint indices directly or joint names from BVH.
    When names are used, `joint_names` must be provided.
    """
    resolved: Dict[str, np.ndarray] = {}
    name_to_index = None if joint_names is None else {str(name): i for i, name in enumerate(joint_names)}

    for group_name, refs in joint_groups.items():
        if len(refs) == 0:
            raise ValueError(f"joint_groups[{group_name!r}] must contain at least one joint.")

        indices = []
        for ref in refs:
            if isinstance(ref, (int, np.integer)):
                indices.append(int(ref))
                continue

            if name_to_index is None:
                raise ValueError(
                    "joint_names must be provided when joint_groups contain joint names."
                )
            ref_name = str(ref)
            if ref_name not in name_to_index:
                raise KeyError(f"Unknown joint name {ref_name!r} in group {group_name!r}.")
            indices.append(name_to_index[ref_name])

        resolved[group_name] = np.asarray(indices, dtype=np.int64)

    return resolved


def estimate_skeleton_height(
    joints: Optional[Sequence[object]] = None,
    *,
    parents: Optional[Sequence[int]] = None,
    offsets: Optional[np.ndarray] = None,
) -> float:
    """
    Estimate skeleton height from the skeleton hierarchy in rest pose.

    Supports either:
    - `joints`: objects with `.parent` and `.offset`
    - `parents` and `offsets`: current BVH-style arrays from this project
    """
    if joints is not None:
        rest_positions = np.zeros((len(joints), 3), dtype=np.float64)
        for index, joint in enumerate(joints):
            parent = int(joint.parent)
            offset = np.asarray(joint.offset, dtype=np.float64)
            if parent == -1:
                rest_positions[index] = offset
            else:
                rest_positions[index] = rest_positions[parent] + offset
        return max(float(rest_positions[:, 1].max() - rest_positions[:, 1].min()), 1.0e-6)

    if parents is None or offsets is None:
        raise ValueError("Provide either `joints` or both `parents` and `offsets`.")

    parents_arr = np.asarray(parents, dtype=np.int64)
    offsets_arr = np.asarray(offsets, dtype=np.float64)
    if parents_arr.ndim != 1:
        raise ValueError("parents must be a 1D array.")
    if offsets_arr.ndim != 2 or offsets_arr.shape != (parents_arr.shape[0], 3):
        raise ValueError("offsets must have shape [J, 3] matching parents.")

    rest_positions = np.zeros_like(offsets_arr, dtype=np.float64)
    for index, parent in enumerate(parents_arr):
        if int(parent) < 0 or int(parent) == index:
            rest_positions[index] = offsets_arr[index]
        else:
            rest_positions[index] = rest_positions[int(parent)] + offsets_arr[index]

    return max(float(rest_positions[:, 1].max() - rest_positions[:, 1].min()), 1.0e-6)


def remove_short_contacts(mask: np.ndarray, min_frames: int = 2) -> np.ndarray:
    """
    Remove contact segments shorter than `min_frames`.
    """
    cleaned = np.asarray(mask, dtype=bool).copy()
    if cleaned.ndim != 1:
        raise ValueError("mask must be a 1D boolean array.")
    if min_frames <= 1 or cleaned.size == 0:
        return cleaned

    for start, end in contact_segments(cleaned):
        if end - start + 1 < min_frames:
            cleaned[start : end + 1] = False

    return cleaned


def estimate_floor_height(
    joints_world: np.ndarray,
    foot_indices: Sequence[int],
) -> float:
    """
    Estimate floor height from lower foot-joint heights.

    The current project uses `joints_world[T, J, 3]`, where axis 1 is height.
    """
    validate_joints_world(joints_world)
    foot_idx = np.asarray(foot_indices, dtype=np.int64)
    if foot_idx.ndim != 1 or foot_idx.size == 0:
        raise ValueError("foot_indices must be a non-empty 1D sequence.")

    min_heights = joints_world[:, foot_idx, 1].min(axis=1).astype(np.float64)
    sorted_heights = np.sort(min_heights)
    low = int(0.05 * len(sorted_heights))
    high = max(low + 1, int(0.25 * len(sorted_heights)))
    return float(sorted_heights[low:high].mean())


def detect_foot_contacts(
    joints_world: np.ndarray,
    foot_groups: Mapping[str, Sequence[JointRef]],
    *,
    skeleton_height: Optional[float] = None,
    joints: Optional[Sequence[object]] = None,
    parents: Optional[Sequence[int]] = None,
    offsets: Optional[np.ndarray] = None,
    joint_names: Optional[Sequence[str]] = None,
    height_threshold: Optional[float] = None,
    velocity_threshold: Optional[float] = None,
    min_contact_frames: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Detect foot-floor contacts using foot height and planar velocity.

    Parameters
    ----------
    joints_world:
        Global joint positions with shape `[T, J, 3]`.
    foot_groups:
        Mapping like `{"left": [...], "right": [...]}`. Group members may be
        joint indices or joint names from `joint_names`.
    skeleton_height:
        Optional skeleton height. If omitted, it is estimated from the BVH
        hierarchy using `joints` or `parents`+`offsets`.
    joints:
        Optional sequence of joints with `.parent` and `.offset`, matching the
        API from the other project.
    parents, offsets:
        Current project-friendly rest hierarchy representation. Use these when
        `joints` objects are not available.
    joint_names:
        Names list from `load_bvh_as_world_joints(...)`. Required only when
        `foot_groups` contains names instead of indices.
    height_threshold:
        Maximum distance from the estimated floor to count as contact.
    velocity_threshold:
        Maximum planar speed in the XZ plane to count as contact.
    min_contact_frames:
        Removes short isolated detections after thresholding.
    """
    validate_joints_world(joints_world)
    resolved_groups = resolve_joint_groups(foot_groups, joint_names=joint_names)
    all_indices = np.concatenate(list(resolved_groups.values()))

    floor_height = estimate_floor_height(joints_world, all_indices)
    clip_height = (
        float(skeleton_height)
        if skeleton_height is not None
        else estimate_skeleton_height(joints=joints, parents=parents, offsets=offsets)
    )

    height_limit = (
        float(height_threshold)
        if height_threshold is not None
        else max(0.035 * clip_height, 0.02)
    )
    velocity_limit = (
        float(velocity_threshold)
        if velocity_threshold is not None
        else max(0.025 * clip_height, 0.015)
    )

    masks: Dict[str, np.ndarray] = {}
    for group_name, group_indices in resolved_groups.items():
        foot_xz = joints_world[:, group_indices][:, :, [0, 2]].mean(axis=1)
        velocity = np.linalg.norm(
            np.diff(foot_xz, axis=0, prepend=foot_xz[:1]),
            axis=1,
        )
        if velocity.size > 1:
            velocity[0] = velocity[1]

        foot_height = joints_world[:, group_indices, 1].min(axis=1)
        mask = (foot_height <= floor_height + height_limit) & (velocity <= velocity_limit)
        masks[group_name] = remove_short_contacts(mask, min_frames=min_contact_frames)

    return masks
