"""Foot contact and foot-lock helpers for BVH motion processing."""

#TODO: convert to generic functions; keep some default settings as a custom case workflow
# and maybe move this file to the ops module?

from __future__ import annotations

from typing import Sequence

import numpy as np

from broom import Hierarchy, Joint
from broom.channels import joint_name_matches
from broom.math_helpers import smoothstep
from broom.ops.skeleton_geometry import estimate_height

DEFAULT_FOOT_JOINTS = {
    "left": ("LeftFoot", "LeftToe"),
    "right": ("RightFoot", "RightToe"),
}


def apply_root_foot_lock(
    root_points: np.ndarray,
    original_root_points: np.ndarray,
    global_positions: np.ndarray,
    hierarchy: Hierarchy,
    blend_frames: int,
    height_threshold: float | None,
    velocity_threshold: float | None,
    *,
    up_axis: str = "Y",
    floor_axes: Sequence[int] = (0, 2),
    include_end_sites: bool = False,
) -> np.ndarray:
    """Apply foot locking to the root points based on detected foot contacts.

    :param root_points: An array of shape (N, 2) representing the current
        root X/Z points to be corrected.
    :param original_root_points: An array of shape (N, 2) representing the
        original root X/Z points before correction.
    :param global_positions: An array of shape (N, J, 3) representing the
        global positions of all joints.
    :param hierarchy: Skeleton hierarchy matching ``global_positions``.
    :param blend_frames: The number of frames to blend the foot lock correction
        in and out. Even values are rounded up to the next odd value.
    :param height_threshold: height threshold for foot contact detection.
        If None, a default threshold based on the skeleton height will be used.
    :param velocity_threshold: velocity threshold for foot contact detection.
        If None, a default threshold based on the skeleton height will be used.
    :param up_axis: Coordinate axis used to measure height above the floor.
    :param floor_axes: Coordinate dimensions spanning the floor plane.
    """
    foot_groups = find_foot_groups(hierarchy.joints)
    if not foot_groups:
        return root_points.copy()

    skeleton_height = estimate_height(
        hierarchy,
        include_end_sites=include_end_sites,
        up_axis=up_axis,
    )
    contact_masks = detect_foot_contacts(
        global_positions=global_positions,
        foot_groups=foot_groups,
        skeleton_height=skeleton_height,
        height_threshold=height_threshold,
        velocity_threshold=velocity_threshold,
        velocity_dimensions=floor_axes,
        height_dimension="XYZ".index(up_axis.upper()),
    )
    if not any(mask.any() for mask in contact_masks.values()):
        return root_points.copy()

    base_delta = root_points - original_root_points
    frame_count = root_points.shape[0]
    correction_sum = np.zeros((frame_count, 2), dtype=np.float64)
    correction_weight = np.zeros(frame_count, dtype=np.float64)

    for group_name, group_indices in foot_groups.items():
        mask = contact_masks[group_name]
        foot_points = global_positions[:, group_indices, :][
            :, :, floor_axes
        ].mean(axis=1)
        foot_points = foot_points + base_delta

        for start, end in contact_segments(mask):
            segment = foot_points[start : end + 1]
            anchor = segment.mean(axis=0)
            corrections = anchor[None, :] - segment
            correction_sum[start : end + 1] += corrections
            correction_weight[start : end + 1] += 1.0

            first = corrections[0]
            last = corrections[-1]
            for distance in range(1, max(0, blend_frames) + 1):
                weight = smoothstep(
                    (blend_frames - distance + 1) / (blend_frames + 1)
                )
                left = start - distance
                right = end + distance
                if left >= 0:
                    correction_sum[left] += first * weight
                    correction_weight[left] += weight
                if right < frame_count:
                    correction_sum[right] += last * weight
                    correction_weight[right] += weight

    corrected = root_points.copy()
    active = correction_weight > 0.0
    corrected[active] += (
        correction_sum[active] / correction_weight[active, None]
    )
    return corrected

def find_foot_groups(joints: Sequence[Joint]) -> dict[str, np.ndarray]:
    """Identify foot joint groups based on the joint names.

    :param joints: The joints to search through.
    """
    groups = {}
    for group_name, target_names in DEFAULT_FOOT_JOINTS.items():
        indices = [
            index
            for index, joint in enumerate(joints)
            if any(
                joint_name_matches(joint.name, target)
                for target in target_names
            )
        ]
        if indices:
            groups[group_name] = np.asarray(indices, dtype=np.int64)
    return groups


def detect_foot_contacts(
    global_positions: np.ndarray,
    foot_groups: dict[str, np.ndarray],
    skeleton_height: float,
    height_threshold: float | None,
    velocity_threshold: float | None,
    velocity_dimensions: Sequence[int],
    height_dimension: int,
) -> dict[str, np.ndarray]:
    """Detect foot contacts based on foot heights and planar velocity.

    :param global_positions: An array of shape (N, J, 3) representing the
        global positions of all joints.
    :param foot_groups: A dictionary mapping foot group names to joint indices.
    :param skeleton_height: The estimated height of the skeleton for thresholds.
    :param height_threshold: height threshold for foot contact detection.
        If None, a default threshold based on the skeleton height will be used.
    :param velocity_threshold: velocity threshold for foot contact detection.
        If None, a default threshold based on the skeleton height will be used.
    :param velocity_dimensions: Coordinate dimensions used for foot velocity,
        e.g. ``(0, 2)`` for X/Z.
    :param height_dimension: Coordinate dimension used for foot height.
    """
    velocity_dimensions = tuple(velocity_dimensions)
    all_indices = np.concatenate(list(foot_groups.values()))
    floor_height = estimate_floor_height(
        global_positions=global_positions,
        foot_indices=all_indices,
        height_dimension=height_dimension,
    )
    height_limit = (
        float(height_threshold)
        if height_threshold is not None
        else max(0.035 * skeleton_height, 0.02) #TODO: qualify magic numbers
    )
    velocity_limit = (
        float(velocity_threshold)
        if velocity_threshold is not None
        else max(0.025 * skeleton_height, 0.015) #TODO: qualify magic numbers
    )

    masks = {}
    for group_name, group_indices in foot_groups.items():
        foot_points = global_positions[:, group_indices, :][
            :, :, velocity_dimensions
        ].mean(axis=1)
        velocity = np.linalg.norm(
            np.diff(foot_points, axis=0, prepend=foot_points[:1]), axis=1
        )
        if len(velocity) > 1:
            velocity[0] = velocity[1]
        foot_height = global_positions[:, group_indices, height_dimension].min(
            axis=1
        )
        mask = (foot_height <= floor_height + height_limit) & (
            velocity <= velocity_limit
        )
        masks[group_name] = remove_short_contacts(mask, min_frames=2)
    return masks


def estimate_floor_height(
    global_positions: np.ndarray,
    foot_indices: np.ndarray,
    height_dimension: int,
) -> float:
    """Estimate floor height from the lower quantile of foot joint heights.

    :param global_positions: The global joint positions with shape
        (frames, joints, 3).
    :param foot_indices: The indices of the foot joints to consider.
    :param height_dimension: Coordinate dimension used for foot height.
    """
    min_heights = global_positions[:, foot_indices, height_dimension].min(
        axis=1
    )
    low = int(0.05 * len(min_heights)) #TODO: qualify magic numbers
    high = max(low + 1, int(0.25 * len(min_heights))) #TODO: qualify magic numbers
    sorted_heights = np.sort(min_heights)
    return float(sorted_heights[low:high].mean())


def remove_short_contacts(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Remove contact segments shorter than min_frames.

    :param mask: A boolean array where True indicates contact frames.
    :param min_frames: Minimum number of frames to keep a contact segment.
    """
    cleaned = mask.copy()
    for start, end in contact_segments(mask):
        if end - start + 1 < min_frames:
            cleaned[start : end + 1] = False
    return cleaned


def contact_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find continuous segments of True values in a boolean mask.

    :param mask: A 1D boolean array to analyze.
    """
    segments = []
    start = None
    for index, value in enumerate(mask):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments
