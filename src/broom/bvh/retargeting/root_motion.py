"""Root-motion transfer helpers for BVH retargeting."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from broom.bvh.kinematics import compute_global_positions
from broom.bvh.ops.motion import with_motion_values
from broom.bvh.ops.skeleton import estimate_skeleton_height
from broom.bvh.schemas import BVHDocument


# TODO: Review name-based joint selection; these substrings are not universal
# skeleton semantics and can exclude relevant joints from floor estimation.
_FLOOR_JOINT_NAME_MARKERS = ("foot", "toe", "ankle", "heel")


#TODO: Consider renaming function because of possible confusion with scale_skeleton from bvh.ops.skeleton
def skeleton_scale(
    source_document: BVHDocument,
    target_document: BVHDocument,
) -> float:
    """Return target/source skeleton size ratio for root translation."""

    source_size = estimate_skeleton_height(source_document.joints)
    target_size = estimate_skeleton_height(target_document.joints)
    if source_size <= 1.0e-8 or target_size <= 1.0e-8:
        return 1.0
    return float(target_size / source_size)


def transfer_root_translation(
    source_document: BVHDocument,
    target_document: BVHDocument,
    target_motion: np.ndarray,
    scale: float,
) -> None:
    """Copy source root translation into the output target motion."""

    source_root = _root_position_channels(source_document)
    target_root = _root_position_channels(target_document)
    if not source_root or not target_root:
        return

    source_positions = _extract_positions(
        motion_values=source_document.motion_values,
        channels=source_root,
    )
    root_delta = (source_positions - source_positions[0:1]) * float(scale)
    _write_positions(
        motion_values=target_motion,
        channels=target_root,
        positions=root_delta,
    )


# TODO: Unify floor estimation in analysis and retargeting, with analysis as
# the source of truth. Keep the retargeting implementation for now.
def estimate_floor_level(
    document: BVHDocument,
    *,
    use_rest_pose: bool = True,
    first_frame_only: bool = False,
) -> float:
    """Return the minimum world Y of foot/toe/ankle/heel origins in the selected poses.

    Use the whole clip unless first_frame_only selects just its first frame.
    Fall back to all joints when no foot/toe/ankle/heel names exist. With use_rest_pose,
    also include a pose with zero rotations and the first frame's translations.
    Without animation frames, use zero channels for that rest pose.
    The document is not modified; the returned value is a level, not a shift.
    """
    if not document.joints:
        raise ValueError("Floor estimation requires at least one joint.")

    # TODO: Clarify the original use case for first-frame-only floor estimation.
    values = document.motion_values[:1] if first_frame_only else document.motion_values
    if use_rest_pose:
        rest = (
            values[:1].copy() if values.shape[0]
            else np.zeros((1, document.total_channels), dtype=values.dtype)
        )
        for joint in document.joints:
            for offset, channel in enumerate(joint.channels):
                if channel.endswith("rotation"):
                    rest[0, joint.channel_start + offset] = 0.0
        values = np.concatenate((values, rest), axis=0)
    positions = compute_global_positions(replace(document, motion_values=values))
    floor_indices = [
        index
        for index, joint in enumerate(document.joints)
        if any(marker in joint.name.casefold() for marker in _FLOOR_JOINT_NAME_MARKERS)
    ]
    if not floor_indices:
        floor_indices = list(range(len(document.joints)))

    return float(positions[:, floor_indices, 1].min())


def align_root_to_floor(
    document: BVHDocument,
    floor_height: float = 0.0,
    *,
    use_rest_pose: bool = True,
    first_frame_only: bool = False,
) -> BVHDocument:
    """Return a copy shifted by one constant root Y translation to the floor.

    Estimate over all frames or just the first, optionally including rest pose
    as in estimate_floor_level. Shift the entire clip in either case; preserve
    frame count, timing and all other channels. Requires nonempty motion.
    """
    if document.motion_values.shape[0] == 0:
        raise ValueError("Floor alignment requires at least one motion frame.")
    if not np.isfinite(floor_height):
        raise ValueError("floor_height must be finite.")
    root_y_channel = _root_position_channels(document).get(1)
    if root_y_channel is None:
        raise ValueError("Floor alignment requires a root Yposition channel.")
    level = estimate_floor_level(
        document, use_rest_pose=use_rest_pose, first_frame_only=first_frame_only,
    )
    values = document.motion_values.copy()
    values[:, root_y_channel] += float(floor_height) - level
    return with_motion_values(document, values)


def _root_position_channels(document: BVHDocument) -> dict[int, int]:
    root_joint = next((joint for joint in document.joints if joint.parent == -1), None)
    if root_joint is None:
        return {}

    result = {}
    for offset, channel in enumerate(root_joint.channels):
        if not channel.endswith("position"):
            continue
        dimension = {"X": 0, "Y": 1, "Z": 2}[channel[0]]
        result[dimension] = root_joint.channel_start + offset
    return result


def _extract_positions(
    motion_values: np.ndarray,
    channels: Mapping[int, int],
) -> np.ndarray:
    positions = np.zeros((motion_values.shape[0], 3), dtype=np.float64)
    for dimension, channel_index in channels.items():
        positions[:, dimension] = motion_values[:, channel_index]
    return positions


def _write_positions(
    motion_values: np.ndarray,
    channels: Mapping[int, int],
    positions: np.ndarray,
) -> None:
    for dimension, channel_index in channels.items():
        motion_values[:, channel_index] = positions[:, dimension]
