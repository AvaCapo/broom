"""Root-motion transfer helpers for BVH retargeting."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from broom.bvh.kinematics import compute_global_positions
from broom.bvh.ops.skeleton import estimate_skeleton_height
from broom.bvh.schemas import BVHDocument


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


def align_root_to_floor(
    document: BVHDocument,
    motion_values: np.ndarray,
    floor_height: float = 0.0,
) -> float:
    """Offset root Y translation so the first-frame feet rest on the floor."""

    root_y_channel = _root_position_channels(document).get(1)
    if root_y_channel is None:
        return 0.0

    animated_document = replace(document, motion_values=motion_values)
    positions = compute_global_positions(animated_document)
    floor_indices = _floor_joint_indices(document)
    if floor_indices.size == 0:
        floor_indices = np.arange(len(document.joints), dtype=np.int64)

    current_floor = float(positions[0, floor_indices, 1].min())
    correction = float(floor_height) - current_floor
    motion_values[:, root_y_channel] += correction
    return correction


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


def _floor_joint_indices(document: BVHDocument) -> np.ndarray:
    indices = [
        index
        for index, joint in enumerate(document.joints)
        if "foot" in joint.name.casefold() or "toe" in joint.name.casefold()
    ]
    return np.asarray(indices, dtype=np.int64)
