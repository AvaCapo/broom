"""Channel and joint selection helpers for Broom motions."""
#TODO: consider rewriting these functions as Hierarchy class properties 
# or derive explicitely where such logic is needed from existing class methods

from __future__ import annotations
from typing import Sequence

import numpy as np

from broom import Joint, Motion


def root_channel_indices(
    root_channels: Sequence[str], axes: Sequence[str]
) -> list[int]:
    """Return root-channel indices for the requested channel names."""

    indices = []
    for axis in axes:
        if axis not in root_channels:
            raise ValueError(
                f"Root channel '{axis}' was not found. "
                f"Available root channels: {', '.join(root_channels)}"
            )
        indices.append(root_channels.index(axis))
    return indices


def root_points(motion: Motion, axes: Sequence[str]) -> np.ndarray:
    """Return selected root channels as a 2D array copy."""

    root_index = motion.hierarchy.root
    root_channels = motion.hierarchy.root_joint.channels
    channel_indices = root_channel_indices(root_channels, axes)
    start = motion.hierarchy.channel_start(root_index)
    return motion.values[:, [start + index for index in channel_indices]].copy()


def position_channel_dimension(channel: str) -> int:
    """Return dimension index for a position channel."""

    if channel.startswith("X"):
        return 0
    if channel.startswith("Y"):
        return 1
    if channel.startswith("Z"):
        return 2
    raise ValueError(f"Unsupported position channel '{channel}'.")


def joint_name_matches(name: str, target: str) -> bool:
    """Match joint names case-insensitively and tolerate namespaces."""

    name = name.lower()
    target = target.lower()
    return (
        name == target or name.endswith(":" + target) or name.endswith(target)
    )


def select_body_joints(
    joints: Sequence[Joint],
    weighted_targets: Sequence[tuple[str, float]],
    use_all_joints: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Select body joints by name and assign analysis weights."""

    if use_all_joints:
        indices = np.arange(len(joints), dtype=np.int64)
        weights = np.ones(len(joints), dtype=np.float64)
        return indices, weights

    selected: dict[int, float] = {}
    for target_name, weight in weighted_targets:
        for index, joint in enumerate(joints):
            if joint_name_matches(joint.name, target_name):
                selected[index] = max(selected.get(index, 0.0), float(weight))

    if not selected:
        raise ValueError("No body joints matched the BVH hierarchy.")

    indices = np.asarray(sorted(selected), dtype=np.int64)
    weights = np.asarray(
        [selected[int(index)] for index in indices], dtype=np.float64
    )
    if float(weights.sum()) <= 0.0:
        raise ValueError("Body PCA weights must sum to a positive value.")
    return indices, weights


def weighted_body_points(
    global_positions: np.ndarray,
    selected_indices: np.ndarray,
    weights: np.ndarray,
    dimensions: Sequence[int],
) -> np.ndarray:
    """Calculate a weighted body trajectory from global joint positions."""

    selected = global_positions[:, selected_indices, :][:, :, dimensions]
    return np.average(selected, axis=1, weights=weights)
