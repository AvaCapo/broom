"""Forward kinematics helpers for BVH motion data."""

from __future__ import annotations

import numpy as np

from broom.bvh.channels import position_channel_dimension
from broom.bvh.schemas import BVHDocument


def compute_global_positions(document: BVHDocument) -> np.ndarray:
    """Compute global joint positions for each frame."""

    frames = document.motion_values.shape[0]
    joint_count = len(document.joints)
    positions = np.zeros((frames, joint_count, 3), dtype=np.float64)
    rotations = np.zeros((frames, joint_count, 3, 3), dtype=np.float64)

    identity = np.eye(3, dtype=np.float64)
    rotations[:] = identity

    for joint_index, joint in enumerate(document.joints):
        local_position = np.repeat(joint.offset[None, :], frames, axis=0)
        local_rotation = np.repeat(identity[None, :, :], frames, axis=0)

        for channel_offset, channel in enumerate(joint.channels):
            values = document.motion_values[
                :, joint.channel_start + channel_offset
            ]
            values = np.nan_to_num(values, nan=0.0)
            if channel.endswith("position"):
                local_position[
                    :, position_channel_dimension(channel)
                ] += values
            elif channel.endswith("rotation"):
                local_rotation = np.matmul(
                    local_rotation,
                    axis_rotation_matrices(channel[0], np.deg2rad(values)),
                )

        if joint.parent == -1:
            positions[:, joint_index, :] = local_position
            rotations[:, joint_index, :, :] = local_rotation
        else:
            parent_rotation = rotations[:, joint.parent, :, :]
            parent_position = positions[:, joint.parent, :]
            rotations[:, joint_index, :, :] = np.matmul(
                parent_rotation, local_rotation
            )
            positions[:, joint_index, :] = parent_position + np.einsum(
                "fij,fj->fi", parent_rotation, local_position
            )

    return positions


def axis_rotation_matrices(axis: str, radians: np.ndarray) -> np.ndarray:
    """Return rotation matrices for an axis and per-frame angles."""

    matrices = np.zeros((len(radians), 3, 3), dtype=np.float64)
    cos_v = np.cos(radians)
    sin_v = np.sin(radians)

    if axis == "X":
        matrices[:, 0, 0] = 1.0
        matrices[:, 1, 1] = cos_v
        matrices[:, 1, 2] = -sin_v
        matrices[:, 2, 1] = sin_v
        matrices[:, 2, 2] = cos_v
    elif axis == "Y":
        matrices[:, 0, 0] = cos_v
        matrices[:, 0, 2] = sin_v
        matrices[:, 1, 1] = 1.0
        matrices[:, 2, 0] = -sin_v
        matrices[:, 2, 2] = cos_v
    elif axis == "Z":
        matrices[:, 0, 0] = cos_v
        matrices[:, 0, 1] = -sin_v
        matrices[:, 1, 0] = sin_v
        matrices[:, 1, 1] = cos_v
        matrices[:, 2, 2] = 1.0
    else:
        raise ValueError(f"Unsupported rotation axis '{axis}'.")

    return matrices
