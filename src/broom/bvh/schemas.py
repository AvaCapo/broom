"""Common data structures for BVH."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BVHJoint:
    """Parsed BVH joint metadata.

    :param name: Joint name.
    :param parent: Index of the parent joint, or -1 for the root.
    :param offset: Joint offset as a 3D vector.
    :param channels: Tuple of channel names for this joint.
    :param channel_start: Starting index of this joint's
        channels in the motion values.
    :param dof: Number of constrained angular degrees of freedom.
    :param min_values: Minimum rotation values in degrees, ordered like this
        joint's rotation channels.
    :param max_values: Maximum rotation values in degrees, ordered like this
        joint's rotation channels.
    """

    name: str
    parent: int
    offset: np.ndarray
    channels: tuple[str, ...]
    channel_start: int
    dof: int = 0
    min_values: tuple[float, ...] = ()
    max_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class BVHDocument:
    """BVH data split into immutable metadata and motion values.

    :param path: Original file path of the BVH document.
    :param prefix_lines: Lines from the BVH file before the motion data.
    :param motion_rows: Tuple of (channel names, line) for each motion row.
    :param motion_values: Motion values as a 2D array of shape (frames, channels).
    :param joints: Tuple of BVHJoint objects representing the joint hierarchy.
    :param total_channels: Total number of channels in the motion data.
    :param root_name: Name of the root joint.
    :param root_channels: Tuple of channel names for the root joint.
    :param frame_time: Declared duration of a single frame, if available.
    :param declared_frames: Declared frame count from the MOTION header,
        if available.
    """

    path: Path
    prefix_lines: tuple[str, ...]
    motion_rows: tuple[tuple[tuple[str, ...], str], ...]
    motion_values: np.ndarray
    joints: tuple[BVHJoint, ...]
    total_channels: int
    root_name: str
    root_channels: tuple[str, ...]
    frame_time: float | None = None
    declared_frames: int | None = None

    @property
    def frame_count(self) -> int:
        """Actual number of loaded motion frames."""

        return int(self.motion_values.shape[0])

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Joint names in hierarchy order."""

        return tuple(joint.name for joint in self.joints)

    @property
    def joint_index(self) -> dict[str, int]:
        """Mapping from joint name to hierarchy index."""

        return {joint.name: index for index, joint in enumerate(self.joints)}
