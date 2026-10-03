"""Skeleton hierarchy editing operations."""

from __future__ import annotations

import math
from dataclasses import replace

from broom.hierarchy import Hierarchy


def scale_offsets(hierarchy: Hierarchy, factor: float) -> Hierarchy:
    """Return a hierarchy with rest offsets scaled by ``factor``.

    Joint offsets, including the root offset, and End Site offsets are scaled
    relative to the skeleton origin. Topology, channel declarations, names and
    local orientations are preserved.
    """
    if not isinstance(hierarchy, Hierarchy):
        raise TypeError("hierarchy must be a Hierarchy.")
    if isinstance(factor, bool):
        raise ValueError("factor must be a finite positive number.")
    try:
        scale = float(factor)
    except (TypeError, ValueError) as error:
        raise ValueError("factor must be a finite positive number.") from error
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("factor must be a finite positive number.")

    joints = tuple(
        replace(
            joint,
            offset=tuple(scale * value for value in joint.offset),
            end_site_offset=(
                None
                if joint.end_site_offset is None
                else tuple(scale * value for value in joint.end_site_offset)
            ),
        )
        for joint in hierarchy.joints
    )
    return Hierarchy(joints)


def scale_joint_offset(hierarchy: Hierarchy, joint_name: str, factor: float) -> Hierarchy:
    """Return a hierarchy with one joint's incoming rest offset scaled.

    The offset describes the connection from the joint's parent to the named
    joint. End Site offsets and all other joints are preserved. The root is
    not accepted because its offset defines skeleton placement rather than a
    bone length.
    """
    if not isinstance(hierarchy, Hierarchy):
        raise TypeError("hierarchy must be a Hierarchy.")
    joint_index = hierarchy.joint_index(joint_name)
    if joint_index == hierarchy.root:
        raise ValueError("The root joint offset cannot be scaled as a bone length.")
    if isinstance(factor, bool):
        raise ValueError("factor must be a finite non-negative number.")
    try:
        scale = float(factor)
    except (TypeError, ValueError) as error:
        raise ValueError("factor must be a finite non-negative number.") from error
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("factor must be a finite non-negative number.")

    joints = list(hierarchy.joints)
    joint = joints[joint_index]
    joints[joint_index] = replace(
        joint,
        offset=tuple(scale * value for value in joint.offset),
    )
    return Hierarchy(tuple(joints))
