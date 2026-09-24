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
