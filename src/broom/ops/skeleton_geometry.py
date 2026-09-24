"""Geometric measurements of skeleton rest poses."""

from __future__ import annotations

import numpy as np

from broom.hierarchy import Hierarchy
from broom.kinematics import compute_global_transforms_from_local

_UP_AXIS_INDICES = {"X": 0, "Y": 1, "Z": 2}


def _up_axis_index(up_axis: str) -> int:
    """Return the coordinate index for an up-axis name."""
    if not isinstance(up_axis, str):
        raise ValueError("up_axis must be one of 'X', 'Y', or 'Z'.")
    try:
        return _UP_AXIS_INDICES[up_axis.upper()]
    except KeyError as error:
        raise ValueError("up_axis must be one of 'X', 'Y', or 'Z'.") from error


def estimate_height(
    hierarchy: Hierarchy,
    *,
    up_axis: str = "Y",
    include_end_sites: bool = True,
) -> float:
    """Return the rest-pose span along an up axis.

    End Site positions are included when ``include_end_sites`` is true.
    """
    if not isinstance(hierarchy, Hierarchy):
        raise TypeError("hierarchy must be a Hierarchy.")
    if not isinstance(include_end_sites, bool):
        raise ValueError("include_end_sites must be a bool.")
    axis = _up_axis_index(up_axis)
    positions, rotations = compute_global_transforms_from_local(hierarchy)
    rest_positions = positions[0]
    if include_end_sites:
        end_site_positions = [
            rest_positions[index] + rotations[0, index] @ joint.end_site_offset
            for index, joint in enumerate(hierarchy.joints)
            if joint.end_site_offset is not None
        ]
        if end_site_positions:
            rest_positions = np.vstack((rest_positions, end_site_positions))
    return float(np.ptp(rest_positions[:, axis]))


def estimate_joint_height(
    hierarchy: Hierarchy,
    joint_index: int,
    *,
    up_axis: str = "Y",
    floor_height: float | None = None,
    include_end_sites: bool = True,
) -> float:
    """Return a rest-pose joint's signed height above a floor.

    If ``floor_height`` is None, the floor is the lowest rest-pose point
    along ``up_axis``. End Site positions participate in that choice when
    ``include_end_sites`` is true.
    """
    if not isinstance(hierarchy, Hierarchy):
        raise TypeError("hierarchy must be a Hierarchy.")
    if isinstance(joint_index, bool) or not isinstance(joint_index, int):
        raise IndexError("joint_index must be a joint index.")
    if not 0 <= joint_index < hierarchy.joint_count:
        raise IndexError("joint_index is outside hierarchy.")
    if not isinstance(include_end_sites, bool):
        raise ValueError("include_end_sites must be a bool.")
    axis = _up_axis_index(up_axis)
    positions, rotations = compute_global_transforms_from_local(hierarchy)
    rest_positions = positions[0]
    if floor_height is None:
        floor_positions = rest_positions
        if include_end_sites:
            end_site_positions = [
                rest_positions[index] + rotations[0, index] @ joint.end_site_offset
                for index, joint in enumerate(hierarchy.joints)
                if joint.end_site_offset is not None
            ]
            if end_site_positions:
                floor_positions = np.vstack((floor_positions, end_site_positions))
        floor = float(floor_positions[:, axis].min())
    else:
        floor = float(floor_height)
        if not np.isfinite(floor):
            raise ValueError("floor_height must be finite or None.")
    return float(rest_positions[joint_index, axis] - floor)



# TODO: Evaluate whether hips detection belongs in Broom or a consumer preset.
DEFAULT_HIPS_NAMES = (
    "hips",
    "pelvis",
    "mixamorig:hips",
    "root",
)

def estimate_hips_height(
    hierarchy: Hierarchy,
    hips_name: str | None = None,
    *,
    up_axis: str = "Y",
    floor_height: float | None = None,
    include_end_sites: bool = True,
) -> float:
    """Return the detected hips joint's rest-pose height above a floor."""
    return estimate_joint_height(
        hierarchy,
        find_hips_joint_index(hierarchy, hips_name=hips_name),
        up_axis=up_axis,
        floor_height=floor_height,
        include_end_sites=include_end_sites,
    )


def find_hips_joint_index(
    hierarchy: Hierarchy, hips_name: str | None = None
) -> int:
    """Return the index of a hips or pelvis joint selected by name heuristics."""
    if not isinstance(hierarchy, Hierarchy):
        raise TypeError("hierarchy must be a Hierarchy.")
    joint_names = tuple(name.casefold() for name in hierarchy.joint_names)

    # search by provided hips joint name
    if hips_name is not None:
        if not isinstance(hips_name, str):
            raise ValueError("hips_name must be a string or None.")
        hips_name_normalized = hips_name.casefold()
        for index, joint_name in enumerate(joint_names):
            if joint_name == hips_name_normalized:
                return index
        raise ValueError(f"Hips joint {hips_name!r} was not found.")

    # exact search default known hip joint names
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

    # fallback to the root joint if no hips joint was found
    return 0
