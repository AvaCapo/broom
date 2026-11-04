"""Support-area bounds for BVH joint trajectories."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .foot_contacts import JointRef, estimate_floor_height, resolve_joint_groups
from ._validation import validate_joints_world


_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class SupportBounds:
    axis0_min: np.ndarray
    axis0_max: np.ndarray
    axis1_min: np.ndarray
    axis1_max: np.ndarray
    bounds: np.ndarray
    active_contacts: Dict[str, np.ndarray]
    plane: tuple[str, str]
    floor_height: float


def _resolve_group_scalar(
    value: Optional[float | Mapping[str, float]],
    group_name: str,
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if group_name not in value:
            return None
        return float(value[group_name])
    return float(value)


def _validate_plane(plane: Sequence[str]) -> tuple[str, str]:
    if len(plane) != 2:
        raise ValueError("plane must contain exactly 2 axes, e.g. ('x', 'z').")
    axis0 = str(plane[0]).lower()
    axis1 = str(plane[1]).lower()
    if axis0 not in _AXIS_TO_INDEX or axis1 not in _AXIS_TO_INDEX:
        raise ValueError("plane axes must be chosen from {'x', 'y', 'z'}.")
    if axis0 == axis1:
        raise ValueError("plane axes must be different.")
    return axis0, axis1


def _validate_contact_masks(
    contact_masks: Mapping[str, np.ndarray],
    frame_count: int,
) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for group_name, mask in contact_masks.items():
        arr = np.asarray(mask, dtype=bool)
        if arr.ndim != 1 or arr.shape[0] != frame_count:
            raise ValueError(
                f"contact_masks[{group_name!r}] must be a boolean array with shape [{frame_count}]."
            )
        masks[group_name] = arr
    return masks


def compute_support_bounds(
    joints_world: np.ndarray,
    foot_groups: Mapping[str, Sequence[JointRef]],
    contact_masks: Mapping[str, np.ndarray],
    *,
    joint_names: Optional[Sequence[str]] = None,
    plane: Sequence[str] = ("x", "z"),
    foot_lengths: Optional[float | Mapping[str, float]] = None,
    foot_widths: Optional[float | Mapping[str, float]] = None,
    length_padding: float = 0.0,
    width_padding: float = 0.0,
    floor_height: Optional[float] = None,
) -> SupportBounds:
    """
    Compute per-frame support bounds on a chosen plane.

    Each foot group must contain at least two joints:
    - first joint: rear anchor, e.g. `Foot`
    - second joint: front anchor, e.g. `Toe`

    For every active foot contact we build a simple oriented foot rectangle,
    then return only its global min/max bounds on the selected plane.
    """
    frame_count, _ = validate_joints_world(joints_world)
    resolved_groups = resolve_joint_groups(foot_groups, joint_names=joint_names)
    masks = _validate_contact_masks(contact_masks, frame_count)
    axis0_name, axis1_name = _validate_plane(plane)
    axis0_idx = _AXIS_TO_INDEX[axis0_name]
    axis1_idx = _AXIS_TO_INDEX[axis1_name]

    if set(resolved_groups) != set(masks):
        missing_masks = sorted(set(resolved_groups) - set(masks))
        extra_masks = sorted(set(masks) - set(resolved_groups))
        raise ValueError(
            f"foot_groups/contact_masks mismatch. missing_masks={missing_masks}, extra_masks={extra_masks}"
        )

    for group_name, group_indices in resolved_groups.items():
        if group_indices.size < 2:
            raise ValueError(
                f"foot_groups[{group_name!r}] must contain at least two joints: rear and front anchors."
            )

    all_indices = np.concatenate(list(resolved_groups.values()))
    support_floor_height = (
        estimate_floor_height(joints_world, all_indices)
        if floor_height is None
        else float(floor_height)
    )

    axis0_min = np.full((frame_count,), np.nan, dtype=np.float64)
    axis0_max = np.full((frame_count,), np.nan, dtype=np.float64)
    axis1_min = np.full((frame_count,), np.nan, dtype=np.float64)
    axis1_max = np.full((frame_count,), np.nan, dtype=np.float64)

    for frame_idx in range(frame_count):
        frame_points: list[np.ndarray] = []

        for group_name, group_indices in resolved_groups.items():
            if not bool(masks[group_name][frame_idx]):
                continue

            rear_idx = int(group_indices[0])
            front_idx = int(group_indices[1])

            rear_point = joints_world[frame_idx, rear_idx, [axis0_idx, axis1_idx]].astype(np.float64)
            front_point = joints_world[frame_idx, front_idx, [axis0_idx, axis1_idx]].astype(np.float64)

            direction = front_point - rear_point
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1.0e-8:
                direction = np.array([1.0, 0.0], dtype=np.float64)
                direction_norm = 0.0
            else:
                direction = direction / direction_norm

            normal = np.array([-direction[1], direction[0]], dtype=np.float64)

            foot_length = _resolve_group_scalar(foot_lengths, group_name)
            foot_width = _resolve_group_scalar(foot_widths, group_name)

            effective_length = (
                direction_norm
                if foot_length is None
                else max(float(foot_length), 1.0e-6)
            )
            effective_width = (
                max(0.35 * direction_norm, 1.0e-6)
                if foot_width is None
                else max(float(foot_width), 1.0e-6)
            )

            extra_length = max(0.0, effective_length - direction_norm)
            half_width = 0.5 * effective_width + float(width_padding)

            rear_center = rear_point - direction * (0.5 * extra_length + float(length_padding))
            front_center = front_point + direction * (0.5 * extra_length + float(length_padding))

            frame_points.extend(
                [
                    rear_center + normal * half_width,
                    front_center + normal * half_width,
                    front_center - normal * half_width,
                    rear_center - normal * half_width,
                ]
            )

        if not frame_points:
            continue

        frame_points_arr = np.asarray(frame_points, dtype=np.float64)
        axis0_min[frame_idx] = frame_points_arr[:, 0].min()
        axis0_max[frame_idx] = frame_points_arr[:, 0].max()
        axis1_min[frame_idx] = frame_points_arr[:, 1].min()
        axis1_max[frame_idx] = frame_points_arr[:, 1].max()

    bounds = np.column_stack([axis0_min, axis0_max, axis1_min, axis1_max])

    return SupportBounds(
        axis0_min=axis0_min,
        axis0_max=axis0_max,
        axis1_min=axis1_min,
        axis1_max=axis1_max,
        bounds=bounds,
        active_contacts=masks,
        plane=(axis0_name, axis1_name),
        floor_height=support_floor_height,
    )
