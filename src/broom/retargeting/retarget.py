"""Motion retargeting between different skeleton hierarchies."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from broom import Hierarchy, Motion
from broom.io import load_bvh, write_bvh
from broom.ops.skeleton_geometry import estimate_skeleton_scale_ratio
from broom.retargeting.mapping import (
    load_joint_mapping,
    map_joints,
    validate_joint_mapping,
)
from broom.retargeting.root_motion import (
    align_root_to_floor,
    transfer_mapped_translations,
)
from broom.retargeting.rotation_transfer import transfer_fk_rotations
from broom.retargeting.schemas import RetargetResult


def retarget_mapped_motion(
    source_motion: Motion,
    target_hierarchy: Hierarchy,
    joint_map: Mapping[str, str],
    *,
    scale: float = 1.0,
    rotation_correction: str = "rest_pose",
    floor_height: float | None = None,
    floor_up_axis: str = "Y",
    floor_use_rest_pose: bool = False,
    floor_first_frame_only: bool = True,
) -> Motion:
    """Transfer a Motion through an explicit source-to-target joint mapping.

    Position and rotation channels are transferred only for mapped joints. All
    transferred position values use ``scale``; root translation is relative to
    the first source frame. When ``floor_height`` is not None, the result is
    aligned to that coordinate along ``floor_up_axis``. The returned Motion
    uses ``target_hierarchy`` and the source frame time.
    """

    validate_joint_mapping(
        source_motion.hierarchy, target_hierarchy, joint_map
    )
    try:
        scale = float(scale)
    except (TypeError, ValueError) as error:
        raise ValueError("scale must be a finite positive number.") from error
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be a finite positive number.")

    if not isinstance(rotation_correction, str):
        raise ValueError("rotation_correction must be 'rest_pose' or 'none'.")
    rotation_correction = rotation_correction.lower()
    if rotation_correction not in {"rest_pose", "none"}:
        raise ValueError("rotation_correction must be 'rest_pose' or 'none'.")

    target_values = np.zeros(
        (source_motion.frame_count, target_hierarchy.total_channels),
        dtype=np.float64,
    )
    source_to_target = dict(joint_map)
    transfer_fk_rotations(
        source_motion=source_motion,
        target_hierarchy=target_hierarchy,
        source_to_target=source_to_target,
        target_motion=target_values,
        rotation_correction=rotation_correction,
    )
    transfer_mapped_translations(
        source_motion=source_motion,
        target_hierarchy=target_hierarchy,
        source_to_target=source_to_target,
        target_values=target_values,
        scale=scale,
    )
    output_motion = Motion(
        target_hierarchy, target_values, source_motion.frame_time
    )
    if floor_height is not None:
        output_motion = align_root_to_floor(
            output_motion,
            floor_height=floor_height,
            up_axis=floor_up_axis,
            use_rest_pose=floor_use_rest_pose,
            first_frame_only=floor_first_frame_only,
        )
    return output_motion


def retarget_motion(
    source_motion: Motion,
    target_hierarchy: Hierarchy,
    known_joint_map: Mapping[str, str] | None = None,
    *,
    min_score: float = 0.65,
    scale: float | None = None,
    auto_scale: bool = True,
    rotation_correction: str = "rest_pose",
    floor_align: bool = True,
    floor_height: float = 0.0,
    strict: bool = False,
    floor_up_axis: str = "Y",
    floor_use_rest_pose: bool = False,
    floor_first_frame_only: bool = True,
) -> RetargetResult:
    """Automatically map and retarget source motion onto a target hierarchy.

    ``known_joint_map`` fixes explicit source-to-target pairs before automatic
    mapping. Choose exactly one scaling policy: set ``auto_scale=True`` without
    ``scale`` to estimate skeleton scale, or set ``auto_scale=False`` together
    with an explicit positive ``scale``. When ``floor_align`` is enabled,
    ``floor_height`` selects the output floor coordinate.
    """

    if not isinstance(auto_scale, bool):
        raise ValueError("auto_scale must be a bool.")
    if auto_scale and scale is not None:
        raise ValueError("Pass either auto_scale=True or an explicit scale, not both.")
    if not auto_scale and scale is None:
        raise ValueError("scale is required when auto_scale is False.")

    mapping_result = map_joints(
        source_hierarchy=source_motion.hierarchy,
        target_hierarchy=target_hierarchy,
        known_joint_map=known_joint_map,
        min_score=min_score,
    )
    if strict and (
        mapping_result.unmapped_sources or mapping_result.unmapped_targets
    ):
        raise ValueError(
            "Retargeting mapping is incomplete. "
            "Unmapped source joints: "
            f"{', '.join(mapping_result.unmapped_sources) or 'none'}. "
            "Unmapped target joints: "
            f"{', '.join(mapping_result.unmapped_targets) or 'none'}."
        )

    resolved_scale = (
        estimate_skeleton_scale_ratio(
            source_hierarchy=source_motion.hierarchy,
            target_hierarchy=target_hierarchy,
        )
        if auto_scale
        else scale
    )
    output_motion = retarget_mapped_motion(
        source_motion=source_motion,
        target_hierarchy=target_hierarchy,
        joint_map=mapping_result.mapping,
        scale=resolved_scale,
        rotation_correction=rotation_correction,
        floor_height=floor_height if floor_align else None,
        floor_up_axis=floor_up_axis,
        floor_use_rest_pose=floor_use_rest_pose,
        floor_first_frame_only=floor_first_frame_only,
    )
    return RetargetResult(
        motion=output_motion,
        joint_map=mapping_result.mapping,
        unmapped_source_joints=mapping_result.unmapped_sources,
        unmapped_target_joints=mapping_result.unmapped_targets,
        scale=float(resolved_scale),
    )


def retarget_bvh_file(
    source_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    known_joint_map: Mapping[str, str] | None = None,
    mapping_path: str | Path | None = None,
    min_score: float = 0.65,
    scale: float | None = None,
    auto_scale: bool = True,
    rotation_correction: str = "rest_pose",
    floor_align: bool = True,
    floor_height: float = 0.0,
    strict: bool = False,
    precision: int = 6,
    *,
    floor_up_axis: str = "Y",
    floor_use_rest_pose: bool = False,
    floor_first_frame_only: bool = True,
) -> RetargetResult:
    """Load BVH motions, retarget onto the target hierarchy, and write BVH."""

    if known_joint_map is not None and mapping_path is not None:
        raise ValueError("Pass either known_joint_map or mapping_path, not both.")

    source_motion = load_bvh(source_path)
    target_hierarchy = load_bvh(target_path).hierarchy
    mapping = load_joint_mapping(mapping_path) if mapping_path else known_joint_map

    result = retarget_motion(
        source_motion=source_motion,
        target_hierarchy=target_hierarchy,
        known_joint_map=mapping,
        min_score=min_score,
        scale=scale,
        auto_scale=auto_scale,
        rotation_correction=rotation_correction,
        floor_align=floor_align,
        floor_height=floor_height,
        strict=strict,
        floor_up_axis=floor_up_axis,
        floor_use_rest_pose=floor_use_rest_pose,
        floor_first_frame_only=floor_first_frame_only,
    )
    write_bvh(result.motion, output_path, precision=precision)
    return result
