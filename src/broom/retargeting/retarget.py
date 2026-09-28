"""Motion retargeting between different skeleton hierarchies."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from broom import Hierarchy, Motion
from broom.io import load_bvh, write_bvh
from broom.ops.skeleton_geometry import estimate_skeleton_scale_ratio
from broom.retargeting.mapping import (
    load_joint_mapping,
    map_joints,
    unmapped_sources,
    unmapped_targets,
)
from broom.retargeting.root_motion import (
    align_root_to_floor,
    transfer_root_translation,
)
from broom.retargeting.rotation_transfer import (
    transfer_fk_rotations,
)
from broom.retargeting.schemas import RetargetResult


def retarget_motion(
    source_motion: Motion,
    target_motion: Motion,
    joint_map: dict[str, str] | None = None,
    root_translation: str = "scaled",
    root_scale: float | None = None,
    rotation_correction: str = "rest_pose",
    initial_pose: str = "zero",
    floor_align: bool = True,
    strict: bool = False,
    *,
    floor_up_axis: str = "Y",
    floor_use_rest_pose: bool = False,
    floor_first_frame_only: bool = True,
) -> RetargetResult:
    """Transfer source motion onto a target skeleton.

    ``joint_map`` uses source joint names as keys and target joint names as
    values. If it is omitted, joints are matched by normalized names.
    When floor_align is enabled, floor_first_frame_only selects the first
    output frame instead of the whole clip for floor estimation;
    floor_use_rest_pose additionally includes rest pose. floor_up_axis selects
    the coordinate axis normal to the floor. Defaults preserve first-frame-only
    estimation without rest pose and a Y-up floor.
    """

    if source_motion.frame_count <= 0:
        raise ValueError("Source Motion does not contain motion frames.")

    root_translation = root_translation.lower()
    if root_translation not in {"scaled", "copy", "none"}:
        raise ValueError("root_translation must be one of: 'scaled', 'copy', 'none'.")

    rotation_correction = rotation_correction.lower()
    if rotation_correction not in {"rest_pose", "none"}:
        raise ValueError("rotation_correction must be one of: 'rest_pose', 'none'.")

    initial_pose = initial_pose.lower()
    if initial_pose not in {"zero", "first_frame"}:
        raise ValueError("initial_pose must be one of: 'zero', 'first_frame'.")

    mapping_result = map_joints(
        source_hierarchy=source_motion.hierarchy,
        target_hierarchy=target_motion.hierarchy,
        joint_map=joint_map,
    )
    source_to_target = mapping_result.mapping
    if strict:
        _raise_if_unmapped(
            source_hierarchy=source_motion.hierarchy,
            target_hierarchy=target_motion.hierarchy,
            source_to_target=source_to_target,
        )

    if initial_pose == "zero":
        target_values = np.zeros(
            (source_motion.frame_count, target_motion.hierarchy.total_channels),
            dtype=np.float64,
        )
    else:
        if target_motion.frame_count <= 0:
            raise ValueError(
                "Target Motion must contain at least one motion frame "
                "when initial_pose='first_frame'."
            )
        template_pose = np.nan_to_num(target_motion.values[0], nan=0.0)
        target_values = np.repeat(
            template_pose[None, :],
            source_motion.frame_count,
            axis=0,
        )
    transfer_fk_rotations(
        source_motion=source_motion,
        target_hierarchy=target_motion.hierarchy,
        source_to_target=source_to_target,
        target_motion=target_values,
        rotation_correction=rotation_correction,
    )

    scale = float(root_scale) if root_scale is not None else 1.0
    if root_translation == "scaled":
        if root_scale is None:
            scale = estimate_skeleton_scale_ratio(
                source_hierarchy=source_motion.hierarchy,
                target_hierarchy=target_motion.hierarchy,
            )
        transfer_root_translation(
            source_motion=source_motion,
            target_hierarchy=target_motion.hierarchy,
            target_motion=target_values,
            scale=scale,
        )
    elif root_translation == "copy":
        transfer_root_translation(
            source_motion=source_motion,
            target_hierarchy=target_motion.hierarchy,
            target_motion=target_values,
            scale=1.0,
        )
    output_motion = Motion(
        target_motion.hierarchy,
        target_values,
        source_motion.frame_time,
    )
    if floor_align:
        output_motion = align_root_to_floor(
            output_motion,
            up_axis=floor_up_axis,
            use_rest_pose=floor_use_rest_pose,
            first_frame_only=floor_first_frame_only,
        )
    return RetargetResult(
        motion=output_motion,
        joint_map=source_to_target,
        unmapped_source_joints=unmapped_sources(
            source_motion.hierarchy, source_to_target
        ),
        unmapped_target_joints=unmapped_targets(
            target_motion.hierarchy, source_to_target
        ),
        root_scale=scale,
    )


def retarget_bvh_file(
    source_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    joint_map: dict[str, str] | None = None,
    mapping_path: str | Path | None = None,
    root_translation: str = "scaled",
    root_scale: float | None = None,
    rotation_correction: str = "rest_pose",
    initial_pose: str = "zero",
    floor_align: bool = True,
    strict: bool = False,
    precision: int = 6,
    *,
    floor_up_axis: str = "Y",
    floor_use_rest_pose: bool = False,
    floor_first_frame_only: bool = True,
) -> RetargetResult:
    """Load two BVH files, retarget the motion, and write the output BVH.

    Floor alignment options are forwarded to retarget_motion.
    """

    if joint_map is not None and mapping_path is not None:
        raise ValueError("Pass either joint_map or mapping_path, not both.")

    source_motion = load_bvh(source_path)
    target_motion = load_bvh(target_path)
    mapping = load_joint_mapping(mapping_path) if mapping_path else joint_map

    result = retarget_motion(
        source_motion=source_motion,
        target_motion=target_motion,
        joint_map=mapping,
        root_translation=root_translation,
        root_scale=root_scale,
        rotation_correction=rotation_correction,
        initial_pose=initial_pose,
        floor_align=floor_align,
        strict=strict,
        floor_up_axis=floor_up_axis,
        floor_use_rest_pose=floor_use_rest_pose,
        floor_first_frame_only=floor_first_frame_only,
    )
    write_bvh(result.motion, output_path, precision=precision)
    return result


def _raise_if_unmapped(
    source_hierarchy: Hierarchy,
    target_hierarchy: Hierarchy,
    source_to_target: dict[str, str],
) -> None:
    missing_sources = unmapped_sources(source_hierarchy, source_to_target)
    missing_targets = unmapped_targets(target_hierarchy, source_to_target)
    if missing_sources or missing_targets:
        raise ValueError(
            "Retargeting mapping is incomplete. "
            f"Unmapped source joints: {', '.join(missing_sources) or 'none'}. "
            f"Unmapped target joints: {', '.join(missing_targets) or 'none'}."
        )
