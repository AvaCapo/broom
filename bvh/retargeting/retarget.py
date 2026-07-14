"""BVH motion retargeting between different skeleton hierarchies."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from broom.bvh.io import (
    load_bvh_document,
    write_bvh_with_motion_values,
)
from broom.bvh.ops.motion import (
    with_motion_values,
)
from broom.bvh.schemas import BVHDocument
from broom.bvh.retargeting.mapping import (
    load_joint_mapping,
    map_joints,
    unmapped_sources,
    unmapped_targets,
)
from broom.bvh.retargeting.root_motion import (
    align_root_to_floor,
    skeleton_scale,
    transfer_root_translation,
)
from broom.bvh.retargeting.rotation_transfer import (
    transfer_fk_rotations,
)
from broom.bvh.retargeting.schemas import RetargetResult


def retarget_motion(
    source_document: BVHDocument,
    target_document: BVHDocument,
    joint_map: dict[str, str] | None = None,
    root_translation: str = "scaled",
    root_scale: float | None = None,
    rotation_correction: str = "rest_pose",
    initial_pose: str = "zero",
    floor_align: bool = True,
    strict: bool = False,
) -> RetargetResult:
    """Transfer source BVH motion onto a target BVH skeleton.

    ``joint_map`` uses source joint names as keys and target joint names as
    values. If it is omitted, joints are matched by normalized names.
    """

    if source_document.frame_count <= 0:
        raise ValueError("Source BVH document does not contain motion frames.")

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
        source_document=source_document,
        target_document=target_document,
        joint_map=joint_map,
    )
    source_to_target = mapping_result.mapping
    if strict:
        _raise_if_unmapped(
            source_document=source_document,
            target_document=target_document,
            source_to_target=source_to_target,
        )

    if initial_pose == "zero":
        target_motion = np.zeros(
            (source_document.frame_count, target_document.total_channels),
            dtype=np.float64,
        )
    else:
        if target_document.frame_count <= 0:
            raise ValueError(
                "Target BVH document must contain at least one motion frame "
                "when initial_pose='first_frame'."
            )
        template_pose = np.nan_to_num(target_document.motion_values[0], nan=0.0)
        target_motion = np.repeat(
            template_pose[None, :],
            source_document.frame_count,
            axis=0,
        )
    transfer_fk_rotations(
        source_document=source_document,
        target_document=target_document,
        source_to_target=source_to_target,
        target_motion=target_motion,
        rotation_correction=rotation_correction,
    )

    scale = float(root_scale) if root_scale is not None else 1.0
    if root_translation == "scaled":
        if root_scale is None:
            scale = skeleton_scale(
                source_document=source_document,
                target_document=target_document,
            )
        transfer_root_translation(
            source_document=source_document,
            target_document=target_document,
            target_motion=target_motion,
            scale=scale,
        )
    elif root_translation == "copy":
        transfer_root_translation(
            source_document=source_document,
            target_document=target_document,
            target_motion=target_motion,
            scale=1.0,
        )
    if floor_align:
        align_root_to_floor(
            document=target_document,
            motion_values=target_motion,
        )

    output_document = with_motion_values(
        document=target_document,
        motion_values=target_motion,
        frame_time=source_document.frame_time,
    )
    return RetargetResult(
        document=output_document,
        motion_values=target_motion,
        joint_map=source_to_target,
        unmapped_source_joints=unmapped_sources(source_document, source_to_target),
        unmapped_target_joints=unmapped_targets(target_document, source_to_target),
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
) -> RetargetResult:
    """Load two BVH files, retarget the motion, and write the output BVH."""

    if joint_map is not None and mapping_path is not None:
        raise ValueError("Pass either joint_map or mapping_path, not both.")

    source_document = load_bvh_document(source_path)
    target_document = load_bvh_document(target_path)
    mapping = load_joint_mapping(mapping_path) if mapping_path else joint_map

    result = retarget_motion(
        source_document=source_document,
        target_document=target_document,
        joint_map=mapping,
        root_translation=root_translation,
        root_scale=root_scale,
        rotation_correction=rotation_correction,
        initial_pose=initial_pose,
        floor_align=floor_align,
        strict=strict,
    )
    write_bvh_with_motion_values(
        document=result.document,
        output_path=output_path,
        motion_values=result.motion_values,
        precision=precision,
    )
    return result


def _raise_if_unmapped(
    source_document: BVHDocument,
    target_document: BVHDocument,
    source_to_target: dict[str, str],
) -> None:
    missing_sources = unmapped_sources(source_document, source_to_target)
    missing_targets = unmapped_targets(target_document, source_to_target)
    if missing_sources or missing_targets:
        raise ValueError(
            "Retargeting mapping is incomplete. "
            f"Unmapped source joints: {', '.join(missing_sources) or 'none'}. "
            f"Unmapped target joints: {', '.join(missing_targets) or 'none'}."
        )
