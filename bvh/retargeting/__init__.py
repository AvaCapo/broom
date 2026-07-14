"""BVH retargeting helpers."""

from bvh.retargeting.mapping import (
    build_joint_mapping,
    load_joint_mapping,
    map_joints,
    score_joint_match,
)
from bvh.retargeting.retarget import (
    retarget_bvh_file,
    retarget_motion,
)
from bvh.retargeting.root_motion import (
    align_root_to_floor,
    skeleton_scale,
    transfer_root_translation,
)
from bvh.retargeting.rotation_transfer import (
    rotation_channels_by_name,
    transfer_fk_rotations,
)
from bvh.retargeting.schemas import (
    JointMatch,
    MappingResult,
    RetargetResult,
)

__all__ = (
    "JointMatch",
    "MappingResult",
    "RetargetResult",
    "align_root_to_floor",
    "build_joint_mapping",
    "load_joint_mapping",
    "map_joints",
    "rotation_channels_by_name",
    "retarget_bvh_file",
    "retarget_motion",
    "score_joint_match",
    "skeleton_scale",
    "transfer_fk_rotations",
    "transfer_root_translation",
)
