# Joint mapping and retargeting

Source and target must share coordinate axes; `scale` must account for any difference in length units as well as the desired motion scaling. The target is a `Hierarchy`; the output is a motion on that hierarchy.

## Suggest, review, then transfer

```python
from broom.io import load_bvh
from broom.retargeting import map_joints, validate_joint_mapping, retarget_mapped_motion
from broom.ops.skeleton_geometry import estimate_skeleton_scale_ratio

source = load_bvh("source.bvh")
target = load_bvh("target.bvh").hierarchy
proposal = map_joints(source.hierarchy, target)
for match in proposal.matches:
    print(match.source, match.target, match.score, match.reason)

mapping = dict(proposal.mapping)  # Review/edit source-name -> target-name pairs here.
validate_joint_mapping(source.hierarchy, target, mapping)
scale = estimate_skeleton_scale_ratio(source.hierarchy, target)
result = retarget_mapped_motion(source, target, mapping, scale=scale)
```

The mapper is a heuristic suggestion, not a semantic guarantee. Unmatched joints are reported in `MappingResult`. Explicit transfer does not guess more pairs. Mappings are injective: multiple source joints cannot map to one target joint.

The low-level transfer preserves target rest geometry and scales mapped translations. Source root placement is scaled too; vertical standing-height correction is described below. `rotation_correction="local_orientation"` is the default: it transfers rotation deltas between global rest-orientation bases computed by FK. `"none"` only converts Euler order. `"rest_pose"` is an explicit geometric heuristic that estimates rotation bases from bone directions, not the actual joint coordinate frames. Optional `floor_height` requests a separate constant floor alignment. Unmapped channel values start at zero; their global transforms still depend on mapped ancestors.

For non-root source joints, `local_orientation` transfers translations using `d_target = scale * B_target_parent.T @ B_source_parent @ d_source`, where each B is a global rest rotation. A virtual root parent has identity orientation. Missing source components are zero; joints without position channels leave their target translations unchanged. Static basis conversion does not compensate for different animated parent motion. In `none` and `rest_pose`, non-root translations are copied by matching axis and scaled. Source root trajectory transfer follows the separate rule below in all three modes.

Missing target position channels discard unsupported components; unsupported rotation channel groups are skipped. This may lose motion and is currently allowed; compensation through IK or another algorithm remains future work.

`retarget_motion` is the convenience entry point that also attempts mapping. It returns `RetargetResult` with `.motion`, `.joint_map`, unmatched names and `.scale`. Use explicit transfer when a user has already approved the mapping.

## Refine chains

```python
from broom.retargeting import refine_mapped_chains

# Example names: replace with endpoints in the source and ensure both are mapped.
chain_motion = refine_mapped_chains(
    source, result, mapping, chains=[("Shoulder", "Hand")],
)
```

Each pair defines the source start and end. Full paths are found in both hierarchies. Mapped intermediate joints split paths into segments; unmapped intermediate joints still participate. This supports different link counts.

Chain refinement uses source world geometry and orientations; it assumes compatible orientation frames. A rest-pose correction is not a substitute for coordinate calibration. Chain starts may rotate. Translation channels remain unchanged. Shared boundaries are fitted jointly; conflicting overlaps are rejected. Internal target joints need three rotation axes. This is a geometric initializer, not IK: endpoint positions, floor contacts and temporal smoothness are not guaranteed.

`retarget_mapped_chains` combines mapped transfer and chain refinement and accepts `source_root_height`, `target_mapped_root_height` and `up_axis`. Refinement preserves translation channels, but can move the mapped root recipient through ancestor rotations; its final world trajectory is not constrained.

A prepared motion can be passed to `solve_motion_spacetime` with explicit constraints built from the original source. See [Space-time usage](../spacetime_usage.md). Different-topology quality depends on mapping, chain preparation and constraints; there is no automatic full-body contact-preserving pipeline.

## Root trajectory and standing height

The source structural root transfers its trajectory to its mapped target joint, which need not be target root. An unmapped source root transfers no trajectory. Source and target must share coordinate axes; `scale` includes any unit conversion. `up_axis="Y"` selects the vertical axis (X/Y/Z, case-insensitive) for both transfer and optional floor alignment.

Horizontal positions are scaled relative to the origin. Along the vertical axis, the desired recipient position is `target_mapped_root_height + scale * (source_position - source_root_height)`. Heights refer to these specific joints in a standing rest pose, not total character height. Source height uses source units; target height uses target units. The first animation sample is never subtracted, so initial jumps and crouches remain relative to the standing baseline.

Both heights default to `None`: use the respective joint's global rest coordinate along `up_axis`, including ancestor offsets and orientations. This assumes standing rest poses with ground at zero and source animation using that same ground plane. No minimum-joint floor estimate or name heuristic is used. Explicit finite heights override these baselines without modifying either hierarchy or the source samples.

For a service clip with Hips OFFSET zero, specify its known standing root height:

```python
result = retarget_mapped_motion(
    source, target, mapping,
    scale=1.0,
    source_root_height=1.05,  # Known standing pelvis height in source units.
    up_axis="Y",
)
```

If the mapped target pelvis has rest height `0.9`, source Y samples `[1.05, 1.35]` become `[0.9, 1.2]`. Supply `target_mapped_root_height` as well if its rest coordinates do not encode the desired standing height. These baselines are skeleton properties; do not estimate them from an arbitrary first animation frame.

The desired skeleton-space position is written as `d_target = R_parent(t).T @ (desired_position - p_parent(t)) - target_offset`, using actual animated target parent transforms. Thus an additional target root does not receive the source trajectory automatically. Missing target channels still discard unsupported components. Other source joints retain their existing displacement transfer.

`retarget_motion`, `retarget_bvh_file` and `retarget_mapped_chains` accept the same height options. Floor alignment is a separate post-correction using the same `up_axis`; disable it with `floor_align=False` in the convenience wrappers or leave `floor_height=None` in explicit mapped transfer to preserve the computed trajectory. Height correction alone does not guarantee foot contacts on different skeletons.
