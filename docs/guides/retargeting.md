# Joint mapping and retargeting

Source and target must use compatible length units and world axes. The target is a `Hierarchy`; the output is a motion on that hierarchy.

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

The low-level transfer scales mapped translations and rebases root translation to the first source frame. `rotation_correction="rest_pose"` is the default; `"none"` disables this correction. Optional `floor_height` requests floor alignment. Unmapped channel values start at zero; their global transforms still depend on mapped ancestors.

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

A prepared motion can be passed to `solve_motion_spacetime` with explicit constraints built from the original source. See [Space-time usage](../spacetime_usage.md). Different-topology quality depends on mapping, chain preparation and constraints; there is no automatic full-body contact-preserving pipeline.
