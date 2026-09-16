# Space-Time Retargeting: Quick Start

`retarget_motion_spacetime` adapts a BVH motion to a target skeleton with the same joint hierarchy, joint order, and channel layout. The skeletons may have different offsets or segment lengths.

The solver starts from the source motion on the target skeleton, scales root translation from the relative rest-pose height, then optimizes smooth root and rotation corrections to satisfy supplied constraints.

## Minimal retarget

Load a source animation and a compatible target skeleton. The target may be a BVH with any motion because its hierarchy and offsets are what matter.

```python
from pathlib import Path

import numpy as np

from broom.bvh.io import load_bvh_document, write_bvh_with_motion_values
from broom.bvh.kinematics import compute_global_positions
from broom.bvh.retargeting import align_root_to_floor, retarget_motion_spacetime

source = load_bvh_document(Path("source_walk.bvh"))
target = load_bvh_document(Path("target_skeleton.bvh"))

# Put the source in a known floor coordinate system before its positions are used as targets.
# Set use_rest_pose=True when rest pose should also affect the floor estimate.
source = align_root_to_floor(source, use_rest_pose=False)
source_positions = compute_global_positions(source)
```

Define contact intervals yourself. In this example `LeftToeBase` is planted on frames 12 through 20. Replace the joint name and interval with the contact data for the clip.

```python
joint_name = "LeftToeBase"
joint_index = source.joint_index[joint_name]
contact_frames = np.arange(12, 21)

constraints = [
    # Keep the contact at the source height without restoring its source XZ footprint.
    # The target may choose a stride length that fits its skeleton.
    {
        "type": "position",
        "name": "left_contact_height",
        "joint": joint_name,
        "frames": contact_frames,
        "axes": "Y",
        "positions": source_positions[contact_frames, joint_index, 1:2],
        "weight": 10.0,
    },
    # Keep the planted foot from sliding horizontally during this interval.
    {
        "type": "stationary",
        "name": "left_contact_xz",
        "joint": joint_name,
        "frames": contact_frames,
        "axes": "XZ",
        "weight": 10.0,
    },
    # Prevent this contact point from passing below the Y=0 floor.
    {
        "type": "floor",
        "name": "left_contact_floor",
        "joint": joint_name,
        "frames": contact_frames,
        "normal": [0.0, 1.0, 0.0],
        "offset": 0.0,
        "weight": 100.0,
    },
]

result = retarget_motion_spacetime(
    source,
    target,
    constraints,
    control_point_spacing=8,
    max_nfev=100,
)

print(result.success, result.message)
for name, residual in result.constraint_residuals.items():
    print(name, np.abs(residual).max())

write_bvh_with_motion_values(
    result.document,
    Path("output_retargeted.bvh"),
    result.motion_values,
    precision=8,
)
```

Add equivalent constraints for every known contact interval. A `floor` constraint is one-sided: it prevents penetration but allows a point to hover. Use a `position` constraint on `"Y"` when the contact must have a known height.

## Automatic floor and contact detection

`broom.bvh.analysis.foot_contacts` can estimate a floor from designated foot joints and propose contact intervals. It is a heuristic, so inspect the resulting intervals before using them as constraints.

```python
from broom.bvh.analysis.foot_contacts import contact_segments, detect_foot_contacts, estimate_floor_height

foot_groups = {
    "left": ["LeftToeBase", "LeftToe_End"],
    "right": ["RightToeBase", "RightToe_End"],
}
foot_indices = [source.joint_index[name] for names in foot_groups.values() for name in names]

floor_y = estimate_floor_height(source_positions, foot_indices)
contacts = detect_foot_contacts(
    source_positions,
    foot_groups,
    joints=source.joints,
    joint_names=source.joint_names,
    min_contact_frames=2,
)

constraints = []
for side, mask in contacts.items():
    joint_name = foot_groups[side][0]
    joint_index = source.joint_index[joint_name]
    for start, end in contact_segments(mask):
        frames = np.arange(start, end + 1)
        constraints.extend([
            {
                "type": "position",
                "name": f"{side}_height_{start}_{end}",
                "joint": joint_name,
                "frames": frames,
                "axes": "Y",
                "positions": source_positions[frames, joint_index, 1:2],
                "weight": 10.0,
            },
            {
                "type": "stationary",
                "name": f"{side}_xz_{start}_{end}",
                "joint": joint_name,
                "frames": frames,
                "axes": "XZ",
                "weight": 10.0,
            },
            {
                "type": "floor",
                "name": f"{side}_floor_{start}_{end}",
                "joint": joint_name,
                "frames": frames,
                "normal": [0.0, 1.0, 0.0],
                "offset": floor_y,
                "weight": 100.0,
            },
        ])
```

If the source was aligned with `align_root_to_floor(..., floor_height=0.0)`, `floor_y` should be near zero. Estimating it explicitly still makes the constraint coordinate system clear.

## Constraint inputs

Every constraint has a unique `name` and frame indices. Most constraints use one target `joint`; `relational` uses target endpoints `joint_a` and `joint_b`. `weight` is optional for `position`, `stationary`, and `floor`: when omitted, the solver derives it from `constraint_tolerance`. An explicit non-negative `weight` overrides that default. Weights are soft penalties, so inspect residuals rather than assuming every constraint was met exactly.

| Type | Required fields | Meaning |
| --- | --- | --- |
| `position` | `positions`, optional `axes` | Match selected world coordinates to a target. `axes` defaults to `"XYZ"`; allowed values are `"X"`, `"Y"`, `"Z"`, `"XY"`, `"XZ"`, `"YZ"`, `"XYZ"`. |
| `stationary` | optional `axes` | Keep selected world coordinates unchanged between consecutive listed frames. |
| `floor` | optional `normal`, `offset` | Penalize positions below a plane. Defaults to the Y=0 plane. |
| `joint_limit` | `minimum`, `maximum` | Penalize BVH Euler rotation channels outside declared bounds. |
| `relational` | `joint_a`, `joint_b`, `source_normalized_distance`, `target_path_length`; optional `activation` | Match the endpoint distance normalized by the target rest-pose kinematic path length to the source-normalized distance. `activation` defaults to `1.0`. |

For `position`, `positions` can be one constant point or one point per frame. It may contain full XYZ coordinates or only the selected axes. A Y-only per-frame target must have shape `(frame_count, 1)`, as in the examples.

Use a full `"XYZ"` position constraint when a joint must remain at a specific world-space location, such as a stair tread or a marked ground point. For free walking, Y position plus XZ stationary contact is usually the better starting constraint: it preserves ground contact without requiring target steps to land in the source footprints.

By default, `constraint_tolerance=0.005` represents a 5 mm positional error. The solver scales it by the resolved root scale and uses `1 / tolerance_target**2` as the spatial constraint weight. This assumes source units represent meters before the scale is applied. For `stationary`, the tolerance applies to each adjacent selected-frame displacement, not accumulated drift across the interval. `joint_limit` remains degree-valued and defaults to weight `1.0`.

## Quick single-file experiment

When no separate target skeleton is available, make a copy of the source skeleton with changed offsets. This is useful only as a local experiment; a real target should come from its own BVH hierarchy.

```python
from dataclasses import replace

offset_scale = {
    "LeftUpLeg": 0.8,
    "LeftLeg": 0.8,
    "RightUpLeg": 0.8,
    "RightLeg": 0.8,
}

target = replace(
    source,
    joints=tuple(
        replace(joint, offset=joint.offset * offset_scale.get(joint.name, 1.0))
        for joint in source.joints
    ),
)
```

Then use `target` in the minimal-retarget call above. Keep both legs symmetric when testing scale effects, otherwise asymmetry becomes part of the experiment.

## Main parameters

- `scale`: optional explicit root-translation scale. Omit it to use the target and source rest-pose height ratio.
- `control_point_spacing`: frames between cubic B-spline breakpoints. Smaller values allow more local corrections (higher frequency); larger values enforce broader changes.
- `control_weight`: penalty on correction control points. Higher values keep the result closer to the initial transferred motion.
- `parameter_weights`: one value for every optimized root-translation or rotation parameter. Rotation corrections are measured in radians. To compare with an older degree-based angular penalty, a starting conversion is `(180 / np.pi) ** 2` for rotation parameters.
- `constraint_tolerance`: source-space spatial tolerance used only when `position`, `stationary`, or `floor` omit `weight`. The default `0.005` is 5 mm; it is multiplied by the resolved root scale before deriving the target-space weight.
- `max_nfev`: solver evaluation budget. Check `success`, `message`, and residuals after every run.

## Limitations
 
The solver is currently for equal-topology skeletons only. It does not infer joint mapping, foot contacts, scene geometry, or physically correct gait.
