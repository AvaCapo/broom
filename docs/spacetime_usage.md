# Space-time retargeting

Space-time retargeting transfers motion to a skeleton with different proportions, then optimizes smooth corrections over time to satisfy explicit constraints. Typical constraints preserve foot contacts while allowing the root and joint rotations to adjust. All constraints are **soft penalties**, not exact guarantees.

## Load source motion and target skeleton

```python
import numpy as np
from broom.io import load_bvh, write_bvh
from broom.kinematics import compute_global_positions
from broom.retargeting import retarget_motion_spacetime

source = load_bvh("walk.bvh")
target_hierarchy = load_bvh("target.bvh").hierarchy
source_positions = compute_global_positions(source)
```

Only the hierarchy of `target.bvh` is used. The source and target must have matching joint counts, parent indices and channel layouts; their offsets can differ. The source needs at least two frames. For different topologies, see [Refine a prepared motion](#refine-a-prepared-motion) below.

Use consistent units and world coordinates. The following example assumes meters, Y up, and a ground surface at Y=0. BVH itself does not specify units; see [Coordinates and timing](concepts/coordinates.md).

## Define foot-contact constraints

Contact intervals should come from the **original source animation**: annotate them or detect them before retargeting. Here frames 12 through 20 are assumed to be a known left-foot contact. Replace the names and interval with your data.

```python
source_foot = source.hierarchy.joint_index("LeftToeBase")
target_foot_name = "LeftToeBase"
contact_frames = np.arange(12, 21)
assert contact_frames[-1] < source.frame_count
all_frames = np.arange(source.frame_count)
floor_y = 0.0

constraints = [
    dict(
        type="position", name="left_contact_height",
        joint=target_foot_name, frames=contact_frames, axes="Y",
        positions=source_positions[contact_frames, source_foot, 1:2],
    ),
    dict(
        type="stationary", name="left_contact_stationary",
        joint=target_foot_name, frames=contact_frames, axes="XZ",
    ),
    dict(
        type="floor", name="left_no_penetration",
        joint=target_foot_name, frames=all_frames,
        normal=[0, 1, 0], offset=floor_y,
    ),
]
```

These constraints serve different purposes:

- **Contact height** preserves the source joint's vertical coordinates during contact. Use `positions=[floor_y]` instead if this joint should lie exactly on the floor. An ankle joint usually sits above the sole, so its intended height is not necessarily zero.
- **Stationary contact** penalizes changes in X and Z between adjacent listed frames. It lets the optimizer choose where the foot is planted; it does not copy the source's absolute horizontal trajectory.
- **Floor** penalizes penetration at all listed frames, including outside contact intervals. It does not force contact or prevent horizontal sliding.

Add separate height/stationary constraints for every continuous contact interval and each foot. Do not combine separated intervals into one stationary constraint: that also links the last frame of one interval to the first of the next. Choose corresponding source/target contact points with comparable meaning.

**Y and XZ are choices in this example, not restrictions.** Position and stationary constraints accept `X`, `Y`, `Z`, `XY`, `XZ`, `YZ`, or `XYZ`. For Z-up data, use Z for height, XY for stationary contact, and `[0, 0, 1]` as the floor normal. A floor constraint can represent any plane, including a slope. Coordinate-axis constraints do not automatically rotate into that plane's tangent frame.

## Run and inspect the result

```python
# Rotations are optimized in radians. This optional weighting makes their
# coefficient penalty comparable to a penalty on angles measured in degrees.
parameter_weights = [
    (180 / np.pi) ** 2 if channel.endswith("rotation") else 1.0
    for joint in target_hierarchy.joints
    for channel in joint.channels
    if channel.endswith("rotation")
    or (joint.parent == -1 and channel.endswith("position"))
]
result = retarget_motion_spacetime(
    source,
    target_hierarchy,
    constraints,
    scale=None,
    control_point_spacing=4,
    parameter_weights=parameter_weights,
    constraint_tolerance=0.005,
    max_nfev=60,
)
print(result.success, result.message)
for name, residual in result.constraint_residuals.items():
    print(name, "maximum weighted residual:", np.max(np.abs(residual)))
write_bvh(result.motion, "retargeted.bvh")
```

The function performs three stages:

1. Copy source samples onto the target hierarchy and scale root translations. `scale=None` uses the target/source rest-height ratio; `scale=1.0` disables scaling. It scales root translation values, not only frame-to-frame travel.
2. Fit root translation to position constraints, interpolate the required shifts over time, and smooth them. This preparation is always performed; stationary and floor constraints alone do not provide positional anchors for this step.
3. Optimize spline corrections to root translation and joint rotations.

Non-root translations are copied unchanged: they are neither scaled nor optimized. Check them explicitly when changing skeleton proportions. The function does not infer contacts or apply an independent final floor shift.

The returned SciPy `OptimizeResult` includes:

| Field | Meaning |
| --- | --- |
| `motion` | Final target animation. |
| `initial_motion_values` | Samples **after root fitting, before optimization**; not a separate FK-retarget baseline. |
| `constraint_residuals` | Named weighted residual arrays. |
| `control_points`, `parameter_names` | Fitted spline coefficients and their parameter identities. |
| `scale`, `constraint_tolerance` | Resolved scale and spatial tolerance in target units. |
| `success`, `message`, `nfev`, `cost` | Solver termination and optimization diagnostics. |

Weighted residuals are not distances in meters. Measure actual errors with FK:

```python
positions = compute_global_positions(result.motion)
foot = target_hierarchy.joint_index(target_foot_name)
contact = positions[contact_frames, foot]
height_error = contact[:, 1] - source_positions[contact_frames, source_foot, 1]
sliding = np.linalg.norm(np.diff(contact[:, [0, 2]], axis=0), axis=1)
penetration = np.maximum(floor_y - positions[:, foot, 1], 0)
print("maximum height error:", np.max(np.abs(height_error)))
print("maximum contact displacement per frame:", sliding.max())
print("maximum floor penetration:", penetration.max())
```

Inspect both these errors and playback. `success=True` means a solver termination criterion was met, not that every contact is acceptable. A low evaluation limit can stop before convergence. A final manual floor shift can invalidate solved position constraints.

## Detect contact intervals

The contact detector uses source world positions, foot-joint groups, and height and displacement thresholds. This is an alternative to manual annotation above:

```python
from broom.foot_lock import detect_foot_contacts, contact_segments
from broom.ops.skeleton_geometry import estimate_height

foot_groups = {
    "left": np.array([source.hierarchy.joint_index("LeftToeBase")]),
    "right": np.array([source.hierarchy.joint_index("RightToeBase")]),
}
contact_masks = detect_foot_contacts(
    source_positions,
    foot_groups,
    skeleton_height=estimate_height(source.hierarchy),
    height_threshold=None,
    velocity_threshold=None,
    velocity_dimensions=(0, 2),
    height_dimension=1,
)
for side, mask in contact_masks.items():
    for start, end in contact_segments(mask):
        frames = np.arange(start, end + 1)  # Both endpoints are included.
        print(side, frames)
```

Use each interval to construct constraints as above, with the corresponding target joint. Groups can contain multiple points on one foot. The detector uses their minimum height and the displacement of their mean horizontal position.

Despite the parameter name, `velocity_threshold` is a **displacement per frame**, not a speed per second. Defaults depend on skeleton height and include absolute length thresholds suited to meters. The detector estimates a floor from the clip; it does not read the plane passed to the solver. Review its output for airborne clips, unusual frame rates, and other units. For Z up, set `height_dimension=2` and `velocity_dimensions=(0, 1)`.

## Constraint reference

Every constraint contains `type` and `frames`. Joint names refer to the **target** hierarchy; frame indices refer to the motion being optimized. Optional `name` identifies residuals and must be unique; optional nonnegative `weight` controls its penalty. Frames must be valid, nonempty integer indices.

### Position

Fields: `joint`, `positions`, optional `axes` (default `XYZ`). Match selected world coordinates at each listed frame. With N frames and K selected axes, `positions` accepts `(K,)` for a constant target or `(N, K)` for per-frame targets. Full XYZ vectors `(3,)` or `(N, 3)` are also accepted; selected axes are extracted. For one constant height, use `[height]`, not a scalar.

A full XYZ position target pins a point in world space. Use it when that absolute location is intentional; it can conflict with proportion-dependent root travel.

### Stationary

Fields: `joint`, optional `axes` (default `XYZ`). Penalize coordinate differences between consecutive listed frames. Requires at least two strictly increasing frames. It does not specify an absolute location and does not divide differences by elapsed time. Small per-step errors can accumulate into drift.

### Floor

Fields: `joint`, optional `normal` (default `[0, 1, 0]`) and `offset` (default 0). The allowed side of the plane is `dot(position, unit_normal) >= offset`. Supply a unit normal and a signed plane offset in the motion's length units. The solver normalizes the normal but does not rescale the offset.

### Joint limits

Fields: `joint`, `minimum`, `maximum`. Bounds are **Euler angles in degrees**, in the joint's declared rotation-channel order, with one value per rotation channel. They are explicit constraints; BVH loading does not install limits automatically. The solver supports one-axis hinges and three-axis rotation groups, not two-axis groups. Euler limits retain Euler singularity/branch limitations even though rotational corrections are optimized in radians.

### Relational distance

Fields: `joint_a`, `joint_b`, `source_normalized_distance`, `target_path_length`, optional `activation`. The residual compares `distance(target_a, target_b) / target_path_length` with the supplied normalized source distance. `target_path_length` must be positive. Distance targets and nonnegative activation can be scalar or have one value per listed frame; activation defaults to 1 and multiplies the penalty weight.

For example, a normalized target of 0.1 and target path length of 1.2 meters request a separation of 0.12 meters. This constrains distance, not orientation or an exact shared contact point. Source-derived relations must be constructed explicitly; they are not inferred from an already-retargeted result. See the [constraint builders and solver API](api/spacetime.md).

## Tuning

- `control_point_spacing` is in frames. Smaller spacing permits more local corrections; larger spacing couples more frames. Splines smooth the **correction**, not necessarily jitter already present in the input motion.
- `control_weight` and `parameter_weights` penalize spline coefficients. Root translations use input length units; rotations use radians. The example's degree-to-radian weighting is a choice, not a universal optimum. Weights follow optimized channels in hierarchy/channel order: root translations and rotations.
- For position/stationary/floor constraints without explicit weights, retargeting uses `1 / (constraint_tolerance * scale)**2`. The default tolerance is 0.005; in meters, with scale 1, that is 5 mm. It sets a penalty scale, not a hard error bound. Explicit per-constraint weights override this default.
- `smoothing_sigma` is the root-preparation Gaussian width in frames.
- `max_nfev` limits evaluations; `ftol`, `xtol`, and `gtol` control termination. Increase the budget only after checking that the requested constraints are compatible and reachable.

## Refine a prepared motion

`solve_motion_spacetime` optimizes an animation already on the target hierarchy. For example, it can refine `result.motion` using additional target constraints:

```python
from broom.retargeting import solve_motion_spacetime

prepared_motion = result.motion
refined = solve_motion_spacetime(
    prepared_motion,
    constraints,
    parameter_weights=parameter_weights,
    constraint_tolerance=result.constraint_tolerance,
    control_point_spacing=4,
    max_nfev=60,
)
```

This is an alternative entry point, not a required second solve after retargeting. It shares mandatory root fitting and the optimizer but does not rescale again. Its default spatial weight is `1 / constraint_tolerance**2`; passing the resolved retarget tolerance above preserves the previous default penalty scale.

For different topologies, [mapped transfer and chain refinement](guides/retargeting.md) can prepare the target motion. Build constraints for **target joint indices/names and target dimensions**, while taking contact events and desired relations from the original source. The prepared motion supplies the reference for corrections, not just an arbitrary starting point. This workflow depends on the quality of that preparation and is not a guarantee of equivalent cross-topology retargeting.
