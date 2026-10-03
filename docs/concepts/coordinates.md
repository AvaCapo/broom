# Coordinates and timing

## Local transforms

For joint `j`, Broom applies animation to the rest transform:

```text
t_local = offset + d
R_local = O @ Q
p_global = p_parent + R_parent @ t_local
R_global = R_parent @ R_local
```

`O` is the rest orientation; `Q` is the animation rotation. `d` is expressed in **parent axes**, or skeleton axes for the root. Neither `O` nor `Q` rotates the joint's own translation. Rotating a joint moves its descendants; it does not move that joint's origin.

Motion rotation channels are Euler angles in **degrees**, composed in their declared intrinsic order. `Joint.local_orientation` uses **wxyz** quaternions; SciPy's default quaternion order is **xyzw**.

## Length units and external data

BVH does not prescribe length units: the same numeric coordinates may represent meters, centimeters, or another convention chosen by the exporter. Loading does not infer units or change coordinate axes. Use consistent units for offsets, position channels, targets, floor levels, and solver tolerances. Use meters for the human-motion workflows documented here.

Some algorithms, including space-time retargeting, have default tolerances intended for meter-scale human motion. Other units do not by themselves cause an exception, but the same numeric tolerance then describes a different physical error. This changes constraint weighting and can affect convergence and the resulting motion. Check tolerances, distance thresholds, and weights when using another scale.

```python
from broom.ops.motion_editing import scale_skeleton

# Only if the input is in centimeters:
meters = scale_skeleton(motion, 0.01)
```

This scales all offsets and translation channels, including non-root channels. Changing only hierarchy offsets is a different operation: animated translations remain unchanged. See [Editing](../guides/editing.md).

If an external exporter stores absolute root positions instead of additive root displacements, convert explicitly, once:

```python
from broom.ops.motion_editing import subtract_root_offset_from_translation

canonical = subtract_root_offset_from_translation(motion)
```

Do not apply this to data already using `offset + d`. The inverse operation is `add_root_offset_to_translation`. Neither operation aligns the character to a floor or changes units. Host-specific orientation and axis conversions belong at the import/export boundary; the BVH loader does not infer them.

## Time

`frame_time` is in seconds; FPS is `1 / frame_time`. `duration = (frame_count - 1) * frame_time`, the interval from the first sample to the last. A pose has zero duration. Resampling endpoint policies are described in [Interpolation](../guides/interpolation.md).
